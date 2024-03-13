"""
Helper classes for managing feature layers and datasets.  These class are not created by users directly.
Instances of this class, are available as a properties of feature layers and make it easier to manage them.
"""
from __future__ import absolute_import, annotations
import os
import json
import time
import logging
import tempfile
import collections
from enum import Enum
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import GIS, _GISResource, Item, ItemDependency
import concurrent.futures as _cf
from typing import Optional, Any, Union
from arcgis.auth.tools import LazyLoader
from dataclasses import dataclass
import datetime as _dt


features = LazyLoader("arcgis.features")
_version = LazyLoader("arcgis.features._version")
_common_utils = LazyLoader("arcgis._impl.common._utils")
re = LazyLoader("re")

_log = logging.getLogger()

# pylint: disable=protected-access
###########################################################################


class WebHookEvents(Enum):
    """
    Provides the allowed webhook enumerations for the captured events.
    """

    ALL = "*"
    FEATURESCREATED = "FeaturesCreated"
    FEATURESUPDATED = "FeaturesUpdated"
    FEATURESDELETED = "FeaturesDeleted"
    FEATURESEDITED = "FeaturesEdited"
    ATTACHMENTSCREATED = "AttachmentsCreated"
    ATTACHMENTSUPDATED = "AttachmentsUpdated"
    ATTACHMENTSDELETED = "AttachmentsDeleted"
    LAYERSCHEMACHANGED = "LayerSchemaChanged"
    LAYERDEFINITIONCHANGED = "LayerDefinitionChanged"
    FEATURESERVICEDEFINITIONCHANGED = "FeatureServiceDefinitionChanged"


###########################################################################
@dataclass
class WebHookScheduleInfo:
    """
    This dataclass provides information on how to schedule a webhook.


    =====================================    ===========================================================================
    **Parameter**                             **Description**
    -------------------------------------    ---------------------------------------------------------------------------
    name                                     Required string.  The name of the scheduling task.
    -------------------------------------    ---------------------------------------------------------------------------
    start_at                                 Required datetime.datetime. The start date.
    -------------------------------------    ---------------------------------------------------------------------------
    state                                    Optional String. The state of the task, this can be `enabled` or `disabled`.
    -------------------------------------    ---------------------------------------------------------------------------
    frequency                                Optional String. The default is `minute`. The time interval to run each task.
                                             The allows values are: second, minute, hour, day, week, month, year.
    -------------------------------------    ---------------------------------------------------------------------------
    interval                                 Optional Integer. The value for with the frequency describes.
    =====================================    ===========================================================================


    """

    name: str
    start_at: _dt.datetime
    state: str = "enabled"
    frequency: str = "minute"
    interval: int = 5

    def as_dict(self) -> dict[str, Any]:
        """returns the dataclass as a dictionary"""
        return {
            "name": self.name,
            "startAt": int(self.start_at.timestamp() * 1000),
            "recurrenceInfo": {
                "frequency": self.frequency,
                "interval": self.interval,
            },
        }


###########################################################################
class AttachmentManager(object):
    """
    Manager class for manipulating feature layer attachments.

    This class can be created by the user directly if a version is to be specified.

    Otherwise, an instance of this class, called 'attachments',
    is available as a property of the FeatureLayer object, if the layer supports attachments.
    Users call methods on this 'attachments' object to manipulate (create, get, list, delete) attachments.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    layer                   Required :class:`~arcgis.features.FeatureLayer` . The Feature Layer
                            that supports attachments.
    ---------------------   -------------------------------------------
    version                 Required Version or string. The `Version` class where
                            the branch version will take place or the
                            version name.
    =====================   ===========================================

    """

    def __init__(
        self,
        layer: features.FeatureLayer,
        version: str | _version.Version = None,
    ):
        self._layer = layer
        if isinstance(version, str):
            self._version = version
        elif isinstance(version, _version.Version):
            self._version = version.properties.versionName
        else:
            self._version = None

    def count(
        self,
        where: str | None = None,
        attachment_where: str | None = None,
        object_ids: str | None = None,
        global_ids: str | None = None,
        attachment_types: str | None = None,
        size: tuple[int] | list[int] | None = None,
        keywords: str | None = None,
    ) -> int:
        """"""
        url: str = "{}/{}".format(self._layer.url, "queryAttachments")
        if object_ids is None:
            object_ids = []
        if global_ids is None:
            global_ids = []
        if attachment_types is None:
            attachment_types = []
        if where is None:
            where = ""
        if keywords is None:
            keywords = []
        params: dict[str, Any] = {
            "f": "json",
            "definitionExpression": where,
            "attachmentTypes": ",".join(attachment_types),
            "objectIds": ",".join([str(v) for v in object_ids]),
            "globalIds": ",".join([str(v) for v in global_ids]),
            "definitionExpression": where,
            "attachmentsDefinitionExpression": attachment_where or "",
            "keywords": ",".join([str(v) for v in keywords]),
            "size": size,
            "returnCountOnly": True,
        }
        res = self._layer._con._session.get(url=url, params=params)
        res.raise_for_status()
        data: dict[str, Any] = res.json()
        if "attachmentGroups" in data:
            return sum([grp["count"] for grp in res.json()["attachmentGroups"]])
        elif "error" in data:
            raise Exception(data["error"])
        else:
            raise Exception(
                "Could not obtain the attachment counts, verify that attachments is enabled."
            )

    def search(
        self,
        where: str = "1=1",
        object_ids: str | None = None,
        global_ids: str | None = None,
        attachment_types: str | None = None,
        size: tuple[int] | list[int] | None = None,
        keywords: str | None = None,
        show_images: bool = False,
        as_df: bool = False,
        return_metadata: bool = False,
        return_url: bool = False,
        max_records: int | None = None,
        offset: int | None = None,
        *,
        attachment_where: str | None = None,
    ):
        """

        The `search` method allows querying the layer for its attachments and returns the results as
        a Pandas DataFrame or dict


        =========================   ===============================================================
        **Parameter**                **Description**
        -------------------------   ---------------------------------------------------------------
        where                       Required string.  The definition expression to be applied to
                                    the related layer/table. From the list of records that are
                                    related to the specified object Ids, only those records that
                                    conform to this expression will be returned.

                                    Example:

                                        where="STATE_NAME = 'Alaska'".

                                    The query results will return all attachments in Alaska.
        -------------------------   ---------------------------------------------------------------
        object_ids                  Optional list/string. The object IDs of this layer/table to be
                                    queried.

                                    Syntax:

                                        object_ids = <object_id1>, <object_id2>

                                    Example:

                                        object_ids = 2

                                    The query results will return attachments only for the specified object id.
        -------------------------   ---------------------------------------------------------------
        global_ids                  Optional list/string. The global IDs of this layer/table to be
                                    queried.

                                    Syntax:

                                        global_ids = <globalIds1>,<globalIds2>

                                    Example:

                                        global_ids = 6s430c5a-kb75-4d52-a0db-b30bg060f0b9, 35f0d027-8fc0-4905-a2f6-373c9600d017

                                    The query results will return attachments only for specified
                                    global id.
        -------------------------   ---------------------------------------------------------------
        attachment_types            Optional list/string. The file format that is supported by
                                    query attachment.

                                    Supported attachment types:
                                    bmp, ecw, emf, eps, ps, gif, img, jp2, jpc, j2k, jpf, jpg,
                                    jpeg, jpe, png, psd, raw, sid, tif, tiff, wmf, wps, avi, mpg,
                                    mpe, mpeg, mov, wmv, aif, mid, rmi, mp2, mp3, mp4, pma, mpv2,
                                    qt, ra, ram, wav, wma, doc, docx, dot, xls, xlsx, xlt, pdf, ppt,
                                    pptx, txt, zip, 7z, gz, gtar, tar, tgz, vrml, gml, json, xml,
                                    mdb, geodatabase

                                    Example:

                                        attachment_types='image/jpeg'
        -------------------------   ---------------------------------------------------------------
        size                        Optional tuple/list. The file size of the attachment is
                                    specified in bytes. You can enter a file size range
                                    (1000,15000) to query for attachments with the specified range.

                                    Example:

                                        size= (1000,15000)

                                    The query results will return all attachments within the
                                    specified file size range (1000 - 15000) bytes.
        -------------------------   ---------------------------------------------------------------
        keywords                    Optional string.  When attachments are uploaded, keywords can
                                    be assigned to the uploaded file.  By passing a keyword value,
                                    the values will be searched.

                                    Example:

                                        keywords='airplanes'
        -------------------------   ---------------------------------------------------------------
        show_images                 Optional bool. The default is False, when the value is True,
                                    the results will be displayed as a HTML table. If the as_df is
                                    set to False, this parameter will be ignored.
        -------------------------   ---------------------------------------------------------------
        as_df                       Optional bool. Default is False, if True, the results will be
                                    a Pandas' DataFrame.  If False, the values will be a list of
                                    dictionary values.
        -------------------------   ---------------------------------------------------------------
        return_metadata             Optional Boolean. If true, metadata stored in the `exifInfo`
                                    column will be returned for attachments that have `exifInfo`.
                                    This option is supported only when "name": "exifInfo" in the
                                    layer's attachmentProperties includes "isEnabled": true. When
                                    set to false, or not set, None is returned for `exifInfo`.
        -------------------------   ---------------------------------------------------------------
        return_url                  Optional Boolean. Specifies whether to return the attachment
                                    URL. The default is false. This parameter is supported if the
                                    `supportsQueryAttachmentsWithReturnUrl` property is true on the
                                    layer. Applications can use this URL to download the attachment
                                    image.
        -------------------------   ---------------------------------------------------------------
        max_records                 Optional Integer. This option fetches query results up to the
                                    `resultRecordCount` specified. When `resultOffset` is specified
                                    and this parameter is not, the feature service defaults to the
                                    `maxRecordCount`. The maximum value for this parameter is the
                                    value of the layer's `maxRecordCount` property. This parameter
                                    only applies if `supportPagination` is true.
        -------------------------   ---------------------------------------------------------------
        offset                      Optional Integer. This parameter is designed to be used in
                                    conjunction with `max_records` to page through a long list of
                                    attachments, one request at a time. This option fetches query
                                    results by skipping a specified number of records. The query
                                    results start from the next record (i.e., resultOffset + 1).
                                    The default value is 0. This parameter only applies when
                                    `supportPagination` is true. You can use this option to fetch
                                    records that are beyond `maxRecordCount` property.
        -------------------------   ---------------------------------------------------------------
        attachment_where            Optional str. The definition expression to be applied to the
                                    attachments table. Only those records that conform to this
                                    expression will be returned. You can get the attachments table
                                    field names to use in the expression by checking the layer's
                                    `attachmentProperties`.
        =========================   ===============================================================

        :return: A Pandas DataFrame or Dict of the attachments of the :class:`~arcgis.features.FeatureLayer`

        """
        import copy

        columns = [
            col.upper()
            for col in [
                "ParentObjectid",
                "ParentGlobalId",
                "Id",
                "Name",
                "GlobalId",
                "ContentType",
                "Size",
                "KeyWords",
                "URL",
                "IMAGE_PREVIEW",
            ]
        ]
        result_offset = 0
        if keywords is None:
            keywords = []
        elif isinstance(keywords, str):
            keywords = keywords.split(",")
        if object_ids is None:
            object_ids = []
        elif isinstance(object_ids, str):
            object_ids = object_ids.split(",")
        if global_ids is None:
            global_ids = []
        elif isinstance(global_ids, str):
            global_ids = global_ids.split(",")
        if attachment_types is None:
            attachment_types = []
        elif isinstance(attachment_types, str):
            attachment_types = attachment_types.split(",")
        if isinstance(size, (tuple, list)):
            size = ",".join(list([str(s) for s in size]))
        elif size is None:
            size = None
        if (
            self._layer._gis._portal.is_arcgisonline == False
            and self._layer.properties.hasAttachments
        ):
            rows = []

            query = self._layer.query(
                where=where,
                object_ids=",".join(map(str, object_ids)),
                global_ids=",".join(global_ids),
                return_ids_only=True,
            )
            if "objectIds" in query:
                token = self._layer._con.token
                for i in query["objectIds"]:
                    attachments = self.get_list(oid=i)
                    for att in attachments:
                        if not token is None:
                            att_path = "{}/{}/attachments/{}?token={}".format(
                                self._layer.url,
                                i,
                                att["id"],
                                self._layer._con.token,
                            )
                        else:
                            att_path = "{}/{}/attachments/{}".format(
                                self._layer.url, i, att["id"]
                            )
                        preview = None
                        if att["contentType"].find("image") > -1:
                            preview = (
                                '<img src="' + att_path + '" width=150 height=150 />'
                            )

                        row = {
                            "PARENTOBJECTID": i,
                            "PARENTGLOBALID": "N/A",
                            "ID": att["id"],
                            "NAME": att["name"],
                            "CONTENTTYPE": att["contentType"],
                            "SIZE": att["size"],
                            "KEYWORDS": "",
                            "IMAGE_PREVIEW": preview,
                        }
                        if "globalId" in att:
                            row["GLOBALID"] = att["globalId"]
                        if as_df and show_images:
                            row["DOWNLOAD_URL"] = (
                                '<a href="%s" target="_blank">DATA</a>' % att_path
                            )
                        else:
                            row["DOWNLOAD_URL"] = "%s" % att_path
                        rows.append(row)

                if (
                    attachment_types is not None and len(attachment_types) > 0
                ):  # performs contenttype search
                    if isinstance(attachment_types, str):
                        attachment_types = attachment_types.split(",")
                    rows = [
                        row
                        for row in rows
                        if os.path.splitext(row["NAME"])[1][1:] in attachment_types
                        or row["CONTENTTYPE"] in attachment_types
                    ]
        else:
            url = "{}/{}".format(self._layer.url, "queryAttachments")

            params = {
                "f": "json",
                "attachmentTypes": ",".join(attachment_types),
                "objectIds": ",".join([str(v) for v in object_ids]),
                "globalIds": ",".join([str(v) for v in global_ids]),
                "definitionExpression": where,
                "keywords": ",".join([str(v) for v in keywords]),
                "size": size,
                "returnMetadata": return_metadata,
                "returnUrl": return_url,
                "resultRecordCount": max_records,
                "resultOffset": offset,
            }
            if offset:
                params["offset"] = offset
            if attachment_where:
                params["attachmentsDefinitionExpression"] = attachment_where or ""

            iterparams = copy.copy(params)
            for k, v in iterparams.items():
                if k in ["objectIds", "globalIds", "attachmentTypes"] and v == "":
                    del params[k]
                elif k == "size" and v is None:
                    del params[k]

            results = self._layer._con.post(url, params)
            rows = []
            if "attachmentGroups" not in results:
                return []
            for result in results["attachmentGroups"]:
                for data in result["attachmentInfos"]:
                    token = self._layer._con.token
                    if not token is None:
                        att_path = "{}/{}/attachments/{}?token={}".format(
                            self._layer.url,
                            result["parentObjectId"],
                            data["id"],
                            self._layer._con.token,
                        )
                    else:
                        att_path = "{}/{}/attachments/{}".format(
                            self._layer.url,
                            result["parentObjectId"],
                            data["id"],
                        )
                    preview = None
                    if data["contentType"].find("image") > -1:
                        preview = '<img src="' + att_path + '" width=150 height=150 />'

                    row = {
                        "PARENTOBJECTID": result["parentObjectId"],
                        "PARENTGLOBALID": result["parentGlobalId"],
                        "ID": data["id"],
                        "NAME": data["name"],
                        "CONTENTTYPE": data["contentType"],
                        "SIZE": data["size"],
                        "KEYWORDS": data["keywords"],
                        "IMAGE_PREVIEW": preview,
                    }
                    if "globalId" in data:
                        row["GLOBALID"] = data["globalId"]
                    if as_df and show_images:
                        row["DOWNLOAD_URL"] = (
                            '<a href="%s" target="_blank">DATA</a>' % att_path
                        )
                    else:
                        row["DOWNLOAD_URL"] = "%s" % att_path
                    rows.append(row)
                    del row

        if as_df == True:
            import pandas as pd

            if show_images:
                from IPython.display import HTML

                pd.set_option("display.max_colwidth", -1)
                return HTML(pd.DataFrame.from_dict(rows).to_html(escape=False))
            else:
                if len(rows) == 0:
                    return pd.DataFrame()
                df = pd.DataFrame.from_dict(rows)
                df.drop(
                    ["DOWNLOAD_URL", "IMAGE_PREVIEW"],
                    axis=1,
                    inplace=True,
                    errors="ignore",
                )
                return df
        else:
            return rows

    def _download_all(self, object_ids=None, save_folder=None, attachment_types=None):
        """
        Downloads all attachments to a specific folder

        =========================   ===============================================================
        **Arguement**               **Description**
        -------------------------   ---------------------------------------------------------------
        object_ids                  optional list. A list of object_ids to download data from.
        -------------------------   ---------------------------------------------------------------
        save_folder                 optional string. Path to save data to.
        -------------------------   ---------------------------------------------------------------
        attachment_types            optional string.  Allows the limitation of file types by passing
                                    a string of the item type.

                                    **Example:** image/jpeg
        =========================   ===============================================================

        :return: path to the file where the attachements have downloaded

        """
        results = []
        if save_folder is None:
            save_folder = os.path.join(tempfile.gettempdir(), "attachment_download")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        attachments = self.search(
            object_ids=object_ids,
            attachment_types=attachment_types,
            as_df=True,
        )
        for row in attachments.to_dict("records"):
            dlpath = os.path.join(
                save_folder,
                "%s" % int(row["PARENTOBJECTID"]),
                "%s" % int(row["ID"]),
            )
            if os.path.isdir(dlpath) == False:
                os.makedirs(dlpath)
            path = self.download(
                oid=int(row["PARENTOBJECTID"]),
                attachment_id=int(row["ID"]),
                save_path=dlpath,
            )
            results.append(path[0])
            del row
        return results

    def get_list(self, oid: str):
        """
        Get the list of attachements for a given OBJECT ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        oid                 Required string of the object id
        ===============     ====================================================================

        :result:
            A list of attachements

        """
        return self._layer._list_attachments(oid)["attachmentInfos"]

    def download(
        self,
        oid: str | None = None,
        attachment_id: str | None = None,
        save_path: str | None = None,
    ):
        """
        Downloads attachment and returns its path on disk.

        The download tool works as follows:

        * If nothing is given, all attachments will be downloaded

          Example: download()

        * If a single oid and attachment_id are given, the single file will download

        * If a list of oid values are given, all the attachments for those object ids will be saved locally.

        =========================   ===============================================================
        **Argument**                **Description**
        -------------------------   ---------------------------------------------------------------
        oid                         Optional list/string. A list of object Ids or a single value
                                    to download data from.
        -------------------------   ---------------------------------------------------------------
        attachment_id               Optional string. Id of the attachment to download. This is only
                                    honored if return_all is False.
        -------------------------   ---------------------------------------------------------------
        save_path                   Optional string. Path to save data to.
        =========================   ===============================================================

        :return: A path to the folder where the attachement are saved


        """
        return_all = False
        if isinstance(oid, str):
            oid = oid.split(",")
            oid_len = len(oid)
        elif isinstance(oid, int):
            oid = str(oid).split(",")
            oid_len = 1
        elif isinstance(oid, (tuple, list)):
            oid_len = len(oid)
        elif oid is None:
            oid_len = 0
        else:
            raise ValueError("oid must be of type list or string")
        if isinstance(attachment_id, str):
            attachment_id = [int(att) for att in attachment_id.split(",")]
            att_len = len(attachment_id)
        elif isinstance(attachment_id, int):
            attachment_id = str(attachment_id).split(",")
            att_len = 1
        elif isinstance(attachment_id, (tuple, list)):
            att_len = len(attachment_id)
        elif attachment_id is None:
            att_len = 0
        else:
            raise ValueError("attachment_id must be of type list or string")
        if oid_len == 1 and att_len > 0:
            return_all = False
        elif oid_len > 1 and att_len > 0:
            raise ValueError(
                "You cannot provide more than one oid when providing attachment_id values."
            )
        else:
            return_all = True

        if not return_all:
            oid = oid[0]
            paths = []
            for att in attachment_id:
                att_path = "{}/{}/attachments/{}".format(self._layer.url, oid, att)
                att_list = self.get_list(int(oid))

                # get attachment file name
                desired_att = [att2 for att2 in att_list if att2["id"] == int(att)]
                if len(desired_att) == 0:  # bad attachment id
                    raise RuntimeError
                else:
                    att_name = desired_att[0]["name"]

                if not save_path:
                    save_path = tempfile.gettempdir()
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                path = self._layer._con.get(
                    path=att_path,
                    try_json=False,
                    out_folder=save_path,
                    file_name=att_name,
                    token=self._layer._token,
                    force_bytes=False,
                )
                paths.append(path)
            return paths
        else:
            return self._download_all(object_ids=oid, save_folder=save_path)

    def add(
        self,
        oid: str,
        file_path: str,
        keywords: str | None = None,
        return_moment: bool = False,
    ) -> bool:
        """
        Adds an attachment to a :class:`~arcgis.features.FeatureLayer`. Adding an attachment
        is a feature update, but is support with either the Update or the Create capability.
        The add operation is performed on a feature service feature resource.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        oid                 Required string of the object ID.
        ---------------     --------------------------------------------------------------------
        file_path           Required string. Path to the file to be uploaded as a new feature
                            attachment. The content type, size, and name of the attachment will
                            be derived from the uploaded file.
        ---------------     --------------------------------------------------------------------
        keywords            Optional string. Sets a text value that is stored as the keywords
                            value for the attachment. This parameter can be set when the layer has
                            an attachmentProperties property that includes "name": "keywords"
                            with "isEnabled": True. If the attachments have keywords enabled
                            and the layer also includes the attachmentFields property,
                            you can use it to understand properties like keywords field length.
        ---------------     --------------------------------------------------------------------
        return_moment       Optional bool. Specify whether the response will report the time
                            attachments were added. If True, the server will return the time
                            in the response's `editMoment` key. The default is False.
        ===============     ====================================================================

        :return:
            A JSON Response stating 'success' or 'error'

        """
        return self._layer._add_attachment(
            oid,
            file_path,
            keywords=keywords,
            return_moment=return_moment,
            version=self._version,
        )

    def delete(
        self,
        oid: str,
        attachment_id: str,
        return_moment: bool = False,
        rollback_on_failure: bool = True,
    ) -> bool:
        """
        Removes an attachment from a :class:`~arcgis.features.FeatureLayer` . Deleting an attachment is a feature update;
        it requires the Update capability. The deleteAttachments operation is performed on a
        feature service feature resource. This operation is available only if the layer has advertised that it has attachments.
        A layer has attachments if its hasAttachments property is true.

        ===================     ====================================================================
        **Parameter**            **Description**
        -------------------     --------------------------------------------------------------------
        oid                     Required string of the object ID
        -------------------     --------------------------------------------------------------------
        attachment_id           Required string. Ids of attachment to delete.

                                Syntax:

                                    attachment_id = "<attachmentId1>, <attachmentId2>"
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

        :result:
           JSON response stating 'success' or 'error'
        """
        return self._layer._delete_attachment(
            oid,
            attachment_id,
            return_moment=return_moment,
            rollback_on_failure=rollback_on_failure,
            version=self._version,
        )

    def update(
        self,
        oid: str,
        attachment_id: str,
        file_path: str,
        return_moment: bool = False,
    ) -> bool:
        """
        Updates an existing attachment with a new file

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        oid                 Required string of the object ID.
        ---------------     --------------------------------------------------------------------
        attachment_id       Required string. Id of the attachment to update.
        ---------------     --------------------------------------------------------------------
        file_path           Required string. Path to file to be uploaded as the updated feature
                            attachment.
        ---------------     --------------------------------------------------------------------
        return_moment       Optional boolean. Specify whether the response will report the time
                            attachments were deleted. If True, the server will report the time
                            in the response's `editMoment` key. The default value is False.
        ===============     ====================================================================

        :result:
           JSON response stating 'success' or 'error'
        """
        return self._layer._update_attachment(
            oid,
            attachment_id,
            file_path,
            return_moment=return_moment,
            version=self._version,
        )


###########################################################################
class SyncManager(object):
    """
    Manager class for manipulating replicas for syncing disconnected editing of :class:`~arcgis.features.FeatureLayer` .
    This class is not created by users directly.
    An instance of this class, called 'replicas', is available as a property of the :class:`~arcgis.features.FeatureLayerCollection` object,
    if the layer is sync enabled / supports disconnected editing.
    Users call methods on this 'replicas' object to manipulate (create, synchronize, unregister) replicas.
    """

    # http://services.arcgis.com/help/fsDisconnectedEditing.html
    def __init__(self, featsvc):
        self._fs = featsvc

    def get_list(self):
        """returns all the replicas for the feature layer collection"""
        return self._fs._replicas

    # ----------------------------------------------------------------------
    def unregister(self, replica_id: str):
        """
        unregisters a replica from a feature layer collection

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        replica_id          The replicaID returned by the feature service when the replica was created.
        ===============     ====================================================================

        """
        return self._fs._unregister_replica(replica_id)

    # ----------------------------------------------------------------------
    def get(self, replica_id: str):
        """
        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        replica_id          Required string. replica_id returned by the feature service when
                            the replica was created.
        ===============     ====================================================================

        :return:
            The replica information
        """
        return self._fs._replica_info(replica_id)

    # ----------------------------------------------------------------------
    def create(
        self,
        replica_name: str,
        layers: list[int],
        layer_queries: dict[str, Any] | None = None,
        geometry_filter: dict[str, str] | None = None,
        replica_sr: dict[str, Any] | int | None = None,
        transport_type: str = "esriTransportTypeUrl",
        return_attachments: bool = False,
        return_attachments_databy_url: bool = False,
        asynchronous: bool = False,
        attachments_sync_direction: str = "none",
        sync_model: str = "none",
        data_format: str = "json",
        replica_options: dict[str, Any] | None = None,
        wait: bool = False,
        out_path: str | None = None,
        sync_direction: str | None = None,
        target_type: str = "client",
        transformations: list[str] | None = None,
        time_reference_unknown_client: bool | None = None,
    ):
        """
        The create operation is performed on a :class:`~arcgis.features.FeatureLayerCollection` resource.
        This operation creates the replica between the feature dataset and a client based on a client-supplied
        replica definition. It requires the Sync capability. See Sync overview for more
        information on sync. The response for create includes replicaID, replica generation
        number, and data similar to the response from the :meth:`~arcgis.features.FeatureLayerCollection.query`
        operation. The create operation returns a response of type esriReplicaResponseTypeData,
        as the response has data for the layers in the replica. If the operation is called to
        register existing data by using replicaOptions, the response type will be
        esriReplicaResponseTypeInfo, and the response will not contain data for the layers in
        the replica.


        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        replica_name                        Required string. Name of the replica.
        -----------------------------       --------------------------------------------------------------------
        layers                              Required list. A list of layers and tables to include in the replica.
        -----------------------------       --------------------------------------------------------------------
        layer_queries                       Optional dictionary. In addition to the layers and geometry
                                            parameters, the layer_queries parameter can be used to further define
                                            what is replicated. This parameter allows you to set properties on a
                                            per layer or per table basis. Only the properties for the layers and
                                            tables that you want changed from the default are required.

                                            Example:

                                                | layer_queries = {"0":{"queryOption": "useFilter", "useGeometry": true,
                                                | "where": "requires_inspection = Yes"}}
        -----------------------------       --------------------------------------------------------------------
        geometry_filter                     Optional dictionary. Spatial filter from :mod:`~arcgis.geometry.filters` module
                                            to filter results by a spatial relationship with another geometry.
        -----------------------------       --------------------------------------------------------------------
        replica_sr                          Optional WKID or a spatial reference JSON object. The spatial
                                            reference of the replica geometry.
        -----------------------------       --------------------------------------------------------------------
        transport_type                      The transport_type represents the response format. If the
                                            transport_type is esriTransportTypeUrl, the JSON response is contained
                                            in a file, and the URL link to the file is returned. Otherwise, the
                                            JSON object is returned directly. The default is esriTransportTypeUrl.
                                            If async is true, the results will always be returned as if
                                            transport_type is esriTransportTypeUrl. If dataFormat is sqlite, the
                                            transportFormat will always be esriTransportTypeUrl regardless of how
                                            the parameter is set.

                                            Values:

                                                esriTransportTypeUrl | esriTransportTypeEmbedded.
        -----------------------------       --------------------------------------------------------------------
        return_attachments                  If True, attachments are added to the replica and returned in the
                                            response. Otherwise, attachments are not included. The default is
                                            False. This parameter is only applicable if the feature service has
                                            attachments.
        -----------------------------       --------------------------------------------------------------------
        return_attachments_databy_url       If True, a reference to a URL will be provided for each attachment
                                            returned from create method. Otherwise, attachments are embedded in
                                            the response. The default is True. This parameter is only applicable
                                            if the feature service has attachments and if return_attachments is
                                            True.
        -----------------------------       --------------------------------------------------------------------
        asynchronous                        If True, the request is processed as an asynchronous job, and a URL
                                            is returned that a client can visit to check the status of the job.
                                            See the topic on asynchronous usage for more information. The default
                                            is False.
        -----------------------------       --------------------------------------------------------------------
        attachments_sync_direction          Client can specify the attachmentsSyncDirection when creating a
                                            replica. AttachmentsSyncDirection is currently a createReplica property
                                            and cannot be overridden during sync.

                                            Values:

                                                none, upload, bidirectional
        -----------------------------       --------------------------------------------------------------------
        sync_model                          This parameter is used to indicate that the replica is being created
                                            for per-layer sync or per-replica sync. To determine which model types
                                            are supported by a service, query the supportsPerReplicaSync,
                                            supportsPerLayerSync, and supportsSyncModelNone properties of the Feature
                                            Service. By default, a replica is created for per-replica sync.
                                            If syncModel is perReplica, the syncDirection specified during sync
                                            applies to all layers in the replica. If the syncModel is perLayer, the
                                            syncDirection is defined on a layer-by-layer basis.

                                            If sync_model is perReplica, the response will have replicaServerGen.
                                            A perReplica sync_model requires the replicaServerGen on sync. The
                                            replicaServerGen tells the server the point in time from which to send
                                            back changes. If sync_model is perLayer, the response will include an
                                            array of server generation numbers for the layers in layerServerGens. A
                                            perLayer sync model requires the layerServerGens on sync. The
                                            layerServerGens tell the server the point in time from which to send
                                            back changes for a specific layer. sync_model=none can be used to export
                                            the data without creating a replica. Query the supportsSyncModelNone
                                            property of the feature service to see if this model type is supported.

                                            See the `RollbackOnFailure and Sync Models <https://developers.arcgis.com/rest/services-reference/enterprise/rollbackonfailure-and-sync-models.htm>`_ topic for more details.

                                            Values:

                                                perReplica | perLayer | none

                                            Example:

                                                sync_model = perLayer
        -----------------------------       --------------------------------------------------------------------
        data_format                         The format of the replica geodatabase returned in the response. The
                                            default is json.

                                            Values:

                                                filegdb, json, sqlite, shapefile
        -----------------------------       --------------------------------------------------------------------
        replica_options                     This parameter instructs the create operation to create a new replica
                                            based on an existing replica definition (refReplicaId). It can be used
                                            to specify parameters for registration of existing data for sync. The
                                            operation will create a replica but will not return data. The
                                            responseType returned in the create response will be
                                            esriReplicaResponseTypeInfo.
        -----------------------------       --------------------------------------------------------------------
        wait                                If async, wait to pause the process until the async operation is completed.
        -----------------------------       --------------------------------------------------------------------
        out_path                            out_path - folder path to save the file.
        -----------------------------       --------------------------------------------------------------------
        sync_direction                      Defaults to bidirectional when the targetType is client and download
                                            when the target_type is server. If set, only bidirectional is supported
                                            when target_type is client. If set, only upload or download are
                                            supported when target_type is server.

                                            Values:

                                                download | upload | bidirectional

                                            Example:

                                                sync_direction=download
        -----------------------------       --------------------------------------------------------------------
        target_type                         Can be set to either server or client. If not set, the default is
                                            client. This option was added at 10.5.1.
        -----------------------------       --------------------------------------------------------------------
        transformations                     Optional List. Introduced at 10.8. This parameter applies a datum
                                            transformation on each layer when the spatial reference used in
                                            geometry is different from the layer's spatial reference.
        -----------------------------       --------------------------------------------------------------------
        time_reference_unknown_client       Setting time_reference_unknown_client as true indicates that the client is
                                            capable of working with data values that are not in UTC. If its not set
                                            to true, and the service layer's datesInUnknownTimeZone property is true,
                                            then an error is returned. The default is false

                                            It's possible to define a service's time zone of date fields as unknown.
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
        =============================       ====================================================================


        :return:
           JSON response if POST request made successfully. Otherwise, return None.


        .. code-block:: python  (optional)

           # USAGE EXAMPLE: Create a replica on server with geometry_filter specified.

           geom_filter = {'geometry':'8608022.3,1006191.2,8937015.9,1498443.1',
                          'geometryType':'esriGeometryEnvelope'}

           fs.replicas.create(replica_name='your_replica_name',
                              layers=[0],
                              geometry_filter=geom_filter,
                              attachments_sync_direction=None,
                              transport_type="esriTransportTypeUrl",
                              return_attachments=True,
                              return_attachments_databy_url=True,
                              asynchronous=True,
                              sync_model="perLayer",
                              target_type="server",
                              data_format="sqlite",
                              out_path=r'/arcgis/home',
                              wait=True)

        """
        if geometry_filter is None:
            extents = self._fs.properties["fullExtent"]
            extents_str = ",".join(
                format(x, "10.3f")
                for x in [
                    extents["xmin"],
                    extents["ymin"],
                    extents["xmax"],
                    extents["ymax"],
                ]
            )
            geometry_filter = {"geometryType": "esriGeometryEnvelope"}
            geometry_filter.update({"geometry": extents_str})
        if not layer_queries:
            # Assures correct number of record counts are returned
            layer_queries = {}
            for layer in layers:
                layer_queries[str(layer)] = {"queryOption": "all"}

        return self._fs._create_replica(
            replica_name=replica_name,
            layers=layers,
            layer_queries=layer_queries,
            geometry_filter=geometry_filter,
            replica_sr=replica_sr,
            transport_type=transport_type,
            return_attachments=return_attachments,
            return_attachments_data_by_url=return_attachments_databy_url,
            asynchronous=asynchronous,
            sync_direction=sync_direction,
            target_type=target_type,
            attachments_sync_direction=attachments_sync_direction,
            sync_model=sync_model,
            data_format=data_format,
            replica_options=replica_options,
            wait=wait,
            out_path=out_path,
            transformations=transformations,
            time_reference_unknown_client=time_reference_unknown_client,
        )

    # ----------------------------------------------------------------------
    def cleanup_change_tracking(
        self,
        layers: list[int],
        retention_period: int,
        period_unit: str = "days",
        min_server_gen: str | None = None,
        replica_id: str | None = None,
        future: bool = False,
    ):
        """

        Change tracking information stored in each feature service layer
        (enabled for Change Tracking) might grow very large. The change
        tracking info used by the feature service to determine the change
        generation number and the features that have changed for a
        particular generation. Clients can purge the change tracking
        content if the changes are already synced-up to all clients and the
        changes are no longer needed.

        Only the owner or the organization administrator can cleanup change
        tracking information.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        layers                 Required list. A list of layers and tables to include in the replica.
        ------------------     --------------------------------------------------------------------
        retention_period       Optional Integer. The retention period to use when cleaning up the
                               change tracking information. Change tracking information will be
                               cleaned up if they are older than the retention period.
        ------------------     --------------------------------------------------------------------
        period_unit            Optional String.  The units of the retention period.

                               Values:

                                    `days`, `seconds`, `minutes`, or `hours`

        ------------------     --------------------------------------------------------------------
        min_server_gen         Optional String.  In addition to the retention period, the change
                               tracking can be cleaned by its generation numbers. Older tracking
                               information that has older generation number than the
                               `min_server_gen` will be cleaned.
        ------------------     --------------------------------------------------------------------
        replica_id             Optional String.  The change tracking can also be cleaned by the
                               `replica_id` in addition to the `retention_period` and the
                               `min_server_gen`.
        ------------------     --------------------------------------------------------------------
        future                 Optional boolean. If True, a future object will be returned and the process
                               will not wait for the task to complete. The default is False, which means wait for results.
        ==================     ====================================================================


        :return:
            Boolean or If ``future = True``, then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.

        """
        return self._fs._cleanup_change_tracking(
            layers=layers,
            retention_period=retention_period,
            period_unit=period_unit,
            min_server_gen=min_server_gen,
            replica_id=replica_id,
            future=future,
        )

    # ----------------------------------------------------------------------
    def synchronize(
        self,
        replica_id: str,
        transport_type: str = "esriTransportTypeUrl",
        replica_server_gen: int | None = None,
        return_ids_for_adds: bool = False,
        edits: list[dict[str, Any]] | None = None,
        return_attachment_databy_url: bool = False,
        asynchronous: bool = False,
        sync_direction: str = "snapshot",
        sync_layers: str = "perReplica",
        edits_upload_id: dict | None = None,
        edits_upload_format: str | None = None,
        data_format: str = "json",
        rollback_on_failure: bool = True,
    ):
        """
        synchronizes replica with feature layer collection
        See `Synchronize Replica <https://developers.arcgis.com/rest/services-reference/synchronize-replica.htm>`_
        """
        # TODO:
        return self._fs._synchronize_replica(
            replica_id=replica_id,
            transport_type=transport_type,
            replica_server_gen=replica_server_gen,
            return_ids_for_adds=return_ids_for_adds,
            edits=edits,
            return_attachment_databy_url=return_attachment_databy_url,
            asynchronous=asynchronous,
            sync_direction=sync_direction,
            sync_layers=sync_layers,
            edits_upload_id=edits_upload_id,
            edits_upload_format=edits_upload_format,
            data_format=data_format,
            rollback_on_failure=rollback_on_failure,
            close_replica=False,
            out_path=None,
        )

    def create_replica_item(
        self,
        replica_name: str,
        item: Item,
        destination_gis: GIS,
        layers: list[int] | None = None,
        extent: dict[str, Any] | None = None,
    ):
        """
        Creates a replicated service from a parent to another GIS.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        replica_name        Optional string. Name for replicated item in other GIS
        ---------------     --------------------------------------------------------------------
        item                Required Item to replicate
        ---------------     --------------------------------------------------------------------
        destination_gis     Required :class:`~arcgis.gis.GIS` object
        ---------------     --------------------------------------------------------------------
        layers              Optional list. Layers to replicate in the item
        ---------------     --------------------------------------------------------------------
        extent              Optional dict. Depicts the geometry extent for an item.
        ===============     ====================================================================

        :return:
            The published replica item created
        """
        import tempfile
        import os
        from ..gis import Item

        fs = item.layers[0].container
        if layers is None:
            ls = fs.properties["layers"]
            ts = fs.properties["tables"]
            layers = ""
            for i in ls + ts:
                layers += str(i["id"])
        if extent is None:
            extent = fs.properties["fullExtent"]
            if "spatialReference" in extent:
                del extent["spatialReference"]
        extents_str = ",".join(
            format(x, "10.3f")
            for x in [
                extent["xmin"],
                extent["ymin"],
                extent["xmax"],
                extent["ymax"],
            ]
        )
        geom_filter = {"geometryType": "esriGeometryEnvelope"}
        geom_filter.update({"geometry": extents_str})

        out_path = tempfile.gettempdir()
        from . import FeatureLayerCollection

        isinstance(fs, FeatureLayerCollection)
        db = fs._create_replica(
            replica_name=replica_name,
            layers=layers,
            geometry_filter=geom_filter,
            attachments_sync_direction=None,
            transport_type="esriTransportTypeUrl",
            return_attachments=True,
            return_attachments_data_by_url=True,
            asynchronous=True,
            sync_model="perLayer",
            target_type="server",
            data_format="sqlite",
            # target_type="server",
            out_path=out_path,
            wait=True,
        )
        if os.path.isfile(db) == False:
            raise Exception("Could not create the replica")
        destination_content = destination_gis.content
        item = destination_content.add(
            item_properties={
                "type": "SQLite Geodatabase",
                "tags": "replication",
                "title": replica_name,
            },
            data=db,
        )
        published = item.publish()
        return published

    def sync_replicated_items(self, parent: Item, child: Item, replica_name: str):
        """
        Synchronizes two replicated items between portals

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        parent              Required :class:`~arcgis.gis.Item` that points to the feature service
                            that is the parent dataset. (source)
        ---------------     --------------------------------------------------------------------
        child               Required :class:`~arcgis.gis.Item` that points to the feature service
                            that is the child dataset. (target)
        ---------------     --------------------------------------------------------------------
        replica_name        Required string. Name of either parent or child Item
        ===============     ====================================================================

        :result:
            Boolean value. True means service is up to date/synchronized,
            False means the synchronization failed.

        """
        from ..gis import Item

        if isinstance(parent, Item) == False:
            raise ValueError("parent must be an Item")
        if isinstance(child, Item) == False:
            raise ValueError("child must be an Item")
        child_fs = child.layers[0].container
        parent_fs = parent.layers[0].container
        child_replicas = child_fs.replicas
        parent_replicas = parent_fs.replicas
        if child_replicas and parent_replicas:
            child_replica_id = None
            parent_replica_id = None
            child_replica = None
            parent_replica = None
            for replica in child_replicas.get_list():
                if replica["replicaName"].lower() == replica_name.lower():
                    child_replica_id = replica["replicaID"]
                    break
            for replica in parent_replicas.get_list():
                if replica["replicaName"].lower() == replica_name.lower():
                    parent_replica_id = replica["replicaID"]
                    break
            if child_replica_id and parent_replica_id:
                import tempfile
                import os

                child_replica = child_replicas.get(replica_id=child_replica_id)
                parent_replica = parent_replicas.get(replica_id=parent_replica_id)
                delta = parent_fs._synchronize_replica(
                    replica_id=parent_replica_id,
                    transport_type="esriTransportTypeUrl",
                    close_replica=False,
                    return_ids_for_adds=False,
                    return_attachment_databy_url=True,
                    asynchronous=False,
                    sync_direction="download",
                    sync_layers=parent_replica["layerServerGens"],
                    edits_upload_format="sqlite",
                    data_format="sqlite",
                    rollback_on_failure=False,
                    out_path=tempfile.gettempdir(),
                )
                if os.path.isfile(delta) == False:
                    return True
                work, message = child_fs.upload(path=delta)
                if (
                    isinstance(message, dict)
                    and "item" in message
                    and "itemID" in message["item"]
                ):
                    syncLayers_child = child_replica["layerServerGens"]
                    syncLayers_parent = parent_replica["layerServerGens"]
                    for i in range(len(syncLayers_parent)):
                        syncLayers_child[i]["serverSibGen"] = syncLayers_parent[i][
                            "serverGen"
                        ]
                        syncLayers_child[i]["syncDirection"] = "upload"
                    child_fs._synchronize_replica(
                        replica_id=child_replica_id,
                        sync_layers=syncLayers_child,
                        sync_direction=None,
                        edits_upload_id=message["item"]["itemID"],
                        return_ids_for_adds=False,
                        data_format="sqlite",
                        asynchronous=False,
                        edits_upload_format="sqlite",
                        rollback_on_failure=False,
                    )
                    return True
                else:
                    return False
            else:
                raise ValueError(
                    "Could not find replica name %s in both services" % replica_name
                )
        else:
            return False


###########################################################################
class WebHook(object):
    """
    The Webhook represents a single hook instance.
    """

    _properties = None
    _url = None
    _gis = None
    # ----------------------------------------------------------------------

    def __init__(self, url, gis):
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        """returns the class as a string"""
        return f"< WebHook @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """
        Returns the WebHook's properties

        :return: PropertyMap
        """
        if self._properties is None:
            self._properties = PropertyMap(
                self._gis._con.post(self._url, {"f": "json"})
            )
        return self._properties

    # ----------------------------------------------------------------------
    def edit(
        self,
        name: str | None = None,
        change_types: WebHookEvents | str | None = None,
        hook_url: str | None = None,
        signature_key: str | None = None,
        active: bool | None = None,
        schedule_info: WebHookScheduleInfo | dict[str, Any] | None = None,
        payload_format: str | None = None,
    ) -> dict:
        """
        Updates the existing WebHook's Properties.

        =====================================    ===========================================================================
        **Parameter**                             **Description**
        -------------------------------------    ---------------------------------------------------------------------------
        name                                     Optional String. Use valid name for a webhook. This name needs to be unique per service.
        -------------------------------------    ---------------------------------------------------------------------------
        hook_url                                 Optional String.  The URL to which the payloads will be delivered.
        -------------------------------------    ---------------------------------------------------------------------------
        change_types                             Optional :class:`~arcgis.features.managers.WebHookEvents` or String.
                                                 The default is "*", which means all events.  This is a
                                                 comma separated list of values that will fire off the web hook.  The list
                                                 each supported type is below.
        -------------------------------------    ---------------------------------------------------------------------------
        signature_key                            Optional String. If specified, the key will be used in generating the HMAC
                                                 hex digest of value using sha256 hash function and is return in the
                                                 x-esriHook-Signature header.
        -------------------------------------    ---------------------------------------------------------------------------
        active                                   Optional bool. Enable or disable call backs when the webhook is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        schedule_info                            Optional :class:`~arcgis.features.managers.WebHookScheduleInfo` or Dict.
                                                 Allows the trigger to be used as a given schedule.

                                                 Example Dictionary:


                                                     | {
                                                     |    "name" : "Every-5seconds",
                                                     |    "startAt" : 1478280677536,
                                                     |    "state" : "enabled",
                                                     |    "recurrenceInfo" : {
                                                     |     "frequency" : "second",
                                                     |     "interval" : 5
                                                     |   }
                                                     | }

        -------------------------------------    ---------------------------------------------------------------------------
        payload_format                           Optional String. The payload can be sent in pretty format or standard.
                                                 The default is `json`.
        =====================================    ===========================================================================


        A list of allowed web hook triggers is shown below.

        =====================================    ===========================================================================
        **Name**                                 **Triggered When**
        -------------------------------------    ---------------------------------------------------------------------------
        `*`                                      Wildcard event. Any time any event is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesCreated`                        A new feature is created
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesUpdated`                        Any time a feature is updated
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesDeleted`                        Any time a feature is deleted
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesEdited`                         Any time a feature is edited (insert or update or delete)
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsCreated`                     Any time adding a new attachment to a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsUpdated`                     Any time updating a feature attachment
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsDeleted`                     Any time an attachment is deleted from a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerSchemaChanged`                     Any time a schema is changed in a layer
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerDefinitionChanged`                 Any time a layer definition is changed
        -------------------------------------    ---------------------------------------------------------------------------
        `FeatureServiceDefinitionChanged`        Any time a feature service is changed
        =====================================    ===========================================================================


        :return: Response of edit as a dict.

        """
        props = dict(self.properties)
        url = f"{self._url}/edit"
        if isinstance(schedule_info, WebHookScheduleInfo):
            schedule_info = schedule_info.as_dict()
        if isinstance(change_types, list):
            ctypes = []
            for ct in change_types:
                if isinstance(ct, WebHookEvents):
                    ctypes.append(ct.value)
                else:
                    ctypes.append(ct)

            change_types = ",".join(ctypes)
        elif isinstance(change_types, WebHookEvents):
            change_types = change_types.value
        elif change_types is None:
            change_types = ",".join(self.properties["changeTypes"])
        params = {
            "f": "json",
            "name": name,
            "changeTypes": change_types,
            "signatureKey": signature_key,
            "hookUrl": hook_url,
            "active": active,
            "scheduleInfo": schedule_info,
            "payloadFormat": payload_format,
        }
        for k in list(params.keys()):
            if params[k] is None:
                params.pop(k)
            del k
        props.update(params)
        resp = self._gis._con.post(url, props)
        self._properties = PropertyMap(resp)
        return resp

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes the current webhook from the system

        :return: Boolean, True if successful
        """
        url = f"{self._url}/delete"
        params = {"f": "json"}
        resp = self._gis._con.post(url, params)
        return resp["status"] == "success"


###########################################################################
class WebHookServiceManager(object):
    """
    The `WebHookServiceManager` allows owners and administrators wire feature
    service specific events to :class:`~arcgis.features.FeatureLayerCollection`.
    """

    _fc = None
    _url = None
    _gis = None
    # ----------------------------------------------------------------------

    def __init__(self, url, fc, gis) -> None:
        self._url = url
        self._fc = fc
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        """returns the class as a string"""
        return f"< WebHookServiceManager @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """
        Gets the properties for the WebHook Service Manager and returns
        a PropertyMap object
        """
        return PropertyMap(self._gis._con.post(self._url, {"f": "json"}))

    # ----------------------------------------------------------------------
    @property
    def list(self) -> tuple:
        """
        Get a list of web hooks on the :class:`~arcgis.features.FeatureLayerCollection`

        :return: tuple[:class:`~arcgis.features.managers.WebHook`]

        """
        resp = self._gis._con.post(self._url, {"f": "json"})
        ret = [
            WebHook(url=self._url + f"/{d['globalId']}", gis=self._gis) for d in resp
        ]
        return ret

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        hook_url: str,
        change_types: WebHookEvents | str = WebHookEvents.ALL,
        signature_key: str | None = None,
        active: bool = False,
        schedule_info: dict[str, Any] | WebHookScheduleInfo | None = None,
        payload_format: str = "json",
        content_type: str | None = None,
    ) -> WebHook:
        """

        Creates a new Feature Collection Web Hook


        =====================================    ===========================================================================
        **Parameter**                             **Description**
        -------------------------------------    ---------------------------------------------------------------------------
        name                                     Required String. Use valid name for a webhook. This name needs to be unique per service.
        -------------------------------------    ---------------------------------------------------------------------------
        hook_url                                 Required String.  The URL to which the payloads will be delivered.
        -------------------------------------    ---------------------------------------------------------------------------
        change_types                             Optional WebHookEvents or String.  The default is "WebHookEvents.ALL", which means all events.  This is a
                                                 comma separated list of values that will fire off the web hook.  The list
                                                 each supported type is below.
        -------------------------------------    ---------------------------------------------------------------------------
        signature_key                            Optional String. If specified, the key will be used in generating the HMAC
                                                 hex digest of value using sha256 hash function and is return in the
                                                 x-esriHook-Signature header.
        -------------------------------------    ---------------------------------------------------------------------------
        active                                   Optional bool. Enable or disable call backs when the webhook is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        schedule_info                            Optional Dict or `WebHookScheduleInfo`. Allows the trigger to be used as a given schedule.

                                                 Example Dictionary:

                                                     | {
                                                     |   "name" : "Every-5seconds",
                                                     |   "startAt" : 1478280677536,
                                                     |   "state" : "enabled"
                                                     |   "recurrenceInfo" : {
                                                     |     "frequency" : "second",
                                                     |     "interval" : 5
                                                     |   }
                                                     | }

        -------------------------------------    ---------------------------------------------------------------------------
        payload_format                           Optional String. The payload can be sent in pretty format or standard.
                                                 The default is `json`.
        -------------------------------------    ---------------------------------------------------------------------------
        content_type                             Optional String. The Content Type is used to indicate the media type of the
                                                 resource. The media type is a string sent along with the file indicating
                                                 the format of the file.
        =====================================    ===========================================================================


        A list of allowed web hook triggers is shown below.

        =====================================    ===========================================================================
        **Name**                                 **Triggered When**
        -------------------------------------    ---------------------------------------------------------------------------
        `*`                                      Wildcard event. Any time any event is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesCreated`                        A new feature is created
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesUpdated`                        Any time a feature is updated
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesDeleted`                        Any time a feature is deleted
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesEdited`                         Any time a feature is edited (insert or update or delete)
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsCreated`                     Any time adding a new attachment to a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsUpdated`                     Any time updating a feature attachment
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsDeleted`                     Any time an attachment is deleted from a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerSchemaChanged`                     Any time a schema is changed in a layer
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerDefinitionChanged`                 Any time a layer definition is changed
        -------------------------------------    ---------------------------------------------------------------------------
        `FeatureServiceDefinitionChanged`        Any time a feature service is changed
        =====================================    ===========================================================================

        :return:
            A :class:`~arcgis.features.managers.WebHook` object

        """
        url = f"{self._url}/create"
        if isinstance(change_types, list):
            ctnew = []
            for ct in change_types:
                if isinstance(ct, str):
                    ctnew.append(ct)
                elif isinstance(ct, WebHookEvents):
                    ctnew.append(ct.value)
            change_types = ",".join(ctnew)
        elif isinstance(change_types, WebHookEvents):
            change_types = change_types.value
        if isinstance(schedule_info, WebHookScheduleInfo):
            schedule_info = schedule_info.as_dict()
        params = {
            "f": "json",
            "name": name,
            "changeTypes": change_types,
            "signatureKey": signature_key,
            "hookUrl": hook_url,
            "active": active,
            "scheduleInfo": schedule_info,
            "payloadFormat": payload_format,
        }
        if content_type:
            params["contentType"] = content_type
        resp = self._gis._con.post(url, params)
        if not "url" in resp:
            hook_url = self._url + f"/{resp['globalId']}"
            return WebHook(url=hook_url, gis=self._gis)
        else:
            return WebHook(url=resp["url"], gis=self._gis)

    # ----------------------------------------------------------------------
    def enable_hooks(self) -> bool:
        """
        The `enable_hooks` operation restarts a deactivated webhook. When
        activated, payloads will be delivered to the payload URL when the
        webhook is invoked.

        :return: Bool, True if successful

        """
        url = f"{self._url}/activateAll"
        params = {"f": "json"}
        return self._gis._con.post(url, params).get("status", "failed") == "success"

    # ----------------------------------------------------------------------
    def disable_hooks(self) -> bool:
        """
        The `disable_hooks` will turn off all web hooks for the current service.

        :return: Bool, True if successful

        """
        url = f"{self._url}/deactivateAll"
        params = {"f": "json"}
        return self._gis._con.post(url, params).get("status", "failed") == "success"

    # ----------------------------------------------------------------------
    def delete_all_hooks(self) -> bool:
        """
        The `delete_all_hooks` operation will permanently remove the specified webhook.

        :return: Bool, True if successful

        """
        url = f"{self._url}/deleteAll"
        params = {"f": "json"}
        return self._gis._con.post(url, params).get("status", "failed") == "success"


###########################################################################
class FeatureLayerCollectionManager(_GISResource):
    """
    Allows updating the definition (if access permits) of a :class:`~arcgis.features.FeatureLayerCollection`.
    This class is not created by users directly.
    An instance of this class, called 'manager', is available as a property of the :class:`~arcgis.features.FeatureLayerCollection` object.

    Users call methods on this 'manager' object to manage the feature layer collection.
    """

    _layers = None
    _tables = None

    def __init__(self, url, gis=None, fs=None):
        super(FeatureLayerCollectionManager, self).__init__(url, gis)
        self._fs = fs
        self._wh = None
        self._tp = _cf.ThreadPoolExecutor(5)

    @property
    def layers(self) -> list:
        """
        Returns a list of :class:`~arcgis.features.managers.FeatureLayerManager` to work with FeatureLayers

        :returns: List[:class:`~arcgis.features.managers.FeatureLayerManager`]
        """
        self._layers = []
        if "layers" in self.properties:
            for table in self.properties.layers:
                try:
                    self._layers.append(
                        FeatureLayerManager(
                            self.url + "/" + str(table["id"]), self._gis
                        )
                    )
                except Exception as e:
                    _log.error(str(e))

        return self._layers

    @property
    def tables(self) -> list:
        """
        Returns a list of :class:`~arcgis.features.managers.FeatureLayerManager` to work with tables

        :returns: List[:class:`~arcgis.features.managers.FeatureLayerManager`]
        """
        self._tables = []
        if "tables" in self.properties:
            for table in self.properties.tables:
                try:
                    self._tables.append(
                        FeatureLayerManager(
                            self.url + "/" + str(table["id"]), self._gis
                        )
                    )
                except Exception as e:
                    _log.error(str(e))
        return self._tables

    def _populate_layers(self):
        """
        populates layers and tables in the managed feature service
        """
        return

    @property
    def webhook_manager(self) -> WebHookServiceManager:
        """
        :return:
            :class:`~arcgis.features.managers.WebHookServiceManager`
        """
        if self._gis.version >= [8, 2] and self._gis._portal.is_arcgisonline:
            if self._wh is None:
                self._wh = WebHookServiceManager(
                    url=self._url + "/WebHooks", fc=self._fs, gis=self._gis
                )
            return self._wh
        elif self._gis.version >= [8, 2] and self._gis._portal.is_arcgisonline == False:
            return self._fs.service.webhook_manager
        return None

    # ----------------------------------------------------------------------
    def refresh(self):
        """refreshes a feature layer collection"""
        params = {"f": "json"}
        refresh_url = self._url + "/refresh"
        res = self._con.post(refresh_url, params)

        super(FeatureLayerCollectionManager, self)._refresh()
        self._populate_layers()

        self._fs._refresh()
        self._fs._populate_layers()

        return res

    # ----------------------------------------------------------------------
    @property
    def generate_service_definition(self):
        """
        Returns a dictionary can be used for service generation.

        :return: dict or None (if not supported on the service)

        """
        return self._generate_mapservice_definition()

    # ----------------------------------------------------------------------
    def _generate_mapservice_definition(self):
        """
        This operation returns a map service JSON that can be used to
        create a service.

        If a service does not support this operation, None is returned.

        :return:
           dictionary
        """
        params = {
            "f": "json",
        }
        url = "%s/generateMapServiceDefinition" % self._url
        try:
            res = self._con.post(url, params)
        except:
            res = None
        return res

    # ----------------------------------------------------------------------
    def insert_layer(self, data_path: str, name: str = None):
        """
        This method will create a feature layer or table and insert it into the existing feature service.
        If your data path will publish more than one layer or table, only the first will be added.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        data_path              Required string. The path to the data to be inserted.

                               .. note::
                                   Shapefiles and file geodatabases must be in a .zip file.
        ------------------     --------------------------------------------------------------------
        name                   Optional string. The name of the layer or table to be created.
        ==================     ====================================================================
        """
        # Check that the user is the owner of both the source and the published item or has administrative privileges

        from ..gis._impl._content_manager._import_data import _perform_insert

        orig_item = self._gis.content.get(self.properties.serviceItemId)
        if (
            self._gis.users.me.username != orig_item.owner
            and "portal:admin:updateItems" not in self._gis.users.me.privileges
        ):
            raise AssertionError(
                "You must own the service to insert data to it or have administrative privileges."
            )
        # Get the data related
        related_items = orig_item.related_items(rel_type="Service2Data")
        for i in related_items:
            if (
                self._gis.users.me.username != i.owner
                and "portal:admin:updateItems" not in self._gis.users.me.privileges
            ):
                raise AssertionError(
                    "You must own the service data to insert data or have the administrative privilege to update items (portal:admin:updateItems)."
                )

        # Get the name for new service if None passed, ensure data_path has all special characters removed and spaces removed
        data_path = data_path.replace(" ", "_")
        data_path = re.sub(r"[^a-zA-Z0-9_/\.\\:]", "", data_path)
        if name is None:
            name = os.path.basename(data_path)

        # Get the file type
        file_type = os.path.splitext(data_path)[1]
        file_types = {
            ".csv": "CSV",
            ".sqlite": "SQLite",
            ".xls": "Excel",
            ".xlsx": "Excel",
            ".xml": "XML",
            ".sd": "Service Definition",
            ".zip": "Zipfile",
        }
        file_type = file_types.get(file_type, None)
        if file_type is None:
            raise ValueError(
                "File type not supported. Supported file types are: zipped shapefiles, zipped file geodatabases, CSV, Excel, XML, SQLite, and Service Definition."
            )

        # Check if the zipfile is a shapefile or file geodatabase
        if file_type == "Zipfile":
            shapefile = _common_utils._is_shapefile(data_path)
            if shapefile:
                file_type = "Shapefile"
            else:
                file_type = "File Geodatabase"

        # Add to the same folder as the service
        folder_id = orig_item.ownerFolder
        if folder_id is not None:
            folder_name = self._gis.content.get_folder(folder_id)
        else:
            folder_name = None

        # Add the file as an item to portal
        file_item = self._gis.content.add(
            item_properties={
                "type": file_type,
                "title": name,
                "tags": "inserted",
            },
            data=data_path,
            owner=self._gis.users.me.username,
            folder=folder_name,
        )

        # Analyze the file to get publish parameters
        if file_type == "CSV" or file_type == "Excel":
            publish_parameters = self._gis.content.analyze(item=file_item)[
                "publishParameters"
            ]
        else:
            # start creating publish params from new file item
            publish_parameters = {
                "hasStaticData": True,
                "name": os.path.splitext(file_item["name"])[0],
                "maxRecordCount": 2000,
                "layerInfo": {"capabilities": "Query"},
                "targetSR": {"wkid": 102100, "latestWkid": 3857},
            }

        # Publish the item
        new_item = file_item.publish(publish_parameters=publish_parameters)

        # Insert layer or table
        source_info = self._gis.content.analyze(item=file_item)["publishParameters"]
        if len(new_item.layers) > 0:
            publish_parameters = new_item.layers[0].properties
            index = _perform_insert(self, publish_parameters)
            if (
                file_type == "File Geodatabase"
                and "filegdb"
                in orig_item.layers[index].properties.supportedAppendFormats
            ) or file_type != "File Geodatabase":
                if file_type == "File Geodatabase":
                    upload_format = "filegdb"
                else:
                    upload_format = file_type.lower()
                # Workflow for all file types and file geo databases that support append
                ItemDependency(orig_item).add("itemid", file_item.id)
                orig_item.layers[index].append(
                    item_id=file_item.id,
                    upload_format=upload_format,
                    source_info=source_info,
                )
            elif file_type == "File Geodatabase":
                # When filegdb not supported through append, use edit features
                features = new_item.layers[0].query().features
                orig_item.layers[index].edit_features(adds=features)
        elif len(new_item.tables) > 0:
            publish_parameters = new_item.tables[0].properties
            index = _perform_insert(self, publish_parameters)
            ItemDependency(orig_item).add("itemid", file_item.id)
            orig_item.tables[index].append(
                item_id=file_item.id,
                upload_format=file_type,
                source_info=source_info,
            )

        # Add relationship between service and data
        orig_item.add_relationship(rel_item=file_item, rel_type="Service2Data")

        # Remove newly published item since inserted into service
        new_item.delete()
        return orig_item

    # ----------------------------------------------------------------------
    def create_view(
        self,
        name: str,
        spatial_reference: dict[str, Any] | None = None,
        extent: dict[str, int] | None = None,
        allow_schema_changes: bool = True,
        updateable: bool = True,
        capabilities: str = "Query",
        view_layers: list[int] | None = None,
        view_tables: list[int] | None = None,
        *,
        description: str | None = None,
        tags: str | None = None,
        snippet: str | None = None,
        overwrite: bool | None = None,
        set_item_id: str | None = None,
        preserve_layer_ids: bool = False,
    ):
        """
        Creates a view of an existing feature service. You can create a view, if you need a different view of the data
        represented by a hosted feature layer, for example, you want to apply different editor settings, apply different
        styles or filters, define which features or fields are available, or share the data to different groups than
        the hosted feature layer  create a hosted feature layer view of that hosted feature layer.

        When you create a feature layer view, a new hosted feature layer item is added to Content. This new layer is a
        view of the data in the hosted feature layer, which means updates made to the data appear in the hosted feature
        layer and all of its hosted feature layer views. However, since the view is a separate layer, you can change
        properties and settings on this item separately from the hosted feature layer from which it is created.

        For example, you can allow members of your organization to edit the hosted feature layer but share a read-only
        feature layer view with the public.

        To learn more about views visit: https://doc.arcgis.com/en/arcgis-online/share-maps/create-hosted-views.htm

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        name                     Required string. Name of the new view item
        --------------------     --------------------------------------------------------------------
        spatial_reference        Optional dict. Specify the spatial reference of the view
        --------------------     --------------------------------------------------------------------
        extent                   Optional dict. Specify the extent of the view
        --------------------     --------------------------------------------------------------------
        allow_schema_changes     Optional bool. Default is True. Determines if a view can alter a
                                 service's schema.
        --------------------     --------------------------------------------------------------------
        updateable               Optional bool. Default is True. Determines if view can update values
        --------------------     --------------------------------------------------------------------
        capabilities             Optional string. Specify capabilities as a comma separated string.
                                 For example "Query, Update, Delete". Default is 'Query'.
        --------------------     --------------------------------------------------------------------
        view_layers              Optional list. Specify list of layers present in the FeatureLayerCollection
                                 that you want in the view.
        --------------------     --------------------------------------------------------------------
        view_tables              Optional list. Specify list of tables present in the FeatureLayerCollection
                                 that you want in the view.
        --------------------     --------------------------------------------------------------------
        description              Optional String. A user-friendly description for the published dataset.
        --------------------     --------------------------------------------------------------------
        tags                     Optional String. The comma separated string of descriptive words.
        --------------------     --------------------------------------------------------------------
        snippet                  Optional String. A short description of the view item.
        --------------------     --------------------------------------------------------------------
        overwrite                Optional Boolean.  If true, the view is overwritten, False is the default.
        --------------------     --------------------------------------------------------------------
        set_item_id              Optional String. If set, the ItemId is defined by the user, not the system.
        --------------------     --------------------------------------------------------------------
        preserve_layer_ids       Optional Boolean. Preserves the layer's `id` on it's definition when `True`.  The default is `False`.
        ====================     ====================================================================

        .. code-block:: python  (optional)

           USAGE EXAMPLE: Create a veiw from a hosted feature layer

           crime_fl_item = gis.content.search("2012 crime")[0]
           crime_flc = FeatureLayerCollection.fromitem(crime_fl_item)

           # Create a view with just the first layer
           crime_view = crime_flc.manager.create_view(name='Crime in 2012", updateable=False,
                                                        view_layers=[crime_flc.layers[0]])

        .. code-block:: python (optional)

            USAGE EXAMPLE: Create an editable view

            crime_fl_item = gis.content.search("2012 crime")[0]
            crime_flc = FeatureLayerCollection.fromitem(crime_fl_item)
            crime_view = crime_flc.manager.create_view(name=uuid.uuid4().hex[:9], # create random name
                                                       updateable=True,
                                                       allow_schema_changes=False,
                                                       capabilities="Query,Update,Delete")

        :return:
            Returns the newly created :class:`~arcgis.gis.Item` for the view.
        """

        import os
        from . import FeatureLayerCollection

        regex = r"^[-a-zA-Z0-9_]*$"
        if len(re.findall(regex, name)) == 0:
            raise ValueError(
                "The service `name` cannot contain any spaces or special characters except underscores."
            )
        gis = self._gis
        content = gis.content
        if "serviceItemId" not in self.properties:
            raise Exception(
                "A registered hosted feature service is required to use create_view"
            )
        item_id = self.properties["serviceItemId"]
        item = content.get(itemid=item_id)
        url = item.url
        fs = FeatureLayerCollection(url=url, gis=gis)
        if gis._url.lower().find("sharing/rest") < 0:
            url = gis._url + "/sharing/rest"
        else:
            url = gis._url

        if "serviceItemId" in self.properties:
            # get the owner of the service
            user = gis.content.get(self.properties["serviceItemId"])["owner"]
        else:
            # if no service item id then default to logged in user
            user = gis.users.me.username

        url = "%s/content/users/%s/createService" % (url, user)
        if spatial_reference is None:
            # handle for tables
            if "spatialReference" in fs.properties:
                spatial_reference = fs.properties["spatialReference"]
            # else it stays the spatial reference given or None
        params = {
            "f": "json",
            "isView": True,
            "createParameters": json.dumps(
                {
                    "name": name,
                    "isView": True,
                    "sourceSchemaChangesAllowed": allow_schema_changes,
                    "isUpdatableView": updateable,
                    "spatialReference": spatial_reference,
                    "initialExtent": extent or fs.properties["initialExtent"],
                    "capabilities": capabilities or fs.properties["capabilties"],
                    "preserveLayerIds": preserve_layer_ids,
                }
            ),
            "tags": tags if tags else item.tags,
            "snippet": snippet if snippet else item.snippet,
            "description": description if description else item.description,
            "outputType": "featureService",
        }
        if set_item_id:
            params["itemIdToCreate"] = set_item_id
        if overwrite:
            logging.warning(
                "overwrite is currently not supported on this platform, and will not be honored"
            )

        res = gis._con.post(path=url, postdata=params)
        view = content.get(res["itemId"])
        fs_view = FeatureLayerCollection(url=view.url, gis=gis)
        add_def = {"layers": [], "tables": []}

        def is_none_or_empty(view_param):
            if not view_param:
                return True
            if isinstance(view_param, list) and len(view_param) == 0:
                return True
            if isinstance(view_param, dict):
                for k, v in view_param.items():
                    if view_param[k] is not None:
                        return False
                return True
            return False

        if is_none_or_empty(view_layers) and is_none_or_empty(view_tables):
            # When view_layers and view_tables are not specified, create a view from all layers and tables
            for lyr in fs.layers:
                lyr_id = lyr.manager.properties.serviceItemId
                data_path = "content/items/" + lyr_id + "/data"
                data = item._portal.con.get(path=data_path)
                add_def["layers"].append(
                    {
                        "adminLayerInfo": {
                            "popupInfo": data["layers"][0]["popupInfo"]
                            if "layers" in data
                            else None,
                            "viewLayerDefinition": {
                                "sourceServiceName": os.path.basename(
                                    os.path.dirname(fs.url)
                                ),
                                "sourceLayerId": lyr.manager.properties["id"],
                                "sourceLayerFields": "*",
                            },
                        },
                        "name": lyr.manager.properties["name"],
                    }
                )
            for tbl in fs.tables:
                add_def["tables"].append(
                    {
                        "adminLayerInfo": {
                            "viewLayerDefinition": {
                                "sourceServiceName": os.path.basename(
                                    os.path.dirname(fs.url)
                                ),
                                "sourceLayerId": tbl.manager.properties["id"],
                                "sourceLayerFields": "*",
                            }
                        },
                        "id": tbl.manager.properties["id"],
                        "name": tbl.manager.properties["name"],
                        "type": "Table",
                    }
                )
        else:
            # when view_layers is specified
            if view_layers:
                if isinstance(view_layers, list):
                    for lyr in view_layers:
                        lyr_id = lyr.manager.properties.serviceItemId
                        data_path = "content/items/" + lyr_id + "/data"
                        data = item._portal.con.get(path=data_path)
                        def_lyr = dict(lyr.properties)
                        def_lyr["adminLayerInfo"] = {
                            "popupInfo": data["layers"][0]["popupInfo"]
                            if "layers" in data
                            else None,
                            "viewLayerDefinition": {
                                "sourceServiceName": os.path.basename(
                                    os.path.dirname(fs.url)
                                ),
                                "sourceLayerId": lyr.manager.properties["id"],
                                "sourceLayerFields": "*",
                            },
                        }
                        for k in {
                            "indexes",
                            "relationships",
                            "geometryProperties",
                            "hasGeometryProperties",
                            "serviceItemId",
                            "supportsMultiScaleGeometry",
                            "fields",
                            "isView",
                        }:
                            if k in def_lyr:
                                del def_lyr[k]
                        if self._gis._con.token:
                            def_lyr["url"] = lyr.url + f"?token={self._gis._con.token}"
                        add_def["layers"].append(def_lyr)
                else:
                    import logging

                    _log = logging.getLogger(__name__)
                    from arcgis.features.layer import Layer

                    if isinstance(view_layers, dict):
                        if "layers" in view_layers:
                            add_def["layers"] = view_layers["layers"]
                        else:
                            add_def["layers"].append(view_layers)
                    elif isinstance(view_layers, Layer):
                        add_def["layers"].append(
                            {
                                "adminLayerInfo": {
                                    "viewLayerDefinition": {
                                        "sourceServiceName": os.path.basename(
                                            os.path.dirname(fs.url)
                                        ),
                                        "sourceLayerId": view_layers.manager.properties[
                                            "id"
                                        ],
                                        "sourceLayerFields": "*",
                                    }
                                },
                                "name": view_layers.manager.properties["name"],
                            }
                        )
                    else:
                        _log.error("Unable to parse the view_layers parameter")

            # when view_tables is specified
            if view_tables:
                if isinstance(view_tables, list):
                    for tbl in view_tables:
                        tbl_def = {
                            "adminLayerInfo": {
                                "viewLayerDefinition": {
                                    "sourceServiceName": os.path.basename(
                                        os.path.dirname(fs.url)
                                    ),
                                    "sourceLayerId": tbl.manager.properties["id"],
                                    "sourceLayerFields": "*",
                                }
                            },
                            "id": tbl.manager.properties["id"],
                            "name": tbl.manager.properties["name"],
                            "type": "Table",
                        }
                        tbl_def.update(dict(tbl.properties))
                        for k in {
                            "isView",
                            "sourceSchemaChangesAllowed",
                            "fields",
                            "serviceItemId",
                            "relationships",
                            "indexes",
                            "isUpdatableView",
                            "viewSourceHasAttachments",
                        }:
                            if k in tbl_def:
                                del tbl_def[k]
                        add_def["tables"].append(tbl_def)
                else:
                    import logging

                    _log = logging.getLogger(__name__)

                    from arcgis.features.layer import Table

                    if isinstance(view_tables, dict):
                        if "tables" in view_tables:
                            add_def["tables"] = view_tables["tables"]
                        else:
                            add_def["tables"].append(view_tables)
                    elif isinstance(view_tables, Table):
                        add_def["tables"].append(
                            {
                                "adminLayerInfo": {
                                    "viewLayerDefinition": {
                                        "sourceServiceName": os.path.basename(
                                            os.path.dirname(fs.url)
                                        ),
                                        "sourceLayerId": view_tables.manager.properties[
                                            "id"
                                        ],
                                        "sourceLayerFields": "*",
                                    }
                                },
                                "name": view_tables.manager.properties["name"],
                            }
                        )
                    else:
                        _log.error("Unable to parse the view_tables parameter")

        fs_view.manager.add_to_definition(add_def)
        if extent and fs_view.layers:
            for vw_lyr in fs_view.layers:
                vw_lyr.manager.update_definition(
                    {
                        "viewLayerDefinition": {
                            "filter": {
                                "operator": "esriSpatialRelIntersects",
                                "value": {
                                    "geometryType": "esriGeometryEnvelope",
                                    "geometry": extent,
                                },
                            }
                        }
                    }
                )

        if view_layers:
            data = item.get_data()
            if "layers" in data:
                item_upd_dict = {
                    "layers": [
                        ilyr
                        for ilyr in item.get_data()["layers"]
                        for lyr in view_layers
                        if int(lyr.url[-1]) == ilyr["id"]
                    ]
                }
                view.update(data=item_upd_dict)
        else:
            view.update(data=item.get_data())

        return content.get(res["itemId"])

    # ----------------------------------------------------------------------
    def _check_status(self, url: str) -> dict:
        """Internal method to check the status of the definition change.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required String. The URL endpoint to check the status
        ===============     ====================================================================


        :return:
           The status dictionary
        """
        sleep_time = 1
        count = 1

        params = {"f": "json"}
        con = self._gis._con
        job_response = con.post(url, params)
        if "status" in job_response:
            while "status" in job_response and not job_response.get("status") in [
                "completed",
                "Completed",
            ]:
                time.sleep(sleep_time * count)
                job_response = con.post(url, params)
                if (
                    job_response.get("status") in ("esriJobFailed", "failed")
                    or job_response.get("status").lower().find("error") > -1
                ):
                    if "error" in job_response:
                        raise Exception(job_response["error"])
                    else:
                        raise Exception(f"Job failed: {job_response}")
                elif job_response.get("status") == "esriJobCancelled":
                    raise Exception("Job cancelled.")
                elif job_response.get("status") == "esriJobTimedOut":
                    raise Exception("Job timed out.")
                count += 1

        else:
            raise Exception("No job results.")
        return job_response

    # ----------------------------------------------------------------------
    def _refresh_callback(self, *args, **kwargs):
        """function to refresh the service post add or update definition for async operations"""
        try:
            self._hydrated = False
            self.refresh()
        except:
            self._hydrated = False

    # ----------------------------------------------------------------------
    def add_to_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The add_to_definition operation supports adding a definition
        property to a hosted feature layer collection service. The result of this
        operation is a response indicating success or failure with error
        code and description.

        This function will allow users to change or add additional values
        to an already published service.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
        ---------------     --------------------------------------------------------------------
        future              Optional, If True, a future object will be returns and the process
                            will not wait for the task to complete.
                            The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON message as dictionary when `future=False` else If ``future = True``,
           then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.

        """

        if isinstance(json_dict, PropertyMap):
            json_dict = dict(json_dict)

        params = {
            "f": "json",
            "addToDefinition": json.dumps(json_dict),
            "async": json.dumps(future),
        }
        adddefn_url = self._url + "/addToDefinition"
        res = self._con.post(adddefn_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def update_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The update_definition operation supports updating a definition
        property in a hosted feature layer collection service. The result of this
        operation is a response indicating success or failure with error
        code and description.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
        ---------------     --------------------------------------------------------------------
        future              Optional boolean. If True, a future object will be returns and the process
                            will not wait for the task to complete.
                            The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON message as dictionary when `future=False`
           when `future=True`, `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object is returned. Call ``result()`` to get the response.

        """
        definition = None
        if json_dict is not None:
            if isinstance(json_dict, PropertyMap):
                definition = dict(json_dict)
            if isinstance(json_dict, collections.OrderedDict):
                definition = json_dict
            else:
                definition = collections.OrderedDict()
                if "hasStaticData" in json_dict:
                    definition["hasStaticData"] = json_dict["hasStaticData"]
                if "allowGeometryUpdates" in json_dict:
                    definition["allowGeometryUpdates"] = json_dict[
                        "allowGeometryUpdates"
                    ]
                if "capabilities" in json_dict:
                    definition["capabilities"] = json_dict["capabilities"]
                if "editorTrackingInfo" in json_dict:
                    definition["editorTrackingInfo"] = collections.OrderedDict()
                    if "enableEditorTracking" in json_dict["editorTrackingInfo"]:
                        definition["editorTrackingInfo"][
                            "enableEditorTracking"
                        ] = json_dict["editorTrackingInfo"]["enableEditorTracking"]

                    if (
                        "enableOwnershipAccessControl"
                        in json_dict["editorTrackingInfo"]
                    ):
                        definition["editorTrackingInfo"][
                            "enableOwnershipAccessControl"
                        ] = json_dict["editorTrackingInfo"][
                            "enableOwnershipAccessControl"
                        ]

                    if "allowOthersToUpdate" in json_dict["editorTrackingInfo"]:
                        definition["editorTrackingInfo"][
                            "allowOthersToUpdate"
                        ] = json_dict["editorTrackingInfo"]["allowOthersToUpdate"]

                    if "allowOthersToDelete" in json_dict["editorTrackingInfo"]:
                        definition["editorTrackingInfo"][
                            "allowOthersToDelete"
                        ] = json_dict["editorTrackingInfo"]["allowOthersToDelete"]

                    if "allowOthersToQuery" in json_dict["editorTrackingInfo"]:
                        definition["editorTrackingInfo"][
                            "allowOthersToQuery"
                        ] = json_dict["editorTrackingInfo"]["allowOthersToQuery"]
                    if isinstance(json_dict["editorTrackingInfo"], dict):
                        for key, val in json_dict["editorTrackingInfo"].items():
                            if key not in definition["editorTrackingInfo"]:
                                definition["editorTrackingInfo"][key] = val
                if isinstance(json_dict, dict):
                    for key, val in json_dict.items():
                        if key not in definition:
                            definition[key] = val

        params = {
            "f": "json",
            "updateDefinition": json.dumps(obj=definition, separators=(",", ":")),
            "async": json.dumps(future),
        }
        u_url = self._url + "/updateDefinition"
        res = self._con.post(u_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def delete_from_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The delete_from_definition operation supports deleting a
        definition property from a hosted feature layer collection service. The result of
        this operation is a response indicating success or failure with
        error code and description.
        See `Delete From Definition (Feature Service) <https://developers.arcgis.com/rest/services-reference/delete-from-definition-feature-service-.htm>`_
        for additional information on this function.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
        ---------------     --------------------------------------------------------------------
        future              Optional boolean. If True, a future object will be returns and the process
                            will not wait for the task to complete.
                            The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON message as dictionary when `future=False` else If ``future = True``,
           then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.

        """
        params = {
            "f": "json",
            "deleteFromDefinition": json.dumps(json_dict),
            "async": json.dumps(future),
        }
        u_url = self._url + "/deleteFromDefinition"

        res = self._con.post(u_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def overwrite(self, data_file: str):
        """
        Overwrite all the features and layers in a hosted feature layer collection service. This operation removes
        all features but retains the properties (such as metadata, itemID) and capabilities configured on the service.
        There are some limits to using this operation:

        1. Only hosted feature layer collection services can be overwritten

        2. The original data used to publish this layer should be available on the portal

        3. The data file used to overwrite should be of the same format and filename as the original that was used to publish the layer

        4. The schema (column names, column data types) of the data_file should be the same as original. You can have additional or fewer rows (features).

        In addition to overwriting the features, this operation also updates the data of the item used to published this
        layer.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        data                Required string. Path to the file used to overwrite the hosted
                            feature layer collection.
        ===============     ====================================================================

        :return: JSON message as dictionary such as {'success':True} or {'error':'error message'}
        """
        # check for outstanding replicas
        if hasattr(self._fs, "replicas") and bool(self._fs.replicas.get_list()):
            raise Exception(
                "Service cannot be overwritten if Sync is enabled and replicas exist."
            )

        # region Get Item associated with the service
        if "serviceItemId" in self.properties.keys():
            feature_layer_item = self._gis.content.get(self.properties["serviceItemId"])
        else:
            return {
                "Error": "Can only overwrite a Hosted Feature Layer Collection (Feature Service)"
            }
        # endregion

        # region find data item related to this hosted feature layer
        related_data_items = feature_layer_item.related_items("Service2Data", "forward")
        if len(related_data_items) > 0:
            related_data_item = related_data_items[0]
        else:
            return {
                "Error": "Cannot find related data item used to publish this Feature Layer"
            }

        # find if we are overwritting only a hosted table
        hosted_table = False
        if not feature_layer_item.layers and feature_layer_item.tables:
            hosted_table = True
        # endregion

        params = None
        # overwritting for online and enterprise is different
        # if online or hosted table then use minimal parameters
        if (
            related_data_item.type
            in ["CSV", "Shapefile", "File Geodatabase", "Microsoft Excel"]
            and self._gis._portal.is_arcgisonline
            or hosted_table is True
        ):
            # construct a full publishParameters that is a combination of existing Feature Layer definition
            # and original publishParameters.json used for publishing the service the first time

            # get old publishParameters.json
            path = (
                "content/items/"
                + feature_layer_item.itemid
                + "/info/publishParameters.json"
            )
            postdata = {"f": "json"}

            old_publish_parameters = self._gis._con.post(path, postdata)

            # get FeatureServer definition
            feature_service_def = dict(self.properties)

            # Get definition of each layer and table, remove fields in the dict
            layers_dict = []
            tables_dict = []
            for layer in self.layers:
                layer_def = dict(layer.properties)
                if "fields" in layer_def.keys():
                    layer_def.pop("fields")
                layers_dict.append(layer_def)

            for table in self.tables:
                table_def = dict(table.properties)
                if "fields" in table_def.keys():
                    table_def.pop("fields")
                tables_dict.append(table_def)

            # Splice the detailed table and layer def with FeatuerServer def
            feature_service_def["layers"] = layers_dict
            feature_service_def["tables"] = tables_dict
            from pathlib import Path

            service_name = Path(self.url).parts[-2]  # get service name from url
            feature_service_def["name"] = service_name

            # combine both old publish params and full feature service definition
            publish_parameters = feature_service_def
            publish_parameters.update(old_publish_parameters)

        # enterprise overwriting params creation
        elif related_data_item.type in [
            "CSV",
            "Shapefile",
            "File Geodatabase",
            "Microsoft Excel",
        ]:
            path = (
                "content/items/"
                + feature_layer_item.itemid
                + "/info/publishParameters.json"
            )
            postdata = {"f": "json"}

            old_publish_parameters = self._gis._con.post(path, postdata)
            base_url = feature_layer_item.privateUrl
            lyr_url_info = "%s/layers" % base_url
            fs_url = "%s" % base_url
            # layer_info gets information on the layers and tables of an item
            layer_info = self._gis._con.get(lyr_url_info, {"f": "json"})
            [lyr.pop("fields") for lyr in layer_info["layers"]]
            [lyr.pop("fields") for lyr in layer_info["tables"]]
            feature_service_def = self._gis._con.get(fs_url, {"f": "json"})
            feature_service_def["tables"] = []
            feature_service_def["layers"] = []
            feature_service_def.update(layer_info)
            publish_parameters = feature_service_def
            publish_parameters.update(old_publish_parameters)
        else:
            # overwriting a SD case - no need for detailed publish parameters
            publish_parameters = None

        if related_data_item.update(item_properties=params, data=data_file):
            published_item = related_data_item.publish(
                publish_parameters, overwrite=True
            )
            if published_item is not None:
                return {"success": True}
            else:
                return {
                    "error": "Unable to overwrite the hosted feature layer collection"
                }
        else:
            return {"error": "Unable to update related data item with new data"}

    # ----------------------------------------------------------------------

    def _gen_overwrite_publishParameters(self, flc_item):
        """
        This internal method generates publishParameters for overwriting a hosted feature layer collection. This is used
        by Item.publish() method when user wants to originate the overwrite process from the data item instead of
        the hosted feature layer.

        :param flc_item: The Feature Layer Collection Item object that is being overwritten
        :return: JSON message as dictionary with to be used as publishParameters payload in the publish REST call.
        """

        # region Get Item associated with the service
        if "serviceItemId" in self.properties.keys():
            feature_layer_item = self._gis.content.get(self.properties["serviceItemId"])
        else:
            return {"error": "Can only overwrite a hosted feature layer collection"}
        # endregion

        # region find data item related to this hosted feature layer
        related_data_items = feature_layer_item.related_items("Service2Data", "forward")
        if len(related_data_items) > 0:
            related_data_item = related_data_items[0]
        else:
            return {
                "error": "Cannot find related data item used to publish this feature layer"
            }

        # endregion

        # region Construct publish parameters for Portal / Enterprise
        params = None
        if (
            related_data_item.type
            in ["CSV", "Shapefile", "File Geodatabase", "Microsoft Excel"]
            and self._gis._portal.is_arcgisonline == False
        ):
            params = {
                "name": related_data_item.name,
                "title": related_data_item.title,
                "tags": related_data_item.tags,
                "type": related_data_item.type,
                "overwrite": True,
                "overwriteService": "on",
                "useDescription": "on",
            }
            base_url = feature_layer_item.privateUrl
            lyr_url_info = "%s/layers" % base_url
            fs_url = "%s" % base_url
            layer_info = self._gis._con.get(lyr_url_info, {"f": "json"})
            [lyr.pop("fields") for lyr in layer_info["layers"]]
            [lyr.pop("fields") for lyr in layer_info["tables"]]
            feature_service_def = self._gis._con.get(fs_url, {"f": "json"})
            feature_service_def["tables"] = []
            feature_service_def["layers"] = []
            feature_service_def.update(layer_info)
            publish_parameters = feature_service_def
            publish_parameters["name"] = feature_layer_item.title
            publish_parameters["_ssl"] = False
            for idx, lyr in enumerate(publish_parameters["layers"]):
                lyr["parentLayerId"] = -1
                for k in {
                    "sourceSpatialReference",
                    "isCoGoEnabled",
                    "parentLayer",
                    "isDataArchived",
                    "cimVersion",
                }:
                    lyr.pop(k, None)
            for idx, lyr in enumerate(publish_parameters["tables"]):
                lyr["parentLayerId"] = -1
                for k in {
                    "sourceSpatialReference",
                    "isCoGoEnabled",
                    "parentLayer",
                    "isDataArchived",
                    "cimVersion",
                }:
                    lyr.pop(k, None)
        # endregion

        # region Construct publish parameters for AGO
        elif (
            related_data_item.type
            in ["CSV", "Shapefile", "File Geodatabase", "Microsoft Excel"]
            and self._gis._portal.is_arcgisonline
        ):
            # construct a full publishParameters that is a combination of existing Feature Layer definition
            # and original publishParameters.json used for publishing the service the first time

            # get old publishParameters.json
            path = (
                "content/items/"
                + feature_layer_item.itemid
                + "/info/publishParameters.json"
            )
            postdata = {"f": "json"}

            old_publish_parameters = self._gis._con.post(path, postdata)

            # get FeatureServer definition
            feature_service_def = dict(self.properties)

            # Get definition of each layer and table, remove fields in the dict
            layers_dict = []
            tables_dict = []
            for layer in self.layers:
                layer_def = dict(layer.properties)
                if "fields" in layer_def.keys():
                    dump = layer_def.pop("fields")
                layers_dict.append(layer_def)

            for table in self.tables:
                table_def = dict(table.properties)
                if "fields" in table_def.keys():
                    dump = table_def.pop("fields")
                tables_dict.append(table_def)

            # Splice the detailed table and layer def with FeatuerServer def
            feature_service_def["layers"] = layers_dict
            feature_service_def["tables"] = tables_dict
            from pathlib import Path

            service_name = Path(self.url).parts[-2]  # get service name from url
            feature_service_def["name"] = service_name

            # combine both old publish params and full feature service definition
            publish_parameters = feature_service_def
            publish_parameters.update(old_publish_parameters)
        else:
            # overwriting a SD case - no need for detailed publish parameters
            publish_parameters = None
        # endregion

        return (publish_parameters, params)


###########################################################################
class FeatureLayerManager(_GISResource):
    """
    Allows updating the definition (if access permits) of a :class:`~arcgis.features.FeatureLayer`.
    This class is not created by users
    directly.
    An instance of this class, called 'manager', is available as a property of the :class:`~arcgis.features.FeatureLayer`
    object, if the layer can be managed by the user.
    Users call methods on this 'manager' object to manage the feature layer.
    """

    def __init__(self, url, gis=None):
        super(FeatureLayerManager, self).__init__(url, gis)
        self._hydrate()

    # ----------------------------------------------------------------------
    @property
    def contingent_values(self) -> dict[str, Any]:
        """returns the contingent values for the service endpoint"""
        url: str = f"{self._url}/contingentValues"
        params: dict[str, Any] = {"f": "json"}
        return self._gis._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def field_groups(self) -> dict[str, Any]:
        """returns the field groups for the service endpoint"""
        url: str = f"{self._url}/fieldGroups"
        params: dict[str, Any] = {"f": "json"}
        return self._gis._con.get(url, params)

    # ----------------------------------------------------------------------
    @classmethod
    def fromitem(cls, item: Item, layer_id: int = 0):
        """
        Creates a :class:`~arcgis.features.managers.FeatureLayerManager` object from a GIS Item.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required of type :class:`~arcgis.features.FeatureService` that represents
                            a :class:`~arcgis.features.FeatureLayerCollection` .
        ---------------     --------------------------------------------------------------------
        layer_id            Required int. Id of the layer in the
                            :class:`~arcgis.features.FeatureLayerCollection`
        ===============     ====================================================================

        :return:
            :class:`~arcgis.features.FeatureLayer` created from the layer provided.

        """
        if item.type != "Feature Service":
            raise TypeError("item must be a of type Feature Service, not " + item.type)
        from arcgis.features import FeatureLayer

        return FeatureLayer.fromitem(item, layer_id).manager

    # ----------------------------------------------------------------------
    def refresh(self):
        """refreshes a service"""
        params = {"f": "json"}
        u_url = self._url + "/refresh"
        res = self._con.post(u_url, params)

        super(FeatureLayerManager, self)._refresh()

        return res

    # ----------------------------------------------------------------------
    def add_to_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The addToDefinition operation supports adding a definition
        property to a hosted feature layer.

        This function will allow users to change add additional values
        to an already published service.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
        ---------------     --------------------------------------------------------------------
        future              Optional boolean. If True, a future object will be returned and the process
                            will not wait for the task to complete. The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON message as dictionary indicating 'success' or 'error'. If ``future = True``,
           then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.
        """

        if isinstance(json_dict, PropertyMap):
            json_dict = dict(json_dict)

        params = {
            "f": "json",
            "addToDefinition": json.dumps(json_dict),
            "async": json.dumps(future),
        }
        u_url = self._url + "/addToDefinition"

        res = self._con.post(u_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def update_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The `update_definition` operation supports updating a definition
        property in a hosted feature layer. The result of this
        operation is a response indicating success or failure with error
        code and description.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
        ---------------     --------------------------------------------------------------------
        future              Optional, If True, a future object will be returns and the process
                            will not wait for the task to complete.
                            The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON Message as dictionary indicating 'success' or 'error'. If ``future = True``,
           then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.
        """

        if isinstance(json_dict, PropertyMap):
            json_dict = dict(json_dict)

        params = {
            "f": "json",
            "updateDefinition": json.dumps(json_dict),
            "async": json.dumps(future),
        }

        u_url = self._url + "/updateDefinition"

        res = self._con.post(u_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def delete_from_definition(self, json_dict: dict[str, Any], future: bool = False):
        """
        The deleteFromDefinition operation supports deleting a
        definition property from a hosted feature layer. The result of
        this operation is a response indicating success or failure with
        error code and description.
        See: `Delete From Definition (Feature Service) <https://developers.arcgis.com/rest/services-reference/delete-from-definition-feature-service-.htm>`_
        for additional information on this function.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict. The part to add to the hosted service. The format
                            can be derived from the `properties` property.
                            For layer level modifications, run updates on each individual feature
                            service layer object.
                            Only include the items you want to remove from the FeatureService or layer.
        ---------------     --------------------------------------------------------------------
        future              Optional, If True, a future object will be returns and the process
                            will not wait for the task to complete.
                            The default is False, which means wait for results.
        ===============     ====================================================================

        :return:
           JSON Message as dictionary indicating 'success' or 'error'. If ``future = True``,
           then the result is a `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object. Call ``result()`` to get the response.

        """

        if isinstance(json_dict, PropertyMap):
            json_dict = dict(json_dict)

        params = {
            "f": "json",
            "deleteFromDefinition": json.dumps(json_dict),
            "async": json.dumps(future),
        }
        u_url = self._url + "/deleteFromDefinition"

        res = self._con.post(u_url, params)
        if future and "statusURL" in res:
            executor = _cf.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._check_status, **{"url": res.get("statusURL")}
            )
            futureobj.add_done_callback(self._refresh_callback)
            executor.shutdown(False)
            return futureobj
        self.refresh()
        return res

    # ----------------------------------------------------------------------
    def truncate(
        self,
        attachment_only: bool = False,
        asynchronous: bool = False,
        wait: bool = True,
    ):
        """
        The truncate operation supports deleting all features or attachments
        in a hosted feature service layer. The result of this operation is a
        response indicating success or failure with error code and description.
        See `Truncate (Feature Layer) <https://developers.arcgis.com/rest/services-reference/online/truncate-feature-layer-.htm>`_
        for additional information on this method.

        .. note::
            The `truncate` method is restricted to
            :class:`layers <arcgis.features.FeatureLayer>` that:

            - do not serve as the origin in a relationship with other
              layers
            - do not reference the same underlying database tables that are
              referenced by other layers (for example, if the layer was
              published from a layer with a definition query and a
              separate layer has also been published from that source)
            - do not have `sync` enabled

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment_only     Optional boolean. If True, deletes all the attachments for this layer.
                            None of the layer features will be deleted.
        ---------------     --------------------------------------------------------------------
        asynchronous        Optional boolean. If True, supports asynchronous processing. The
                            default is False. It is recommended to set asynchronous=True for
                            large datasets.
        ---------------     --------------------------------------------------------------------
        wait                Optional boolean. If True, then wait to pause the process until
                            asynchronous operation is completed. Default is True.
        ===============     ====================================================================

        :return:
           JSON Message as dictionary indicating `success` or `error`

        """
        params = {
            "f": "json",
            "attachmentOnly": attachment_only,
            "async": asynchronous,
        }
        u_url = self._url + "/truncate"

        if asynchronous:
            if wait:
                job = self._con.post(u_url, params)
                status = self._get_status(url=job["statusURL"])
                while status["status"] not in (
                    "Completed",
                    "CompletedWithErrors",
                    "Failed",
                ):
                    # wait before checking again
                    time.sleep(2)
                    status = self._get_status(url=job["statusURL"])

                res = status
                self.refresh()
            else:
                res = self._con.post(u_url, params)
                # Leave calling refresh to user since wait is false
        else:
            res = self._con.post(u_url, params)
            self.refresh()
        return res

    # ----------------------------------------------------------------------
    def _check_status(self, url: str) -> dict:
        """
        Internal method to check the status of the definition change.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required String. The URL endpoint to check the status
        ===============     ====================================================================


        :return:
           The status dictionary
        """
        sleep_time = 1
        count = 1

        params = {"f": "json"}
        con = self._gis._con
        job_response = con.post(url, params)
        if "status" in job_response:
            while "status" in job_response and not job_response.get("status") in [
                "completed",
                "Completed",
            ]:
                if count > 10:
                    count = 10
                time.sleep(sleep_time * count)
                job_response = con.post(url, params)
                if job_response.get("status") in ("esriJobFailed", "failed"):
                    if "error" in job_response:
                        raise Exception(job_response["error"])
                    else:
                        raise Exception("Job failed.")
                elif job_response.get("status") == "esriJobCancelled":
                    raise Exception("Job cancelled.")
                elif job_response.get("status") == "esriJobTimedOut":
                    raise Exception("Job timed out.")
                count += 1

        else:
            raise Exception("No job results.")
        return job_response

    # ----------------------------------------------------------------------
    def _refresh_callback(self, *args, **kwargs):
        """function to refresh the service post add or update definition for async operations"""
        try:
            self._hydrated = False
            self.refresh()
        except:
            self._hydrated = False

    # ----------------------------------------------------------------------
    def _get_status(self, url):
        """gets the status when exported async set to True"""
        params = {"f": "json"}
        url += "/status"
        return self._con.get(url, params)
