from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List


########################################################################
class Uploads(_BaseKube):
    """
    This resource is a collection of all the items that have been uploaded
    to the kubernetes site.

    There are two ways to upload items. You can upload complete items using
    the Upload Item operation. If a particular item is made up of many
    chunks (parts), you need to first register the item and subsequently
    upload the individual parts using the Upload Part operation. Item
    uploads are filtered by a whitelist of filename extensions. This is the
    default list: soe, sd, sde, odc, csv, txt, zshp, kmz. The default list
    can be overridden by setting the uploadFileExtensionWhitelist property
    with the kubernetes site properties API.

    """

    _uploads = None
    _con = None
    _json = None
    _json_dict = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, initialize=False):
        """Constructor"""
        if url.lower().find("uploads") < -1:
            self._url = url + "/uploads"
        else:
            self._url = url
        self._con = gis
        self._json_dict = {}
        self._json = ""

    # ----------------------------------------------------------------------
    @property
    def uploads(self):
        """
        returns a collection of all the items that have been uploaded to
        the kubernetes site.

        There are two ways to upload items. You can upload complete items
        using the Upload Item operation. If a particular item is made up of
        many chunks (parts), you need to first register the item and
        subsequently upload the individual parts using the Upload Part
        operation. Item uploads are filtered by a whitelist of filename
        extensions. This is the default list: soe, sd, sde, odc, csv, txt,
        zshp, kmz. The default list can be overridden by setting the
        uploadFileExtensionWhitelist property with the kubernetes site properties API.

        """
        params = {"f": "json"}
        return self._con.get(path=self._url, params=params)

    # ----------------------------------------------------------------------
    def delete(self, item_id):
        """
        Deletes the uploaded item and its configuration.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. unique ID of the item
        ===============     ====================================================================


        :return: boolean

        """
        url = self._url + "/%s/delete" % item_id
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def download(self, item_id: str) -> str:
        """
        Downloads a previously uploaded file.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. unique ID of the item
        ===============     ====================================================================

        :returns: str
        """
        url = self._url + "/%s/download" % item_id
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def item(self, item_id):
        """
        This resource represents an item that has been uploaded to the
        kubernetes site. Various workflows upload items and then process them on the
        kubernetes site. For example, when publishing a GIS service from ArcGIS for
        Desktop or ArcGIS kubernetes site Manager, the application first uploads the
        service definition (.SD) to the kubernetes site and then invokes the
        publishing geoprocessing tool to publish the service.
        Each uploaded item is identified by a unique name (item_id). The
        pathOnkubernetes site property locates the specific item in the ArcGIS
        kubernetes site system directory.
        The committed parameter is set to true once the upload of
        individual parts is complete.

        Parameters:
         :item_id: uploaded id identifier
        """
        url = self._url + "/%s" % item_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def upload(self, path, description=None):
        """
        Uploads a new item to the kubernetes site. Once the operation is completed
        successfully, the JSON structure of the uploaded item is returned.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        path                Required string. The file location to upload
        ---------------     --------------------------------------------------------------------
        description         Optional string. Description of the upload.
        ===============     ====================================================================


        :return: boolean


        """
        url = self._url + "/upload"
        params = {"f": "json"}
        files = {}
        files["itemFile"] = path
        if description:
            params["description"] = description
        res = self._con.post(path=url, postdata=params, files=files)
        if "status" in res and res["status"] == "success":
            return True, res
        return False, res

    # ----------------------------------------------------------------------
    def _service_configuration(self, upload_id):
        """gets the serviceconfiguration.json info for an uploaded sd file"""
        url = self._url + "/%s/serviceconfiguration.json" % upload_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def _initial_cache_settings(self, upload_id):
        """gets the initial cache settings for a given uploaded sd file"""
        url = self._url + "/%s/serviceconfiguration.json" % upload_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def upload_by_part(self, item_id, part_number, part):
        """
        Uploads a new item to the kubernetes site. Once the operation is completed
        successfully, the JSON structure of the uploaded item is returned.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. Item ID to upload to.
        ---------------     --------------------------------------------------------------------
        part_number         Required int. An integer value associated with the part.
        ---------------     --------------------------------------------------------------------
        part                Required string. File path to the part to upload.
        ===============     ====================================================================


        :return: dict

        """
        url = self._url + "{iid}/uploadPart".format(iid=item_id)
        params = {"f": "json", "partNumber": part_number}
        files = {}
        files["partFile"] = part
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def commit(self, item_id, parts=None):
        """
        Use this operation to complete the upload of all the parts that
        make an item. The parts parameter indicates to the kubernetes site all the
        parts that make up the item.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. Item ID to commit.
        ---------------     --------------------------------------------------------------------
        parts               Optional list. An optional comma-separated ordered list of all the
                            parts that make the item. If this parameter is not provided, the
                            default order of the parts is used.
        ===============     ====================================================================


        :return: Boolean

        """
        params = {"f": "json"}
        url = self._url + "/{iid}/commit".format(iid=item_id)
        if parts:
            params["parts"] = parts
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res
