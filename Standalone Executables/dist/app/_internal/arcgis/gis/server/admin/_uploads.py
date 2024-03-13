"""
This resource is a collection of all the items that have been uploaded
to the server.

There are two ways to upload items. You can upload complete items using
the Upload Item operation. If a particular item is made up of many
chunks (parts), you need to first register the item and subsequently
upload the individual parts using the Upload Part operation. Item
uploads are filtered by a whitelist of filename extensions. This is the
default list: soe, sd, sde, odc, csv, txt, zshp, kmz. The default list
can be overridden by setting the uploadFileExtensionWhitelist property
with the server properties API.

"""
from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
from arcgis.gis import GIS
from typing import Optional


########################################################################
class Uploads(BaseServer):
    """
    This resource is a collection of all the items that have been uploaded
    to the server.

    There are two ways to upload items. You can upload complete items using
    the Upload Item operation. If a particular item is made up of many
    chunks (parts), you need to first register the item and subsequently
    upload the individual parts using the Upload Part operation. Item
    uploads are filtered by a whitelist of filename extensions. This is the
    default list: soe, sd, sde, odc, csv, txt, zshp, kmz. The default list
    can be overridden by setting the uploadFileExtensionWhitelist property
    with the server properties API.

    """

    _uploads = None
    _con = None
    _json = None
    _json_dict = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
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
    def uploads(self) -> dict:
        """
        returns a collection of all the items that have been uploaded to
        the server.

        There are two ways to upload items. You can upload complete items
        using the Upload Item operation. If a particular item is made up of
        many chunks (parts), you need to first register the item and
        subsequently upload the individual parts using the Upload Part
        operation. Item uploads are filtered by a whitelist of filename
        extensions. This is the default list: soe, sd, sde, odc, csv, txt,
        zshp, kmz. The default list can be overridden by setting the
        uploadFileExtensionWhitelist property with the server properties API.

        """
        params = {"f": "json"}
        return self._con.get(path=self._url, params=params)

    # ----------------------------------------------------------------------
    def delete(self, item_id: str) -> bool:
        """
        Deletes the uploaded item and its configuration.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. unique ID of the item
        ===============     ====================================================================


        :return: Boolean. True if successful else False.

        """
        url = self._url + "/%s/delete" % item_id
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def item(self, item_id: str) -> dict:
        """
        This resource represents an item that has been uploaded to the
        server. Various workflows upload items and then process them on the
        server. For example, when publishing a GIS service from ArcGIS for
        Desktop or ArcGIS Server Manager, the application first uploads the
        service definition (.SD) to the server and then invokes the
        publishing geoprocessing tool to publish the service.
        Each uploaded item is identified by a unique name (item_id). The
        pathOnServer property locates the specific item in the ArcGIS
        Server system directory.
        The committed parameter is set to true once the upload of
        individual parts is complete.

        Parameters:
         :item_id: uploaded id identifier
        """
        url = self._url + "/%s" % item_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def upload(self, path: str, description: Optional[str] = None) -> bool:
        """
        Uploads a new item to the server. Once the operation is completed
        successfully, the JSON structure of the uploaded item is returned.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        path                Required string. The file location to upload
        ---------------     --------------------------------------------------------------------
        description         Optional string. Description of the upload.
        ===============     ====================================================================


        :return: Boolean. True if successful else False.


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
    def _service_configuration(self, upload_id: str) -> dict:
        """gets the serviceconfiguration.json info for an uploaded sd file"""
        url = self._url + "/%s/serviceconfiguration.json" % upload_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def _initial_cache_settings(self, upload_id: str) -> dict:
        """gets the initial cache settings for a given uploaded sd file"""
        url = self._url + "/%s/serviceconfiguration.json" % upload_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def upload_by_part(self, item_id: str, part_number: int, part: str) -> dict:
        """
        Uploads a new item to the server. Once the operation is completed
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


        :return: Dictionary indicating 'success' or 'error'

        """
        url = self._url + "{iid}/uploadPart".format(iid=item_id)
        params = {"f": "json", "partNumber": part_number}
        files = {}
        files["partFile"] = part
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def commit(self, item_id: str, parts: Optional[list] = None) -> bool:
        """
        Use this operation to complete the upload of all the parts that
        make an item. The parts parameter indicates to the server all the
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


        :return: Boolean. True if successful else False.

        """
        params = {"f": "json"}
        url = self._url + "/{iid}/commit".format(iid=item_id)
        if parts:
            params["parts"] = parts
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res
