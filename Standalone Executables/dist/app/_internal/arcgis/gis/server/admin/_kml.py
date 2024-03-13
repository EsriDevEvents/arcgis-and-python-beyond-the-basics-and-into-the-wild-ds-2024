"""
This resource is a container for all the KMZ files created on the
server.s
"""
from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
from arcgis.gis import GIS


########################################################################
class KML(BaseServer):
    """
    This resource is a container for all the KMZ files created on the
    server.
    """

    _con = None
    _url = None
    _json_dict = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """
        Constructor

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required string. The administration URL for the ArcGIS Server.
        ---------------     --------------------------------------------------------------------
        gis                 Required Server object. Connection object.
        ---------------     --------------------------------------------------------------------
        initialize          Optional boolean. If true, information loaded at object
        ===============     ====================================================================

        """
        super(KML, self).__init__(gis=gis, url=url)
        self._con = gis
        self._url = url
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def create_KMZ(self, kmz_as_json: dict) -> dict:
        """
        Creates a KMZ file from json.
        See https://developers.arcgis.com/rest/enterprise-administration/server/createkmz.htm
        for more information.
        """
        url = self._url + "/createKmz"
        params = {"f": "json", "kml": kmz_as_json}
        return self._con.post(path=url, postdata=params)
