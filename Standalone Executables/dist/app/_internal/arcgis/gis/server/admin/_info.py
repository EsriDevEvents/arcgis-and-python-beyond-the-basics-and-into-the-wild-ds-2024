from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
from arcgis.gis import GIS


########################################################################
class Info(BaseServer):
    """
    A read-only resource that returns meta information about the server.
    """

    _con = None
    _json_dict = None
    _url = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor
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
        super(Info, self).__init__(gis=gis, url=url)
        self._con = gis
        self._url = url
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def available_time_zones(self) -> dict:
        """
        Returns an enumeration of all the time zones of which the server
        is aware. This is used by the GIS service publishing tools
        """
        url = self._url + "/getAvailableTimeZones"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)
