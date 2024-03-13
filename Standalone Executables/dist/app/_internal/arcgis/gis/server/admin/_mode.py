"""
ArcGIS Server site mode that allows you to control changes to your site.
You can set the site mode to READ_ONLY to disallow the publishing of new
services and block most administrative operations. Your existing services
will continue to function as they did previously. Note that certain
administrative operations such as adding and removing machines from a
site are still available in READ_ONLY mode.
"""
from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
from arcgis._impl.common._deprecate import deprecated
from arcgis.gis import GIS


###########################################################################
class Mode(BaseServer):
    """
    ArcGIS Server site mode that allows you to control changes to your site.
    You can set the site mode to READ_ONLY to disallow the publishing of new
    services and block most administrative operations. Your existing services
    will continue to function as they did previously. Note that certain
    administrative operations such as adding and removing machines from a
    site are still available in READ_ONLY mode.
    """

    _url = None
    _con = None
    _json_dict = None
    _json = None
    _siteMode = None
    _copyConfigLocal = None
    _lastModified = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor"""
        super(Mode, self).__init__(gis=gis, url=url)
        if url.lower().endswith("/mode"):
            self._url = url
        else:
            self._url = url + "/mode"
        self._con = gis
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    @deprecated(
        deprecated_in="1.7.1",
        removed_in=None,
        current_version="2.2.0",
        details="Use `Mode.update_mode` instead.",
    )
    def update(self, siteMode: str, runAsync: bool = False) -> bool:
        """
        The update operation is used to move between the two types of site
        modes. Switching to READ_ONLY mode will restart all your services
        as the default behavior. Moving to EDITABLE mode will not restart
        services.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        siteMode:           Required string. The mode you will set your site to. Values:
                            READ_ONLY or EDITABLE.
        ---------------     --------------------------------------------------------------------
        runAsync            Optional boolean. Determines if this operation must run asynchronously.
        ===============     ====================================================================


        :return: Boolean

        """
        params = {"siteMode": siteMode, "runAsync": runAsync, "f": "json"}
        url = self._url + "/update"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def update_mode(
        self,
        site_mode: str,
        run_async: bool = False,
        allow_editing: bool = True,
    ) -> bool:
        """
        The update operation is used to move between the two types of site
        modes. Switching to READ_ONLY mode will restart all your services
        as the default behavior. Moving to EDITABLE mode will not restart
        services.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        site_mode           Required string. The mode you will set your site to. Values:
                            READ_ONLY or EDITABLE.
        ---------------     --------------------------------------------------------------------
        run_async           Optional boolean. Determines if this operation must run asynchronously.
        ---------------     --------------------------------------------------------------------
        allow_editing       Optional boolean. Specifies if edits to feature services are allowed
                            while a Server is in read-only mode. The default value is true.
        ===============     ====================================================================


        :return: Boolean

        """
        params = {
            "siteMode": site_mode,
            "runAsync": run_async,
            "allowEditingViaServices": allow_editing,
            "f": "json",
        }
        url = self._url + "/update"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res
