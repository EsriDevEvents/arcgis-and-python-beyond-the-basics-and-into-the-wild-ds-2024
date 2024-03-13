"""
Updates the Federation Settings to Portal
"""
from typing import Optional
from .. import GIS
from ._base import BasePortalAdmin

########################################################################


class Federation(BasePortalAdmin):
    """
    This resource returns information about the ArcGIS Servers registered
    with Portal for ArcGIS.
    """

    _gis = None
    _url = None
    _con = None
    _portal = None
    # ----------------------------------------------------------------------

    def __init__(self, url, gis):
        """Constructor"""
        if isinstance(gis, GIS):
            url = url.replace("http://", "https://")
            self._url = url
            self._gis = gis
            self._portal = gis._portal
            self._con = gis._con
        else:
            raise ValueError("gis object must be of type GIS")

    # ----------------------------------------------------------------------
    def federate(self, url: str, admin_url: str, username: str, password: str):
        """
        This operation enables ArcGIS Servers to be federated with Portal
        for ArcGIS.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        url:                            Required string. The URL of the GIS server used by external users
                                        when accessing the ArcGIS Server site. If the site includes the Web
                                        Adaptor, the URL includes the Web Adaptor address, for example,
                                        ``https://webadaptor.domain.com/arcgis``. If you've added ArcGIS
                                        Server to your organization's reverse proxy server, the URL is
                                        the reverse proxy server address (for example,
                                        ``https://reverseproxy.domain.com/myorg``). Note that the federation
                                        operation will perform a validation check to determine if the
                                        provided URL is accessible from the server site. If the resulting
                                        validation check fails, a warning will be generated in the Portal
                                        for ArcGIS logs. However, federation will not fail if the URL is
                                        not validated, as the URL may not be accessible from the server
                                        site, such as is the case when the server site is behind a
                                        firewall.
        ---------------------------     --------------------------------------------------------------------
        admin_url                       Required string. The URL used for accessing ArcGIS Server when
                                        performing administrative operations on the internal network, for
                                        example, ``https://gisserver.domain.com:6443/arcgis``.
        ---------------------------     --------------------------------------------------------------------
        username                        Required string. The username of the primary site administrator account
        ---------------------------     --------------------------------------------------------------------
        password                        Required string. password of the username above.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'
        """
        fedurl = "%s/servers/federate" % self._url
        params = {
            "f": "json",
            "url": url,
            "adminUrl": admin_url,
            "username": username,
            "password": password,
        }
        res = self._con.post(path=fedurl, postdata=params)
        if "status" in res:
            return res["status"]
        return res

    # ----------------------------------------------------------------------
    @property
    def servers(self):
        """
        This resource returns detailed information about the ArcGIS Servers
        registered with Portal for ArcGIS, such as the ID of the server,
        name of the server, ArcGIS Web Adaptor URL, administration URL, and
        if the server is set as a hosting server.
        """
        url = "%s/servers" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def unfederate(self, server_id: str):
        """
        This operation unfederates an ArcGIS Server from Portal for ArcGIS.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        server_id                       Required string. The unique ID of the server
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = "%s/servers/%s/unfederate" % (self._url, server_id)
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    def update(self, server_id: str, role: str, function: Optional[str] = None):
        """
        This operation allows you to set an ArcGIS Server federated with
        Portal for ArcGIS as the hosting server or to enforce fine-grained
        access control to a federated server. You can also remove hosting
        server status from an ArcGIS Server. You can also remove hosting
        server status from an ArcGIS Server. To set a hosting server, an
        enterprise geodatabase must be registered as a managed database
        with the ArcGIS Server.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        server_id                       Required string. The unique ID of the server
        ---------------------------     --------------------------------------------------------------------
        role                            Required string. Whether the server is a hosting server for the
                                        portal, a federated server, or a server with restricted access to
                                        publishing. The allowed values are:
                                        FEDERATED_SERVER, FEDERATED_SERVER_WITH_RESTRICTED_PUBLISHING,
                                        or HOSTING_SERVER.
        ---------------------------     --------------------------------------------------------------------
        function                        Optional string. This is the purpose of the ArcGIS Server.
                                        Values are: GeoAnalytics, RasterAnalytics, ImageHosting, NotebookServer, MissionServer, WorkflowManager, or None
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        role_allow = [
            "FEDERATED_SERVER",
            "FEDERATED_SERVER_WITH_RESTRICTED_PUBLISHING",
            "HOSTING_SERVER",
        ]
        function_allow = [
            "GeoAnalytics",
            "RasterAnalytics",
            "ImageHosting",
            "NotebookServer",
            "MissionServer",
            "WorkflowManager",
        ]
        if role.upper() in role_allow:
            role = role.upper()
        else:
            raise ValueError("Invalid role type")
        if function and function not in function_allow:
            raise ValueError("Invalid function")
        params = {
            "f": "json",
            "serverRole": role,
        }
        if function:
            params["serverFunction"] = function
        url = "%s/servers/%s/update" % (self._url, server_id)
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def validate(self, server_id: str):
        """
        This operation provides status information about a specific ArcGIS
        Server federated with Portal for ArcGIS.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        server_id                       Required string. The unique ID of the server
        ===========================     ====================================================================

        :return: Dictionary

        """
        params = {"f": "json"}
        url = "%s/servers/%s/validate" % (self._url, server_id)
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def validate_all(self):
        """
        This operation returns information on the status of ArcGIS Servers
        registered with Portal for ArcGIS.
        """
        params = {"f": "json"}
        url = "%s/servers/validate" % (self._url)
        return self._con.get(path=url, params=params)
