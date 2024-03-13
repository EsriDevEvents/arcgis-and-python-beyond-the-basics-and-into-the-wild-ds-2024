from typing import Optional
import arcgis
from arcgis import gis
from arcgis._impl.common._isd import InsensitiveDict
from .admin.administration import Server
import logging
from functools import lru_cache

_log = logging.getLogger(__name__)


###########################################################################
class ServerManager(object):
    """
    Helper class for managing your ArcGIS Servers. This class is not created
    by users directly. An instance of this class, called 'servers',
    is available as a property of the gis.admin object. Administrators call methods
    on this :class:`ServerManager` object to manage and interrogate ArcGIS Servers.
    """

    _gis = None
    _catalog_list = None
    _server_list = None
    _portal = None
    _gis = None
    _pa = None
    _federation = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal
        self._pa = gis.admin
        self._federation = self._pa.federation
        self._server_list = None
        self._catalog_list = None

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._pa._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._pa._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        The :class:`~arcgis.gis.server.ServerManager` properties

        :return: Dict
        """
        res = self._portal.con.post("portals/self/servers", {"f": "json"})
        return InsensitiveDict(res)

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=100)
    def list(self):
        """
        The ``list`` method retrieves all servers in a :class:`~arcgis.gis.GIS`, retrieving a list of admin services.

        .. note::
           This method is not to be confused with the :attr:`~arcgis.server.ServicesDirectory.list` method, in the
           :class:`~arcgis.server.ServicesDirectory` class, which returns a variety of services, such as a ``Feature Service``,
           ``Map Service``, ``Vector Tile``, ``Geoprocessing Service``, etc.

        :return:
           A list of all servers (in the form of admin service objects) found in the :class:`~arcgis.gis.GIS`.
        """

        from . import ServicesDirectory

        if self._server_list is not None:
            return self._server_list
        self._server_list = []
        self._catalog_list = []
        res = self._portal.con.post("portals/self/servers", {"f": "json"})
        servers = res["servers"]
        admin_url = None
        public_url = None
        for server in servers:
            admin_url = server["adminUrl"]
            public_url = server["url"]
            try:
                if server["serverFunction"] == "NotebookServer":
                    try:
                        from arcgis.gis.nb import NotebookServer

                        nbs = NotebookServer(url=admin_url, gis=self._gis)
                        nbs.info
                        self._server_list.append(nbs)
                    except:
                        from arcgis.gis.nb import NotebookServer

                        nbs = NotebookServer(url=public_url, gis=self._gis)
                        nbs.info
                        self._server_list.append(nbs)
                elif server["serverFunction"] == "MissionServer":
                    from arcgis.gis.mission import MissionServer

                    try:
                        ms = MissionServer(url=admin_url, gis=self._gis)
                        ms.info
                        self._server_list.append(ms)
                    except:
                        ms = MissionServer(url=public_url, gis=self._gis)
                        ms.info
                        self._server_list.append(ms)
                else:
                    try:
                        c = ServicesDirectory(
                            url=admin_url, portal_connection=self._gis._portal.con
                        )
                        c.admin.logs
                        self._server_list.append(c.admin)
                        self._catalog_list.append(c)
                    except:
                        c = ServicesDirectory(
                            url=public_url, portal_connection=self._gis._portal.con
                        )
                        self._server_list.append(c.admin)
                        self._catalog_list.append(c)

            except:
                _log.warning("Could not access the server at " + admin_url)

        return self._server_list

    # ----------------------------------------------------------------------
    def get(self, role: Optional[str] = None, function: Optional[str] = None):
        """
        Retrieves the ArcGIS Server(s) by role or function. While each argument is optional,
        at least one argument must be set with an allowed value other than None.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        role                   Optional string. Limits the returned ArcGIS Servers based on the
                               server's role as either a hosting server for the portal, a federated server,
                               or a server with restricted access to publishing. The allowed values
                               are HOSTING_SERVER, FEDERATED_SERVER, or FEDERATED_SERVER_WITH_RESTRICTED_PUBLISHING,
                               respectively.
        ------------------     --------------------------------------------------------------------
        function               Optional string. Limits the returned ArcGIS Servers based on the
                               server's function. Provide a comma-separated list of values. The
                               allowed values are GeoAnalytics, RasterAnalytics, and ImageHosting.
        ==================     ====================================================================

        :return:
           The ArcGIS Server(s) discovered that match the criteria.
        """
        servers = []
        if role is None and function is None:
            raise ValueError("A role or function must be provided")
        for server in self._federation.servers["servers"]:
            if str(role).lower() == server["serverRole"].lower():
                servers.append(Server(url=server["adminUrl"], gis=self._gis))
            elif str(function).lower() in server["serverFunction"].lower():
                servers.append(Server(url=server["adminUrl"], gis=self._gis))
        return servers

    # ----------------------------------------------------------------------
    @property
    def _server_info(self):
        """
        Gets federation information for all servers associated the WebGIS.
        """
        return self._federation.servers["servers"]

    # ----------------------------------------------------------------------
    def _federate(self, url, admin_url, username, password):
        """
        This operation enables ArcGIS Servers to be federated with Portal
        for ArcGIS.

        .. note::
            For the url argument - If the site includes the Web Adaptor,
            the URL includes the Web Adaptor address, for example,
            https://webadaptor.domain.com/arcgis. If you've added ArcGIS Server
            to your organization's reverse proxy server, the URL is the reverse
            proxy server address (for example,
            https://reverseproxy.domain.com/myorg). Note that the federation
            operation will perform a validation check to determine if the
            provided URL is accessible from the server site. If the resulting
            validation check fails, a warning will be generated in the Portal
            for ArcGIS logs. However, federation will not fail if the URL is
            not validated, as the URL may not be accessible from the server
            site, such as is the case when the server site is behind a firewall.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The URL of the GIS server used by external users when
                               accessing the ArcGIS Server site.  See note above.
        ------------------     --------------------------------------------------------------------
        admin_url              Required string. The URL used for accessing ArcGIS Server when
                               performing administrative operations on the internal network, for
                               example, https://gisserver.domain.com:6443/arcgis.
        ------------------     --------------------------------------------------------------------
        username               Required string. The username of the primary site administrator account.
        ------------------     --------------------------------------------------------------------
        password               Required integer. The password of the username above.
        ==================     ====================================================================


        :return:
           The server response with server ID.
        """
        res = self._federation.federate(url, admin_url, username, password)
        self._server_list = None
        return res

    # ----------------------------------------------------------------------
    def _unfederate(self, server_id):
        """
        This operation unfederates an ArcGIS Server from Portal for ArcGIS.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        server_id              Required string. The unique ID of the server.
        ==================     ====================================================================


        :return:
           The server response with server ID.
        """
        res = self._federation(server_id)
        self._server_list = None
        return res

    # ----------------------------------------------------------------------
    def update(self, server: str, role: str, function: Optional[str] = None):
        """
        This operation allows you to set an ArcGIS Server federated with
        Portal for ArcGIS as the hosting server or to enforce fine-grained
        access control to a federated server. You can also remove hosting
        server status from an ArcGIS Server. To set a hosting server, an
        enterprise geodatabase must be registered as a managed database
        with the ArcGIS Server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        server                 Required string. The arcgis.gis.Server object.
        ------------------     --------------------------------------------------------------------
        role                   Required string. State whether the server is either a hosting server
                               for the portal, a federated server, or a server with restricted access
                               to publishing. The allowed values are HOSTING_SERVER, FEDERATED_SERVER,
                               or FEDERATED_SERVER_WITH_RESTRICTED_PUBLISHING, respectively.
        ------------------     --------------------------------------------------------------------
        function               Optional string. The specific function associated with this server. Provide a
                               comma-separated list of values, but it is not recommend that a single
                               server have all the server functions. The allowed values are GeoAnalytics,
                               RasterAnalytics, and ImageHosting.
        ==================     ====================================================================


        :return:
           A status message of 'success' with the ID of the ArcGIS Server.
        """
        if isinstance(server, Server) == False:
            raise ValueError("server must be of type arcgis.gis.Server")
        roles = [
            "FEDERATED_SERVER",
            "FEDERATED_SERVER_WITH_RESTRICTED_PUBLISHING",
            "HOSTING_SERVER",
        ]
        functions = {
            "geoanalytics": "GeoAnalytics",
            "rasteranalytics": "RasterAnalytics",
            "imagehosting": "ImageHosting",
            "none": None,
        }
        if not role.upper() in roles:
            raise ValueError("Invalid role, allowed values: %s" % ",".join(roles))
        if not str(function).lower() in functions.keys():
            raise ValueError(
                "Invalid function, allowed values: %s" % ",".join(functions.keys())
            )
        else:
            function = functions[str(function).lower()]
        server_id = None
        from urllib.parse import urlparse

        b = urlparse(url=server._admin_url).netloc.lower()
        for s in self._server_info:
            if b == urlparse(s["adminUrl"].lower()).netloc:
                server_id = s["id"]
                break
            del s
        return self._federation.update(server_id, role, function)

    # ----------------------------------------------------------------------
    def validate(self):
        """
        This operation returns information on the status of ArcGIS Servers
        registered with Portal for ArcGIS.

        :return:
           True if all servers are functioning as expected, False if there is an
           issue with 1 or more of the Federated Servers.
        """
        return self._federation.validate_all()["status"] == "success"
