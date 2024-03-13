"""
The System resource is a collection of miscellaneous server-wide
resources such as server properties, server directories, the
configuration store, Web Adaptors, and licenses.
"""
from __future__ import annotations
from __future__ import absolute_import
from __future__ import print_function
from .._common import BaseServer
from arcgis.gis import GIS
from arcgis.gis._impl._con import Connection
from typing import Optional


########################################################################
class SystemManager(BaseServer):
    """
    The System resource is a collection of miscellaneous server-wide
    resources such as server properties, server directories, the
    configuration store, Web Adaptors, and licenses.
    """

    _json = None
    _json_dict = None
    _con = None
    _url = None
    _resources = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """

        super(SystemManager, self).__init__(gis=gis, url=url)
        self._con = gis
        if url.lower().endswith("/system"):
            self._url = url
        else:
            self._url = url + "/system"
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def server_properties(self) -> "ServerProperties":
        """
        Gets the server properties for the site as an object.

        :return:
            :class:`~arcgis.gis.server.ServerProperties` object

        """
        return ServerProperties(
            url=self._url + "/properties", connection=self._con, initialize=True
        )

    # ----------------------------------------------------------------------
    @property
    def _directories(self) -> list:
        """
        Gets the server directory object as a list.

        """
        directs = []
        url = self._url + "/directories"
        params = {"f": "json", "private": False}
        res = self._con.get(path=url, params=params)
        for direct in res["directories"]:
            directs.append(
                ServerDirectory(
                    url=url + "/%s" % direct["name"],
                    connection=self._con,
                    initialize=True,
                )
            )
        return directs

    # ----------------------------------------------------------------------
    @property
    def directories(self) -> "DirectoryManager":
        """
        :return:
            The :class:`~arcgis.gis.server.ServerDirectory` object in a list.
        """
        return DirectoryManager(system=self)

    # ----------------------------------------------------------------------
    def _get_directory(self, name: str) -> "ServerDirectory":
        """
        Retrieves a single directory registered with ArcGIS Server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the registered directory.
        ==================     ====================================================================

        :return:
            The ArcGIS Server directory as an object or None.


        """
        url = self._url + "/directories"
        params = {"f": "json"}
        res = self._con.get(path=url, params=params)
        for direct in res["directories"]:
            if name.lower() == direct["name"].lower():
                return ServerDirectory(
                    url=url + "/%s" % direct["name"],
                    connection=self._con,
                    initialize=True,
                )
        return None

    # ----------------------------------------------------------------------
    def _register(
        self,
        name: str,
        physical_path: str,
        directory_type: str,
        max_age: int,
        cleanup_mode: str = "NONE",
        description: Optional[str] = None,
    ) -> bool:
        """
        Registers a new server directory. While registering the server
        directory, you can also specify the directory's cleanup parameters.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the server directory.
        ------------------     --------------------------------------------------------------------
        physical_path          Required string. The absolute physical path of the server directory.
        ------------------     --------------------------------------------------------------------
        directory_type         Required string. The type of server directory.
        ------------------     --------------------------------------------------------------------
        max_age                Required integer. The length of time a file in the directory needs
                               to be kept before it is deleted.
        ------------------     --------------------------------------------------------------------
        cleanup_mode           Optional string. Defines if files in the server directory needs to
                               be cleaned up. The default is None.
        ------------------     --------------------------------------------------------------------
        description            Optional string. An optional description for the server directory.
                               The default is None.
        ==================     ====================================================================


        :return:
             A boolean indicating success (True).

        """
        url = self._url + "/directories/register"
        params = {
            "f": "json",
            "name": name,
            "physicalPath": physical_path,
            "directoryType": directory_type,
            "cleanupMode": cleanup_mode,
            "maxFileAge": max_age,
        }
        if description:
            params["description"] = description
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def licenses(self) -> dict:
        """
        Gets the license resource list.  The licenses resource lists the
        current license level of ArcGIS for Server and all authorized
        extensions. Contact Esri Customer Service if you have questions
        about license levels or expiration properties.
        """
        url = self._url + "/licenses"
        params = {"f": "json"}
        return self._con.get(path=url, postdata=params)

    # ----------------------------------------------------------------------
    @property
    def platform_services(self) -> "PlatformServiceManager":
        """
        Provides access to the platform services that are associated with GeoAnalytics.
        """
        url = self._url + "/platformservices"
        return PlatformServiceManager(url=url, connection=self._con)

    # ----------------------------------------------------------------------
    @property
    def jobs(self) -> "Jobs":
        """
        Gets the Jobs object.

        :return:
            :class:`~arcgis.gis.server.Jobs` object
        """
        url = self._url + "/jobs"
        return Jobs(url=url, connection=self._con, initialize=True)

    # ----------------------------------------------------------------------
    @property
    def web_adaptors(self) -> dict:
        """
        Gets a list of all the Web Adaptors that have been registered
        with the site. The server will trust all these Web Adaptors and
        will authorize calls from these servers.

        To configure a new Web Adaptor with the server, you'll need to use
        the configuration web page or the command line utility. For full
        instructions, see Configuring the Web Adaptor after installation.
        """
        url = self._url + "/webadaptors"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def web_adaptors_configuration(self) -> dict:
        """
        Gets the Web Adaptors configuration which is a resource of all the
        configuration parameters shared across all the Web Adaptors in the
        site. Most importantly, this resource lists the shared key that is
        used by all the Web Adaptors to encrypt key data bits for the
        incoming requests to the server.
        """
        url = self._url + "/webadaptors/config"
        params = {"f": "json"}
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def update_web_adaptors_configuration(self, config: str) -> bool:
        """
        You can use this operation to change the Web Adaptor configuration
        and the sharedkey attribute. The sharedkey attribute must be present
        in the request.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required string. The configuration items to be updated for this web
                               adaptor. Always include the web adaptor's sharedkey attribute.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True), else a Python dictionary containing an error message.

        """
        url = self._url + "/webadaptors/config/update"
        params = {"f": "json", "webAdaptorConfig": config}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def update_web_adaptor(
        self, wa_id: str, description: str, http_port: int, https_port: int
    ) -> bool:
        """
        This operation allows you to update the description, HTTP port, and
        HTTPS port of a Web Adaptor that is registered with the server.

        .. note::
            This operation is only meant to change the descriptive properties
            of the Web Adaptor and does not affect the configuration of the web
            server that deploys your Web Adaptor.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        wa_id                  Required string. The web adaptor ID.
        ------------------     --------------------------------------------------------------------
        description            Required string. A description of this web adaptor.
        ------------------     --------------------------------------------------------------------
        http_port              Required integer. The HTTP port of the web server.
        ------------------     --------------------------------------------------------------------
        https_port             Required integer. The HTTPS (SSL) port of the web server that runs
                               the web adaptor.
        ==================     ====================================================================


        :return:
            A boolean indicating success (True), else a Python dictionary containing an error message.

        """
        url = self._url + "/webadaptors/{w}/update".format(w=wa_id)
        params = {
            "f": "json",
            "description": description,
            "httpPort": http_port,
            "httpsPort": https_port,
        }
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def unregister_webadaptor(self, wa_id: str) -> bool:
        """
        Unregistering a Web Adaptor removes the Web Adaptor from the ArcGIS
        Server's trusted list. The Web Adaptor can no longer submit requests
        to the server.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        wa_id                  Required string. The web adaptor ID.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/webadaptors/{waid}/update".format(waid=wa_id)
        params = {
            "f": "json",
        }
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def configuration_store(self) -> "ConfigurationStore":
        """
        Gets the ConfigurationStore object for this site.

        :return:
            :class:`~arcgis.gis.server.ConfigurationStore`

        """
        url = self._url + "/configstore"

        return ConfigurationStore(url=url, connection=self._con)

    # ----------------------------------------------------------------------
    def clear_cache(self) -> bool:
        """
        This operation clears the cache on all REST handlers in the system.
        While the server typically manages the REST cache for you, use this
        operation to get explicit control over the cache.

        :return:
            A boolean indicating success (True).

        """
        params = {"f": "json"}
        url = self._url + "/handlers/rest/cache/clear"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def deployment(self) -> dict:
        """
        Gets the load balancing value for this site.  Load balancing is an
        ArcGIS Server deployment configuration resource that can
        control the load balancing functionality between GIS server
        machines. A value of True means that load balancing is disabled.

        singleClusterMode - At 10.4, in large sites with a single cluster,
        the site is configured to prevent load balancing between GIS
        server machines. This reduces network traffic between machines
        in the site and helps reduce load on your network. Upgrades
        from earlier versions will  set this property to true if the
        site uses a single cluster. Sites with multiple clusters cannot
        use singleClusterMode.

        To prevent load balancing, the following criteria must be met:

         - All machines in the site must participate in a single
           cluster. Multiple clusters cannot exist.
         - An external load balancer or ArcGIS Web Adaptor must be
           configured to forward requests to the site. If no external
           gateway exists, requests will only be handled by the
           machine designated in the request.

        To enable load balancing, set this property to false. Updating
        this property will restart all machines in the site.
        """
        url = self._url + "/deployment"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def soap_config(self) -> dict:
        """
        The `soap_config` resource lists the URLs for domains allowed to
        make cross-domain requests, including SOAP and OGC service requests.
        If the value for `origins` is not updated, no restrictions on
        cross-domain requests will be made.

        The `set` operation allows you to restrict cross-domain requests to
        specific domains, including SOAP and OGC service requests. By default,
        no domains are restricted.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        origins                Optional String. A comma-separated list of URLs of domains allowed
                               to make requests. The default value, *, denotes all domains, meaning
                               none are restricted.
        ==================     ====================================================================

        :returns: dict
        """
        url = self._url + "/handlers/soap/soaphandlerconfig"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @soap_config.setter
    def soap_config(self, origins: str):
        """
        The `set` operation allows you to restrict cross-domain requests to
        specific domains, including SOAP and OGC service requests. By default,
        no domains are restricted.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        origins                Optional String. A comma-separated list of URLs of domains allowed
                               to make requests. The default value, *, denotes all domains, meaning
                               none are restricted.
        ==================     ====================================================================

        :returns: dict
        """
        params = {"f": "json", "allowedOrigins": origins}
        url = self._url + "/handlers/soap/soaphandlerconfig/edit"
        self._con.post(url, params)

    # ----------------------------------------------------------------------
    def _edit_services_directory(
        self,
        allowed_origins: str,
        arcgis_com_map: str,
        arcgis_com_map_text: str,
        jsapi_arcgis: str,
        jsapi_arcgis_css: str,
        jsapi_arcgis_css2: str,
        jsapi_arcgis_sdk: str,
        service_dir_enabled: str,
        callback_functions: bool | None = None,
        map_text: str | None = None,
        arcgis_map: str | None = None,
    ) -> bool:
        """
        Allows you to update the Services Directory configuration.  You can do such thing as
        enable or disable the HTML view of ArcGIS REST API, or adjust the JavaScript and map viewer
        previews of services in the Services Directory so that they work with your own locally
        hosted JavaScript API and map viewer.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        allowed_origins          Required string. A comma-separated list of URLs of domains allowed to
                                 make requests. An asterisk (*) can be used to denote all domains.
        --------------------     --------------------------------------------------------------------
        arcgis_com_map           Required string. URL of the map viewer application used for service
                                 previews. Defaults to the ArcGIS.com map viewer but could be used
                                 to point at your own Portal for ArcGIS map viewer.
        --------------------     --------------------------------------------------------------------
        arcgis_com_map_text      Required string. The text to use for the preview link that opens
                                 the map viewer.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis             Required string. The URL of the JavaScript API to use for service
                                 previews. Defaults to the online ArcGIS API for JavaScript, but
                                 could be pointed at your own locally-installed instance of the
                                 JavaScript API.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_css         Required string. The CSS file associated with the ArcGIS API for
                                 JavaScript. Defaults to the online Dojo tundra.css.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_css2        Required string. An additional CSS file associated with the ArcGIS
                                 API for JavaScript. Defaults to the online esri.css.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_sdk         Required string. The URL of the ArcGIS API for JavaScript help.
        --------------------     --------------------------------------------------------------------
        service_dir_enabled      Required string. Flag to enable/disable the HTML view of the
                                 services directory.
        --------------------     --------------------------------------------------------------------
        callback_functions       Optional boolean. Introduced at 11.0. The flag to enable or disable
                                 the ability to make JSONP callback requests. The JSONP callback
                                 feature is enabled by default (true) and allows older clients a way
                                 to make CORS requests without being restricted by the same-origin
                                 policy. This is useful for older browsers or other clients that do
                                 not supports CORS requests.
        ====================     ====================================================================


        :return:
            A boolean indicating success (True).

        """
        params = {
            "f": "json",
            "allowedOrigins": allowed_origins,
            "arcgis.com.map": arcgis_com_map,
            "arcgis.com.map.text": arcgis_com_map_text,
            "jsapi.arcgis": jsapi_arcgis,
            "jsapi.arcgis.css": jsapi_arcgis_css,
            "jsapi.arcgis.css2": jsapi_arcgis_css2,
            "jsapi.arcgis.sdk": jsapi_arcgis_sdk,
            "servicesDirEnabled": service_dir_enabled,
            "callbackFunctionsEnabled": callback_functions,
        }
        keys = list(params.keys())
        for key in keys:
            if params[key] is None:
                del params[key]
        url = self._url + "/handlers/rest/servicesdirectory/edit"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def _services_directory(self) -> dict:
        """returns the Server directory properties"""
        url = self._url + "/handlers/rest/servicesdirectory"
        params = {"f": "json"}
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    @property
    def handlers(self) -> dict:
        """
        Gets the handler of this server. A handler exposes the GIS capabilities of ArcGIS Server through a
        specific interface/API. There are two types of handlers currently
        available in the server:

          - Rest -- Exposes the REST-ful API
          - Soap -- Exposes the SOAP API

        The Rest Handler resource exposes some of the administrative
        operations on the REST handler such as clearing the cache.
        """
        url = self._url + "/handlers"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @property
    def rest_handler(self) -> dict:
        """
        Gets a list of resources accessible throught the REST API.
        """
        url = self._url + "/handlers/rest"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)


########################################################################
class PlatformServiceManager(BaseServer):
    """
    The Platform Services Manager allows you to view and manage your platform services in the ArcGIS
    Server Administrator Directory. The Compute Platform and Synchronization services are used by
    GeoAnalytics Server. For each platform service, you can start it, stop it, and view its status.
    While a health check operation is exposed for all three platform services, only Compute Platform
    Health Check returns valuable information.


    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string. The service URL.
    ------------------     --------------------------------------------------------------------
    connection             Required Connection. The connection object.
    ------------------     --------------------------------------------------------------------
    initialize             Optional string. Denotes whether to load the service properties at
                           creation (True). Default is False.
    ==================     ====================================================================

    """

    _cp = None
    _mb = None
    _ss = None
    _con = None
    _url = None
    _json = None
    _json_dict = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The service URL.
        ------------------     --------------------------------------------------------------------
        connection             Required Connection. The connection object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the service properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(PlatformServiceManager, self).__init__(connection=connection, url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def _init(self, connection: Connection = None):
        """loads the properties into the class"""
        from arcgis._impl.common._mixins import PropertyMap

        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def get(self, service: str) -> PlatformService:
        """
        Returns a single instance of a Platform Service

        ================   =================================================
        **Arguements**     **Description**
        ----------------   -------------------------------------------------
        service            optional string. This is the name of the service
                           that you want to examine.
                           Allowed values: SYNCHRONIZATION_SERVICE,
                           MESSAGE_BUS, or COMPUTE_PLATFORM
        ================   =================================================

        :returns PlatformService objects

        """
        services = []

        if "platformservices" in self._json_dict:
            for ps in self._json_dict["platformservices"]:
                if ps["type"].lower() == service.lower():
                    return PlatformService(
                        url="%s/%s" % (self._url, ps["id"]), connection=self._con
                    )
        return None

    # ----------------------------------------------------------------------
    def list(self) -> list:
        """
        Returns all Platform Services on the enterprise configuration.

        :returns list of PlatformService objects

        """
        services = []
        service = None
        if service is None:
            if "platformservices" in self._json_dict:
                for ps in self._json_dict["platformservices"]:
                    services.append(
                        PlatformService(
                            url="%s/%s" % (self._url, ps["id"]), connection=self._con
                        )
                    )
        else:
            if "platformservices" in self._json_dict:
                for ps in self._json_dict["platformservices"]:
                    if ps["type"].lower() == service.lower():
                        services.append(
                            PlatformService(
                                url="%s/%s" % (self._url, ps["id"]),
                                connection=self._con,
                            )
                        )
        return services


########################################################################
class PlatformService(BaseServer):
    """
    The platform service can be a message_bus, computre_platform, or synchronization_service.
    Each platform service has the following operations: start, stop, status, and health check.

    - The Compute Platform resource allows you to view and manage your compute platform service for ArcGIS GeoAnalytics Server in the ArcGIS Server Administrator Directory. You can start or stop the service, view its status, and run a health check.
    - The Message Bus resource allows you to view and manage your message bus service in the ArcGIS Server Administrator Directory. You can start or stop the service, and view its status. While a health check operation is exposed, it will not return any valuable information.
    - The Synchronization Service resource allows you to view and manage your synchronization service for ArcGIS GeoAnalytics Server in the ArcGIS Server Administrator Directory. You can start or stop the service and view its status. While a health check operation is exposed for the synchronization service, it will not return any valuable information.


    """

    _url = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The service URL.
        ------------------     --------------------------------------------------------------------
        connection             Required string. The connection string.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the properties at creation
                               (True). Default is False.
        ==================     ====================================================================


        """
        initialize = True
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def start(self) -> dict:
        """
        The Start method allows for the running of the service.
        """
        params = {"f": "json"}
        url = "%s/start" % self._url
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def stop(self) -> dict:
        """
        The Start method allows for the running of the service.
        """
        params = {"f": "json"}
        url = "%s/stop" % self._url
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def status(self) -> dict:
        """
        The status resource allows you to view the status of the service.
        The
        """
        params = {"f": "json"}
        url = "%s/status" % self._url
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def health(self) -> dict:
        """
        The health check operation allows you to view the health of the service.
        """
        params = {"f": "json"}
        url = "%s/health" % self._url
        return self._con.get(url, params)


########################################################################
class ConfigurationStore(BaseServer):
    """
    A utility class for managing the Configuration Store of this server.
    """

    _con = None
    _url = None
    _json = None
    _json_dict = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        connection             Required string. The connection string.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(ConfigurationStore, self).__init__(connection=connection, url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def recover(self) -> dict:
        """
        Recovers the Configuration Store of the site.

        If the shared configuration store for a site is unavailable, a site
        in read-only mode will operate in a degraded capacity that allows
        access to the ArcGIS Server Administrator Directory. You can recover
        a site if the shared configuration store is permanently lost. The
        site must be in read-only mode, and the site configuration files
        must have been copied to the local repository when switching site
        modes. The recover operation will copy the configuration store from
        the local repository into the shared configuration store location.
        The copied local repository will be from the machine in the site
        where the recover operation is performed.

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/recover"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def edit(
        self,
        type_value: str,
        connection: GIS,
        move: bool = True,
        run_async: bool = False,
        *,
        local_path: Optional[str] = None,
    ) -> bool:
        """
        You can use this operation to update the configuration store.
        Typically, this operation is used to change the location of the
        configuration store.

        When ArcGIS Server is installed, the default configuration store
        uses local paths. As the site grows (more server machines are
        added), the location of the store must be updated to use a shared
        file system path. On the other hand, if you know at the onset that
        your site will have two or more server machines, you can start from
        a shared path while creating a site and skip this step altogether.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        type_value             Required string. The type of the configuration store. Values: FILESYSTEM
        ------------------     --------------------------------------------------------------------
        gis                    Required string. A file path or connection URL to the physical
                               location of the store.
        ------------------     --------------------------------------------------------------------
        move                   Optional string. A boolean to indicate if you want
                               to move the content of the current store to the new store. The
                               default True (move the content).
        ------------------     --------------------------------------------------------------------
        run_async              Optional string. Determines if this operation must run asynchronously.
                               The default is False (doesn not have to run asynchronously).
        ------------------     --------------------------------------------------------------------
        local_path             Optional String. A file path or connection URL to the physical
                               location of the local repository for when the site is in read-only
                               mode.
        ==================     ====================================================================


        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/edit"
        params = {
            "f": "json",
            "type": type_value,
            "connectionString": connection,
            "move": move,
            "runAsync": run_async,
        }
        if local_path:
            params["localRepositoryPath"] = local_path
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class Jobs(BaseServer):
    """
    This resource is a collection of all the administrative jobs
    (asynchronous operations) created within your site. When operations
    that support asynchronous execution are run, the server creates a new
    job entry that can be queried for its current status and messages.
    """

    _con = None
    _json = None
    _jobs = None
    _json_dict = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        connection             Required string. The connection string.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(Jobs, self).__init__(connection=connection, url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    @property
    def jobs(self) -> list:
        """
        Gets the job IDs.
        """
        if self._jobs is None:
            self._init()
        return self._jobs

    # ----------------------------------------------------------------------
    def get(self, job_id: str) -> dict:
        """
        A job represents the asynchronous execution of an operation. You
        can acquire progress information by periodically querying the job.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        job_id                 Required string. The ID of the job.
        ==================     ====================================================================

        :return:
            A JSON dictionary containing progress information for the job ID.

        """
        url = self._url + "/%s" % job_id
        params = {"f": "json"}
        return self._con.get(path=url, params=params)


########################################################################
class ServerProperties(BaseServer):
    """
    The Server has configuration parameters that can govern some of its
    intricate behavior. This Server Properties resource is a container for
    such properties. These properties are available to all server objects
    and extensions through the server environment interface.

    The properties include:

    - ``CacheSizeForSecureTileRequests`` -- An integer that specifies the
      number of users whose token information will be cached. This
      increases the speed of tile retrieval for cached services. If not
      specified, the default cache size is 200,000. Both REST and SOAP
      services honor this property. You'll need to manually restart
      ArcGIS Server in order for this change to take effect.

    - ``DisableAdminDirectoryCache`` -- Disables browser caching of the
      Administrator Directory pages. The default is False. To disable
      browser caching, set this property to True.

    - ``disableIPLogging`` -- When a possible cross-site request forgery
      (CSRF) attack is detected, the server logs a message containing
      the possible IP address of the attacker. If you do not want IP
      addresses listed in the logs, set this property to True. Also,
      HTTP request referrers are logged at FINE level by the REST and
      SOAP handlers unless this property is set to True.

    - ``javaExtsBeginPort`` -- Specifies a start port of the port range used
      for debugging Java server object extensions.
      Example: ``8000``

    - ``javaExtsEndPort`` -- Specifies an end port of the port range used for
      debugging Java server object extensions.
      Example: ``8010``

    - ``localTempFolder`` -- Defines the local folder on a machine that can
      be used by GIS services and objects. If this property is not
      explicitly set, the services and objects will revert to using the
      system's default temporary directory.

      .. note::
          If this property is used, you must create the temporary directory
          on every server machine in the site. Example: /tmp/arcgis.

    - ``messageFormat`` -- Defines the transmission protocol supported by
      the services catalog in the server.

      Values:
      - esriServiceCatalogMessageFormatBin,
      - esriServiceCatalogMessageFormatSoap,
      - esriServiceCatalogMessageFormatSoapOrBin

    - ``messageVersion`` -- Defines the version supported by the services
      catalog in the server.
      Example: esriArcGISVersion101

    - ``PushIdentityToDatabase`` -- Propogates the credentials of the logged-in
      user to make connections to an Oracle database. This
      property is only supported for use with Oracle databases.
      Values: True | False

    - ``suspendDuration`` -- Specifies the duration for which the ArcGIS
      service hosting processes should suspend at startup. This
      duration is specified in milliseconds. This is an optional
      property that takes effect when suspendServiceAtStartup is set
      to True. If unspecified and suspension of service at startup is
      requested, then the default suspend duration is 30 seconds.
      Example: 10000 (meaning 10 seconds)

    - ``suspendServiceAtStartup`` -- Suspends the ArcGIS service hosting
      processes at startup. This will enable attaching to those
      processes and debugging code that runs early in the lifecycle of
      server extensions soon after they are instantiated.
      Values: True | False

    - ``uploadFileExtensionWhitelist`` -- This specifies what files are
      allowed to be uploaded through the file upload API by
      identifying the allowable extensions. It is a list of comma-separated
      extensions without dots. If this property is not
      specified, a default list is used. This is the default list: soe,
      sd, sde, odc, csv, txt, zshp, kmz, and geodatabase.

      .. note::
          Updating this list overrides the default list completely. This
          means if you set this property to a subset of the default list
          then only those items in the subset will be accepted for upload.
          Example: sd, so, sde, odc.

    - ``uploadItemInfoFileExtensionWhitelist`` -- This specifies what files
      are allowed to be uploaded through the service iteminfo upload
      API by identifying the allowable extensions. It should be a list
      of comma-separated extensions without dots. If this property is
      not specified, a default list is used. This is the default list:
      xml, img, png, gif, jpg, jpeg, bmp.

      .. note::
          This list overrides the default list completely. This means if you
          set this property to a subset of the default list then only those
          items in the subset will be accepted for upload. Example: png, svg,
          gif, jpg, tiff, bmp.

    - ``WebContextURL`` -- Defines the web front end as seen by your users.
      Example: ``http://mycompany.com/gis``

    """

    _con = None
    _url = None
    _json = None
    _json_dict = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        connection             Required string. The connection string.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(ServerProperties, self).__init__(connection=connection, url=url)
        if url.lower().endswith("/properties"):
            self._url = url
        else:
            self._url = url + "/properties"
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def update(self, properties: str) -> bool:
        """
        This operation allows you to update the server properties. See the ServerProperties
        class description for all possible properties.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        properties             Required string. A Python dictionary of server properties to be updated.
                               To reset the properties, pass in `None`.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/update"
        if properties is None:
            properties = {}
        params = {"f": "json", "properties": properties}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class DirectoryManager(object):
    """
    A collection of all the server directories is listed under this
    resource.

    You can add a new directory using the Register Directory operation. You
    can then configure GIS services to use one or more of these
    directories. If you no longer need the server directory, you must
    remove the directory by using the Unregister Directory operation.
    """

    _system = None

    def __init__(self, system):
        self._system = system

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._system._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._system._url)

    # ----------------------------------------------------------------------
    def all(self) -> list:
        """
        Provides a configuration of this server directory.
        Server directories are used by GIS services as a location to output
        items such as map images, tile caches, and geoprocessing results.
        In addition, some directories contain configurations that power the
        GIS services.

        :return:
            A dictionary of this server directory configuration properties.
        """
        return self._system._directories

    # ----------------------------------------------------------------------
    def edit_services_directory(
        self,
        allowedOrigins: str,
        arcgis_com_map: str,
        arcgis_com_map_text: str,
        jsapi_arcgis: str,
        jsapi_arcgis_css: str,
        jsapi_arcgis_css2: str,
        jsapi_arcgis_sdk: str,
        serviceDirEnabled: str,
    ) -> bool:
        """
        Allows you to update the Services Directory configuration.  You can do such thing as
        enable or disable the HTML view of ArcGIS REST API, or adjust the JavaScript and map viewer
        previews of services in the Services Directory so that they work with your own locally
        hosted JavaScript API and map viewer.


        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        allowed_origins          Required string. A comma-separated list of URLs of domains allowed to
                                 make requests. An asterisk (*) can be used to denote all domains.
        --------------------     --------------------------------------------------------------------
        arcgis_com_map           Required string. URL of the map viewer application used for service
                                 previews. Defaults to the ArcGIS.com map viewer but could be used
                                 to point at your own Portal for ArcGIS map viewer.
        --------------------     --------------------------------------------------------------------
        arcgis_com_map_text      Required string. The text to use for the preview link that opens
                                 the map viewer.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis             Required string. The URL of the JavaScript API to use for service
                                 previews. Defaults to the online ArcGIS API for JavaScript, but
                                 could be pointed at your own locally-installed instance of the
                                 JavaScript API.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_css         Required string. The CSS file associated with the ArcGIS API for
                                 JavaScript. Defaults to the online Dojo tundra.css.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_css2        Required string. An additional CSS file associated with the ArcGIS
                                 API for JavaScript. Defaults to the online esri.css.
        --------------------     --------------------------------------------------------------------
        jsapi_arcgis_sdk         Required string. The URL of the ArcGIS API for JavaScript help.
        --------------------     --------------------------------------------------------------------
        service_dir_enabled      Required string. Flag to enable/disable the HTML view of the
                                 services directory.
        ====================     ====================================================================


        :return:
            A boolean indicating success (True).

        """
        return self._system._edit_services_directory(
            allowedOrigins,
            arcgis_com_map,
            arcgis_com_map_text,
            jsapi_arcgis,
            jsapi_arcgis_css,
            jsapi_arcgis_css2,
            jsapi_arcgis_sdk,
            serviceDirEnabled,
        )

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> dict:
        """
        returns the current service directory properties for the server.

        :return: Dict
        """
        return self._system._services_directory

    # ----------------------------------------------------------------------
    def get(self, name: str) -> dict:
        """
        Retrieves a single directory registered with ArcGIS Server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the registered directory.
        ==================     ====================================================================

        :return:
            The ArcGIS Server :class:`~arcgis.gis.server.ServerDirectory` object

        """
        return self._system._get_directory(name=name)

    # ----------------------------------------------------------------------
    def add(
        self,
        name: str,
        physicalPath: str,
        directoryType: str,
        maxFileAge: int,
        cleanupMode: str = "NONE",
        description: Optional[str] = None,
    ) -> bool:
        """
        Registers a new server directory. While registering the server
        directory, you can also specify the directory's cleanup parameters.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the server directory.
        ------------------     --------------------------------------------------------------------
        physical_path          Required string. The absolute physical path of the server directory.
        ------------------     --------------------------------------------------------------------
        directory_type         Required string. The type of server directory.
        ------------------     --------------------------------------------------------------------
        max_age                Required integer. The length of time a file in the directory needs
                               to be kept before it is deleted.
        ------------------     --------------------------------------------------------------------
        cleanup_mode           Optional string. Defines if files in the server directory needs to
                               be cleaned up. The default is None.
        ------------------     --------------------------------------------------------------------
        description            Optional string. An optional description for the server directory.
                               The default is None.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True).

        """
        return self._system._register(
            name, physicalPath, directoryType, maxFileAge, cleanupMode, description
        )


########################################################################
class ServerDirectory(BaseServer):
    """
    Server directories are used by GIS services as a location to output
    items such as map images, tile caches, and geoprocessing results. In
    addition, some directories contain configurations that power the GIS
    services.
    In a Site with more than one server machine these directories must be
    available on network shares, accessible to every machine in the site.

    The following directory types can be registered with the server:

    - Output -- Stores various information generated by services, such as map
      images. Instances: One or more

    - Cache -- Stores tile caches used by map, globe, and image services for
      rapid performance. Instances: One or more

    - Jobs -- Stores results and other information from geoprocessing
      services. Instances: One or more

    - System -- Stores files that are used internally by the GIS server.
      Instances: One

    Server directories that contain output of various GIS
    services can be periodically cleaned to remove old unused files. By
    using the cleanup mode and maximum file age parameters, you control
    when when you would like the files in these directories to be
    cleaned.

    All the output server directories are automatically virtualized (they
    can be accessed over a URL) for you through the ArcGIS Server REST API.
    """

    _con = None
    _url = None
    _json = None
    _json_dict = None
    _url = None
    _name = None
    _physicalPath = None
    _directoryType = None
    _cleanupMode = None
    _maxFileAge = None
    _description = None
    _virtualPath = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, connection: Connection, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        connection             Required string. The connection string.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(ServerDirectory, self).__init__(connection=connection, url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def edit(
        self,
        physical_path: str,
        cleanup_mode: str,
        max_age: int,
        description: str,
        *,
        use_local_dir: Optional[bool] = None,
        local_dir: Optional[str] = None,
    ):
        """
        The server directory's edit operation allows you to change the path
        and clean up properties of the directory. This operation updates
        the GIS service configurations (and points them to the new path)
        that are using this directory, causing them to restart. It is
        therefore recommended that any edit to the server directories be
        performed when the server is not under load.

        This operation is mostly used when growing a single machine site to
        a multiple machine site configuration, which requires that the
        server directories and configuration store be put on a
        network-accessible file share.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        physical_path          Required string. The absolute physical path of the server directory.
        ------------------     --------------------------------------------------------------------
        cleanup_mode           Optional string. Defines if files in the server directory needs to
                               be cleaned up. The default is None.
        ------------------     --------------------------------------------------------------------
        max_age                Required integer. The length of time a file in the directory needs
                               to be kept before it is deleted.
        ------------------     --------------------------------------------------------------------
        description            Optional string. An optional description for the server directory.
                               The default is None.
        ------------------     --------------------------------------------------------------------
        use_local_dir          Optional Boolean. When `True` the local directory will be used to
                               store results.  This is useful for `HA` configurations to reduce
                               copying over the local network.  The directory must exist on the
                               server already.
        ------------------     --------------------------------------------------------------------
        local_dir              Optional String. The local directory path to be used for the service.
        ==================     ====================================================================


        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/edit"
        params = {
            "f": "json",
            "physicalPath": physical_path,
            "cleanupMode": cleanup_mode,
            "maxFileAge": max_age,
            "description": description,
        }
        if not use_local_dir is None and isinstance(use_local_dir, bool):
            params["useLocalDir"] = use_local_dir
        if not local_dir is None:
            params["localDirectoryPath"] = local_dir
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def clean(self) -> bool:
        """
        Cleans the content (files and folders) within the directory that
        have passed their expiration date. Every server directory has the
        max file age and cleanup mode parameter that govern when a file
        created inside is supposed to be cleaned up. The server directory
        cleaner automatically cleans up the content within server
        directories at regular intervals. However, you can explicitly clean
        the directory by invoking this operation.

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/clean"
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def recover(self) -> dict:
        """
        Recovers the shared server directories of the site.

        If the shared server directories for a site are unavailable, a site
        in read-only mode will operate in a degraded capacity that allows
        access to the ArcGIS Server Administrator Directory. You can recover
        a site if the shared server directories are permanently lost. The
        site must be in read-only mode, and the site configuration files
        must have been copied to the local repository when switching site
        modes. The recover operation will copy the server directories from
        the local repository into the shared server directories location.
        The copied local repository will be from the machine in the site
        where the recover operation is performed.

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/recover"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def unregister(self) -> bool:
        """
        Unregisters a server directory. Once a directory has been
        unregistered, it can no longer be referenced (used) from within a
        GIS service.

        :return:
            A boolean indicating success (True).

        """
        url = self._url + "/unregister"
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res
