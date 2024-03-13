"""
Modifies a local portal's system settings.
"""
from typing import Optional
from .._impl._con import Connection
from .. import GIS
from ._base import BasePortalAdmin
from ..._impl.common._mixins import PropertyMap


###########################################################################
class Indexer(BasePortalAdmin):
    """
    This resource contains connection information to the default indexing service.
    """

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(Indexer, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    @property
    def status(self):
        """
        `status` allows you to view the status of the indexing service. You
        can view the number of users, groups, and search items in both the
        database (store) and the index. If the database and index do not
        match, indexing is either in progress or there is a problem with
        the index. It is recommended that you reindex to correct any
        issues. If indexing is in progress, you can monitor the status by
        refreshing the page.

        :return: dict

        """
        params = {"f": "json"}
        url = f"{self._url}/status"
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def reindex(self, mode, includes=None):
        """
        The operation allows you to generate or update the indexes for content, such as users, groups, and items stored in the database store.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        mode                Required String. The mode in which the indexer should run.
                            Values: USER_MODE, GROUP_MODE, SEARCH_MODE, or FULL_MODE
        ---------------     --------------------------------------------------------------------
        includes            Optional String. A comma separated list of elements to include in
                            the index. This is useful if you want to only index certain items
                            or user accounts.
        ===============     ====================================================================

        :return: Boolean

        """
        url = f"{self._url}/reindex"
        params = {"f": "json", "mode": mode, "includes": includes}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] in ["success", "suceess"]
        return res

    # ----------------------------------------------------------------------
    def reconfigure(self) -> bool:
        """
        This operation recreates the index service metadata, schema, and data in the event it becomes corrupted.

        :returns: Boolean

        """
        params = {"f": "json"}
        url = f"{self._url}/reconfigure"
        res = self._con.post(url, params)
        return res.get("status", "failed") == "success"


########################################################################
class EmailManager(BasePortalAdmin):
    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        super(EmailManager, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        result = connection.get(path=self._url, params=params)
        try:
            if "status" in result and result["status"] == "error":
                self._properties = None
                self._json_dict = None
            else:
                self._json_dict = result
                self._properties = PropertyMap(self._json_dict)
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def test(self, email: str):
        """
        Sends a test email to a provided email account to ensure the
        configuration is correct.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        email                           Required String. The test email to send to.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.
        """
        params = {"mailTo": email, "f": "json"}
        url = self._url + "/test"
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    def update(
        self,
        server: str,
        from_email: str,
        require_auth: bool,
        email_label: Optional[str] = None,
        port: int = 25,
        encryption: str = "SSL",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Configures the Email Server for Portal

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        server                          Required String. The email address
        ---------------------------     --------------------------------------------------------------------
        from_email                      Required String.  The email address the email originates from.
        ---------------------------     --------------------------------------------------------------------
        require_auth                    Required Boolean.  If True, the smtp requires authentication and
                                        the username and password must be provided.  If False, no
                                        authentication is needed for the smtp server.
        ---------------------------     --------------------------------------------------------------------
        email_label                     Optional String. The email label.
        ---------------------------     --------------------------------------------------------------------
        port                            Optional Integer.  The port number for the smtp server.
        ---------------------------     --------------------------------------------------------------------
        encryption                      Optional String. The encryption method used for the email server. The
                                        allowed values are: SSL, TLS, or NONE.
        ---------------------------     --------------------------------------------------------------------
        username                        Optional String. The username to use to login to the smtp server.
        ---------------------------     --------------------------------------------------------------------
        Password                        Optional String. The password to use to login to the smtp server.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        allowed_encrypt = ["none", "tls", "ssl"]
        if email_label is None:
            email_label = from_email
        if require_auth:
            require_auth = "yes"
            if username is None or password is None:
                raise ValueError(
                    "`username` and `password` are required when require_auth=True"
                )
        else:
            require_auth = "no"

        params = {
            "smtpServer": server,
            "fromEmailAddress": from_email,
            "fromEmailAddressLabel": email_label,
            "authRequired": require_auth,
            "username": username,
            "password": password,
            "smtpPort": 25,
            "encryptionMethod": str(encryption).upper(),
            "f": "json",
        }
        url = self._url + "/update"
        res = self._con.post(url, params)
        if "status" in res:
            self._properties = None
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the current email configuration

        :return: Boolean. True if successful else False.
        """
        url = self._url + "/delete"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            self._properties = None
            return res["status"] == "success"
        else:
            return False


########################################################################
class System(BasePortalAdmin):
    """
    This resource is an umbrella for a collection of system-wide resources
    for your portal. This resource provides access to the ArcGIS Web
    Adaptor configuration, portal directories, database management server,
    indexing capabilities, license information, and the properties of your
    portal.
    """

    _gis = None
    _con = None
    _url = None
    _email = None
    _indexer = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(System, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    @property
    def email(self):
        """
        Provides access to the email configuration setting on enterprise.

        :return: :class:`~arcgis.gis.admin.EmailManager`
        """
        # if "supportsEmail" in self._gis.properties and self._properties.supportsEmail:
        if self._gis.version >= [7, 3] or "supportsEmail" in self._gis.properties:
            if self._email is None:
                self._email = EmailManager(
                    url=self._url + "/emailSettings", gis=self._gis
                )
        else:
            raise Exception(
                "Configuring Email Servers is not supported on this enterprise configuration."
            )
        return self._email

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Gets/Sets the system properties that have been modified to control
        the portal's environment.

        The list of available properties are:
         - privatePortalURL-Informs the portal that it has a front end
           load-balancer/proxy reachable at the URL. This property is
           typically used to set up a highly available portal configuration
         - portalLocalhostName-Informs the portal back-end to advertise the
           value of this property as the local portal machine. This is
           typically used during federation and when the portal machine has
           one or more public host names.
         - httpProxyHost-Specifies the HTTP hostname of the proxy server
         - httpProxyPort-Specifies the HTTP port number of the proxy server
         - httpProxyUser-Specifies the HTTP proxy server username.
         - httpProxyPassword-Specifies the HTTP proxy server password.
         - isHttpProxyPasswordEncrypted-Set this property to false when you
           are configuring the HTTP proxy server password in plain text.
           After configuration, the password will be encrypted and this
           property will be set to true
         - httpsProxyHost-Specifies the HTTPS hostname of the proxy server
         - httpsProxyPort-Specifies the HTTPS port number of the proxy
           server
         - httpsProxyUser-Specifies the HTTPS proxy server username
         - httpsProxyPassword-Specifies the HTTPS proxy server password
         - isHttpsProxyPasswordEncrypted-Set this property to false when
           you are configuring the HTTPS proxy server password in plain
           text. After configuration, the password will be encrypted and
           this property will be set to true.
         - nonProxyHosts-If you want to federate ArcGIS Server and the site
           does not require use of the forward proxy, list the server
           machine or site in the nonProxyHosts property. Machine and
           domain items are separated using a pipe (|).
         - WebContextURL-If you are using a reverse proxy, set this
           property to reverse proxy URL.
         - ldapCertificateValidation Introduced at 10.7. When set to true,
           any encrypted LDAP communication (LDAPS) made from the portal to
           the user or group identity store will enforce certificate
           validation. The default value is false.
        """
        url = "%s/properties" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, properties):
        """
        See main ``properties`` property docstring
        """
        url = "%s/properties/update" % self._url
        params = {"f": "pjson", "properties": properties}
        self._con.post(path=url, params=params)
        self._con._create_session()
        self._con.token

    # ----------------------------------------------------------------------
    @property
    def web_adaptors(self):
        """
        The Web Adaptors resource lists the ArcGIS Web Adaptor configured
        with your portal. You can configure the Web Adaptor by using its
        configuration web page or the command line utility provided with
        the installation.

        :return:
            :class:`~arcgis.gis.admin.WebAdaptors` object

        """
        url = "%s/webadaptors" % self._url
        return WebAdaptors(url=url, gis=self._con)

    # ----------------------------------------------------------------------
    @property
    def directories(self):
        """
        The directories resource is a collection of directories that are
        used by the portal to store and manage content. Beginning at
        10.2.1, Portal for ArcGIS supports five types of directories:
         - Content directory-The content directory contains the data
           associated with every item in the portal.
         - Database directory-The built-in security store and sharing
           rules are stored in a Database server that places files in the
           database directory.
         - Temporary directory - The temporary directory is used as a
           scratch workspace for all the portal's runtime components.
         - Index directory-The index directory contains all the indexes
           associated with the content in the portal. The indexes are used
           for quick retrieval of information and for querying purposes.
         - Logs directory-Errors and warnings are written to text files in
           the log file directory. Each day, if new errors or warnings are
           encountered, a new log file is created.
        If you would like to change the path for a directory, you can use
        the Edit Directory operation.
        """
        res = []
        surl = "%s/directories" % self._url
        params = {"f": "json"}
        for d in self._con.get(path=surl, params=params)["directories"]:
            url = "%s/directories/%s" % (self._url, d["name"])
            res.append(Directory(url=url, gis=self._con))
        return res

    # ----------------------------------------------------------------------
    @property
    def licenses(self):
        """
        Portal for ArcGIS requires a valid license to function correctly.
        This resource returns the current status of the license.
        Starting at 10.2.1, Portal for ArcGIS enforces the license by
        checking the number of registered members and comparing it with the
        maximum number of members authorized by the license. Contact Esri
        Customer Service if you have questions about license levels or
        expiration properties.
        """
        if self._gis.version < [7, 1]:
            url = "%s/licenses" % self._url
            return Licenses(url=url, gis=self._con)
        else:
            import os

            u = os.path.dirname(self._url)
            url = "%s/license" % u
            return PortalLicense(url=url, gis=self._con)
        return None

    # ----------------------------------------------------------------------
    @property
    def database(self):
        """
        The database resource represents the database management system
        (DBMS) that contains all of the portal's configuration and
        relationship rules. This resource also returns the name and version
        of the database server currently running in the portal.
        You can use the properety to update database accounts
        """
        url = "%s/database" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @database.setter
    def database(self, value):
        """
        See main ``database`` property docstring
        """
        url = "%s/database" % self._url
        params = {"f": "json"}
        for k, v in value.items():
            params[k] = v
        self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    @property
    def incremental_backup(self):
        """
        Gets/Sets the Incremental Backup for the Enterprise Configuration


        :return: dict
        """
        url = "%s/database/settings" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @incremental_backup.setter
    def incremental_backup(self, value: bool):
        """
        Gets/Sets the Incremental Backup for the Enterprise Configuration

        :return: Dictionary indicating 'success' or 'error'
        """
        url = "%s/database/settings/edit" % self._url
        params = {"incrementalBackupEnabled": value, "f": "json"}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def index_status(self):
        """
        The status resource allows you to view the status of the indexing
        service. You can view the number of users, groups, and search items
        in both the database (store) and the index.
        If the database and index do not match, indexing is either in
        progress or there is a problem with the index. It is recommended
        that you reindex to correct any issues. If indexing is in progress,
        you can monitor the status by refreshing the page.

        :return: dict

        .. code-block:: python

            USAGE: Prints out current Index Status

            from arcgis.gis import GIS
            gis = GIS("https://yourportal.com/portal", "portaladmin", "password")
            sysmgr = gis.admin.system
            idx_status = sysmgr.index_status
            import json
            print(json.dumps(idx_status, indent=2))

            # Output
            {
              "indexes": [
                {
                  "name": "users",
                  "databaseCount": 51,
                  "indexCount": 51
                },
                {
                  "name": "groups",
                  "databaseCount": 325,
                  "indexCount": 325
                },
                {
                  "name": "search",
                  "databaseCount": 8761,
                  "indexCount": 8761
                }
              ]
            }

        """

        url = "%s/indexer/status" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def reindex(self, mode: str = "FULL", includes: Optional[str] = None):
        """
        This operation allows you to generate or update the indexes for
        content; such as users, groups, and items stored in the database
        (store). During the process of upgrading an earlier version of
        Portal for ArcGIS, you are required to update the indexes by
        running this operation. You can check the status of your indexes
        using the status resource.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        mode                            Optional string. The mode in which the indexer should run.
                                        Values USER_MODE | GROUP_MODE | SEARCH_MODE | FULL
        ---------------------------     --------------------------------------------------------------------
        includes                        Optional string. An optional comma separated list of elements to
                                        include in the index. This is useful if you want to only index
                                        certain items or user accounts.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = "%s/indexer/reindex" % self._url
        if mode.lower() == "full":
            mode = "FULL_MODE"
        params = {"f": "json", "mode": mode}
        if includes:
            params["includes"] = includes
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    @property
    def languages(self):
        """
        This resource gets/sets which languages will appear in portal
        content search results. Use the Update languages operation to
        modify which language'content will be available.
        """
        url = "%s/languages" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @languages.setter
    def languages(self, value: str):
        """
        This resource gets/sets which languages will appear in portal
        content search results. Use the Update languages operation to
        modify which language'content will be available.
        """
        url = "%s/languages/update" % self._url
        params = {"f": "json", "languages": value}
        self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def content_discovery(self):
        """
        This resource allows an administrator to enable or disable external content discovery from the portal website.
        Because some Esri-provided content requires external access to the internet, an administrator may choose to disable the content to prevent requests to ArcGIS Online resources. When disabling the content, a select group of items will be disabled:

        - All basemaps owned by "esri_[lang]"
        - All content owned by "esri_nav"
        - All content owned by "esri"

        This resource will not disable ArcGIS Online utility services or Living Atlas content. For steps to disable these items, refer to the Portal Administrator guide.

        When external content is disabled, System Languages are also disabled.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  required Boolean. If true, external content is enabled, else it is
                               disabled.
        ==================     ====================================================================

        :return: boolean

        """
        url = "%s/content/configuration" % self._url
        params = {"f": "json"}
        res = self._con.get(url, params)
        return res["isExternalContentEnabled"]

    # ----------------------------------------------------------------------
    @content_discovery.setter
    def content_discovery(self, value: bool):
        """
        See main ``content_discovery`` property docstring
        """
        import json

        url = "%s/content/configuration/update" % self._url
        params = {"f": "json", "externalContentEnabled": json.dumps(value)}
        res = self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def indexer(self):
        """
        Allows user to manage the site's indexer

        :return: :class:`~arcgis.gis.admin.Indexer`
        """
        if self._indexer is None:
            url = f"{self._url}/indexer"
            self._indexer = Indexer(url=url, gis=self._gis)
        return self._indexer


########################################################################
class WebAdaptors(BasePortalAdmin):
    """
    The Web Adaptors resource lists the ArcGIS Web Adaptor configured with
    your portal. You can configure the Web Adaptor by using its
    configuration web page or the command line utility provided with the
    installation.
    """

    _gis = None
    _con = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(WebAdaptors, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    def list(self):
        """
        Returns all instances of WebAdaptors

        .. code-block:: python

            USAGE: Get all Web Adaptors and list keys,values of first Web Adaptor object

            from arcgis.gis import GIS
            gis = GIS("https://yourportal.com/portal", "portaladmin", "password")

            # Return a List of Web Adaptor objects
            webadaptors = gis.admin.system.web_adaptors.list()

            # Get the first Web Adaptor object and print out each of its values
            for key, value in dict(webadaptors[0]).items():
                print("{} : {}".format(key, value))

            # Output
            machineName : yourportal.com
            machineIP : 10.11.12.13
            webAdaptorURL : https://yourwebserver.com/portal
            id : ac17d7b9-adbd-4c45-ae13-77b0ad6f14e8
            description :
            httpPort : 80
            httpsPort : 443
            refreshServerListInterval : 1
            reconnectServerOnFailureInterval : 1


        :return:
            List of :class:`~arcgis.gis.admin.WebAdaptor` objects.  Typically, only 1 Web Adaptor will exist for a Portal

        """

        res = []
        if "webAdaptors" in self.properties:
            for wa in self.properties.webAdaptors:
                url = "%s/%s" % (self._url, wa["id"])
                res.append(WebAdaptor(url=url, gis=self._con))
        return res

    # ----------------------------------------------------------------------
    @property
    def configuration(self):
        """
        Gets/Sets the common properties and configuration of the ArcGIS Web
        Adaptor configured with the portal.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        shared_key                      Required string. This property represents credentials that are shared
                                        with the Web Adaptor. The Web Adaptor uses these credentials to
                                        communicate with the portal
        ===========================     ====================================================================
        """
        url = "%s/config" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @configuration.setter
    def configuration(self, shared_key: str):
        """
        See main ``configuration`` property docstring
        """
        url = "%s/config/update" % self._url
        if isinstance(shared_key, str):
            params = {"webAdaptorsConfig": {"sharedkey": shared_key}, "f": "json"}
        elif isinstance(shared_key, dict) and "sharedKey" in shared_key:
            params = {"webAdaptorsConfig": shared_key, "f": "json"}
        return self._con.post(path=url, postdata=params)


########################################################################
class WebAdaptor(BasePortalAdmin):
    """
    The ArcGIS Web Adaptor is a web application that runs in a front-end
    web server. One of the Web Adaptor's primary responsibilities is to
    forward HTTP requests from end users to Portal for ArcGIS. The Web
    Adaptor acts a reverse proxy, providing the end users with an entry
    point to the system, hiding the back-end servers, and providing some
    degree of immunity from back-end failures.
    The front-end web server can authenticate incoming requests against
    your enterprise identity stores and provide specific authentication
    schemes such as Integrated Windows Authentication (IWA), HTTP Basic, or
    Digest.
    Most importantly, a Web Adaptor provides your end users with a well
    defined entry point to your system without exposing the internal
    details of your portal. Portal for ArcGIS will trust requests being
    forwarded by the Web Adaptor and will not challenge the user for any
    credentials. However, the authorization of the request (by looking up
    roles and permissions) is still enforced by the portal's sharing rules.
    """

    _gis = None
    _con = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(WebAdaptor, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    def unregister(self):
        """
        You can use this operation to unregister the ArcGIS Web Adaptor
        from your portal. Once a Web Adaptor has been unregistered, your
        portal will no longer trust the Web Adaptor and will not accept any
        credentials from it. This operation is typically used when you want
        to register a new Web Adaptor or when your old Web Adaptor needs to
        be updated.
        """
        url = "%s/unregister" % self._url
        params = {"f": "json"}
        return self._con.post(path=url, postdata=params)


########################################################################
class Directory(BasePortalAdmin):
    """
    A directory is a file system-based folder that contains a specific type
    of content for the portal. The physicalPath property of a directory
    locates the actual path of the folder on the file system. Beginning at
    10.2.1, Portal for ArcGIS supports local directories and network shares
    as valid locations.
    During the Portal for ArcGIS installation, the setup program asks you
    for the root portal directory (that will contain all the portal's sub
    directories). However, you can change each registered directory through
    this API.
    """

    _gis = None
    _con = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(Directory, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        The properties operation on a directory can be used to change the
        physical path and description properties of the directory. This is
        useful when changing the location of a directory from a local path
        to a network share. However, the API does not copy your content and
        data from the old path to the new path. This has to be done
        independently by the system administrator.
        """
        return PropertyMap(self._json_dict)

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, value):
        """
        The properties operation on a directory can be used to change the
        physical path and description properties of the directory. This is
        useful when changing the location of a directory from a local path
        to a network share. However, the API does not copy your content and
        data from the old path to the new path. This has to be done
        independently by the system administrator.
        """
        url = "%s/edit" % self._url
        params = {"f": "json"}
        if isinstance(value, PropertyMap):
            value = dict(value)
        for k, v in value.items():
            params[k] = v
        return self._con.post(path=url, postdata=params)


########################################################################
class PortalLicense(BasePortalAdmin):
    """
    The Enterprise portal requires a valid license to function correctly.
    This resource returns information for user types that are licensed
    for your organization.

    Starting at 10.7, the Enterprise portal enforces user type licensing.
    Members are assigned a user type which determines the privileges that
    an be granted to the member through a role. Each user type may
    include access to specific apps and app bundles.

    The license information returned for the organization includes the
    total number of registered members that can be added, the current
    number of members in the organization and the Portal for ArcGIS
    version. For each user type, the license information includes the ID,
    the maximum number of registered members that can be assigned, the
    number of members currently assigned the license and the expiration,
    in epoch time. In addition, this resource provides access to the
    Validate License, Import License, Populate License, Update License
    Manager, and Release License operations.

    """

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super().__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    def import_license(self, file: str):
        """
        The `import_license` operation is used to import a new license
        file. The portal license file contains your Enterprise portal's
        user type, app and app bundle licenses. By importing a portal
        license file, you will be applying the licenses in the file to your
        organization.

        Caution:

            Importing a new portal license file will overwrite your
            organization's current user type, app, and app bundle licenses.
            Before importing, verify that the new license file has
            sufficient user type, app, and app bundle licenses.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        file                            Required String. The portal license file.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        file = {"file": file}
        params = {"f": "json"}
        url = "%s/importLicense" % self._url
        res = self._con.post(url, params, files=file)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def populate(self):
        """

        The `populate` operation applies the license information from the
        license file that is used to create or upgrade your portal. This
        operation is only necessary as you create or upgrade your portal
        through the Portal Admin API.

        :return: Boolean. True if successful else False.

        """
        params = {"f": "json"}
        url = "%s/populateLicense" % self._url
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def release_license(self, username: str):
        """
        If a user checks out an ArcGIS Pro license for offline or
        disconnected use, this operation releases the license for the
        specified account. A license can only be used with a single device
        running ArcGIS Pro. To check in the license, a valid access token
        and refresh token is required. If the refresh token for the device
        is lost, damaged, corrupted, or formatted, the user will not be
        able to check in the license. This prevents the user from logging
        in to ArcGIS Pro from any other device. As an administrator, you
        can release the license. This frees the outstanding license and
        allows the user to check out a new license or use ArcGIS Pro in a
        connected environment.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        username	                    Required String. The user name of the account.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.


        """
        params = {
            "f": "json",
        }
        url = "%s/releaseLicense" % self._url
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def update(self, info: dict):
        """
        ArcGIS License Server Administrator works with your portal and
        enforces licenses for ArcGIS Pro. This operation allows you to
        change the license server connection information for your portal.

        You can register a backup license manager for high availability of
        your licensing portal. After configuring the backup license
        manager, Portal for ArcGIS is restarted automatically. When the
        restart completes, the portal is configured with the backup license
        server you specified. When configuring a backup license manager,
        you will need to ensure that the backup is authorized using the
        same license file as your portal.

        :Note:

            Previously, premium apps were licensed individually through the
            portal. Starting at 10.7, there will no longer be separate
            licensing for apps; the portal's user types, apps, and app
            bundles will be licensed using a single portal license file.
            Licensing ArcGIS Pro and Drone2Map requires licensing your
            Enterprise portal's ArcGIS License Server Administrator
            (license manager). Previously, users were required to import a
            .lic file into the portal's license manager. They would then
            generate a .json file through the license manager and import
            the file into portal. Now, users licensing ArcGIS Pro and
            Drone2Map import the same license file used to license their
            portal into their license manager. Users are no longer required
            to generate an additional license file in the license manager.


        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        info                  	        Required Dict. The JSON representation of the license server
                                        connection information.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        .. code-block:: python

            # Example Usage
            >>> gis.admin.system.licenses.update(info={ "hostname": "licensemanager.domain.com,backuplicensemanager.domain.com",
                                                    "port": 27000
                                                  })
            True

        """
        params = {"f": "json", "licenseManagerInfo": info}
        url = "%s/updateLicenseManager" % self._url
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def validate(self, file: str, list_ut: bool = False):
        """
        The `validate` operation is used to validate an input license file.
        Only valid license files can be imported into the Enterprise
        portal. If the provided file is valid, the operation will return
        user type, app bundle, and app information from the license file.
        If the file is invalid, the operation will fail and return an error
        message.


        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        file                            Required String. The portal license file.
        ---------------------------     --------------------------------------------------------------------
        list_ut                         Optional Boolean. Returns a list of user types that are compatible
                                        with the Administrator role. This identifies the user type(s) that
                                        can be assigned to the Initial Administrator Account when creating
                                        a portal.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        file = {"file": file}
        params = {"f": "json", "listAdministratorUserTypes": list_ut}
        url = "%s/validateLicense" % self._url
        res = self._con.post(url, params, files=file)
        return res


########################################################################
class Licenses(BasePortalAdmin):
    """
    Portal for ArcGIS requires a valid license to function correctly. This
    resource returns the current status of the license.
    As of 10.2.1, Portal for ArcGIS enforces the license by checking the
    number of registered members and comparing it with the maximum number
    of members authorized by the license. Contact Esri Customer Service if
    you have questions about license levels or expiration properties.
    Starting at 10.5, Portal for ArcGIS enforces two levels of membership
    for licensing to define sets of privileges for registered members and
    their assigned roles.

    **Deprecated at ArcGIS Enterprise 10.7**

    """

    _gis = None
    _con = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """Constructor"""
        super(Licenses, self).__init__(url=url, gis=gis, **kwargs)
        initialize = kwargs.pop("initialize", False)
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(self._gis)

    # ----------------------------------------------------------------------
    def entitlements(self, app: str = "arcgisprodesktop"):
        """
        This operation returns the currently queued entitlements for a
        product, such as ArcGIS Pro or Navigator for ArcGIS, and applies
        them when their start dates become effective. It's possible that
        all entitlements imported using the Import Entitlements operation
        are effective immediately and no entitlements are added to the
        queue. In this case, the operation returns an empty result.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        app                             Required string. The application lookup.
                                        Allowed values: appstudioweb,arcgisprodesktop,busanalystonline_2,
                                        drone2map,geoplanner,arcgisInsights,LRReporter,
                                        navigator, or RoadwayReporter
        ===========================     ====================================================================

        :return: dict

        """
        allowed = [
            "appstudioweb",
            "arcgisprodesktop",
            "busanalystonline_2",
            "drone2map",
            "geoplanner",
            "arcgisInsights",
            "LRReporter",
            "navigator",
            "RoadwayReporter",
        ]
        params = {"f": "json", "appId": app}
        if app not in allowed:
            raise ValueError("The app value must be: %s" % ",".join(allowed))
        url = "%s/getEntitlements" % self._url
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def remove_entitlement(self, app: str = "arcgisprodesktop"):
        """
        deletes an entitlement from a site

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        app                             Required string. The application lookup.
                                        Allowed values: appstudioweb,arcgisprodesktop,busanalystonline_2,
                                        drone2map,geoplanner,arcgisInsights,LRReporter,
                                        navigator, or RoadwayReporter
        ===========================     ====================================================================

        :return: dict

        """
        allowed = [
            "appstudioweb",
            "arcgisprodesktop",
            "busanalystonline_2",
            "drone2map",
            "geoplanner",
            "arcgisInsights",
            "LRReporter",
            "navigator",
            "RoadwayReporter",
        ]
        params = {"f": "json", "appId": app}
        if app not in allowed:
            raise ValueError("The app value must be: %s" % ",".join(allowed))
        url = "%s/removeAllEntitlements" % self._url
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def update_license_manager(self, info: str):
        """
        ArcGIS License Server Administrator works with your portal and
        enforces licenses for ArcGIS Pro. This operation allows you to
        change the license server connection information for your portal.
        When you import entitlements into portal using the Import
        Entitlements operation, a license server is automatically
        configured for you. If your license server changes after the
        entitlements have been imported, you only need to change the
        license server connection information.
        You can register a backup license manager for high availability of
        your licensing portal. When configuring a backup license manager,
        you need to make sure that the backup license manager has been
        authorized with the same organizational entitlements. After
        configuring the backup license manager, Portal for ArcGIS is
        restarted automatically. When the restart completes, the portal is
        configured with the backup license server you specified.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        info                            Required string. The JSON representation of the license server
                                        connection information.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json", "licenseManagerInfo": info}
        url = "%s/updateLicenseManager" % self._url
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def import_entitlements(self, file: str, application: str):
        """
        This operation allows you to import entitlements for ArcGIS Pro and
        additional products such as Navigator for ArcGIS into your
        licensing portal. Once the entitlements have been imported, you can
        assign licenses to users within your portal. The operation requires
        an entitlements file that has been exported out of your ArcGIS
        License Server Administrator or out of My Esri, depending on the
        product.
        A typical entitlements file will have multiple parts, each
        representing a set of entitlements that are effective at a specific
        date. The parts that are effective immediately will be configured
        to be the current entitlements. Other parts will be added to a
        queue. The portal framework will automatically apply the parts when
        they become effective. You can use the Get Entitlements operation
        to see the parts that are in the queue.
        Each time this operation is invoked, it overwrites all existing
        entitlements, even the ones that are in the queue.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        file                            Required string. The entitlement file to load into Enterprise.
        ---------------------------     --------------------------------------------------------------------
        application                     Required string. The application identifier to be imported
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/importEntitlements" % self._url
        params = {"f": "json", "appId": application}
        files = {"file": file}
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def remove_all(self, application: str):
        """
        This operation removes all entitlements from the portal for ArcGIS
        Pro or additional products such as Navigator for ArcGIS and revokes
        all entitlements assigned to users for the specified product. The
        portal is no longer a licensing portal for that product.
        License assignments are retained on disk. Therefore, if you decide
        to configure this portal as a licensing portal for the product
        again in the future, all licensing assignments will be available in
        the website.
        """
        params = {"f": "json", "appId": application}
        url = "%s/removeAllEntitlements" % self._url
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def release_license(self, username: str):
        """
        If a user checks out an ArcGIS Pro license for offline or
        disconnected use, this operation releases the license for the
        specified account. A license can only be used with a single device
        running ArcGIS Pro. To check in the license, a valid access token
        and refresh token is required. If the refresh token for the device
        is lost, damaged, corrupted, or formatted, the user will not be
        able to check in the license. This prevents the user from logging
        in to ArcGIS Pro from any other device. As an administrator, you
        can release the license. This frees the outstanding license and
        allows the user to check out a new license or use ArcGIS Pro in a
        connected environment.
        """
        params = {"f": "json", "username": username}
        url = "%s/releaseLicense" % self._url
        return self._con.get(path=url, params=params)
