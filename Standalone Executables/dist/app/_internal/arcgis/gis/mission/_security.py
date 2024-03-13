from __future__ import annotations
import os
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


########################################################################
class SecurityManager(object):
    """
    This resource is a container for all resources and operations
    pertaining to security in your ArcGIS Mission Server site.
    Security Manager can be accessed via the
    :attr:`~arcgis.gis.mission.MissionServer.security` property of
    :class:`~arcgis.gis.mission.MissionServer` class
    """

    _url = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< SecurityManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _modify_server_role(self, role: str):
        """
        Allows for the modification of the server role from federated to standalone.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        role                   Required String. The value that determines if the server is federated
                               or standalone.  This allowed values are: "FEDERATED_SERVER" or
                               "STANDALONE_SERVER".
        ==================     ====================================================================

        :return: Dict

        """
        url = self._url + "/config/changeServerRole"
        function = None
        if function is None:
            function = ""
        params = {"f": "json", "serverRole": role, "serverFunction": function}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def configuration(self):
        """
        This resource returns the currently active security configuration
        of your ArcGIS Mission Server site. A security configuration
        involves the following pieces of information, and can be modified
        using the setter.

        **Configuration Properties**

        ===========================     ====================================================================
        **Key**                         **Description**
        ---------------------------     --------------------------------------------------------------------
        allowedAdminAccessIPs           (Allowed Administration Access IPs). A comma separated list of client machine IP addresses that are allowed access to ArcGIS Server. This can be used as an additional security measure to prevent unauthorized access to the site.
        ---------------------------     --------------------------------------------------------------------
        allowDirectAccess               (Allow direct administrator access). A boolean that indicates if a user with administrator privileges can access the server through port 6080. If true, all users with administrative access are allowed to access the Administrator Directory and ArcGIS Server Manager through port 6080. If false, users in the identity store are blocked from accessing the server through port 6080. This forces users to access the site through ArcGIS Web Adaptor. The default value is true.

                                        Before disabling administrative access on port 6080, ArcGIS Server must be configured to use web tier authentication (WEB_ADAPTOR) and at least one user in the identity store must have administrator privileges to the site. The primary site administrator account will still be able to administer the site through port 6080.

                                        To fully disable access on port 6080, you can optionally disabled the primary site administrator account. If ArcGIS Server Manager becomes unavailable or the web server is unable to authenticate users that have administrator privileges, you will be unable to administer your site. To recover from this site, re-enable the primary site administrator account and connect to the site through port 6080 with this account.

        ---------------------------     --------------------------------------------------------------------
        authenticationMode              (Authentication Mode). Specifies the authentication mode used by ArcGIS Server. When ArcGIS Server is federated with Portal for ArcGIS, this property can be included and set to ARCGIS_PORTAL_TOKEN. The default value is ARCGIS_TOKEN.

                                        Values: ARCGIS_TOKEN | ARCGIS_PORTAL_TOKEN | WEB_ADAPTOR_AUTHENTICATION
        ---------------------------     --------------------------------------------------------------------
        authenticationTier              (Authentication Tier). The tier at which requests to access GIS services will be authenticated. It is recommended that you do not modify these values using the Administrator Directory. Instead, use ArcGIS Server Manager to configure web tier authentication or use the Portal for ArcGIS website to federate ArcGIS Server with your portal.

                                        Values: WEB_ADAPTOR | GIS_SERVER | ARCGIS_PORTAL
        ---------------------------     --------------------------------------------------------------------
        HSTSEnabled                     (HSTS Enabled). A boolean that indicates if HTTP Strict Transport Security (HSTS) is being used by the site. See Enforce strict HTTPS communication for more information. In order for this property to be enabled, the Protocol property must already be set to use HTTPS only.
        ---------------------------     --------------------------------------------------------------------
        httpEnabled                     (HTTP Enabled). A boolean that indicates if the site is accessible over HTTP.
        ---------------------------     --------------------------------------------------------------------
        Protocol                        (Protocol). Specifies the HTTP protocol to be used for communication to and from the ArcGIS Server site. If set to HTTP, all communication to and from the site will be over HTTP, with HTTPS communication being unavailable. If HTTP_AND_HTTPS is set, users and client may use either HTTP or HTTPS to connect to the site. If HTTPS, all communication to and from the site will be over HTTPS. Any call made using HTTP will be redirected to use HTTPS instead.
                                        When you initially create your ArcGIS Server site, all communication in the site is sent over HTTP, which is not secure. This means that your credentials sent over an internal network or the Internet are not encrypted and can be intercepted. To prevent the interception of any communication, it's recommended that you configure ArcGIS Server and ArcGIS Server Manager (if installed) to enforce Secure Sockets Layer (SSL). When you initially create your site, you'll see a warning-level message in the logs recommending that you update the communication protocol of your site to use SSL.

                                        Values: HTTP | HTTP_AND_HTTPS | HTTPS
        ---------------------------     --------------------------------------------------------------------
        roleStoreConfig                 (Role Store). Connection information about the currently active role store.

        ---------------------------     --------------------------------------------------------------------
        securityEnabled                 (Security Enabled). A boolean that indicates if security is enabled for any GIS service. The default value is true.
        ---------------------------     --------------------------------------------------------------------
        sslEnabled                      (SSL Enabled). A boolean that indicates if the site is accessible over HTTPS (SSL). The default value is false.
        ---------------------------     --------------------------------------------------------------------
        userStoreConfig                 (User Store). Connection information about the currently active user store.
        ---------------------------     --------------------------------------------------------------------
        virtualDirsSecurityEnabled      (Virtual Directories Security Enabled). A boolean that indicates if the server's virtual directories are secured and require authentication. If true, accessing the content in the arcgisoutput, arcgisjobs, and arcgisinput directories over HTTP will require user authentication. This will negatively impact performance. The default value is false.
        ---------------------------     --------------------------------------------------------------------
        portalProperties                (Portal Properties). Specified when federating ArcGIS Server with Portal for ArcGIS. See the Portal properties for more information.
        ===========================     ====================================================================



        **Portal Properties**

        ===========================     ====================================================================
        **Portal Keys**                 **Description**
        ---------------------------     --------------------------------------------------------------------
        portalMode                      Must be set as ARCGIS_PORTAL_FEDERATION.
        ---------------------------     --------------------------------------------------------------------
        portalSecretKey                 The key obtained after federating ArcGIS Server with Portal for ArcGIS.
        ---------------------------     --------------------------------------------------------------------
        portalURL                       The URL of Portal for ArcGIS.
        ---------------------------     --------------------------------------------------------------------
        referer                         The referer specified when generating the token.
        ---------------------------     --------------------------------------------------------------------
        serverId                        The ID of the server federated with the portal.
        ---------------------------     --------------------------------------------------------------------
        serverURL                       External URL of the server federated with the portal in the following format:
        ---------------------------     --------------------------------------------------------------------
        token                           A token obtained from Portal for ArcGIS for use by ArcGIS Server for initial validation.
        ===========================     ====================================================================

        """
        url = self._url + "/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @configuration.setter
    def configuration(self, settings):
        """
        See main ``configuration`` property docstring.
        """
        url = self._url + "/config/update"
        params = {"f": "json"}
        current = dict(self.configuration)
        for k, v in settings.items():
            if k in current:
                params[k] = settings[k]
                current.pop(k)
        params.update(current)
        return self._con.post(url, params)
