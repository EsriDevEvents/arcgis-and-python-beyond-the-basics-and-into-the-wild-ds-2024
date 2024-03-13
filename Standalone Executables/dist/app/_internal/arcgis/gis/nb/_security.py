import os
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


########################################################################
class SecurityManager(object):
    """
    This resource is a container for all resources and operations
    pertaining to security in your ArcGIS Notebook Server site. An object of this
    class can be created using :attr:`~arcgis.gis.nb.NotebookServer.security` property of the
    :class:`~arcgis.gis.nb.NotebookServer` class
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
    @property
    def configuration(self):
        """
        This resource returns the currently active security configuration
        of your ArcGIS Notebook Server site. A security configuration
        involves the following pieces of information, and can be modified
        using the Update operation.

        **Configuration Properties**

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        httpsProtocols	       The TLS protocols ArcGIS Notebook Server will use. Values must be comma-separated.
        ------------------     --------------------------------------------------------------------
        cipherSuites	       The cipher suites ArcGIS Notebook Server will use. Values must be comma-separated.
        ------------------     --------------------------------------------------------------------
        authenticationMode     Specifies the authentication mode used by ArcGIS Notebook Server. When ArcGIS Notebook Server is federated with an Enterprise portal, this property can be included and set to ARCGIS_PORTAL_TOKEN
                               Values: ARCGIS_TOKEN | ARCGIS_PORTAL_TOKEN
        ------------------     --------------------------------------------------------------------
        authenticationTier     The tier at which requests to access GIS services will be authenticated. You should not use this directory to modify your setting. The default value when your site is first created is NOTEBOOK_SERVER, but when you federate the site with your portal, the value is changed to ARCGIS_PORTAL.
        ------------------     --------------------------------------------------------------------
        tokenServiceKey	       The key used to encrypt tokens.
        ==================     ====================================================================


        **Portal Properties**

        ==========================     ===========================================================================================
        **Parameter**                   **Description**
        --------------------------     -------------------------------------------------------------------------------------------
        portalMode	                   Must be the value ARCGIS_PORTAL_FEDERATION
        --------------------------     -------------------------------------------------------------------------------------------
        portalSecretKey	               The key obtained after federating ArcGIS Notebook Server with portal.
        --------------------------     -------------------------------------------------------------------------------------------
        portalURL	                   The URL of your portal, in the format ``https://webadaptorhost.domain.com/webadaptorname``
        --------------------------     -------------------------------------------------------------------------------------------
        referer	                       The referer specified when generating the token.
        --------------------------     -------------------------------------------------------------------------------------------
        serverId	                   The ID of the server federated with the portal.
        --------------------------     -------------------------------------------------------------------------------------------
        serverUrl	                   The external URL of the federated ArcGIS Notebook Server, in the format ``https://webadaptorhost.domain.com/webadaptorname``
        --------------------------     -------------------------------------------------------------------------------------------
        token                          A token obtained from the portal for initial validation of the ArcGIS Notebook Server.
        --------------------------     -------------------------------------------------------------------------------------------
        webgisServerTrustKey           A key for establishing trust between servers that are federated with the same portal.
        --------------------------     -------------------------------------------------------------------------------------------
        privateHostingServerUrl	       The private URL of the portal's hosting server.
        --------------------------     -------------------------------------------------------------------------------------------
        privatePortalUrl               The private URL of the portal.
        ==========================     ===========================================================================================



        When Setting the `configuration` the following must be defined and provided as a dictionary.

        ==================     ====================================================================
        **Key**                **Value**
        ------------------     --------------------------------------------------------------------
        portalProperties       Required Dict. Portal properties represented as a JSON object.
        ------------------     --------------------------------------------------------------------
        httpsProtocols         Required String. The TLS protocols ArcGIS Server will use. Valid options are TLSv1, TLSv1.1, and TLSv1.2; values must be comma separated. By default, only TLSv1.2 is enabled.
        ------------------     --------------------------------------------------------------------
        cipherSuites	       Required String. The cipher suites ArcGIS Notebook Server will use. Valid options are: TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256, TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA, TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA, TLS_RSA_WITH_AES_128_GCM_SHA256, TLS_RSA_WITH_AES_128_CBC_SHA256, TLS_RSA_WITH_AES_128_CBC_SHA, and TLS_RSA_WITH_3DES_EDE_CBC_SHA.
                               By default, all of the above options are enabled. Values must be comma separated
        ------------------     --------------------------------------------------------------------
        authenticationTier     Required String. The tier at which requests to access GIS services will be authenticated. You should only indicate your setting here, rather than using this operation to modify your setting. The default value when your site is first created is NOTEBOOK_SERVER, but when you federate the site with your portal, the value is changed to ARCGIS_PORTAL. Values: WEB_ADAPTOR | NOTEBOOK_SERVER | ARCGIS_PORTAL
        ==================     ====================================================================

        """
        url = self._url + "/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @configuration.setter
    def configuration(self, settings):
        """
        See main ``configuration`` property docstring
        """
        url = self._url + "/config/update"
        params = {"f": "json"}
        current = dict(self.configuration)
        assert isinstance(settings, (dict, PropertyMap))
        params.update(current)  # Load the current settings
        params.update(settings)  # Load the user settings in from the dictionary
        return self._con.post(url, params)  # post the change.
