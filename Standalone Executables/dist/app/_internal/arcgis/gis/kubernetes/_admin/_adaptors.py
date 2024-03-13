from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS


###########################################################################
class WebAdaptorManager(_BaseKube):
    """
    Provides access to the web adaptor resources defined on the ArcGIS
    Enterprise.
    """

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ---------------------------------------------------------------------
    def web_adaptor(self, adaptor_id: str):
        """
        This resource returns the properties of an individual web adaptor, such as the HTTP and HTTPS ports.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        adaptor_id             Required string. The specific web adaptor to get.
        ==================     ====================================================================
        """
        url = f"{self._url}/{adaptor_id}"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    def unregister_adaptor(self, adaptor_id: str):
        """
        This operation unregisters an ArcGIS Enterprise on Kubernetes Web Adaptor
        from your deployment. Once a web adaptor has been unregistered, the
        web adaptor will no longer be trusted and its credentials will not be
        accepted. This operation is typically used when you want to register
        a new ArcGIS Enterprise on Kubernetes Web Adaptor or when the previous
        one needs to be updated.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        adaptor_id             Required string. The web adaptor to unregister.
        ==================     ====================================================================
        """
        url = f"{self._url}/{adaptor_id}/unregister"
        params = {"f": "json"}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    @property
    def configuration(self):
        """
        This resource is a collection of configuration properties that apply
        to the ArcGIS Enterprise on Kubernetes Web Adaptor configured with your deployment.
        The only supported property is sharedKey, which represents credentials
        that are shared with the web adaptor. The web adaptor will use these credentials
        to communicate with your deployment.
        """
        url = f"{self._url}/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    @configuration.setter
    def configuration(self, configs: dict):
        """
        This operation is used to change the common properties and configurations
        for the ArcGIS Enterprise on Kubernetes Web Adaptor configured with your
        deployment. The properties are stored in a JSON object. Therefore,
        every update must include all necessary properties.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        configs                Required dictionary. The new configs for the web adaptors
        ==================     ====================================================================
        """
        url = f"{self._url}/config/update"
        params = {"f": "json", "webAdaptorsConfig": configs}
        return self._con.post(url, params)
