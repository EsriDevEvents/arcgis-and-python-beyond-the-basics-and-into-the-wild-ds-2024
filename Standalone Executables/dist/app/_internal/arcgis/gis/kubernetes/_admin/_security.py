"""
Contains the base class that all server object inherit from.
"""
from __future__ import annotations
from urllib.request import HTTPError
from arcgis.gis import GIS
from arcgis._impl.common._isd import InsensitiveDict
from typing import Dict, Any, Optional, List


###########################################################################
class KubeSecurityCert(object):
    """
    The certificates resource provides access to child operations and
    resources that can be used to manage all the security certificates
    configured with an organization.
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url: str, gis: GIS) -> "KubeSecurityIngress":
        """class initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value: str):
        """gets/sets the service url"""
        self._url = value
        self._refresh()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def identity_certs(self) -> list:
        """
        Lists all the certificates currently configured with the organization

        :return: List
        """
        url = self._url + "/identity"
        params = {"f": "json"}
        return self._con.get(url, params).get("certificates", [])

    # ----------------------------------------------------------------------
    def remove_identity_cert(self, cert_id: str) -> bool:
        """Deletes an Identity Certificate by ID

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        cert_id                Required String. The unique identifier of the certificate.
        ==================     ====================================================================

        """
        url = self._url + f"/identity/{cert_id}/delete"
        params = {"f": "json"}
        return self._con.post(url, params).get("status", "failed") == "success"

    # ----------------------------------------------------------------------
    def load_identity_cert(self, pfx: str, password: str, name: str) -> bool:
        """
        Imports an existing identity certificate in PKCS #12 (.pfx) format
        into the keystore. An imported certificate can be assigned to the
        Ingress controller by setting the certificate name property via the
        update operation.

        :return: bool
        """
        params = {
            "f": "json",
            "certificateName": name,
            "certificatePassword": password,
        }
        files = {"certificatePfxFile": pfx}
        url = self._url + "/identity/import"
        ret = self._gis._con.post(path=url, postdata=params, files=files)
        return ret.get("status", "false") == "success"

    # ----------------------------------------------------------------------
    @property
    def trust_certs(self) -> list:
        """
        Lists all the trust certificates configured with the organization
        :return: list
        """
        params = {"f": "json"}

        url = self._url + "/trust"
        return self._con.get(url, params).get("certificates", [])

    # ----------------------------------------------------------------------
    def load_trust_cert(self, cert: str, name: str) -> bool:
        """
        This operation imports a trust certificate, in either PEM
        (.cer or .crt files) or a binary (.der) format. Once a trust
        certificate is imported, the corresponding pods that will use the
        certificate are automatically restarted.

        :return: bool
        """
        params = {"f": "json", "certificateName": name}
        files = {"trustCertificateFile": cert}
        url = self._url + "/trust/import"
        ret = self._gis._con.post(path=url, postdata=params, files=files)
        return ret.get("status", "false") == "success"

    def get_cert(self, cert_type: str, cert_id: str) -> dict:
        """
        Obtains a single certificate for a given type and ID

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        cert_type              Required String. The type of certificate to search for. This can be 'trust' or 'identity'.
        ------------------     --------------------------------------------------------------------
        cert_id                Required String. The unique identifier of the certificate.
        ==================     ====================================================================

        :return: Dict
        """
        if cert_type.lower() == "trust":
            url = self._url + f"/trust/{cert_id}"
            params = {"f": "json"}
            return self._con.get(url, params)
        elif cert_type.lower() == "identity":
            url = self._url + f"/identity{cert_id}"
            params = {"f": "json"}
            return self._con.get(url, params)
        else:
            raise ValueError("Invalid certificate type.")

    # ----------------------------------------------------------------------
    def remove_trust_cert(self, cert_id: str) -> bool:
        """Deletes an Identity Certificate by ID

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        cert_id                Required String. The unique identifier of the certificate.
        ==================     ====================================================================

        """
        url = self._url + f"/trust/{cert_id}/delete"
        params = {"f": "json"}
        return self._con.post(url, params).get("status", "failed") == "success"


###########################################################################
class KubeSecuritySAML(object):
    """
    The saml resource returns information about the SAML configuration for
    an organization. If SAML is configured, the enabled property will
    return as true and `identityCertificateName` will show the name of the
    imported identity certificate.
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url: str, gis: GIS) -> "KubeSecurityIngress":
        """class initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value: str):
        """gets/sets the service url"""
        self._url = value
        self._refresh()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        """
        This get/sets the SAML certification information.

        :return: dict

        """
        return dict(self.properties)

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value: dict):
        """
        This get/sets the SAML certification information.

        :return: dict
        """
        url = self.url + "/update"
        params = {"f": "json", "samlSecurityConfig": value}
        res = self._con.post(url, params)
        if res.get("status", "failed") == "success":
            self._refresh()


###########################################################################
class KubeSecurityIngress(object):
    """
    The ingress resource returns the currently configured security
    information for the Ingress controller. You can update ingress security
    configuration properties using the update operation. The update
    operation must be used when adding an imported wildcard certificate for
    the Ingress controller.
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url: str, gis: GIS) -> "KubeSecurityIngress":
        """class initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value: str):
        """gets/sets the service url"""
        self._url = value
        self._refresh()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        """
        gets/sets the ingress configuration properties

        :return: dict

        """
        return dict(self.properties)

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value: dict):
        """
        gets/sets the ingress configuration properties

        :return: dict
        """
        url = self.url + "/update"
        params = {"f": "json", "samlSecurityConfig": value}
        res = self._con.post(url, params)
        if res.get("status", "failed") == "success":
            self._refresh()


###########################################################################
class KubeSecurityConfig(object):
    """
    Allows the user to manage the security configuration for an ArcGIS Enterprise for Kubernetes deployment.
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url: str, gis: GIS) -> "KubeSecurityConfig":
        """class initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value: str):
        """gets/sets the service url"""
        self._url = value
        self._refresh()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        """gets/sets the current secutiry settings for the deployment"""
        return dict(self.properties)

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value: dict):
        """gets/sets the current secutiry settings for the deployment"""
        url = self.url + "/update"
        params = {"f": "json", "securityConfig": value}
        res = self._con.post(url, params)
        if res.get("status", "failed") == "success":
            self._refresh()

    # ----------------------------------------------------------------------
    def test(self, user_store: dict = None, role_store: dict = None) -> bool:
        """
        Users can test the connection to a user or role (group) store.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        user_store             Optional dict. Specifies the user store properties. This parameter
                               accepts as input all the properties as defined in the
                               `user_store` and `role_store` section of the Kubernetes help
                               documentation.
        ------------------     --------------------------------------------------------------------
        role_store             Optional dict. pecifies the role (group) store properties. This parameter
                               accepts as input all the properties as defined in the
                               `user_store` and `role_store` section of the Kubernetes help
                               documentation.
        ==================     ====================================================================

        :return: boolean
        """
        url = f"{self._url}/testIdentityStore"
        params = {
            "f": "json",
        }
        if user_store:
            params["userStoreConfig"] = user_store
        if role_store:
            params["roleStoreConfig"] = role_store
        return self._con.post(url, params).get("status", "fail") == "success"

    # ----------------------------------------------------------------------
    def update_stores(self, user_store: dict = None, role_store: dict = None) -> bool:
        """
        Users can modify the user or role (group) identity stores.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        user_store             Optional dict. Specifies the user store properties. This parameter accepts as input all the properties as defined in the userStoreConfig and roleStoreConfig section of the Kubernetes help doctumentation.
        ------------------     --------------------------------------------------------------------
        role_store             Optional dict. pecifies the role (group) store properties. This parameter accepts as input all the properties as defined in the ArcGIS for Kubernetes help doctumentation.
        ==================     ====================================================================

        :return: boolean
        """
        url = f"{self._url}/updateIdentityStore"
        params = {
            "f": "json",
        }
        if user_store:
            params["userStoreConfig"] = user_store
        if role_store:
            params["roleStoreConfig"] = role_store
        return self._con.post(url, params).get("status", "fail") == "success"


###########################################################################
class KubeSecurity(object):
    """Allows users to configure the Security settings on the kubernetes infrastructure"""

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None
    _config = None
    _ingress = None
    _saml = None
    _certs = None

    def __init__(self, url: str, gis: GIS) -> "KubeSecurity":
        """class initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        returns the object properties
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def url(self):
        """gets/sets the service url"""
        return self._url

    # ----------------------------------------------------------------------
    @url.setter
    def url(self, value):
        """gets/sets the service url"""
        self._url = value
        self._refresh()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """reloads all the properties of a given service"""
        self._init()

    # ----------------------------------------------------------------------
    @property
    def configuration(self) -> "KubeSecurityConfig":
        """Returns the currently active security configuration for an ArcGIS Enterprise for Kubernetes deployment"""
        if self._config is None:
            url = self._url + "/config"
            self._config = KubeSecurityConfig(url=url, gis=self._gis)
        return self._config

    # ----------------------------------------------------------------------
    @property
    def ingress(self) -> "KubeSecurityIngress":
        """Returns a manager to configure the ingress settings.

        :return:
            A KubeSecurityIngress object.
        """
        if self._ingress is None:
            url = self._url + "/ingress"
            self._ingress = KubeSecurityIngress(url, gis=self._gis)
        return self._ingress

    # ----------------------------------------------------------------------
    @property
    def saml(self) -> "KubeSecuritySAML":
        """
        Returns a manager to work with the SAML settings for the organization

        :return: KubeSecuritySAML
        """
        if self._saml is None:
            url = self._url + "/saml"
            self._saml = KubeSecuritySAML(url, gis=self._gis)
        return self._saml

    # ----------------------------------------------------------------------
    @property
    def certificates(self) -> "KubeSecurityCert":
        """
        Provides access to the certificate manager for the Kubernetes infrastructure

        :returns:
            A KubeSecurityCert object.

        """
        if self._certs is None:
            url = self._url + "/certificates"
            self._certs = KubeSecurityCert(url, gis=self._gis)
        return self._certs
