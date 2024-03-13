"""
The **gis** module provides an information model for GIS hosted
within ArcGIS Online or ArcGIS Enterprise, serving as an entry point to the GIS.
This module, the most important in the ArcGIS API for Python, provides functionality to manage
(create, read, update and delete) GIS users, groups and content. The module allows for access to the GIS services using
Python and is an invaluable tool in the API.

"""
from __future__ import absolute_import, annotations
import base64
import json
import locale
import io
import os
import re
import uuid
import time
import shutil
import tempfile
import warnings
import zipfile
from uuid import uuid4
import configparser
from contextlib import contextmanager
import functools
from datetime import datetime, timedelta
import logging
from typing import Any, Optional, Union
from urllib.error import HTTPError
import requests

from arcgis.gis._impl._dataclasses._contentds import (
    ItemTypeEnum,
    ItemProperties,
)
from arcgis.gis._impl import (
    MetadataFormatEnum,
    CreateServiceParameter,
    ViewLayerDefParameter,
)

try:
    import pandas as pd
except:
    pass
try:
    import arcpy

    has_arcpy = True
except ImportError:
    has_arcpy = False
except RuntimeError:
    has_arcpy = False
try:
    import shapefile

    has_pyshp = True
except ImportError:
    has_pyshp = False
import concurrent.futures

from cachetools import cached, TTLCache

from arcgis.auth.tools import LazyLoader
from arcgis.auth import EsriSession

arcgis_env = LazyLoader("arcgis.env")
arcgis = LazyLoader("arcgis")
features = LazyLoader("arcgis.features")
_agoserver = LazyLoader("arcgis.gis.agoserver._api")
_mixins = LazyLoader("arcgis._impl.common._mixins")
_common_utils = LazyLoader("arcgis._impl.common._utils")
_common_deprecated = LazyLoader("arcgis._impl.common._deprecate")
_portalpy = LazyLoader("arcgis.gis._impl._portalpy")
_jb = LazyLoader("arcgis.gis._impl._jb")
_cloner = LazyLoader("arcgis.gis.clone")
_cm_helper = LazyLoader("arcgis.gis._impl._content_manager._import_data")
_log = logging.getLogger(__name__)


class Error(Exception):
    pass


@contextmanager
def _tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write((bytes(data, "UTF-8")))
    temp.close()
    yield temp.name
    os.unlink(temp.name)


def _lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


try:
    from arcgis.features.geo import _is_geoenabled
except:

    def _is_geoenabled(o):
        return False


class GIS(object):
    """
    .. _gis:

    The ``GIS`` class is representative of a single ArcGIS Online organization or an ArcGIS Enterprise deployment.
    The ``GIS`` object provides helper objects to manage (search, create, retrieve) GIS resources such as content, users,
    and groups.

    Additionally, the ``GIS`` object has properties to query its state, which is accessible using the properties attribute.

    .. note::

        The ``GIS`` provides a mapping widget that can be used in the Jupyter Notebook environment for visualizing GIS content
        as well as the results of your analysis. To create a new map, call the :attr:`~arcgis.gis.GIS.map` method.
        IE11 is no longer supported. Please use the latest version of Google Chrome, Mozilla Firefox,
        Apple Safari, or Microsoft Edge.

    The constructor constructs a ``GIS`` object given a url and user credentials to ArcGIS Online
    or an ArcGIS Enterprise portal. User credentials can be passed in using username/password
    pair, or key_file/cert_file pair (in case of PKI). Supports built-in users, LDAP, PKI, Integrated Windows Authentication
    (using NTLM and Kerberos) and Anonymous access.

    If no url is provided, ArcGIS Online is used. If username/password
    or key/cert files are not provided, the currently logged-in user's credentials (IWA) or anonymous access is used.

    Persisted profiles for the GIS can be created by giving the GIS authorization credentials and
    specifying a profile name. The profile stores all of the authorization credentials (except the password) in the
    user's home directory in an unencrypted config file named .arcgisprofile. The profile securely stores the password
    in an O.S. specific password manager through the `keyring <https://pypi.python.org/pypi/keyring>`_ python module.

    .. note::
        Linux systems may need additional software installed and configured for proper security.

    Once a profile has been saved, passing the profile parameter by itself uses the authorization credentials saved
    in the configuration file/password manager by that profile name. Multiple profiles can be created and used in
    parallel.

    See `Working with different authentication schemes
    <https://developers.arcgis.com/python/guide/working-with-different-authentication-schemes/>`_
    in the ArcGIS API for Python guide for examples.


    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    url                 Optional string. If URL is None, then the URL will be ArcGIS
                        Online.  This should be a web address to either an ArcGIS Enterprise portal
                        or to ArcGIS Online in the form:
                        <scheme>://<fully_qualified_domain_name>/<web_adaptor>. An Enterprise example is formatted in
                        the form: https://gis.example.com/portal
    ----------------    ---------------------------------------------------------------
    username            Optional string. The login user name (case-sensitive).
    ----------------    ---------------------------------------------------------------
    password            Optional string. If a username is provided, a password is
                        expected.  This is case-sensitive. If the password is not
                        provided, the user is prompted in the interactive dialog.
    ----------------    ---------------------------------------------------------------
    key_file            Optional string. The file path to a user's key certificate for PKI
                        authentication
    ----------------    ---------------------------------------------------------------
    cert_file           Optional string. The file path to a user's certificate file for PKI
                        authentication. If a PFX or P12 certificate is used, a password is required.
                        If a PEM file is used, the key_file is required.
    ----------------    ---------------------------------------------------------------
    verify_cert         Optional boolean. If a site has an invalid SSL certificate or is
                        being accessed via the IP or hostname instead of the name on the
                        certificate, set this value to ``False``.  This will ensure that all
                        SSL certificate issues are ignored.
                        The default is ``True``.

                        .. warning::
                            Setting the value to ``False`` can be a security risk.
    ----------------    ---------------------------------------------------------------
    set_active          Optional boolean. The default is True.  If True, the GIS object
                        will be used as the default GIS object throughout the whole
                        scripting session.
    ----------------    ---------------------------------------------------------------
    client_id           Optional string. Used for OAuth authentication.  This is the
                        client ID value.
    ----------------    ---------------------------------------------------------------
    profile             Optional string. the name of the profile that the user wishes to use
                        to authenticate, if set, the identified profile will be used to login
                        to the specified GIS.
    ================    ===============================================================

    In addition to explicitly named parameters, the GIS object supports optional key word
    arguments:

    ======================    ===============================================================
    **kwargs**                **Description**
    ----------------------    ---------------------------------------------------------------
    proxy_host                Optional string. The host name of the proxy server used to allow HTTP/S
                              access in the network where the script is run.

                              ex: 127.0.0.1
    ----------------------    ---------------------------------------------------------------
    use_gen_token             Optional Boolean. The default is `False`. Uses generateToken
                              login over OAuth2 login.
    ----------------------    ---------------------------------------------------------------
    proxy_port                Optional integer. The proxy host port.  The default is 80.
    ----------------------    ---------------------------------------------------------------
    token                     Optional string. This is the Enterprise token for built-in
                              logins. This parameter is only honored if the username/password
                              is None and the security for the site uses BUILT-IN security.
    ----------------------    ---------------------------------------------------------------
    api_key                   Optional string.  This is a key generated by the developer site
                              to allow for a limited subset of the REST API functionality.
    ----------------------    ---------------------------------------------------------------
    trust_env                 Optional Boolean. Trust environment settings for proxy
                              configuration, default authentication and similar. If `False`
                              the GIS class will ignore the `netrc` files defined on the
                              system.
    ----------------------    ---------------------------------------------------------------
    proxy                     Optional Dictionary.  If you need to use a proxy, you can
                              configure individual requests with the proxy argument to any
                              request method.  See ```Usage Exmaple 9: Using a Proxy``` for
                              example usage.

                              :Usage Example:


                              {
                                  "http" : "http://10.343.10.22:111",
                                  "https" : "https://127.343.13.22:6443"
                              }
    ----------------------    ---------------------------------------------------------------
    expiration                Optional Integer.  The default is 60 minutes.  The expiration
                              time for a given token.  This is used for user provided tokens
                              and API Keys.
    ----------------------    ---------------------------------------------------------------
    validate_url              Optional Boolean. The default is False. A user can choose to
                              validate the URL on an `Item`'s url.
    ----------------------    ---------------------------------------------------------------
    mutual_authentication     Optional String. Mutual authentication is a security feature in
                              which a client process must prove its identity to a service,
                              and the service must prove its identity to the client, before
                              any application traffic is transmitted over the client/service
                              connection.

                              - REQUIRED - By default, the API will require mutual
                                           authentication from the server, and if a server
                                           emits a non-error response which cannot be
                                           authenticated.
                              - OPTIONAL - This will cause the API to attempt mutual
                                           authentication if the server advertises that it
                                           supports it, and cause a failure if authentication
                                           fails, but not if the server does not support it
                                           at all.
                              - DISABLED - Never attempts mutual authentication, this is not
                                           recommended.

    ----------------------    ---------------------------------------------------------------
    force_preemptive          If you are using Kerberos authentication, it can be forced to
                              preemptively initiate the Kerberos GSS exchange and present a
                              Kerberos ticket on the initial request (and all subsequent).
                              By default, authentication only occurs after a 401 Unauthorized
                              response containing a Kerberos or Negotiate challenge is
                              received from the origin server. This can cause mutual
                              authentication failures for hosts that use a persistent
                              connection (eg, Windows/WinRM), as no Kerberos challenges are
                              sent after the initial auth handshake. This behavior can be
                              altered by setting force_preemptive=True. The default is False
    ----------------------    ---------------------------------------------------------------
    hostname_override         Optional String. If communicating with a host whose DNS name
                              doesn't match its kerberos hostname (eg, behind a content
                              switch or load balancer), the hostname used for the Kerberos
                              GSS exchange can be overridden by setting this value.
    ----------------------    ---------------------------------------------------------------
    delegate                  Optional bool. Kerberos supports credential delegation
                              (GSS_C_DELEG_FLAG). To enable delegation of credentials to a
                              server that requests delegation, pass `delegate=True`. Be
                              careful to only allow delegation to servers you trust as they
                              will be able to impersonate you using the delegated credentials.
    ======================    ===============================================================

    .. code-block:: python

        # Usage Example 1: Anonymous Login to ArcGIS Online

        gis = GIS()

    .. code-block:: python

        # Usage Example 2: Built-in Login to ArcGIS Online

        gis = GIS(username="someuser", password="secret1234")

    .. code-block:: python

        # Usage Example 3: Built-in Login to ArcGIS Enterprise

        gis = GIS(url="http://pythonplayground.esri.com/portal",
              username="user1", password="password1")

    .. code-block:: python

        # Usage Example 4: Built-in Login to ArcGIS Enterprise, ignoring SSL errors

        gis = GIS(url="http://pythonplayground.esri.com/portal", username="user1",
                  password="password1", verify_cert=False)

    .. code-block:: python

        # Usage Example 5: Anonymous ArcGIS Online Login with Proxy

        gis = GIS(proxy_host='127.0.0.1', proxy_port=8888)

    .. code-block:: python

        # Usage Example 6: PKI Login to ArcGIS Enterprise, using PKCS12 user certificate

        gis = GIS(url="https://pkienterprise.esri.com/portal",
                  cert_file="C:\\users\\someuser\\mycert.pfx", password="password1")

    .. code-block:: python

        # Usage Exmaple 7: Login with token (actual token abbreviated for this illustration)

        gis = GIS(token="3G_e-FSoJdwxBgSA0RiOZg7zJVVqlOG-ENw83UtoUzDdz4 ... _L2aQMrthrEq7vKYBn39HGSc.",
                  referer="https://www.arcgis.com")

    .. code-block:: python

        # Usage Exmaple 8: Login with API Key (actual token abbreviated for this illustration)

        gis = GIS(api_key="APKSoJdwxBgSA0RiOZg7zJVVqlOG-ENw83UtoUzDdz4 ... _L2aQMrth39HGSc.",
                  referer="https")

    .. code-block:: python

        # Usage Exmaple 9: Using a Proxy
        proxy = {
            'http': 'http://10.10.1.10:3128',
            'https': 'http://10.10.1.10:1080',
        }
        gis = GIS(proxy=proxy)

    """

    _toolgp = None
    _server_list = None
    _is_hosted_nb_home = False
    _product_version = None
    _is_agol = None
    _pds = None
    _validate_item_url = None
    _properties = None
    _session: EsriSession
    """If 'True', the GIS instance is a GIS('home') from hosted nbs"""

    # admin = None
    # oauth = None
    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        verify_cert: bool = True,
        set_active: bool = True,
        client_id: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs,
    ):
        """
        Constructs a GIS object given a url and user credentials to ArcGIS Online
        or an ArcGIS Enterprise portal. User credentials can be passed in using username/password
        pair, or key_file/cert_file pair (in case of PKI). Supports built-in users, LDAP,
        PKI, Integrated Windows Authentication (using NTLM and Kerberos) and Anonymous access.

        If no url is provided, ArcGIS Online is used. If username/password
        or key/cert files are not provided, logged in user credentials (IWA) or anonymous access is used.

        Persisted profiles for the GIS can be created by giving the GIS authorization credentials and
        specifying a profile name. The profile stores all of the authorization credentials (except the password) in the
        user's home directory in an unencrypted config file named .arcgisprofile. The profile securely stores the password
        in an O.S. specific password manager through the `keyring <https://pypi.python.org/pypi/keyring>`_ python module.
        (Note: Linux systems may need additional software installed and configured for proper security) Once a profile has
        been saved, passing the profile parameter by itself uses the authorization credentials saved in the configuration
        file/password manager by that profile name. Multiple profiles can be created and used in parallel.

        If the GIS uses a secure (https) url, certificate verification is performed. If you are using self signed certificates
        in a testing environment and wish to disable certificate verification, you may specify verify_cert=False to disable
        certificate verification in the Python process. However, this should not be done in production environments and is
        strongly discouraged.
        """
        self._validate_item_url = kwargs.pop("validate_url", False)
        self._use_gen_token = kwargs.pop("use_gen_token", False)
        self._proxy_host = kwargs.pop("proxy_host", None)
        self._proxy_port = kwargs.pop("proxy_port", 80)
        self._referer = kwargs.pop("referer", None)
        self._timeout = kwargs.pop("timeout", 600)  # default timeout is 600 seconds
        custom_auth = kwargs.pop("custom_auth", None)
        custom_adapter = kwargs.pop("adapter", None)
        self._expiration = kwargs.pop("expiration", None)
        security_kwargs = {
            "mutual_authentication": kwargs.pop("mutual_authentication", None),
            "force_preemptive": kwargs.pop("force_preemptive", None),
            "hostname_override": kwargs.pop("hostname_override", None),
            "delegate": kwargs.pop("delegate", None),
            "service": kwargs.pop("service", None),
            "sanitize_mutual_error_response": kwargs.pop(
                "sanitize_mutual_error_response", None
            ),
            "send_cbt": kwargs.pop("send_cbt", None),
        }
        security_kwargs = {k: v for k, v in security_kwargs.items() if not v is None}
        if profile is not None and len(profile) == 0:
            raise ValueError("A `profile` name must not be an empty string.")
        elif profile is not None:
            # Load config
            pm = ProfileManager()

            cfg_file_path = pm._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
                # Check if config file is in the old format
                if not self._config_is_in_new_format(config):
                    answer = input(
                        "Warning: profiles in {} appear to be in "
                        "the <v1.3 format, and must be deleted before "
                        "continuing. Delete? [y/n]".format(cfg_file_path)
                    )
                    if "y" in answer.lower():
                        os.remove(cfg_file_path)
                        config = configparser.ConfigParser()
                    else:
                        raise RuntimeError(
                            "{} not deleted, exiting" "".format(cfg_file_path)
                        )

            # Add any __init__() args to config/keyring store
            if profile not in pm.list():
                _log.info("Adding new profile {} to config...".format(profile))
                pm.create(
                    profile=profile,
                    url=url,
                    username=username,
                    password=password,
                    key_file=key_file,
                    cert_file=cert_file,
                    client_id=client_id,
                )
            elif profile in pm.list():
                # run an update to be safe.
                pm.update(
                    profile,
                    url=url,
                    username=username,
                    password=password,
                    key_file=key_file,
                    cert_file=cert_file,
                    client_id=client_id,
                )
            if (
                profile in pm.list()
            ):  # check if the profile name was successfully added, if so, use the profile credentials
                (
                    url,
                    username,
                    password,
                    key_file,
                    cert_file,
                    client_id,
                ) = pm._retrieve(profile)
            else:
                _log.info(
                    f"Profile {profile} was not saved, using user provided credentials for the `GIS` object."
                )

        if url is None:
            url = "https://www.arcgis.com"
        if (self._uri_validator(url) == False) and (
            str(url).lower() not in ["pro", "home"]
        ):
            raise Exception("Malformed url provided: %s" % url)
        if username is not None and password is None:
            from getpass import getpass

            password = getpass("Enter password: ")
        # Assumes PFX is being passed in cert_file parameter and no key_file is specified
        if (cert_file is not None) and (key_file is None):
            if cert_file.lower().endswith(".pfx") or cert_file.lower().endswith(".p12"):
                if password is None:
                    from getpass import getpass

                    password = getpass("Enter PFX password: ")
                key_file, cert_file = self._pfx_to_pem(cert_file, password)
            else:
                raise Exception(
                    "key_file parameter is required along with cert_file when using PKI authentication."
                )

        self._url = url
        self._username = username
        self._password = password
        self._key_file = key_file
        self._cert_file = cert_file
        self._portal = None
        self._con = None
        self._verify_cert = verify_cert
        self._client_id = client_id
        self._datastores_list = None
        self._utoken = kwargs.pop("token", None)
        client_secret = kwargs.pop("client_secret", None)
        self._api_key = None
        if verify_cert == False:
            log = logging.getLogger()
            log.warning(
                "Setting `verify_cert` to False is a security risk, use at your own risk."
            )
        if self._username is None:
            if "ESRI_API_KEY" in os.environ and self._utoken is None:
                self._utoken = os.environ.get("ESRI_API_KEY", None)
            elif self._utoken is None and not "ESRI_API_KEY" in os.environ:
                self._utoken = kwargs.pop("api_key", None)
            self._api_key = self._utoken

        if self._url.lower() == "home" and not os.getenv("NB_AUTH_FILE", None) is None:
            # configuring for hosted notebooks need to happen before portalpy
            self._try_configure_for_hosted_nb()
            if self._expiration is None:
                self._expiration = 10080
        elif self._url.lower() == "home" and os.getenv("NB_AUTH_FILE", None) is None:
            self._url = "pro"
            url = "pro"
        elif self._expiration is None:  # Keep Default Value
            self._expiration = 60
        try:
            self._portal = _portalpy.Portal(
                self._url,
                self._username,
                self._password,
                self._key_file,
                self._cert_file,
                proxy_host=self._proxy_host,
                proxy_port=self._proxy_port,
                verify_cert=self._verify_cert,
                client_id=self._client_id,
                expiration=self._expiration,
                referer=self._referer,
                custom_auth=custom_auth,  # token=self._utoken,
                client_secret=client_secret,
                trust_env=kwargs.get("trust_env", None),
                timeout=self._timeout,
                proxy=kwargs.get("proxy", None),
                custom_adapter=custom_adapter,
                token=self._utoken,
                api_key=self._api_key,
                is_hosted_nb_home=self._is_hosted_nb_home,
                use_gen_token=self._use_gen_token,
                security_kwargs=security_kwargs,
            )
            if self._portal.is_kubernetes:
                from .kubernetes._sharing import KbertnetesPy

                self._portal = KbertnetesPy(
                    self._url,
                    self._username,
                    self._password,
                    self._key_file,
                    self._cert_file,
                    proxy_host=self._proxy_host,
                    proxy_port=self._proxy_port,
                    verify_cert=self._verify_cert,
                    client_id=self._client_id,
                    expiration=self._expiration,
                    referer=self._referer,
                    custom_auth=custom_auth,
                    trust_env=kwargs.get("trust_env", None),
                    timeout=self._timeout,
                    proxy=kwargs.get("proxy", None),
                    custom_adapter=custom_adapter,
                    token=self._utoken,
                    api_key=self._api_key,
                    is_hosted_nb_home=self._is_hosted_nb_home,
                    use_gen_token=self._use_gen_token,
                    security_kwargs=security_kwargs,
                )
            if self._is_hosted_nb_home:
                self._portal.con._referer = ""
                self._portal.con._session.headers.pop("Referer", None)
        except Exception as e:
            if len(e.args) > 0 and str(type(e.args[0])) == "<class 'ssl.SSLError'>":
                raise RuntimeError(
                    "An untrusted SSL error occurred when attempting to connect to the provided GIS.\n"
                    "If you trust this server and want to proceed, add 'verify_cert=False' as an "
                    "argument when connecting to the GIS."
                )
            else:
                raise e
        try:
            if (
                url.lower().find("arcgis.com") > -1
                and self._portal.is_logged_in
                and self._portal.con._auth.lower() == "oauth"
            ):
                from urllib.parse import urlparse

                props = self._portal.get_properties(force=False)
                url = "%s://%s.%s" % (
                    urlparse(self._url).scheme,
                    props["urlKey"],
                    props["customBaseUrl"],
                )
                self._url = url
                self._portal.resturl = self._portal.resturl.replace(
                    self._portal.url, url
                )
                self._portal.url = url
                self._portal.con.baseurl = self._portal.resturl
                if self._portal.con._auth != "OAUTH":
                    self._portal.con._token = None
            elif url.lower().find("arcgis.com") > -1 and self._portal.is_logged_in:
                from urllib.parse import urlparse

                props = self._portal.get_properties(force=False)
                url = "%s://%s.%s" % (
                    urlparse(self._url).scheme,
                    props["urlKey"],
                    props["customBaseUrl"],
                )
                if self._url != url:
                    self._url = url
                    pp = _portalpy.Portal(
                        url,
                        self._username,
                        self._password,
                        self._key_file,
                        self._cert_file,
                        verify_cert=self._verify_cert,
                        client_id=self._client_id,
                        proxy_port=self._proxy_port,
                        proxy_host=self._proxy_host,
                        expiration=self._expiration,
                        referer=self._referer,
                        custom_auth=custom_auth,
                        trust_env=kwargs.get("trust_env", None),
                        client_secret=client_secret,
                        timeout=self._timeout,
                        proxy=kwargs.get("proxy", None),
                        custom_adapter=custom_adapter,
                        token=self._utoken,
                        api_key=self._api_key,
                        is_hosted_nb_home=self._is_hosted_nb_home,
                        use_gen_token=self._use_gen_token,
                        security_kwargs=security_kwargs,
                    )
                    self._portal = pp
        except:
            pass

        force_refresh = False
        if self._portal.con._auth in ["HOME", "USER_TOKEN"]:
            force_refresh = True

        # If a token was injected, then force refresh to get updated properties
        self._properties = _mixins.PropertyMap(
            self._portal.get_properties(force=force_refresh)
        )
        self._lazy_properties = _mixins.PropertyMap(
            self._portal.get_properties(force=force_refresh)
        )

        self._con = self._portal.con

        if self._url.lower() == "pro":
            self._url = self._portal.url
            if self._con._auth != "ANON":
                self._con._auth = "PRO"

        if self._con._auth != "anon":
            me = self.users.me

        if (
            self._con._auth.lower() != "anon"
            and self._con._auth is not None
            and hasattr(me, "role")
            and me.role == "org_admin"
        ):
            try:
                if self._is_hosted_nb_home:
                    import warnings

                    warnings.warn(
                        "You are logged on as %s with an administrator role, proceed with caution."
                        % self.users.me.username
                    )
                if self.properties.isPortal and self._portal.is_kubernetes:
                    from arcgis.gis.kubernetes._admin.kadmin import (
                        KubernetesAdmin,
                    )

                    url = self._portal.url + "/admin"
                    self.admin = KubernetesAdmin(url=url, gis=self)
                elif (
                    self.properties.isPortal == True
                    and self._portal.is_kubernetes == False
                ):
                    from arcgis.gis.admin.portaladmin import (
                        PortalAdminManager,
                    )

                    self.admin = PortalAdminManager(
                        url="%s/portaladmin" % self._portal.url, gis=self
                    )
                else:
                    from .admin.agoladmin import AGOLAdminManager

                    self.admin = AGOLAdminManager(gis=self)
            except Exception as e:
                pass
        elif (
            self._con._auth.lower() != "anon"
            and self._con._auth is not None
            and hasattr(me, "role")
            and me.role == "org_publisher"
            and self._portal.is_arcgisonline == False
        ):
            try:
                if self._portal.is_kubernetes:
                    from arcgis.gis.kubernetes._admin.kadmin import (
                        KubernetesAdmin,
                    )

                    url = self._portal.url + "/admin"
                    self.admin = KubernetesAdmin(url=url, gis=self)
                else:
                    from .admin.portaladmin import PortalAdminManager

                    self.admin = PortalAdminManager(
                        url="%s/portaladmin" % self._portal.url,
                        gis=self,
                        is_admin=False,
                    )
            except:
                pass
        elif (
            self._con._auth.lower() != "anon"
            and self._con._auth is not None
            and hasattr(me, "privileges")
            and self._portal.is_arcgisonline == False
        ):
            privs = [
                "portal:publisher:publishFeatures",
                "portal:publisher:publishScenes",
                "portal:publisher:publishServerGPServices",
                "portal:publisher:publishServerServices",
                "portal:publisher:publishTiles",
            ]
            for priv in privs:
                if priv in me.privileges:
                    can_publish = True
                    break
                else:
                    can_publish = False
            if can_publish:
                try:
                    if self.properties.isPortal and self._portal.is_kubernetes:
                        from arcgis.gis.kubernetes._admin.kadmin import (
                            KubernetesAdmin,
                        )

                        url = self._portal.url + "/admin"
                        self.admin = KubernetesAdmin(url=url, gis=self)
                    else:
                        from .admin.portaladmin import PortalAdminManager

                        self.admin = PortalAdminManager(
                            url="%s/portaladmin" % self._portal.url,
                            gis=self,
                            is_admin=False,
                        )
                except:
                    pass
        if (
            self._con._auth.lower() != "anon"
            and self._con._auth is not None
            and hasattr(me, "role")
            and me.role == "org_publisher"
            and self._portal.is_arcgisonline == False
        ):
            try:
                if self.properties.isPortal and self._portal.is_kubernetes:
                    from arcgis.gis.kubernetes._admin.kadmin import (
                        KubernetesAdmin,
                    )

                    url = self._portal.url + "/admin"
                    self.admin = KubernetesAdmin(url=url, gis=self)
                else:
                    from .admin.portaladmin import PortalAdminManager

                    self.admin = PortalAdminManager(
                        url="%s/portaladmin" % self._portal.url,
                        gis=self,
                        is_admin=False,
                    )
            except:
                pass
        # self._tools = _Tools(self)
        if set_active:
            from arcgis import env

            env.active_gis = self
        if self._product_version is None:
            self._is_agol = self._portal.is_arcgisonline
            self._product_version = [
                int(i) for i in self._portal.get_version().split(".")
            ]
        self._session: EsriSession = self._con._session

    # ----------------------------------------------------------------------
    @property
    def _tools(self):
        if self._toolgp is None:
            from arcgis._impl.tools import _Tools

            self._toolgp = _Tools(self)
        return self._toolgp

    # ----------------------------------------------------------------------
    @property
    @functools.lru_cache(maxsize=100)
    def _is_arcgisonline(self):
        """Returns true if this portal is ArcGIS Online."""
        return self.properties["portalName"] == "ArcGIS Online" and self._is_multitenant

    # ----------------------------------------------------------------------
    @property
    def session(self) -> EsriSession:
        """
        Provides the raw Esri Session object

        :returns: EsriSession

        """
        if self._session is None:
            self._session = self._con._session
        return self._session

    # ----------------------------------------------------------------------
    @property
    @functools.lru_cache(maxsize=100)
    def _is_multitenant(self) -> bool:
        """Returns true if this portal is multitenant."""
        return self.properties["portalMode"] == "multitenant"

    # ----------------------------------------------------------------------
    @property
    @functools.lru_cache(maxsize=100)
    def _is_kubernetes(self) -> bool:
        """Returns true if this portal is kubernetes."""
        return (
            "portalDeploymentType" in self.properties
            and self._properties["portalDeploymentType"]
            == "ArcGISEnterpriseOnKubernetes"
        )

    # ----------------------------------------------------------------------
    @property
    @functools.lru_cache(maxsize=100)
    def _is_authenticated(self) -> bool:
        """Checks if the GIS is signed in"""
        from ..auth._auth import check_response_for_error

        url: str = self._portal.resturl + "community/self"
        params: dict[str, Any] = {
            "f": "json",
        }
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        try:
            msg = check_response_for_error(resp)
            if msg is None:
                return True
            else:
                _log.info(msg)
                return False
        except:
            return False

    # ----------------------------------------------------------------------
    @_lazy_property
    def api_keys(self):
        """
        The ``api_keys`` property returns an instance of  :class:`~arcgis.gis._impl.APIKeyManager` object which allows
        the User to generate, manage and modify API Keys for controlled application access.

        .. note::
            **The API Key manager is only available for ArcGIS Online**

        :return:
            An :class:`~arcgis.gis._impl.APIKeyManager` object

        """
        if self._portal.is_arcgisonline and self.version >= [8, 2]:
            from arcgis.gis._impl._apikeys import APIKeyManager

            return APIKeyManager(self)
        return None

    # ----------------------------------------------------------------------
    @_lazy_property
    def languages(self) -> list[dict[str, Any]]:
        """
        Lists the available languages.

        :return: List[Dict[str, Any]]
        """
        url = f"{self._portal.resturl}portals/languages"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @_lazy_property
    def regions(self) -> list[dict[str, Any]]:
        """
        Lists the available regions.

        :return: List[Dict[str, Any]]
        """
        url = f"{self._portal.resturl}portals/regions"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def _private_service_url(self, service_url):
        """
        returns the public and private URL for a given registered service

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        service_url         Required string.  The URL to the service.
        ===============     ====================================================================

        :return: dict

        """
        if self.version < [5, 3]:
            return {"serviceUrl": service_url}
        url = ("{base}portals/self" "/servers/computePrivateServiceUrl").format(
            base=self._portal.resturl
        )
        params = {"f": "json", "serviceUrl": service_url}

        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def _pfx_to_pem(self, pfx_path, pfx_password):
        """Decrypts the .pfx file to be used with requests.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        pfx_path            Required string.  File pathname to .pfx file to parse.
        ---------------     --------------------------------------------------------------------
        pfx_password        Required string.  Password to open .pfx file to extract key/cert.
        ===============     ====================================================================

        :return:
           File path to key_file located in a tempfile location
           File path to cert_file located in a tempfile location
        """
        try:
            import OpenSSL.crypto
        except:
            raise RuntimeError(
                "OpenSSL.crypto library is not installed.  You must install this in order "
                + "to use a PFX for connecting to a PKI protected portal."
            )
        key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        k = open(key_file.name, "wb")
        c = open(cert_file.name, "wb")
        try:
            pfx = open(pfx_path, "rb").read()
            p12 = OpenSSL.crypto.load_pkcs12(pfx, pfx_password)
        except OpenSSL.crypto.Error:
            raise RuntimeError("Invalid PFX password.  Unable to parse file.")
        k.write(
            OpenSSL.crypto.dump_privatekey(
                OpenSSL.crypto.FILETYPE_PEM, p12.get_privatekey()
            )
        )
        c.write(
            OpenSSL.crypto.dump_certificate(
                OpenSSL.crypto.FILETYPE_PEM, p12.get_certificate()
            )
        )
        k.close()
        c.close()
        return key_file.name, cert_file.name

    def _config_is_in_new_format(self, config):
        """Any version <= 1.3.0 of the API used a different config file
        formatting that, among other things, did not store the last time
        a profile was modified. Thus, if 'date_modified' is found in at least
        one profile, it is in the new format
        """
        return any(
            [
                profile_data
                for profile_data in config.values()
                if "date_modified" in profile_data
            ]
        )

    def _try_configure_for_hosted_nb(self):
        """If 'home' is specified as the 'url' argument, this func is called"""
        try:
            # Set relevant properties and overrides
            self._is_hosted_nb_home = True
            self._verify_cert = False

            # Get the auth file from environment variables
            nb_auth_file_path = os.getenv("NB_AUTH_FILE", None)
            if not nb_auth_file_path:
                raise RuntimeError(
                    "Environment variable 'NB_AUTH_FILE' " "must be defined."
                )
            elif not os.path.isfile(nb_auth_file_path):
                raise RuntimeError(
                    "'{}' file needed for "
                    "authentication not found.".format(nb_auth_file_path)
                )
            # Open that auth file,
            with open(nb_auth_file_path) as nb_auth_file:
                required_json_keys = set(
                    ["privatePortalUrl", "publicPortalUrl", "referer"]
                )
                json_data = json.load(nb_auth_file)
                assert required_json_keys.issubset(json_data)
                self._url = json_data["privatePortalUrl"]
                self._public_portal_url = json_data["publicPortalUrl"]
                self._referer = json_data.get("referer", "")
                if "token" in json_data:
                    self._utoken = json_data["token"]
                self._expiration = json_data.get("expiration", None)
                if "encryptedToken" in json_data:
                    try:
                        from arcgis.gis._impl._decrypt_nbauth import (
                            get_token,
                        )
                    except ImportError as ie:
                        from arcgis.gis._impl.nbauth import get_token

                    self._utoken = get_token(nb_auth_file_path)

        # Catch errors and re-throw in with more human readable messages
        except json.JSONDecodeError as e:
            self._raise_hosted_nb_error(
                "'{}' file is not " "valid JSON.".format(nb_auth_file.name)
            )
        except AssertionError as e:
            self._raise_hosted_nb_error(
                "Authentication file doesn't contain "
                "required keys {}".format(required_json_keys)
            )
        except Exception as e:
            self._raise_hosted_nb_error(
                "Unexpected exception when authenticating "
                "through 'home' mode: {}".format(e)
            )

    def _raise_hosted_nb_error(self, err_msg):
        """In the event a user can't authenticate in 'home' mode, raise
        an error while also giving a simple mitigation technique of connecting
        to your portal in the standard GIS() way.
        """
        mitigation_msg = (
            "You can still connect to your portal by creating "
            "a GIS() object with the standard user/password, cert_file, etc. "
            "See https://bit.ly/2DT1156 for more information."
        )
        _log.warning(
            'Authenticating in GIS("home") mode failed.' "{}".format(mitigation_msg)
        )
        raise RuntimeError("{}\n-----\n{}".format(err_msg, mitigation_msg))

    def _uri_validator(self, x):
        from urllib.parse import urlparse

        if x is None:
            return False
        try:
            result = urlparse(x)
            return result.scheme != "" and result.netloc != ""
        except:
            return False

    @_lazy_property
    def users(self):
        """
        The ``users`` property is the resource manager for GIS users. See :class:`~arcgis.gis.UserManager` for more
        information.
        """
        return UserManager(self)

    @_lazy_property
    def groups(self):
        """
        The ``groups`` property is resource manager for GIS groups. See :class:`~arcgis.gis.GroupManager` for more
        information.
        """
        return GroupManager(self)

    @_lazy_property
    def content(self):
        """
        The ``content`` property is the resource manager for GIS content. See :class:`~arcgis.gis.ContentManager` for
        more information.
        """
        return ContentManager(self)

    @_lazy_property
    def velocity(self):
        """
        The resource manager for ArcGIS Velocity. See :class:`~arcgis.realtime.velocity.Velocity`
        :return: :class:`~arcgis.realtime.velocity.Velocity`
        """
        if self._portal.is_arcgisonline and self._subscription_information is not None:
            _velocity_url = None
            org_capabilities = self._subscription_information["orgCapabilities"]
            for capabilities in org_capabilities:
                if capabilities["id"] == "velocity":
                    _velocity_url = capabilities["velocityUrl"]
                    if "/iot" not in _velocity_url:
                        _velocity_url += "/iot/"

            if _velocity_url is not None:
                velocity = arcgis.realtime.velocity.Velocity(
                    url=_velocity_url, gis=self
                )
                return velocity
            else:
                raise Exception("Velocity is not available on this organizaiton.")
        else:
            raise Exception("ArcGIS Enterprise does not support Velocity")

    @_lazy_property
    def hub(self):
        """
        The ``hub`` property is the resource manager for GIS hub. See :class:`~arcgis.apps.hub.Hub` for more information.
        """
        if self._portal.is_arcgisonline:
            return arcgis.apps.hub.Hub(self)
        else:
            raise Exception("Hub is currently only compatible with ArcGIS Online.")

    @_lazy_property
    def sites(self):
        """
        The ``sites`` property is the resource manager for Enterprise Sites. See :class:`~arcgis.apps.sites` for more information.
        """
        if not self._portal.is_arcgisonline:
            return arcgis.apps.hub.SiteManager(self)
        else:
            raise Exception("Please access your ArcGIS Online sites through your Hub.")

    @_lazy_property
    def pages(self):
        """
        The ``pages`` property is the resource manager for a Page of an Enterprise Site. See :class:`~arcgis.apps.hub.pages` for more information.
        """
        if not self._portal.is_arcgisonline:
            return arcgis.apps.hub.PageManager(self)
        else:
            raise Exception("Please access your ArcGIS Online pages through your Hub.")

    @_lazy_property
    def notebook_server(
        self,
    ) -> list["NotebookServer"] | list["AGOLNotebookManager"]:
        """
        The ``notebook_server`` property provides access to the :class:`~arcgis.gis.nb.NotebookServer` registered
        with the organization or enterprise.
        :return: `List <https://docs.python.org/3/library/stdtypes.html#lists>`_ [`NotebookServer`]
        """
        if self._portal.is_arcgisonline:
            urls = self._registered_servers()
            url = urls.get("urls", {}).get("notebooks", {}).get("https", None)
            if url:
                from arcgis.gis.agonb import AGOLNotebookManager

                url = f"https://{url[0]}/admin"
                return [AGOLNotebookManager(url=url, gis=self)]
        else:
            try:
                from arcgis.gis.nb import NotebookServer

                res = self._portal.con.post("portals/self/servers", {"f": "json"})

                return [
                    NotebookServer(server["adminUrl"] + "/admin", self)
                    for server in res["servers"]
                    if server["serverFunction"].lower() == "notebookserver"
                ]
            except:
                return []
        return []

    @property
    def symbol_service(self) -> arcgis.mapping._types.SymbolService | None:
        """
        Symbol service is an ArcGIS Server utility service that provides access
        to operations to build and generate images for Esri symbols to be
        consumed by internal and external web applications.

        :return: A :class:`~arcgis.mapping._types.SymbolService` object or None

        """
        try:
            return self._tools.symbol_service
        except:
            return None

    @property
    def datastore(self):
        """
        The ``datastore`` property returns the manager for `user-managed data store
        items <https://enterprise.arcgis.com/en/portal/10.7/use/data-store-items.htm>`_.

        .. note::
            This is only available with ArcGIS Enterprise 10.7+.
            See :class:`~arcgis.gis._impl._datastores.PortalDataStore` for
            more information.

        :return: A :class:`~arcgis.gis._impl._datastores.PortalDataStore` object
        """
        if self.version >= [7, 1] and not self._portal.is_arcgisonline:
            from arcgis.gis._impl._datastores import PortalDataStore

            url = self._portal.resturl + "portals/self/datastores"
            self._pds = PortalDataStore(url=url, gis=self)
        return self._pds

    @_lazy_property
    def _datastores(self):
        """
        The list of datastores resource managers for sites federated with the GIS.
        """
        if self._datastores_list is not None:
            return self._datastores_list

        self._datastores_list = []
        try:
            res = self._portal.con.post("portals/self/servers", {"f": "json"})

            servers = res["servers"]
            admin_url = None
            for server in servers:
                admin_url = server["adminUrl"] + "/admin"
                self._datastores_list.append(DatastoreManager(self, admin_url, server))
        except:
            pass
        return self._datastores_list

    @property
    def properties(self):
        """
        ``properties`` manages the actual properties of the GIS object.
        """
        if self._properties is None:
            self._properties = _mixins.PropertyMap(self._get_properties(force=True))
        return self._properties

    def update_properties(self, properties_dict: dict[str, Any]):
        """The ``update_properties`` method updates the GIS's properties from those in ``properties_dict``. This method
        can be useful for updating the utility services used by the GIS.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        properties_dict     Required dictionary. A dictionary of just those properties and
                            values that are to be updated.
        ===============     ====================================================================

        :return:
           A boolean indicating success (True), or failure (False)


        .. note::
            For examples of the property names and key/values to use when updating utility services,
            refer to the `Common parameters
            <https://developers.arcgis.com/rest/users-groups-and-items/common-parameters.htm>`_
            page in the ArcGIS REST API.

        .. code-block:: python

            # Usage Example: Update the geocode service

            gis = GIS(profile='xyz')
            upd = {'geocodeService': [{
              "singleLineFieldName": "Single Line Input",
              "name": "AtlantaLocator",
              "url": "https://some.server.com/server/rest/services/GeoAnalytics/AtlantaLocator/GeocodeServer",
              "itemId": "abc6e1fc691542938917893c8944606d",
              "placeholder": "",
              "placefinding": "true",
              "batch": "true",
              "zoomScale": 10000}]}

            gis.update_properties(upd)

        """
        postdata = {
            "f": "json",
        }
        postdata.update(properties_dict)
        url: str = self._portal.resturl + "portals/self/update"

        resp = self._con.post(url, postdata)
        if resp:
            self._properties = _mixins.PropertyMap(
                self._portal.get_properties(force=True)
            )
            # delattr(self, '_lazy_properties') # force refresh of properties when queried next
            return resp.get("success")

    @property
    def url(self):
        """The ``url`` property is a read-only URL of your GIS connection."""
        if self._is_hosted_nb_home:
            return self._public_portal_url
        else:
            return self._url

    @property
    def _public_rest_url(self):
        return self.url + "/sharing/rest/"

    # ----------------------------------------------------------------------
    @property
    def _subscription_information(self):
        """
        Returns the ArcGIS Online Subscription Information for a Site.

        :return: dictionary
        """
        if self.version > [6, 4] and self._portal.is_arcgisonline:
            url = "%sportals/self/subscriptionInfo" % self._portal.resturl
            params = {"f": "json"}
            return self._con.get(url, params)
        return None

    # ----------------------------------------------------------------------
    @property
    def subscription_information(self) -> dict:
        """
        Returns the ArcGIS Online Subscription Information for a Site.

        :return: dictionary
        """
        return self._subscription_information

    # ----------------------------------------------------------------------
    @property
    def version(self):
        """The ``version`` property returns the GIS version number"""
        self._is_agol = self._portal.is_arcgisonline
        self._product_version = [int(i) for i in self._portal.get_version().split(".")]
        return self._product_version

    # ----------------------------------------------------------------------
    def _registered_servers(self):
        """returns servers registered with enterprise/portal"""
        params = {"f": "json"}
        if self._portal.is_arcgisonline == False:
            url = f"{self._portal.resturl}portals/self/servers"
        else:
            url = f"{self._portal.resturl}portals/self/urls"
        return self._con.get(url, params=params)

    @property
    def hosting_servers(self) -> list:
        """
        Returns the hosting servers for the GIS

        :returns: list
        """
        if self._portal.is_arcgisonline:
            info = self._registered_servers()
            tile_urls = set(info["urls"].get("tiles", {}).get("https", []))
            feature_urls = set(info["urls"].get("features", {}).get("https", []))
            tile_urls = set(info["urls"].get("tiles", {}).get("https", []))
            tile_urls = [url for url in tile_urls if url not in feature_urls]
            pid = self.properties.id
            feature_urls = [
                _agoserver.AGOLServicesDirectory(
                    f"https://{url}/{pid}/arcgis/rest/services", gis=self
                )
                for url in feature_urls
            ]
            tile_urls = [
                _agoserver.AGOLServicesDirectory(
                    f"https://{url}/tiles/{pid}/arcgis/rest/services",
                    gis=self,
                )
                for url in tile_urls
            ]
            return feature_urls + tile_urls
        else:
            from arcgis.gis.server import ServicesDirectory

            info = self._registered_servers()
            servers = [
                ServicesDirectory(server["url"], portal_connection=self._con, gis=self)
                for server in info["servers"]
                if server.get("serverRole", None) == "HOSTING_SERVER"
            ]
            return servers
        return []

    # ----------------------------------------------------------------------
    @property
    def servers(self) -> dict:
        """
        Returns the servers registered with ArcGIS Entperise.  For ArcGIS
        Online, the return value is `None`.

        :return: dict
        """
        if self._portal.is_arcgisonline:
            info = self._registered_servers()
            return info
        elif self._portal.is_kubernetes or self._portal.is_arcgisonline == False:
            url = self._portal.resturl + f"portals/{self.properties['id']}/servers"
            params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def org_settings(self):
        """
        The portal settings resource is used to return a view of the
        portal's configuration as seen by the current users, either
        anonymous or logged in. Information returned by this resource
        includes helper services, allowed redirect URIs, and the current
        configuration for any access notices or information banners.

        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        settings                   Required Dict.  A dictionary of the settings

                                    ==========================    =============================================
                                    **Fields**                    **Description**
                                    --------------------------    ---------------------------------------------
                                    anonymousAccessNotice         Dict. A JSON object representing a notice that is shown to your organization's anonymous users.
                                                                  Ex: {'title': 'Anonymous Access Notice Title', 'text': 'Anonymous Access Notice Text', 'buttons': 'acceptAndDecline', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    authenticatedAccessNotice     Dict. A JSON object representing a notice that is shown to your organization's authenticated users.
                                                                  Ex: {'title': 'Authenticated Access Notice Title', 'text': 'Authenticated Access Notice Text', 'buttons': 'okOnly', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    informationalBanner           Dict. A JSON object representing the informational banner that is shown at the top of your organization's page.
                                                                  Ex: {'text': 'Header Text', 'bgColor': 'grey', 'fontColor': 'blue', 'enabled': True}
                                    --------------------------    ---------------------------------------------
                                    clearEmptyFields              Bool.  If True, any empty dictionary will be set to null.
                                    ==========================    =============================================

        ======================     ===============================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if self.version >= [7, 4]:
            url = "portals/self/settings"
            params = {"f": "json"}
            return self._con.post(url, params)
        return

    # ----------------------------------------------------------------------
    @org_settings.setter
    def org_settings(self, settings):
        """
        See main ``org_settings`` property docstring
        """
        if self.version >= [7, 4] and isinstance(settings, dict):
            url = "portals/self/settings/update"
            params = {"f": "json"}
            params.update(settings)
            self._con.post(url, params)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "GIS @ {url} version:{version}".format(
            url=self.url,
            version=".".join([str(i) for i in self._product_version]),
        )

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _repr_html_(self):
        """
        HTML Representation for IPython Notebook
        """
        return 'GIS @ <a href="' + self.url + '">' + self.url + "</a>"

    # ----------------------------------------------------------------------
    def _get_properties(self, force=False):
        """Returns the portal properties (using cache unless force=True)."""
        return self._portal.get_properties(force)

    def map(
        self,
        location: Optional[str] = None,
        zoomlevel: Optional[int] = None,
        mode: str = "2D",
        geocoder=None,
    ):
        """
        The ``map`` method creates a map widget centered at the declared location with the specified
        zoom level. If an address is provided, it is geocoded
        using the GIS's configured geocoders. Provided a match is found, the geographic
        extent of the matched address is used as the extent of the map. If a zoomlevel is also
        provided, the map is centered at the matched address instead and the map is zoomed
        to the specified zoomlevel. See :class:`~arcgis.widgets.MapView` for more information.

        .. note::
            The map widget is only supported within a Jupyter Notebook. IE11 is no longer supported.
            Please use the latest version of Google Chrome, Mozilla Firefox, Apple Safari, or Microsoft Edge.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Optional string. The address or lat-long tuple of where the map is to be centered.
        ------------------     --------------------------------------------------------------------
        zoomlevel              Optional integer. The desired zoom level.
        ------------------     --------------------------------------------------------------------
        mode                   Optional string of either '2D' or '3D' to specify map mode. Defaults to '2D'.
        ------------------     --------------------------------------------------------------------
        geocoder               Optional Geocoder. Allows users to specify a geocoder to find a given location.
                               See the `What is geocoding?
                               <https://developers.arcgis.com/python/guide/part1-what-is-geocoding/>`_
                               guide for more information.
        ==================     ====================================================================


        .. note::
            If the Jupyter Notebook server is running over http, you need to
            configure your ArcGIS Enterprise portal or ArcGIS Online organization to allow your host and port; or else
            you will run into CORS issues when displaying this map widget.

            This can be accomplished by signing into your ArcGIS Enterprise portal or ArcGIS Online organization in a
            browser, then navigating to:

            `Organization` > `Settings` > `Security` > `Allow origins` > `Add` > `http://localhost:8888`
            (replace with the host/port you are running on)

        .. code-block:: python

            # Usage Example

            >>> gis = GIS(url="http://pythonplayground.esri.com/portal", username="user1", password="password1")

            >>> gis.map("Durham,NC")

        :return:
          A :class:`map widget <arcgis.widgets.MapView>` (the widget is displayed in Jupyter Notebook when queried).
        """
        try:
            from arcgis.widgets import MapView
            from arcgis.geocoding import get_geocoders, geocode, Geocoder
        except Error as err:
            _log.error("ipywidgets packages is required for the map widget.")
            _log.error("Please install it:\n\tconda install ipywidgets")

        if isinstance(location, Item) and location.type == "Web Map":
            mapwidget = MapView(gis=self, item=location, mode=mode)
        else:
            mapwidget = MapView(gis=self, mode=mode)

            # Geocode the location
            if isinstance(location, str):
                if geocoder and isinstance(geocoder, Geocoder):
                    locations = geocode(
                        location,
                        out_sr=4326,
                        max_locations=1,
                        geocoder=geocoder,
                    )
                    if len(locations) > 0:
                        if zoomlevel is not None:
                            loc = locations[0]["location"]
                            mapwidget.center = loc["y"], loc["x"]
                            mapwidget.zoom = zoomlevel
                        else:
                            mapwidget.extent = locations[0]["extent"]
                else:
                    for geocoder in get_geocoders(self):
                        locations = geocode(
                            location,
                            out_sr=4326,
                            max_locations=1,
                            geocoder=geocoder,
                        )
                        if len(locations) > 0:
                            if zoomlevel is not None:
                                loc = locations[0]["location"]
                                mapwidget.center = loc["y"], loc["x"]
                                mapwidget.zoom = zoomlevel
                            else:
                                if "extent" in locations[0]:
                                    mapwidget.extent = locations[0]["extent"]
                            break

            # Center the map at the location
            elif isinstance(location, (tuple, list)):
                if all(isinstance(el, list) for el in location):
                    extent = {
                        "xmin": location[0][0],
                        "ymin": location[0][1],
                        "xmax": location[1][0],
                        "ymax": location[1][1],
                    }
                    mapwidget.extent = extent
                else:
                    mapwidget.center = location

            elif isinstance(location, dict):  # geocode result
                if "extent" in location and zoomlevel is None:
                    mapwidget.extent = location["extent"]
                elif "location" in location:
                    mapwidget.center = (
                        location["location"]["y"],
                        location["location"]["x"],
                    )
                    if zoomlevel is not None:
                        mapwidget.zoom = zoomlevel

            elif location is not None:
                print(
                    "location must be an address(string) or (lat, long) pair as a tuple"
                )

        if zoomlevel is not None:
            mapwidget.zoom = zoomlevel

        if not location:
            # Set up default extent
            if "defaultExtent" in self.org_settings:
                mapwidget.extent = self.org_settings["defaultExtent"]

        return mapwidget


###########################################################################


class Datastore(dict):
    """
    The ``Datastore`` class represents a data store, either a folder, database
    or bigdata fileshare on a :class:`~arcgis.gis.server.Server` within
    the Enterprise. See :class:`~arcgis.gis.server.Datastore` for more
    information on data stores on a server.
    """

    def __init__(self, datastore, path):
        dict.__init__(self)
        self._datastore = datastore
        self._portal = datastore._portal
        self._admin_url = datastore._admin_url

        self.datapath = path

        params = {"f": "json"}
        path = self._admin_url + "/data/items" + self.datapath

        datadict = self._portal.con.post(path, params, verify_cert=False)

        if datadict:
            self.__dict__.update(datadict)
            super(Datastore, self).update(datadict)

    def __getattr__(
        self, name
    ):  # support group attributes as group.access, group.owner, group.phone etc
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def __getitem__(
        self, k
    ):  # support group attributes as dictionary keys on this object, eg. group['owner']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            params = {"f": "json"}
            path = self._admin_url + "/data/items" + self.datapath

            datadict = self._portal.con.post(path, params, verify_cert=False)
            super(Datastore, self).update(datadict)
            self.__dict__.update(datadict)
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    def __repr__(self):
        return '<%s title:"%s" type:"%s">' % (
            type(self).__name__,
            self.path,
            self.type,
        )

    @property
    def manifest(self):
        """
        The ``manifest`` property retrieves or sets the manifest resource for bigdata fileshares, as a dictionary.
        """
        data_item_manifest_url = (
            self._admin_url + "/data/items" + self.datapath + "/manifest"
        )

        params = {
            "f": "json",
        }
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)
        return res

    @manifest.setter
    def manifest(self, value):
        """
        The ``manifest`` property updates the manifest resource for bigdata file shares.
        """
        manifest_upload_url = (
            self._admin_url + "/data/items" + self.datapath + "/manifest/update"
        )

        with _tempinput(json.dumps(value)) as tempfilename:
            # Build the files list (tuples)
            files = []
            files.append(("manifest", tempfilename, os.path.basename(tempfilename)))

            postdata = {"f": "pjson"}

            resp = self._portal.con.post(
                manifest_upload_url, postdata, files, verify_cert=False
            )

            if resp["status"] == "success":
                return True
            else:
                print(str(resp))
                return False

    @property
    def ref_count(self):
        """
        The ``ref_count`` property gets the total number of references to this data item that exists on the server.
        This property can be used to determine if this data item can be safely deleted or taken down for maintenance.
        """
        data_item_manifest_url = self._admin_url + "/data/computeTotalRefCount"

        params = {"f": "json", "itemPath": self.datapath}
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)
        return res["totalRefCount"]

    def delete(self):
        """
        The ``delete`` method unregisters this data item from the datastore.

        .. code-block:: python

            # Usage Example

            >>> datastore.delete()

        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json", "itempath": self.datapath, "force": True}
        path = self._admin_url + "/data/unregisterItem"

        resp = self._portal.con.post(path, params, verify_cert=False)
        if resp:
            return resp.get("success")
        else:
            return False

    def update(self, item: dict[str, Any]):
        """
        The ``update`` method edits this data item to update its connection information.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required dictionary. The representation of the updated item.
        ===============     ====================================================================


        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json", "item": item}
        path = self._admin_url + "/data/items" + self.datapath + "/edit"

        resp = self._portal.con.post(path, params, verify_cert=False)
        if resp["status"] == "success":
            return True
        else:
            return False

    # ----------------------------------------------------------------------
    def regenerate(self):
        """
        The ``regenerate`` method is used to regenerate the manifest for a big data file share. You can
        regenerate a manifest if you have added new data or if you have
        uploaded a hints file using the edit resource.

        :return:
            A boolean indicating success (True), or failure (False)
        """
        url = self._admin_url + "/data/items" + self.datapath + "/manifest/regenerate"
        params = {"f": "json"}
        res = self._portal.con.post(url, params)
        if isinstance(res, dict):
            if "success" in res:
                return res["success"]
            if "status" in res:
                return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def validate(self):
        """
        The ``validate`` method is used to validate that this data item's path (for file shares) or
        connection string (for databases) is accessible to every server node in the site.

        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json"}
        path = self._admin_url + "/data/items" + self.datapath

        datadict = self._portal.con.post(path, params, verify_cert=False)

        params = {"f": "json", "item": datadict}
        path = self._admin_url + "/data/validateDataItem"

        res = self._portal.con.post(path, params, verify_cert=False)
        if isinstance(res, dict):
            if "success" in res:
                return res["success"]
            if "status" in res:
                return res["status"] == "success"
        return res

    @property
    def datasets(self):
        """
        The ``datasets`` property retrieves the datasets in the data store, returning them as a dictionary
        (currently implemented for big data file shares).

        :return:
            A dictionary
        """
        data_item_manifest_url = (
            self._admin_url + "/data/items" + self.datapath + "/manifest"
        )

        params = {
            "f": "json",
        }
        res = self._portal.con.post(data_item_manifest_url, params, verify_cert=False)

        return res["datasets"]


###########################################################################
class GroupMigrationManager(object):
    """
    The ``GroupMigrationManager`` class provides methods to export all or a
    subset of all supported :class:`items <arcgis.gis.Item>` from an
    ArcGIS Enterprise ``group`` into an export package (see
    `Export Package <https://developers.arcgis.com/rest/users-groups-and-items/items-and-item-types.htm#GUID-57FD13A2-1A35-4F4D-B895-08CB48432E0D>`_)
    that can subsequently be added to another ArcGIS Enterprise and then
    loaded into a :class:`~arcgis.gis.Group`.

    This class is not meant to be initialized directly, but instead an
    object of this class is accessed through the :attr:`~arcgis.gis.Group.migration`
    property on a :class:`~arcgis.gis.Group` object initialized from
    an ArcGIS Enterprise Group

    .. code-block:: python

        # Usage Example: Initializing a ``GroupMigrationManager`` object:

        >>> gis = GIS(profile="your_enterprise_admin_profile")

        >>> ent_grp = gis.groups.search("<group query>")[0]

        >>> grp_mig_mgr = ent_grp.migration
    """

    _con = None
    _gis = None
    _group = None

    def __init__(self, group):
        """initializer"""
        assert isinstance(group, Group)
        self._group = group
        self._gis = group._gis
        self._con = group._gis._con

    # ----------------------------------------------------------------------
    def _from_package(
        self,
        item,
        item_id_list=None,
        preview_only=False,
        run_async=False,
        overwrite=False,
        folder_id=None,
        folder_owner=None,
    ):
        """
        Imports an EPK Item to a Group.  This will import items associated with this group.
        :return: Boolean
        """
        if self._gis.users.me.role == "org_admin":
            try_json = True
            if preview_only:
                try_json = False
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/import"
            if isinstance(item, Item):
                item = item.itemid
            params = {
                "f": "json",
                "itemId": item,
                "itemIdList": "",
                "folderId": "",
                "folderOwnerUsername": "",
                "token": self._con.token,
            }
            if item_id_list:
                params["itemIdList"] = item_id_list
            if overwrite is not None:
                params["overwriteExistingItems"] = overwrite
            if preview_only:
                params["previewOnly"] = preview_only
            if run_async:
                params["async"] = run_async
            if folder_id and self._gis.version >= [8, 4]:
                params["folderId"] = folder_id
            if folder_owner and self._gis.version >= [8, 4]:
                params["folderOwnerUsername"] = folder_owner
            return self._con.post(url, params, try_json=try_json)

        else:
            raise Exception("Must be an administror to perform this action")
        pass

    # ----------------------------------------------------------------------
    def _status(self, job_id, key=None):
        """
        Checks the status of an export job
        """
        import time

        params = {}
        if job_id:
            url = f"{self._gis._portal.resturl}portals/self/jobs/{job_id}"
            params["f"] = "json"
            if key:
                params["key"] = key
            res = self._con.post(url, params)
            while res["status"] not in ["completed", "complete"]:
                res = self._con.post(url, params)
                if res["status"] == "failed":
                    raise Exception(res)
                time.sleep(2)
            return res
        else:
            raise Exception(res)

    # ----------------------------------------------------------------------
    def create(self, items: Optional[list[Item]] = None, future: bool = True):
        """
        The ``create`` method exports supported :class:`~arcgis.gis.Group` content to
        an *Export Package* :class:`~arcgis.gis.Item` (*EPK item*). *EPK Items* can be used to
        migrate content from one ArcGIS Enterprise deployment to another. Once an
        `EPK Item` is created, you can download it, :meth:`~arcgis.gis.ContentManager.add`
        it to a receiving ArcGIS Enterprise and then :meth:`~arcgis.gis.GroupMigrationManager.load`
        it into a :class:`~arcgis.gis.Group` in that Enterprise deployment. The method will
        handle updating service URLs and item IDs used in any web maps,
        web-mapping applications, and/or associated web layers in those items during
        the *load* operation. See full datails in the
        `Export Group Content <https://developers.arcgis.com/rest/users-groups-and-items/export-group-content.htm>`_ documentation.

        .. note::
            There are some limits to this functionality. Packages should be under 10 GB in size
            and only `supported items <https://developers.arcgis.com/rest/users-groups-and-items/export-group-content.htm#ESRI_SECTION2_0C00F2CEA194453D8E91F9E1138CE7E1>`_
            can be exported. You also must have **administrative** privileges to run this
            operation.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        items                  Optional List<:class:`~arcgis.gis.Item`>. A set of items to export
                               from the group.  If argument is not provided, the method will attempt
                               to export all group content items.
        ------------------     --------------------------------------------------------------------
        future                 Optional Boolean.  When `True`, the operation runs asynchronously
                               and returns a :class:`Job <arcgis.gis._impl._jb.StatusJob>` object
                               that can be queried for results. When `False` the operation
                               runs synchronously and returns an *export package*
                               :class:`~arcgis.gis.Item` upon completion.
        ==================     ====================================================================

        :return:
            An *export package* :class:`~arcgis.gis.Item` when `future=False`, or a
            :class:`Job <arcgis.gis._impl._jb.StatusJob>` when `future=True`

        .. code-block:: python

            # Usage Example: Asynchronous execution

            >>> import time
            >>> from arcgis.gis import GIS
            >>>
            >>> gis = GIS(profile="your_enterprise_admin_profile")

            >>> grp = gis.groups.get("<group_id>")
            >>> grp_mig_mgr = grp.migration

            >>> mig_job = grp_mig_mgr.create()

            >>> while mig_job.status != "completed":
            >>>     job_status = mig_job.status
            >>>     if job_status == "failed":
            >>>         break
            >>>     else:
            >>>         print(job_status)
            >>>         time.sleep(3)
            >>> print(f"Job Status: {mig_job.status}")

            processing
            processing
            Job Status: completed

            >>> epk_item = mig_job.result()
        """
        if self._gis.users.me.role == "org_admin":
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/export"
            if items and isinstance(items, (list, tuple)):
                items = ",".join([i.id if isinstance(i, Item) else i for i in items])
            else:
                items = None
            params = {"itemIdList": items}

            params["async"] = json.dumps(True)
            res = self._gis._con.post(url, params)
            if not "jobId" in res:
                raise Exception(
                    f"Either group has no items, or items failed or were skipped: {res}"
                )
            executor = concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._status, **{"job_id": res["jobId"], "key": res["key"]}
            )
            executor.shutdown(False)
            job = _jb.StatusJob(
                future=futureobj,
                op="Export Group Content",
                jobid=res["jobId"],
                gis=self._gis,
                notify=arcgis_env.verbose,
                key=res.get("key", None),
            )
            if future:
                return job
            else:
                return job.result()
        else:
            raise Exception("Must be an administrator to perform this action")

    # ----------------------------------------------------------------------
    def load(
        self,
        epk_item: Item,
        item_ids: Optional[list[str]] = None,
        overwrite: bool = True,
        future: bool = True,
        folder_id: Optional[str] = None,
        folder_owner: Optional[str] = None,
    ):
        """
        The ``load`` method imports the contents of an *export package*
        :class:`~arcgis.gis.Item` into a :class:`~arcgis.gis.Group`.

        See the `Import Group Content <https://developers.arcgis.com/rest/users-groups-and-items/import-group.htm>`_
        documenation for full system details.

        .. note::
            Administrative privileges are required to run this operation.
            Once imported, items will be owned by the importer, and will have
            to be manually reassigned to the proper owner if needed.

        .. warning::
            The receiving ArcGIS Enterprise deployment must be using the same version or later
            of the ArcGIS Enterprise that generated the export package.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        epk_item          Required *export package* :class:`~arcgis.gis.Item`.
        ----------------  -------------------------------------------------------------------------------
        item_ids          Optional list. A list of item IDs to import from the *export package*. If this
                          argument is not provided, the operation will import all supported *items*
                          in the export package to the receiving :class:`~arcgis.gis.Group`.
        ----------------  -------------------------------------------------------------------------------
        overwrite         Optional bool. If *True*, any *items* that already exist in the target
                          organization will be overwritten by the corresponding *item* in the package
                          provided by the `epk_item` argument.
        ----------------  -------------------------------------------------------------------------------
        future            Optional bool. When *True*, the operation will return a
                          :class:`Job <arcgis.gis._impl._jb.StatusJob>` object which can be queried, and
                          the process will not pause so subsequent operations can continue to run.  When
                          `False`, the operation runs synchronously, pausing the process until the
                          job completes and returns a dictionary containing output information.
                          If you are loading large amounts of data, set *future=True* to reduce down time.
                          The *job* can be queried through its :attr:`~arcgis.gis._impl._jb.StatusJob.status`
                          attribute. In addition, the :attr:`~arcgis.gis._impl._jb.StatusJob.messages` and
                          :meth:`~arcgis.gis._impl._jb.StatusJob.result()` attributes will contain
                          information about the output.
        ----------------  -------------------------------------------------------------------------------
        folder_id         Optional String. In ArcGIS Enterprise 10.9 and later, the folder id of
                          the destination Enterprise for the package contents.
        ----------------  -------------------------------------------------------------------------------
        folder_owner      Optional String. In ArcGIS Enterprise 10.9 and later, a *username* for the
                          folder owner.
        ================  ===============================================================================

        :return:
            A dictionary when *future=False* or a :class:`Job <arcgis.gis._impl._jb.StatusJob>` when `future=True`.

        .. code-block:: python

            # Usage Example: Loading package results into a group

            >>> source = GIS(profile="source_enterprise_admin_profile")
            >>> target = GIS(profile="target_enterprise_admin_profile")

            >>> source_grp = source.groups.get("<group_id>")
            >>> source_epk_item = source_grp.migration.create(future=False)

            >>> download_path = source_epk_item.download(save_path="path_on_system",
                                                         file_name="file_name.epk")

            >>> target_epk_item = target.content.add(item_properties={"title": "Group data export item",
                                                                      "tags": "group_content_migration",
                                                                      "snippet": "Sample of loading package.",
                                                                      "type": "Export Package:},
                                                     date=download_path)

            >>> target_grp_mig = target.groups.get("<target_group_id>").migration
            >>> grp_import_job = target_grp_mig.load(epk_item=target_epk_item)

            >>> grp_import_job.messages

            ["Starting import of items from EPK item '<item_id>' to group 'Group Title'.",
             "Starting the import of exported package item '<item_id>' containing 2 items.",
             "Import option to overwrite items if they exist is set to 'true'."]

             >>> grp_import_job.result()

            {'itemsImported': [<Item title:"Item1 title" type:<item1 type> owner:<item_owner>>,
             <Item title:"Item2 title" type:<item2 type> owner:<item_owner>>],
             'itemsSkipped': [],
             'itemsFailedImport': []}

        """

        assert isinstance(epk_item, Item)
        if isinstance(item_ids, list):
            item_ids = ",".join([i.id if isinstance(i, Item) else i for i in item_ids])
        if isinstance(epk_item, Item) and epk_item.type == "Export Package":
            res = self._from_package(
                item=epk_item,
                item_id_list=item_ids,
                preview_only=False,
                run_async=True,
                overwrite=overwrite,
                folder_id=folder_id,
                folder_owner=folder_owner,
            )
            executor = concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._status, **{"job_id": res["jobId"], "key": res["key"]}
            )
            executor.shutdown(False)
            job = arcgis.gis._impl._jb.StatusJob(
                future=futureobj,
                op="Export Group Content",
                jobid=res["jobId"],
                gis=self._gis,
                notify=arcgis_env.verbose,
                key=res.get("key", None),
            )
            if future:
                return job
            else:
                return job.result()
        else:
            raise Exception(f"Invalid Item {epk_item.type}")
        return None

    # ----------------------------------------------------------------------
    def inspect(self, epk_item: Item) -> dict:
        """
        The ``inspect`` method retrieves the contents of an *export package*
        :class:`~arcgis.gis.Item` resulting from the
        :meth:`~arcgis.gis.GroupMigrationManager.create` operation. It outputs
        a report on the contents of the package allowing administrators to
        see the contents of the package.

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        epk_item          Required *export package* :class:`~arcgis.gis.Item`.
        ================  ===============================================================================

        :return:
            A dictionary containing the contents of the EPK Package

        .. code-block:: python

            # Usage Example: Inspecting an export package

            >>> grp = gis.groups.get("<group_id>")

            >>> export_epk_item = grp.migration.create(future=False)
            >>> grp.migration.inspect(epk_item = export_epk_item)

            {'packageSummary': {'id': '68472b95bdfd4efa86531fd202151eda',
            'fileName': 'Group_Data_2023714_032552',
            'packageVersion': '1.0',
            'packageCreated': 1690496752834,
            'sourcePortalInfo': {'httpsUrl': 'https://myserver.company.com/web_adaptor',
             'httpUrl': 'http://myserver.company.com/web_adaptor',
             'version': '11.1.0',
             'portalId': 'e35a6e01-0902-4c77-ef9d-1a84816f530a',
             'isPortal': True}},
           'total': 2,
           'start': 1,
           'num': 2,
           'nextStart': -1,
           'results': [{'id': '4a0dfbfa0eeb415195426eee9131edfa',
             'type': 'Feature Service',
             'title': 'FService Item Name',
             'size': 4204037,
             'exists': True,
             'canImport': True,
             'created': 1689255086965,
             'modified': 1689255155534},
            {'id': 'c9d0f2b3fbf44be8a531c9a47ff160b0',
             'type': 'Feature Service',
             'title': 'FService2 Item name',
             'size': 1985399,
             'exists': True,
             'canImport': True,
             'created': 1689257010004,
             'modified': 1689253036102}]}

        """
        if isinstance(epk_item, Item) and epk_item.type == "Export Package":
            try:
                import time

                self._from_package(epk_item.itemid, preview_only=True, run_async=False)
                time.sleep(2)
            except:
                pass
            url = f"{self._gis._portal.resturl}community/groups/{self._group.groupid}/importPreview/{epk_item.itemid}"
            params = {"f": "json", "start": 1, "num": 25}
            res = self._con.post(url, params)
            results = res["results"]
            while res["nextStart"] > 0:
                params["start"] = res["nextStart"]
                res = self._con.post(url, params)
                results.extend(res["results"])
                if res["nextStart"] == -1:
                    break
            res["results"] = results
            return res

        else:
            raise Exception("Invalid Item Type.")
        return None


###########################################################################
class DatastoreManager(object):
    """
    The ``DatastoreManager`` class is a helper class for managing the data
    store for servers configured within the Enterprise. Depending upon the `server role <https://enterprise.arcgis.com/en/get-started/latest/windows/additional-server-deployment.htm>`_
    an instance of this class can be obtained from helper functions.

    .. note::
        This class is not created directly, but rather the following server roles have
        :class:`datastores <arcgis.gis.Datastore>`, and an instance of the
        :class:`~arcgis.gis.DatastoreManager` for each server is returned by
        the respective `get_datastores()` function:
          * GeoAnalytics Server: :meth:`~arcgis.geoanalytics.get_datastores`
          * Raster Analytics Server: :meth:`~arcgis.raster.analytics.get_datastores`
    """

    def __init__(self, gis, admin_url, server):
        self._gis = gis
        self._portal = gis._portal
        self._admin_url = admin_url
        self._server = server

    def __str__(self):
        return "< %s for %s >" % (type(self).__name__, self._admin_url)

    def __repr__(self):
        return "< %s for %s >" % (type(self).__name__, self._admin_url)

    @property
    def config(self):
        """
        The ``config`` method retrieves and sets the data store configuration properties, which affect the behavior of
        the data holdings of the server. The properties include ``blockDataCopy``. When this property is ``False``, or not
        set at all, copying data to the site when publishing services from a client application is allowed. This is the
        default behavior. When this property is ``True``, the client application is not allowed to copy data to the site
        when publishing. Rather, the publisher is required to register data items through which the service being
        published can reference data.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Required bool.
                                Values: True | False
                                .. note::
                                    If you specify the property as ``True``, users will not be able
                                    to publish ``geoprocessing services`` and ``geocode services``
                                    from composite locators. These service types require data to be
                                    copied to the server.
                                    As a workaround, you can temporarily set the property to ``False``,
                                    publish the service, and then set the property back to ``True``.
        ==================      ====================================================================

        :return: A Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        path = self._admin_url + "/data/config"
        res = self._portal.con.post(path, params, verify_cert=False)
        return res

    @config.setter
    def config(self, value):
        """
        See main ``config`` property docstring
        """
        params = {"f": "json"}
        params["datastoreConfig"] = value
        path = self._admin_url + "/data/config/update"
        res = self._portal.con.post(path, params)
        return res

    def add_folder(
        self, name: str, server_path: str, client_path: Optional[str] = None
    ):
        """
        The ``add_folder`` method registers a folder with the :class:`~arcgis.gis.Datastore`.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique fileshare name on the server.
        ---------------     --------------------------------------------------------------------
        server_path         Required string. The path to the folder from the server (and client, if shared path).
        ---------------     --------------------------------------------------------------------
        client_path         Optional string. If folder is replicated, the path to the folder from the client.
        ===============     ====================================================================

        :return:
           The folder if registered successfully, None otherwise.

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_folder("fileshare name", "folder_path", "clienth_path")
        """
        conn_type = "shared"
        if client_path is not None:
            conn_type = "replicated"

        item = {
            "type": "folder",
            "path": "/fileShares/" + name,
            "info": {
                "path": server_path,
                "dataStoreConnectionType": conn_type,
            },
        }

        if client_path is not None:
            item["clientPath"] = client_path

        params = {"f": "json", "item": item}
        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res["status"] == "success" or res["status"] == "exists":
            return Datastore(self, "/fileShares/" + name)
        else:
            print(str(res))
            return None

    def add_bigdata(
        self,
        name: str,
        server_path: Optional[str] = None,
        connection_type: str = "fileShare",
    ):
        """
        The ``add_bigdata`` method registers a bigdata fileshare with the :class:`~arcgis.gis.Datastore`.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique bigdata fileshare name on the server.
        ---------------     --------------------------------------------------------------------
        server_path         Optional string. The path to the folder from the server.
        ---------------     --------------------------------------------------------------------
        connection_type     Optional string. Allows for the setting of the types of big data store.
                            The value 'fileShare' is used for local big data stores, and for
                            cloud stores, the connection_type should be 'dataStore'. The value
                            'fileShare' is the default value.
        ===============     ====================================================================

        :return:
           The big data fileshare if registered successfully, None otherwise.

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_bigdata("name")
        """
        output = None
        path = self._admin_url + "/data/registerItem"

        pattern = r"\\\\[a-zA-Z]+"
        if (
            re.match(pattern, server_path) is not None
        ):  # starts with double backslash, double the backslashes
            server_path = server_path.replace("\\", "\\\\")

        path_str = '{"path":"' + server_path + '"}'
        params = {
            "f": "json",
            "item": json.dumps(
                {
                    "path": "/bigDataFileShares/" + name,
                    "type": "bigDataFileShare",
                    "info": {
                        "connectionString": path_str,
                        "connectionType": connection_type,
                    },
                }
            ),
        }

        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res["status"] == "success" or res["status"] == "exists":
            output = Datastore(self, "/bigDataFileShares/" + name)

        if res["success"]:
            print("Created Big Data file share for " + name)
        elif res["success"] == False and res["status"] != "exists":
            raise Exception("Could not create Big Data file share: %s" % name)
        elif res["status"] == "exists":
            print("Big Data file share exists for " + name)

        return output

    # ----------------------------------------------------------------------
    def add_amazon_s3(
        self,
        name: str,
        bucket_name: str,
        access_key: str,
        access_secret: str,
        region: str,
        folder: Optional[str] = None,
        default_protocal: str = "https",
    ):
        """

        Allows administrators to registered Amazon S3 Buckets as a :class:`~arcgis.gis.Datastore` object.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the Amazon S3 instance.
        ------------------     --------------------------------------------------------------------
        bucket_name            Required String. The name of the S3 bucket.
        ------------------     --------------------------------------------------------------------
        access_key             Required String. The key value for the S3 Bucket.
        ------------------     --------------------------------------------------------------------
        access_secret          Required String. The access secret value for the S3 bucket.
        ------------------     --------------------------------------------------------------------
        region                 Required String. The Amazon region as a string.
        ------------------     --------------------------------------------------------------------
        folder                 Optional String. The S3 folder within the S3 Bucket.
        ------------------     --------------------------------------------------------------------
        default_protocal       Optional String. The URL scheme to contact the S3 bucket.
        ==================     ====================================================================

        :return:
            A :class:`~arcgis.gis.Datastore` object

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_amazon_s3("bucket_name", "access_key", "access_secret", "region")

        """
        if folder is not None:
            bucket_name = f"{bucket_name}/{folder}"
        path = self._admin_url + "/data/registerItem"
        template = {
            "path": f"/cloudStores/{name}",
            "type": "cloudStore",
            "provider": "amazon",
            "info": {
                "isManaged": False,
                "objectStore": bucket_name,
                "connectionString": {
                    "accessKeyId": f"{access_key}",
                    "secretAccessKey": f"{access_secret}",
                    "region": region,
                    "defaultEndpointsProtocol": default_protocal,
                    "credentialType": "accesskey",
                },
            },
        }
        params = {"f": "json", "item": json.dumps(template)}

        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res["status"] == "success" or res["status"] == "exists":
            output = Datastore(self, "/cloudStores/" + name)

        if res["success"]:
            print("Created cloud store for " + name)
        elif res["success"] == False and res["status"] != "exists":
            raise Exception("Could not create cloud store: %s" % name)
        elif res["status"] == "exists":
            print("Cloud store exists for exists for " + name)
        return output

    # ----------------------------------------------------------------------
    def add_ms_azure_storage(
        self,
        cloud_storage_name: str,
        account_key: str,
        account_name: str,
        container_name: str,
        folder: Optional[str] = None,
    ):
        """
        The ``add_ms_azure_storage`` creates a cloud store with Microsoft Azure.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        cloud_storage_name     Required string. The name of the storage entry.
        ------------------     --------------------------------------------------------------------
        access_key             Required String. The key value for the Azure storage.
        ------------------     --------------------------------------------------------------------
        access_secret          Required String. The access secret value for the Azure storage.
        ------------------     --------------------------------------------------------------------
        container_name         Required String. The container holding the data.
        ------------------     --------------------------------------------------------------------
        folder                 Optional String. The Azure folder within the datastore item.
        ==================     ====================================================================

        :return:
            A :class:`~arcgis.gis.Datastore` object

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_ms_azure_storage("name", "key", "secret", "cont_name")

        """
        path = self._admin_url + "/data/registerItem"
        object_store = ""
        if folder:
            object_store = f"{container_name}/{folder}"
        else:
            object_store = f"{container_name}"
        template = {
            "type": "cloudStore",
            "info": {
                "isManaged": False,
                "connectionString": {
                    "accountKey": account_key,
                    "accountName": account_name,
                    "defaultEndpointsProtocol": "https",
                    "accountEndpoint": "core.windows.net",
                    "credentialType": "accessKey",
                },
                "objectStore": object_store,
            },
            "path": f"/cloudStores/{cloud_storage_name}",
            "provider": "azure",
        }
        params = {"f": "json", "item": json.dumps(template)}

        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res["status"] == "success" or res["status"] == "exists":
            output = Datastore(self, "/cloudStores/" + cloud_storage_name)

        if res["success"]:
            print("Created cloud store for " + cloud_storage_name)
        elif res["success"] == False and res["status"] != "exists":
            raise Exception("Could not create cloud store: %s" % cloud_storage_name)
        elif res["status"] == "exists":
            print("Cloud store exists for exists for " + cloud_storage_name)
        return output

    # ----------------------------------------------------------------------
    def add_cloudstore(
        self,
        name: str,
        conn_str: str,
        object_store: str,
        provider: str,
        managed: bool = False,
        folder: Optional[str] = None,
    ):
        """
        The ``add_cloudstore`` method adds a Cloud Store data :class:`~arcgis.gis.Item`.
        Cloud Store data item represents a connection to a Amazon or Microsoft Azure store.
        Connection information for the data store item is stored within conn_str as a
        stringified JSON. ArcGIS Server encrypts connection string for storage. Connection
        strings that are encrypted will include a {crypt} prefix.

        .. note::
            You can get a :class:`~arcgis.gis.Datastore`
            item with decrypted connection string by passing a decrypt=true parameter in the request
            for a data store item. Data store with decrypted connection string will be returned only for
            requests made with https. The examples below show data stores with decrypted conn_str.
            A valid object_store (S3 bucket or Azure Blob store) is required. Folders within an object
            store are optional.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The name of the cloud store.
        ---------------     --------------------------------------------------------------------
        conn_str            Required string. The connection information for the cloud storage
                            product.
        ---------------     --------------------------------------------------------------------
        object_store        Required string. This is the amazon bucket path or Azuze path.
        ---------------     --------------------------------------------------------------------
        provider            Required string. Values must be azuredatalakestore, amazon,
                            Alibaba, or azure.
        ---------------     --------------------------------------------------------------------
        managed             Optional boolean. When the data store is server only, the database
                            is entirely managed and owned by the server and cannot be accessed
                            by the publisher directly. When this option is chosen, the
                            managed property should be set to true. Otherwise it is false.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. For some Azure cloud stores, an optional folder
                            can be specified.
        ===============     ====================================================================

        :return:
            A :class:`~arcgis.gis.Datastore` object

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_cloudstore("name", "connection_info", "path", "provider")

        """
        path = self._admin_url + "/data/registerItem"
        cs = {
            "path": "/cloudStores/%s" % name,
            "type": "cloudStore",
            "provider": provider,
            "info": {
                "isManaged": managed,
                "connectionString": conn_str,
                "objectStore": object_store,
            },
        }
        if folder is not None:
            cs["info"]["folder"] = folder
        params = {"f": "json", "item": json.dumps(cs)}

        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        res = self._portal.con.post(path, params, verify_cert=False)

        if res["status"] == "success" or res["status"] == "exists":
            output = Datastore(self, "/cloudStores/" + name)

        if res["success"]:
            print("Created cloud store for " + name)
        elif res["success"] == False and res["status"] != "exists":
            raise Exception("Could not create cloud store: %s" % name)
        elif res["status"] == "exists":
            print("Cloud store exists for exists for " + name)

        return output

    def add_database(
        self,
        name: str,
        conn_str: str,
        client_conn_str: Optional[str] = None,
        conn_type: str = "shared",
    ):
        """
        The ``add_database`` method registers a database with the :class:`~arcgis.gis.Datastore`.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The unique database name on the server.
        ---------------     --------------------------------------------------------------------
        conn_str            Required string. The path to the folder from the server (and client
                            if shared or serverOnly database).
        ---------------     --------------------------------------------------------------------
        client_conn_str     Optional string. The connection string for client to connect to replicated enterprise database.
        ---------------     --------------------------------------------------------------------
        conn_type           Optional string. Choice of "<shared|replicated|serverOnly>", shared is the default.
        ===============     ====================================================================

        :return:
           The database if registered successfully, None otherwise.

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add_databse("name", "connection_info")
        """

        item = {
            "type": "egdb",
            "path": "/enterpriseDatabases/" + name,
            "info": {
                "connectionString": conn_str,
                "dataStoreConnectionType": conn_type,
            },
        }

        if client_conn_str is not None:
            item["info"]["clientConnectionString"] = client_conn_str

        is_managed = False
        if conn_type == "serverOnly":
            is_managed = True

        item["info"]["isManaged"] = is_managed

        params = {"f": "json", "item": item}
        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res["status"] == "success" or res["status"] == "exists":
            return Datastore(self, "/enterpriseDatabases/" + name)
        else:
            print(str(res))
            return None

    def add(self, name: str, item: dict[str, Any]):
        """
        The ``add`` method registers a new data :class:`~arcgis.gis.Item` with the:class:`~arcgis.gis.Datastore`.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The name of the item to be added on the server.
        ---------------     --------------------------------------------------------------------
        item                Required dictionary. The dictionary representing the data item.
                            See `Data Item <https://developers.arcgis.com/rest/enterprise-administration/server/dataitem.htm>`_
                            in the ArcGIS REST ApI documentation for more details.
        ===============     ====================================================================

        :return:
           The new data :class:`~arcgis.gis.Item` if registered successfully, None otherwise.

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.add("name", {})

        """
        params = {"f": "json"}

        params["item"] = item
        status, msg = self._validate_item(item=params["item"])
        if status == False:
            raise Exception(msg)
        path = self._admin_url + "/data/registerItem"
        res = self._portal.con.post(path, params, verify_cert=False)
        if res["status"] == "success" or res["status"] == "exists":
            return Datastore(self, "/enterpriseDatabases/" + name)
        else:
            print(str(res))
            return None

    def get(self, path: str):
        """
        The ``get`` method retrieves the data :class:`~arcgis.gis.Item` object at the given path.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        path                Required string. The path for the data item.
        ===============     ====================================================================

        :return:
           The data item object if found, None otherwise.
        """
        params = {"f": "json"}
        urlpath = self._admin_url + "/data/items" + path

        datadict = self._portal.con.post(urlpath, params, verify_cert=False)
        if "status" not in datadict:
            return Datastore(self, path)
        else:
            print(datadict["messages"])
            return None

    def search(
        self,
        parent_path: Optional[str] = None,
        ancestor_path: Optional[str] = None,
        types: Optional[str] = None,
        id: Optional[str] = None,
    ):
        """
           The ``search`` method is used to search through the various data
           items registered in the server's data store. Searching without
           specifying the parent path and other parameters returns a list
           of all registered data items.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        parentPath          Optional string. The path of the parent under which to find items.
                            Pass '/' to get the root data items.
        ---------------     --------------------------------------------------------------------
        ancestorPath        Optional string. The path of the ancestor under which to find items.
        ---------------     --------------------------------------------------------------------
        types               Optional string. A comma separated filter for the type of the items.
                            Types include folder, egdb, bigDataFileShare, datadir.
        ---------------     --------------------------------------------------------------------
        id                  Optional string. A filter to search by the ID of the item.
        ===============     ====================================================================

        :return:
           A list of data items matching the specified query.

        .. code-block:: python

            # Usage Example
            >>> arcgis.geoanalytics.get_datastores.search(parentPath= "parent_path",
            ancestorPath= "ancestor_path", id="id")
        """
        params = {
            "f": "json",
        }
        if (
            parent_path is None
            and ancestor_path is None
            and types is None
            and id is None
        ):
            ancestor_path = "/"
        if parent_path is not None:
            params["parentPath"] = parent_path
        if ancestor_path is not None:
            params["ancestorPath"] = ancestor_path
        if types is not None:
            params["types"] = types
        if id is not None:
            params["id"] = id

        path = self._admin_url + "/data/findItems"

        dataitems = []

        res = self._portal.con.post(path, params, verify_cert=False)
        for item in res["items"]:
            dataitems.append(Datastore(self, item["path"]))
        return dataitems

    def _validate_item(self, item):
        """validates a BDS connection"""
        msg = ""
        url = self._admin_url + "/data/validateDataItem"
        params = {"f": "json", "item": item}
        res = self._portal.con.post(url, params, verify_cert=False)
        try:
            return res["status"] == "success", ""
        except:
            return False, res

    def validate(self):
        """
        The ``validate`` method validates all items in the :class:`~arcgis.gis.Datastore`. In order for a data item
        to be registered and used successfully within the GIS's data store, you need to make sure that the path
        (for file shares) or connection string (for databases) is accessible to every server
        node in the site. To validate all registered data items all
        at once, you can invoke this operation.

        :return:
           A boolean indicating successful validation (True), or failed validation (False)
        """
        params = {"f": "json"}
        path = self._admin_url + "/data/validateAllDataItems"
        res = self._portal.con.post(path, params, verify_cert=False)
        return res["status"] == "success"


###########################################################################
class UserManager(object):
    """
    The ``UserManager`` class is a helper class for managing GIS users. This class is not created by users directly.
    An instance of this class, called 'users', is available as a property of the Gis object.
    Users call methods on this 'users' object to manipulate (create, get, search, etc) users.


    """

    _me = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< UserManager at {url} >".format(url=self._gis._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def delete_users(self, users: list[User]) -> list[str]:
        """
        Allows the administrator to remove users from a portal. Before the
        administrator can remove the user, all of the user's content and
        groups must be reassigned or deleted.

        ================  ====================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------
        users             Required list of :class:`users <arcgis.gis.User>` to
                          delete from the organization.
        ================  ====================================================

        :return:
           list containing the :class:`users <arcgis.gis.User>`
           who could not be removed.
        """
        from arcgis._impl.common._utils import chunks as _chunks

        url = f"{self._gis._portal.resturl}portals/self/removeUsers"
        results = []
        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            jobs = []
            for chunk in _chunks(users, n=100):
                users_str = ",".join([u.username for u in chunk])
                params = {"f": "json", "users": users_str}
                future = executor.submit(
                    self._gis._con.post, **{"path": url, "params": params}
                )
                jobs.append(future)
            for future in concurrent.futures.as_completed(jobs):
                users = future.result().get("notRemoved", [])
                results.extend(users)

        return results

    # ----------------------------------------------------------------------
    @property
    def user_settings(self):
        """
        Gets/sets the user's settings

        The `user_settings` allows administrators to set, and edit, new
        member defaults. Members who create their own built-in accounts and
        members added by an administrator or through automatic account
        creation will be automatically assigned the new member defaults.

        Passing in `None` to the property will delete all the user settings.

        **Settings Key/Value Dictionary**

        ================  ===============================================================================
        **Keys**          **Description**
        ----------------  -------------------------------------------------------------------------------
        role            String/Role. The role ID. To assign a custom role as the new member default,
                          provide a Role object.

                          Values: `administrator`, `publisher`, `editor`, `viewer` or custom `Role` object
        ----------------  -------------------------------------------------------------------------------
        userLicenseType   String. The ID of a user type licensed with your organization. To see which
                          user types are included with your organization's licensing, see the License
                          resource in the Portal Admin API.

                          Values: `creator`, `editor`, `Advanced GIS`, `Basic GIS`, `Standard GIS`,
                          `viewer`, or `fieldWorker`
        ----------------  -------------------------------------------------------------------------------
        groups            List of String/Groups. An array of group ID numbers or `Group` objects that
                          specify the groups new members will be added to.
        ----------------  -------------------------------------------------------------------------------
        userType          String.  This key only applies to `ArcGIS Online`. If new members will have
                          Esri access (both) or if Esri access will be disabled (arcgisonly). The default
                          value is `arcgisonly`.

                          Values: `arcgisonly` or `both`
        ----------------  -------------------------------------------------------------------------------
        apps              List of dictionaries.  An array of an app's itemID and, when applicable, entitlement.
                          Example: `{"apps" :[{"itemId": "f761dd0f298944dcab22d1e888c60293","entitlements": ["Insights"]}]}`
        ----------------  -------------------------------------------------------------------------------
        appBundles        List of dictionaries. An array of an app bundle's ID.

                          Example: `{"appBundles":[{"itemId": "99d7956c7e824ff4ab27422e2a26c2b7}]}`
        ----------------  -------------------------------------------------------------------------------
        clear             Optional Bool. When true, any empty field will reset the value to null.
        ================  ===============================================================================

        :return: Dictionary of the user settings

        """
        if self._gis.version >= [7, 3]:
            url = f"{self._gis._portal.resturl}portals/self/userDefaultSettings"
            params = {"f": "json"}
            return self._gis._con.get(url, params)
        return None

    # ----------------------------------------------------------------------
    @user_settings.setter
    def user_settings(self, settings):
        """
        See main ``user_settings`` property docstring
        """
        clear = settings.pop("clear", None)
        user_li_lu = {
            "creatorUT": "creatorUT",
            "creator": "creatorUT",
            "editor": "editorUT",
            "editorUT": "editorUT",
            "GISProfessionalAdvUT": "GISProfessionalAdvUT",
            "Advanced GIS": "GISProfessionalAdvUT",
            "Basic GIS": "GISProfessionalBasicUT",
            "GISProfessionalBasicUT": "GISProfessionalBasicUT",
            "Standard GIS": "GISProfessionalStdUT",
            "GISProfessionalStdUT": "GISProfessionalStdUT",
            "viewer": "viewerUT",
            "viewerUT": "viewerUT",
            "fieldworker": "fieldWorkerUT",
            "fieldWorkerUT": "fieldWorkerUT",
        }
        role_lu = {
            "administrator": "org_admin",
            "org_admin": "org_admin",
            "publisher": "org_publisher",
            "org_publisher": "org_publisher",
            "user": "org_user",
            "iBBBBBBBBBBBBBBB": "iBBBBBBBBBBBBBBB",
            "editor": "iBBBBBBBBBBBBBBB",
            "viewer": "iAAAAAAAAAAAAAAA",
            "iAAAAAAAAAAAAAAA": "iAAAAAAAAAAAAAAA",
        }
        if self._gis.version > [7, 3]:
            if settings is None or (isinstance(settings, dict) and len(settings) == 0):
                cs = self.user_settings
                if cs and len(cs) > 0:
                    self._delete_user_settings()
            else:
                url = f"{self._gis._portal.resturl}portals/self/setUserDefaultSettings"
                params = {"f": "json"}
                if "role" in settings:
                    if settings["role"] in role_lu:
                        settings["role"] = role_lu[settings["role"].lower()]
                    elif isinstance(settings["role"], Role):
                        settings["role"] = settings["role"].role_id
                if "userLicenseType" in settings:
                    if settings["userLicenseType"].lower() in user_li_lu:
                        settings["userLicenseType"] = user_li_lu[
                            settings["userLicenseType"].lower()
                        ]
                if (
                    "userType" in settings
                    and self._gis._portal.is_arcgisonline == False
                ):
                    del settings["userType"]
                if "groups" in settings:
                    settings["groups"] = [
                        grp.groupid
                        for grp in settings["groups"]
                        if isinstance(grp, Group)
                    ] + [grp for grp in settings["groups"] if isinstance(grp, str)]
                    if len(settings["groups"]) == 0:
                        settings["groups"] = ""
                if clear:
                    settings["clearEmptyFields"] = True
                params.update(settings)
                res = self._gis._con.post(url, params)
                if "success" in res and res["success"] == False:
                    raise Exception(res)

    # ----------------------------------------------------------------------
    def _delete_user_settings(self):
        """
        This operation allows administrators to clear the previously
        configured new member defaults set either through the Set User
        Default Settings operation or from the New Member Defaults tab in
        the Organization Settings of the portal.

        :return: Boolean

        """
        if self._gis.version > [7, 3]:
            url = f"{self._gis._portal.resturl}portals/self/userDefaultSettings/delete"
            params = {"f": "json"}
            res = self._portal.con.post(url, params)
            if "success" in res:
                return res["success"]
            return res
        return None

    # ----------------------------------------------------------------------
    @property
    def license_types(self):
        """
        The ``license_types`` method returns a list of available licenses associated with a given GIS.

        .. note::
            The information returned can help administrators determine what type
            of a user should me based on the bundles associated with each user
            type. Additionally, the ``license_types`` is only available on ArcGIS Enterprise 10.7+.**

        :return:
            A List of available licenses
        """

        if self._gis.version < [6, 4]:
            return []

        url = "portals/self/userLicenseTypes"
        params = {"f": "json", "start": 1, "num": 255}

        res = self._gis._con.get(url, params)
        results = res["userLicenseTypes"]
        while res["nextStart"] > -1:
            params["start"] = res["nextStart"]
            res = self._gis._con.get(url, params)
            results += res["userLicenseTypes"]
        return results

    # ----------------------------------------------------------------------
    def counts(self, type: str = "bundles", as_df: bool = True):
        """
        The ``counts`` method returns a simple report on the number of licenses currently used
        for a given `type`.  A `type` can be a role, app, bundle or user license type.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        type              Required String. The type of data to return.  The following values are valid:

                            + role - returns counts on user roles
                            + app - returns counts on registered applications
                            + bundles - returns counts on application bundles
                            + user_type - returns counts on the user license types
        ----------------  -------------------------------------------------------------------------------
        as_df             Optional boolean. If true, the results are returned as a pandas DataFrame, else
                          it is returned as a list of dictionaries.
        ================  ===============================================================================

        :return:
            Pandas `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
            if ``as_df`` is True. If False, the result is a list of dictionaries.

        .. code-block:: python

            # Usage Example

            >>>gis.users.counts("Role", as_df=True)


            **Example as_df=True**

            >>> df = gis.users.counts('user_type', True)
            >>> df
                count        key
            0     12  creatorUT
            1      2   viewerUT


            **Example as_df=False**


            >>> df = gis.users.counts('user_type', False)
            >>> df
            [{'key': 'creatorUT', 'count': 12}, {'key': 'viewerUT', 'count': 2}]


        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError(
                "`counts` is not implemented at version %s of Enterprise"
                % ".".join([str(i) for i in self._gis.version])
            )

        url = "portals/self/users/counts"
        lu = {
            "roles": "role",
            "role": "role",
            "app": "app",
            "bundles": "appBundle",
            "user_type": "userLicenseType",
            "usertype": "userLicenseType",
        }
        results = []
        params = {
            "f": "json",
            "type": lu[type.lower()],
            "num": 100,
            "start": 1,
        }
        res = self._portal.con.get(url, params, ssl=True)
        results += res["results"]
        while res["nextStart"] != -1:
            if res["nextStart"] == -1:
                break
            params["start"] = res["nextStart"]
            res = self._portal.con.get(url, params, ssl=True)
            results += res["results"]
        if as_df:
            import pandas as pd

            return pd.DataFrame(data=results)
        return results

    # ----------------------------------------------------------------------
    def send_notification(
        self,
        users: Union[list[str], list[User]],
        subject: str,
        message: str,
        type: str = "builtin",
        client_id: Optional[str] = None,
    ) -> bool:
        """
        The ``send_notification`` method creates a user notifcation for a list of users.

        .. note::
            This has been deprecated at Enterprise 10.9 and can only be used with ArcGIS Online.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        users             Required List. A list of strings or User objects to send notifications to.
        ----------------  -------------------------------------------------------------------------------
        subject           Required String. The notification subject line.
        ----------------  -------------------------------------------------------------------------------
        message           Required String. The notification content. This should be in plain text.
        ----------------  -------------------------------------------------------------------------------
        type              Optional String.  The notification can be sent various ways. These include:

                             - builtin - The enterprise built-in system
                             - push - The push notification to send a message to
                             - email - a notification sent to the user's email account
        ----------------  -------------------------------------------------------------------------------
        client_id         Optional String. The client id for push notification.
        ================  ===============================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.users.send_notification([user1,user1234,user123, user1234], "testing notification system",
                                            "This was a test of the sending notification property")


        """
        if self._gis._is_agol:
            if self._gis.version >= [6, 4]:
                susers = []
                for u in users:
                    if isinstance(u, str):
                        susers.append(u)
                    elif isinstance(u, User):
                        susers.append(u.username)
                    del u
                url = "{base}portals/self/createNotification".format(
                    base=self._gis._portal.resturl
                )
                params = {
                    "f": "json",
                    "notificationChannelType": type,
                    "subject": subject,
                    "message": message,
                    "users": ",".join(susers),
                    "clientId": client_id,
                }
                return self._portal.con.post(url, params)["success"]
        else:
            raise NotImplementedError(
                "The current version of the enterprise does not support `send_notification`"
            )
        return False

    # ----------------------------------------------------------------------
    def create(
        self,
        username: str,
        password: str | None,
        firstname: str,
        lastname: str,
        email: str,
        role: str,
        description: Optional[str] = None,
        provider: str = "arcgis",
        idp_username: Optional[str] = None,
        level: int = 2,
        thumbnail: Optional[str] = None,
        user_type: Optional[str] = None,
        credits: float = -1,
        groups: Optional[list[str]] = None,
        email_text: Optional[str] = None,
    ):
        """
        The ``create`` operation is used to create built-in or pre-create organization-specific identity
        store accounts for use in a Web GIS. See the respective documentation for complete details
        about configurating identity stores and managing access to your deployment:

        * ``ArcGIS Enterprise`` - `Manage access to your portal <https://enterprise.arcgis.com/en/portal/latest/administer/windows/managing-access-to-your-portal.htm>`_
        * ``ArcGIS Online`` - `Invite and add members <https://doc.arcgis.com/en/arcgis-online/administer/invite-users.htm>`_

        .. note::
            Only an administrator can call this method.

        .. note::
            When Portal for ArcGIS is connected to an
            `organization specific identity store <https://enterprise.arcgis.com/en/portal/latest/administer/windows/managing-access-to-your-portal.htm#ESRI_SECTION2_4E6A70E10A9444DD92662208198B8876>`_,
            users can sign into portal using their organization specific credentials, also known as
            their enterprise credentials. By default, new installations of Portal for ArcGIS do not
            allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        The `user_type` argument determines which `role` can be assigned to the member. A
        full explanation of `user types` and their compatibility with a particular `role`
        can be found in the
        `User types, roles, and privileges <https://enterprise.arcgis.com/en/portal/latest/administer/windows/roles.htm>`_
        documentation.

        An organization administrator may configure `New Member Defaults`. When a Web GIS is configured
        with these values, any new user will receive these default values unless overridden by the
        corresponding arguments in this method. See the following documentation for additional details:

        * `ArcGIS Online <https://doc.arcgis.com/en/arcgis-online/administer/configure-new-member-defaults.htm>`_
        * `ArcGIS Enterprise <https://enterprise.arcgis.com/en/portal/latest/administer/windows/configure-new-member-defaults.htm>`_

        To query the organization for `New Member Defaults`, run the following code:

        .. code-block:: python

            >>> gis = GIS(profile="your_admin_profile")

            >>> gis.users.user_settings

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Enterprise or
                          in all of ArcGIS Online. Must be between 6 to 24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string if the `provider` argument is `arcgis`. If the argument is
                          `enterprise`, meaning an
                          `organization-specific identity provider
                          <https://enterprise.arcgis.com/en/portal/latest/administer/windows/managing-access-to-your-portal.htm#ESRI_SECTION2_4E6A70E10A9444DD92662208198B8876>`_
                          is configured, the password parameter is optional (or ignored if present).

                          .. note::
                             If creating an ArcGIS Online organization user, the argument can be `None`
                             and users can subsequently set their password by clicking on a link that
                             is emailed to them.

                          .. note::
                             This argument **must** be None if iniviting the user to an ArcGIS Online
                             organization and you want to send an email. Sending email invitations cannot
                             have passwords set by an administrator.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user.
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user.
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important!
        ----------------  -------------------------------------------------------------------------------
        role              Required string. The :class:`role <arcgis.gis.Role>` name or `role_id` value to
                          assign the new member. To assign one of the `default Administrator, Publisher,
                          or User roles <https://enterprise.arcgis.com/en/portal/latest/administer/windows/member-roles.htm#ESRI_SECTION1_C30D73392D964D51A8B606128A8A6E8F>`_
                          enter ``org_admin``, ``org_publisher``, or ``org_user``, respectively.
                          For any other default role, or a custom role within the organization, enter
                          the `role_id` value returned from the :meth:`~arcgis.gis.RoleManager.all` method
                          on the :class:`~arcgis.gis.RoleManager` class.

                          .. code-block:: python

                              >>> from arcgis.gis import GIS

                              >>> gis = GIS(profile="your_org_admin_profile")

                              >>> for org_role in gis.users.roles.all():
                                      print(f"{org_role.name:25}{org_role.role_id}")
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to an image to represent the user.

        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The identity provider for the account. The default value is
                          `arcgis`. Possible values:

                          * `arcgis` - built-in identity provider
                          * `enterprise` - organization-specific identity provider

                          See documentation for managing organizational access for explanation of different
                          identity provider options:

                          * `ArcGIS Enterprise <https://enterprise.arcgis.com/en/portal/latest/administer/windows/managing-access-to-your-portal.htm>`_
                          * `ArcGIS Online <https://doc.arcgis.com/en/arcgis-online/administer/invite-users.htm>`_
        ----------------  -------------------------------------------------------------------------------
        idp_username      Required if `provider` argument is `enterprise`, otherwise not used. The name
                          of the user as stored by the organization-specific identity store.
        ----------------  -------------------------------------------------------------------------------
        level             **Deprecated** Optional integer. The Web GIS system automatically sets this
                          argument based upon the `user_type` and `role` arguments. See
                          `Levels <https://enterprise.arcgis.com/en/portal/10.6/administer/windows/roles.htm#ESRI_SECTION1_08925CEF37334C619D52BC027C3C8DE1>`_
                          for detailed description.

                          .. note::
                              This parameter was deprecated with the 10.7 release.
        ----------------  -------------------------------------------------------------------------------
        user_type         Required string, unless specified in the `New Member Defaults`. The user type
                          license for an organization member. See
                          `user types <https://enterprise.arcgis.com/en/portal/latest/administer/windows/user-types-orgs.htm>`_
                          for detailed descriptions of each `user type`. Each `user_type` is
                          compatible with specific `roles` in the organization. Compatibility is
                          determined by the `privileges` assigned to each `role`. Only certain `role`
                          arguments will work with specific `user types`. The potential values
                          for this argument depend upon the organizational subscription and
                          licensing. Run the following query as an administrator to determine the
                          possible values:

                          .. code-block:: python

                              >>> for utype in gis.users.license_types:
                                      print(f"{utype['id]}")

                          .. note::
                              See the :attr:`~arcgis.gis.UserManager.license_types` property on the
                              :class:`~arcgis.gis.UserManager` class.
        ----------------  -------------------------------------------------------------------------------
        credits           Optional Float. The number of credits to assign a user.  The default is None,
                          unless specified in the `New Member Defaults`.

                          The following code will return the default value if it has been set:

                          .. code-block:: python

                              >>> gis = GIS(profile="your_admin_profile")

                              >>> gis.properties.defaultUserCreditAssignment

                          .. note::
                              Only applies to ArcGIS Online organizations.
        ----------------  -------------------------------------------------------------------------------
        groups            Optional List of :class:`~arcgis.gis.Group` objects to which the new user will
                          be added. If `None`, user will be assigned to any groups specified in the
                          `New Member Defaults`.
        ----------------  -------------------------------------------------------------------------------
        email_text        Optional string. Custom text to include in the invitation email. This text will
                          be appended to the top of the default email text. `ArcGIS Online` only.
        ================  ===============================================================================

        :return:
            The :class:`user <arcgis.gis.User>` if successfully created, None if unsuccessful.

        .. code-block:: python

            #Usage Example 1: New ArcGIS Online user using `New Member Defaults`

            >>> ago = GIS(profile='your_online_admin_profile')

            >>> for k,v in ago.users.user_settings.items():
            >>>     print(f'{k:20}{v}')

            role                org_publisher
            userLicenseType     advancedUT
            groups              ['96c9a826e654481ba2cf8f6d04137b32']
            userType            arcgisonly
            apps                []
            appBundles          []
            categories          []

            >>> new_user = ago.users.create(username= 'new_unique_username',
                                            password= '<strong_password>',
                                            firstname= 'user_firstname',
                                            lastname= 'user_lastname',
                                            email= 'user_email@company.com',
                                            description= 'new user using member defaults'))

            # Usage Example 2: New ArcGIS Online user with custom role and non-default `user_type`

            # Get RoleManager and print `role_id` values for `role` argument
            >>> role_mgr = gis.users.roles

            >>> for role in role_mgr.all():
            >>>     print(f'{role.name}  {role.role_id}')

            Viewer              iAAAAAAAAAAAAAAA
            Data Editor         iBBBBBBBBBBBBBBB
            CustomRole          bKrTCjFF9tKbaFk8

            # Print valid values for `user_type` argument
            >>> [ut['id'] for ut in ago.users.license_types]

            ['advancedUT',
            'basicUT',
            'creatorUT',
            'editorUT',
            'fieldWorkerUT',
            'GISProfessionalAdvUT',
            'GISProfessionalBasicUT',
            'GISProfessionalStdUT',
            'IndoorsUserUT',
            'insightsAnalystUT',
            'liteUT',
            'standardUT',
            'storytellerUT',
            'viewerUT']

            >>> user1 = ago.users.create(username='new_unique_username',
                                         password='<strong_password>',
                                         firstname="user_firstname",
                                         lastname="user_lastname",
                                         email="user_email@company.com",
                                         description="Test user with custom role and non-default user type.",
                                         role='bKrTCjFF9tKbaFk8',
                                         user_type='creatorUT')

            # Usage Example 3: New User invited with an email:

            >>> user_e = ago.users.create(username="new_invited_uk_Q42eklm",
                                          password="S8V3*t4L8tr!&",
                                          firstname="user_firstname",
                                          lastname="user_lastname",
                                          email="user_email@company.com",
                                          description="Test for an invited user to Online.",
                                          user_type="creatorUT",
                                          role="XBH3xJArWxYuK2qX",
                                          email_text="Welcome aboard the Web GIS organization!")
        """
        if any(
            [
                user.username.lower() == username.lower()
                for user in self.search(query=username)
            ]
        ):
            raise Exception(
                "User %s already exists. Please provide a different username."
                % username
            )
        kwargs = {
            "username": username,
            "password": password,
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "description": description,
            "role": role,
            "provider": provider,
            "idp_username": idp_username,
            "level": level,
            "thumbnail": thumbnail,
            "user_type": user_type,
            "credits": credits,
            "groups": groups,
            "email_text": email_text,
        }
        if self._gis.version >= [6, 4]:
            allowed_keys = {
                "username",
                "password",
                "firstname",
                "lastname",
                "email",
                "description",
                "role",
                "provider",
                "idp_username",
                "user_type",
                "thumbnail",
                "credits",
                "groups",
                "level",
                "email_text",
            }
            params = {}
            for k, v in kwargs.items():
                if k in allowed_keys:
                    params[k] = v
            return self._create64plus(**params)
        else:
            allowed_keys = {
                "username",
                "password",
                "firstname",
                "lastname",
                "email",
                "description",
                "role",
                "provider",
                "idp_username",
                "level",
                "thumbnail",
            }
            params = {}
            for k, v in kwargs.items():
                if k in allowed_keys:
                    params[k] = v
            return self._createPre64(**params)
        return None

    # ----------------------------------------------------------------------
    def _createPre64(
        self,
        username: str,
        password: str,
        firstname: str,
        lastname: str,
        email: str,
        description: Optional[str] = None,
        role: str = "org_user",
        provider: str = "arcgis",
        idp_username: Optional[str] = None,
        level: int = 2,
        thumbnail: Optional[str] = None,
    ):
        """
        This operation is used to pre-create built-in or enterprise accounts within the portal,
        or built-in users in an ArcGIS Online organization account. Only an administrator
        can call this method.

        To create a viewer account, choose role='org_viewer' and level=1

        .. note:
            When Portal for ArcGIS is connected to an enterprise identity store, enterprise users sign
            into portal using their enterprise credentials. By default, new installations of Portal for
            ArcGIS do not allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Portal, and
                          6-24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string. The password for the user.  It must be at least 8 characters.
                          This is a required parameter only if
                          the provider is arcgis; otherwise, the password parameter is ignored.
                          If creating an account in an ArcGIS Online org, it can be set as None to let
                          the user set their password by clicking on a link that is emailed to him/her.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to user's image.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The role for the user account. The default value is org_user.
                          Other possible values are org_publisher, org_admin, org_viewer.
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        idp_username      Optional string. The name of the user as stored by the enterprise user store.
                          This parameter is only required if the provider parameter is enterprise.
        ----------------  -------------------------------------------------------------------------------
        level             Optional integer. The account level.
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ================  ===============================================================================

        :return:
            The user if successfully created, None if unsuccessful.

        """
        # map role parameter of a viewer to the internal value for org viewer.
        if role == "org_viewer":
            role = "iAAAAAAAAAAAAAAA"

        if self._gis._portal.is_arcgisonline:
            email_text = (
                """<html><body><p>"""
                + self._gis.properties.user.fullName
                + """ has invited you to join an ArcGIS Online Organization, """
                + self._gis.properties.name
                + """</p>
<p>Please click this link to finish setting up your account and establish your password: <a href="https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@">https://www.arcgis.com/home/newuser.html?invitation=@@invitation.id@@</a></p>
<p>Note that your account has already been created for you with the username, <strong>@@touser.username@@</strong>.  </p>
<p>If you have difficulty signing in, please contact """
                + self._gis.properties.user.fullName
                + "("
                + self._gis.properties.user.email
                + """). Be sure to include a description of the problem, the error message, and a screenshot.</p>
<p>For your reference, you can access the home page of the organization here: <br>"""
                + self._gis.properties.user.fullName
                + """</p>
<p>This link will expire in two weeks.</p>
<p style="color:gray;">This is an automated email. Please do not reply.</p>
</body></html>"""
            )
            params = {
                "f": "json",
                "invitationList": {
                    "invitations": [
                        {
                            "username": username,
                            "firstname": firstname,
                            "lastname": lastname,
                            "fullname": firstname + " " + lastname,
                            "email": email,
                            "role": role,
                            "level": level,
                        }
                    ]
                },
                "message": email_text,
            }
            if idp_username is not None:
                if provider is None:
                    provider = "enterprise"
                params["invitationList"]["invitations"][0][
                    "targetUserProvider"
                ] = provider
                params["invitationList"]["invitations"][0]["idpUsername"] = idp_username
            if password is not None:
                params["invitationList"]["invitations"][0]["password"] = password

            resp = self._portal.con.post("portals/self/invite", params, ssl=True)
            if resp and resp.get("success"):
                if username in resp["notInvited"]:
                    print("Unable to create " + username)
                    _log.error("Unable to create " + username)
                    return None
                else:
                    return self.get(username)
        else:
            createuser_url = self._portal.url + "/portaladmin/security/users/createUser"
            # print(createuser_url)
            params = {
                "f": "json",
                "username": username,
                "password": password,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "description": description,
                "role": role,
                "provider": provider,
                "idpUsername": idp_username,
                "level": level,
            }
            self._portal.con.post(createuser_url, params)
            user = self.get(username)
            if thumbnail is not None:
                ret = user.update(thumbnail=thumbnail)
                if not ret:
                    _log.error("Unable to update the thumbnail for  " + username)
            return user

    # ----------------------------------------------------------------------
    def _create64plus(
        self,
        username,
        password,
        firstname,
        lastname,
        email,
        description=None,
        role="org_user",
        provider="arcgis",
        idp_username=None,
        user_type="creator",
        thumbnail=None,
        credits=None,
        groups=None,
        level=None,
        email_text=None,
    ):
        """
        This operation is used to pre-create built-in or enterprise accounts within the portal,
        or built-in users in an ArcGIS Online organization account. Only an administrator
        can call this method.

        To create a viewer account, choose role='org_viewer' and level='viewer'

        .. note:
            When Portal for ArcGIS is connected to an enterprise identity store, enterprise users sign
            into portal using their enterprise credentials. By default, new installations of Portal for
            ArcGIS do not allow accounts from an enterprise identity store to be registered to the portal
            automatically. Only users with accounts that have been pre-created can sign in to the portal.
            Alternatively, you can configure the portal to register enterprise accounts the first time
            the user connects to the website.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        username          Required string. The user name, which must be unique in the Portal, and
                          6-24 characters long.
        ----------------  -------------------------------------------------------------------------------
        password          Required string. The password for the user.  It must be at least 8 characters.
                          This is a required parameter only if the provider is arcgis; otherwise, the
                          password parameter is ignored.
                          If creating an account in an ArcGIS Online org, it can be set as None to let
                          the user set their password by clicking on a link that is emailed to him/her.
        ----------------  -------------------------------------------------------------------------------
        firstname         Required string. The first name for the user
        ----------------  -------------------------------------------------------------------------------
        lastname          Required string. The last name for the user
        ----------------  -------------------------------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ----------------  -------------------------------------------------------------------------------
        description       Optional string. The description of the user account.
        ----------------  -------------------------------------------------------------------------------
        thumbnail         Optional string. The URL to user's image.
        ----------------  -------------------------------------------------------------------------------
        role              Optional string. The role for the user account. The default value is org_user.
                          Other possible values are org_user, org_publisher, org_admin, viewer,
                          view_only, viewplusedit or a custom role object (from gis.users.roles).

                          .. note::
                            It is recommended to pass in role_id when assigning a custome role to a user. The
                            role name can be used for multiple roles and can lead to issues if more than one
                            custom role has the same role name. Access the role_id through property on the Role class.
        ----------------  -------------------------------------------------------------------------------
        provider          Optional string. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        idp_username      Optional string. The name of the user as stored by the enterprise user store.
                          This parameter is only required if the provider parameter is enterprise.
        ----------------  -------------------------------------------------------------------------------
        user_type         Required string. The account user type. This can be creator or viewer.  The
                          type effects what applications a user can use and what actions they can do in
                          the organization.
                          See http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm
        ----------------  -------------------------------------------------------------------------------
        credits           Optional Float. The number of credits to assign a user.  The default is None,
                          which means unlimited.
        ----------------  -------------------------------------------------------------------------------
        groups            Optional List. An array of Group objects to provide access to for a given user.
        ----------------  -------------------------------------------------------------------------------
        email_text        Optional string. Custom text to include in the invitation email. This text will
                          be appended to the default email text. ArcGIS Online only.
        ================  ===============================================================================

        :return:
            The user if successfully created, None if unsuccessful.

        """
        # map role parameter of a viewer to the internal value for org viewer.
        if self._gis.version >= [7, 2]:
            if self._gis._is_agol:
                if user_type is None and role is None:
                    if self.user_settings and "userLicenseType" in self.user_settings:
                        user_type = self.user_settings["userLicenseType"]
                        role = self.user_settings["role"]
        else:
            if self._gis.version >= [7, 1]:
                if user_type is None and role is None:
                    if "defaultUserTypeIdForUser" in self._gis.admin.security.config:
                        user_type = self._gis.admin.security.config[
                            "defaultUserTypeIdForUser"
                        ]
                        role = self._gis.admin.security.config["defaultRoleForUser"]
        if level == 2 and user_type is None and role is None:
            user_type = "creator"
            role = "publisher"
        elif level == 1 and user_type is None and role is None:
            user_type = "viewer"
            role = "viewer"
        elif level == 1 and user_type is None:
            user_type = "viewer"
        elif level == 1 and role is None:
            role = "viewer"
        elif level == 2 and user_type is None:
            user_type = "creator"
        elif level == 2 and role is None:
            role = "publisher"

        levels = {"creator": "creatorUT", "viewer": "viewerUT"}
        role_lookup = {
            "admin": "org_admin",
            "org_admin": "org_admin",
            "user": "org_user",
            "org_user": "org_user",
            "publisher": "org_publisher",
            "org_publisher": "org_publisher",
            "creator": "org_publisher",
            "view_only": "tLST9emLCNfFcejK",
            "org_viewer": "iAAAAAAAAAAAAAAA",
            "viewer": "iAAAAAAAAAAAAAAA",
            "viewplusedit": "iBBBBBBBBBBBBBBB",
        }

        if groups is None:
            groups = []

        if user_type.lower() in levels:
            user_type = levels[user_type.lower()]

        if isinstance(role, Role):
            role = role.role_id
        elif role and role.lower() in role_lookup:
            role = role_lookup[role.lower()]
        elif isinstance(role, str):
            # lookup the role id to see if it exists, else set to ""
            try:
                role = self._gis.users.roles.get_role(role)
                role = role.role_id
            except:
                # maybe user passed in role name instead of id
                if self._gis.users.roles.exists(role):
                    all_roles = self._gis.users.roles.all()
                    for r in all_roles:
                        if r.name.lower() == role.lower():
                            role = r.role_id
                            break
                else:
                    role = ""
        else:
            role = ""

        # Check if default role provided by org if none given
        if role in ["", None]:
            if self._gis._is_arcgisonline:
                url = (
                    self._gis._public_rest_url
                    + "portals/self/userDefaultSettings?f=json"
                )
            else:
                url = (
                    self._gis._portal.resturl
                    + "portals/self/userDefaultSettings?f=json"
                )
            params = {"f": "json"}
            resp = self._gis._con._session.get(url).json()
            if "role" not in resp or resp["role"] == None:
                raise ValueError(
                    "Role cannot be None since no default role is provided in the org settings. Please provide a valid role."
                )

        if self._gis._portal.is_arcgisonline or (
            self._gis._portal.is_kubernetes
            and provider != "enterprise"
            and self._gis._portal._version != "10.3"
        ):
            if (
                credits == -1
                and self._gis.version >= [7, 2]
                and self._gis.properties["defaultUserCreditAssignment"] != -1
            ):
                credits = self._gis.properties["defaultUserCreditAssignment"]
            if (
                not groups
                and "groups" in self.user_settings
                and self.user_settings["groups"]
            ):
                groups = [
                    self._gis.groups.get(g).id for g in self.user_settings["groups"]
                ]
            params = {
                "f": "json",
                "invitationList": {
                    "invitations": [
                        {
                            "username": username,
                            "firstname": firstname,
                            "lastname": lastname,
                            "fullname": firstname + " " + lastname,
                            "email": email,
                            "role": role,
                            "userLicenseType": user_type,
                            "groups": ",".join([g for g in groups if g]),
                            "userCreditAssignment": credits,
                        }
                    ],
                    "apps": [],
                    "appBundles": [],
                },
            }
            if email_text:
                params["message"] = email_text
            if idp_username is not None:
                if provider is None:
                    provider = "enterprise"
                params["invitationList"]["invitations"][0][
                    "targetUserProvider"
                ] = provider
                params["invitationList"]["invitations"][0]["idpUsername"] = idp_username
            if password is not None:
                params["invitationList"]["invitations"][0]["password"] = password

            resp = self._portal.con.post("portals/self/invite", params, ssl=True)
            if resp and resp.get("success"):
                if username in resp["notInvited"]:
                    print("Unable to create " + username)
                    _log.error("Unable to create " + username)
                    return None
                else:
                    new_user = self.get(username)
                    if (
                        self.user_settings
                        and "userType" in self.user_settings
                        and not self.user_settings["userType"] == "arcgisonly"
                    ):
                        update_url = "community/users/" + username + "/update"
                        user_params = {
                            "f": "json",
                            "token": "token",
                            "userType": self.user_settings["userType"],
                        }
                        self._portal.con.post(update_url, user_params, ssl=True)
                        return new_user
                    else:
                        return new_user
        # If kubernets is 11.1 then need to use the second method, even if provider is arcgis
        elif self._gis._portal.is_kubernetes and (
            provider == "enterprise" or self._gis._portal._version == "10.3"
        ):
            createuser_url = (
                self._portal.url
                + "/admin/orgs/0123456789ABCDEF/security/users/createUser"
            )
            params = {
                "f": "json",
                "username": username,
                "password": password,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "description": description,
                "role": role,
                "provider": provider,
                "idpUsername": idp_username,
                "userLicenseTypeId": user_type,
            }
            self._portal.con.post(createuser_url, params)
            user = self.get(username)
            for grp in groups:
                grp.add_users([username])
            if thumbnail is not None:
                ret = user.update(thumbnail=thumbnail)
                if not ret:
                    _log.error("Unable to update the thumbnail for  " + username)
            return user
        else:
            createuser_url = self._portal.url + "/portaladmin/security/users/createUser"
            params = {
                "f": "json",
                "username": username,
                "password": password,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "description": description,
                "role": role,
                "provider": provider,
                "idpUsername": idp_username,
                "userLicenseTypeId": user_type,
            }
            if "password" in params and params["password"] is None:
                params.pop("password", None)
            self._portal.con.post(createuser_url, params)
            if params["username"].find("\\") > -1:
                d = params["username"].split("\\")
                d.reverse()
                username = "@".join(d)
            user = self.get(username)

            for grp in groups:
                grp.add_users([username])
            if thumbnail is not None:
                ret = user.update(thumbnail=thumbnail)
                if not ret:
                    _log.error("Unable to update the thumbnail for  " + username)
            return user

    # ----------------------------------------------------------------------
    def invite(
        self,
        email: str,
        role: str = "org_user",
        level: int = 2,
        provider: Optional[str] = None,
        must_approve: bool = False,
        expiration: str = "1 Day",
        validate_email: bool = True,
        message_text: Optional[str] = None,
    ):
        """
        The ``invite`` method invites a :class:`~arcgis.gis.User` object to an organization by email.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        email             Required String. The user's email that will be invited to the organization.
        ----------------  -------------------------------------------------------------------------------
        role              Optional String. The role for the user account. The default value is org_user.
                          Other possible values are org_publisher, org_admin, org_viewer.
        ----------------  -------------------------------------------------------------------------------
        level             Optional integer. The account level. The default is 2.
                          See `User types, roles, and privileges
                          <http://server.arcgis.com/en/portal/latest/administer/linux/roles.htm>`_
                          for full details.
        ----------------  -------------------------------------------------------------------------------
        provider          Optional String. The provider for the account. The default value is arcgis.
                          The other possible value is enterprise.
        ----------------  -------------------------------------------------------------------------------
        must_approve      Optional boolean. After a user accepts the invite, if True, and administrator
                          must approve of the individual joining the organization. The default is False.
        ----------------  -------------------------------------------------------------------------------
        expiration        Optional String.  The default is '1 Day'. This is the time the emailed user has
                          to accept the invitation request until it expires.
                          The values are: 1 Day (default), 3 Days, 1 Week, or 2 Weeks.
        ----------------  -------------------------------------------------------------------------------
        validate_email    Optional boolean. If True (default) the Enterprise will ensure that the email
                          is properly formatted. If false, no check will occur
        ----------------  -------------------------------------------------------------------------------
        message_text      Optional String. Added to the message of the invitation and can provide further
                          instructions, a welcome, or any other personalized text to the person invited.
        ================  ===============================================================================

        :return:
            A boolean indicating success (True), or faliure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.users.invite("user1234@email.com", role=org_admin, provider=enterprise)

        """
        time_lookup = {
            "1 Day".upper(): 1440,
            "3 Days".upper(): 4320,
            "1 Week".upper(): 10080,
            "2 Weeks".upper(): 20160,
        }
        if expiration.upper() in time_lookup:
            expiration = time_lookup[expiration.upper()]
        elif not isinstance(expiration, int):
            raise ValueError("Invalid expiration.")

        url = self._portal.resturl + "/portals/self/inviteByEmail"
        msg = "You have been invited you to join an ArcGIS Organization, %s. " % (
            self._gis.properties["name"]
        )
        if message_text:
            msg = msg + "{text}".format(text=message_text)
        params = {
            "f": "json",
            "emails": email,
            "message": msg,
            "role": role,
            "level": level,
            "targetUserProvider": provider or "arcgis",
            "mustApprove": must_approve,
            "expiration": expiration,
            "validateEmail": validate_email,
        }
        res = self._portal.con.post(url, params)
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    @property
    def invitations(self):
        """
        The ``invitations`` property provides access to invitations sent to users using the
        :attr:`~arcgis.gis.UserManager.invite` method.

        **Note** : this is only supported by ArcGIS Online

        :return:
         An :class:`~arcgis.gis.InvitationManager` object

        """

        if self._gis._portal.is_arcgisonline == False:
            raise Exception("This property is only for ArcGIS Online.")
        from ._impl._invitations import InvitationManager

        url = self._portal.resturl + "portals/self/invitations"
        return InvitationManager(url, gis=self._gis)

    # ----------------------------------------------------------------------
    def signup(self, username: str, password: str, fullname: str, email: str):
        """
        The ``signup`` method is used to create a new user account in an ArcGIS Enterprise deployment.

        .. note:
            This method only applies to ArcGIS Enterprise, not ArcGIS
            Online.  The ``signup`` method can be called anonymously, but
            keep in mind that self-signup can also be disabled
            in ArcGIS Enterprise.  It also only creates built-in
            accounts, it does not work with accounts coming from external identity providers
            such as Active Directory, LDAP, or SAML IDPs.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        username          Required string. The desired username, which must be unique in the Portal,
                          and at least 4 characters.
        ----------------  --------------------------------------------------------
        password          Required string. The passowrd, which must be at least 8 characters.
        ----------------  --------------------------------------------------------
        fullname          Required string. The full name of the user.
        ----------------  --------------------------------------------------------
        email             Required string. The email address for the user. This is important to have correct.
        ================  ========================================================

        :return:
            The :class:`~arcgis.gis.User` object if successfully created, None if unsuccessful.

        .. code-block:: python

            # Usage Example

            >>> gis.users.signup("User1234", "password1234","User User", "User1234@email.com")

        """
        success = self._portal.signup(username, password, fullname, email)
        if success:
            return User(self._gis, username)
        else:
            return None

    # ----------------------------------------------------------------------
    def get(self, username: str):
        """
        The ``get`` method retrieves the :class:`~arcgis.gis.User` object for the specified username.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        username               Required string. The user to get as a string. This can be the
                               user's login name or the user's ID.
        ==================     ====================================================================

        :return:
            The :class:`~arcgis.gis.User` object if successfully found, None if unsuccessful.

        .. code-block:: python

            # Usage Example

            >>> gis.users.get("User1234")

        """
        try:
            with _common_utils._DisableLogger():
                user = self._portal.get_user(username)

        except RuntimeError as re:
            if re.args[0].__contains__("User does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as e:
            if e.args[0].__contains__("User does not exist or is inaccessible"):
                return None
            else:
                raise e
        if user is not None:
            return User(self._gis, user["username"], user)
        return None

    # ----------------------------------------------------------------------
    def enable_users(self, users: Union[list[str], list[User]]):
        """
        Thie ``enable_users`` method is a bulk operation that allows administrators to quickly enable large number of
        users in a single call.  It is useful to do this operation if you have multiple users that need
        to be enabled. ``enable_users`` is quite similar to the
        :class:`~arcgis.gis.UserManager.disable_users` method, which disables rather than enables users.

        .. note::
            The ``enable_users`` method is supported on ArcGIS REST API 6.4+.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List. List of :class:`user <arcgis.gis.User>` or UserNames to enable
        ==================     ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.users.enable_users(['user1','user1234','user123', 'user1234'])


        """
        if self._gis.version >= [6, 4]:
            url = "{base}/portals/self/enableUsers".format(base=self._portal.resturl)
            params = {"f": "json", "users": None}
            if isinstance(users, User) or isinstance(users, str):
                users = [users]
            if isinstance(users, (list, tuple)):
                ul = []
                for user in users:
                    if isinstance(user, User):
                        ul.append(user.username)
                    else:
                        ul.append(user)
                results = []
                for chunk in _common_utils.chunks(ul, n=25):
                    params["users"] = ",".join(chunk)
                    res = self._portal.con.post(url, params)
                    results.extend([r["status"] for r in res["results"]])
                return any(results)
            else:
                raise ValueError("Invalid input: must be of type list.")
        return False

    # ----------------------------------------------------------------------
    def disable_users(self, users: Union[list[str], list[User]]):
        """
        The ``disable_users`` method is a bulk disables user operation that allows administrators to quickly disable
        large number of users in a single call.  It is useful to do this operation if you have multiple
        users that need to be disabled. ``disable_users`` is quite similar to the
        :class:`~arcgis.gis.UserManager.enable_users` method, which enables rather than disables users.

        .. note::
            The ``disable_users`` method is supported on ArcGIS REST API 6.4+.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List. List of :class:`user <arcgis.gis.User>` or UserNames to disable
        ==================     ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.users.disable_users(['user1','user1234', 'user123', 'user1234'])

        """
        if self._gis.version >= [6, 4]:
            url = "{base}/portals/self/disableUsers".format(base=self._portal.resturl)
            params = {"f": "json", "users": None}
            if isinstance(users, User) or isinstance(users, str):
                users = [users]
            if isinstance(users, (list, tuple)):
                ul = []
                for user in users:
                    if isinstance(user, User):
                        ul.append(user.username)
                    else:
                        ul.append(user)
                results = []
                for chunk in _common_utils.chunks(ul, n=25):
                    params["users"] = ",".join(chunk)
                    res = self._portal.con.post(url, params)
                    results.extend([r["status"] for r in res["results"]])
                return any(results)

            else:
                raise ValueError("Invalid input: must be of type list.")
        return False

    # ----------------------------------------------------------------------
    def assign_categories(self, users: list[User], categories: list[str]) -> list:
        """Adds categories to :class:`users <arcgis.gis.User>`.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required list of :class:`~arcgis.gis.User` objects to categorize.
        ------------------     --------------------------------------------------------------------
        categories             Required string defining the categories to add to each user in the
                               `users` argument list.
        ==================     ====================================================================
        """
        results = []
        # ensure /Categories is at the start of each string.
        categories = [
            cat if cat.lower().find("/categories") > -1 else f"/Categories/{cat}"
            for cat in categories
        ]
        for user in users:
            results.append({user.username: user.update(categories=categories)})
        return results

    # ----------------------------------------------------------------------
    @property
    def categories(self) -> dict:
        """
        Provides means to get or set categories for members of an organization.
        See `Categorize members <https://doc.arcgis.com/en/arcgis-online/administer/manage-members.htm#ESRI_SECTION1_91337F478F8542D9A6D2F1A7B65E0AFF>`_
        or `Assign Member Category Schema description <https://developers.arcgis.com/rest/users-groups-and-items/assign-member-category-schema.htm>`_
        for additional details.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Required List of strings or dictionary defining the categories to
                               create for assigning to organizational members. If `None` is given,
                               the category schema will be erased.
        ==================     ====================================================================

        :returns: list

        .. code-block:: python

            # Usage example #1: Setting member categories with a list

            >>> gis.users.categories = ["Office Location"]

            # Usage example #2: Setting member categories with dictionary

            >>> category_dict = {"memberCategorySchema": [
                                                           {"title": "Categories",
                                                           "categories": [
                                                                          {"title": "Office Location"},
                                                                          {"title": "Department"}
                                                                         ]
                                                           }
                                                         ]
                                 }
            >>> gis.users.categories = category_dict

        """
        if dict(self._gis.properties).get("hasMemberCategorySchema", False):
            url = f"{self._gis._portal.resturl}portals/self/memberCategorySchema"
            params = {"f": "json"}
            return self._gis._con.get(url, params).get("memberCategorySchema", [])
        return None

    # ----------------------------------------------------------------------
    @categories.setter
    def categories(self, value: list):
        """
        Provides means to get or set categories for members of an organization.
        See `Categorize members <https://doc.arcgis.com/en/arcgis-online/administer/manage-members.htm#ESRI_SECTION1_91337F478F8542D9A6D2F1A7B65E0AFF>`_
        or `Assign Member Category Schema description <https://developers.arcgis.com/rest/users-groups-and-items/assign-member-category-schema.htm>`_
        for additional details.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Required List of strings naming the categories to create for
                               assigning to organizational members. If `None` is given, the
                               categories will be erased.
        ==================     ====================================================================

        :returns: list

        """
        if self._gis.version < [10, 1]:
            return
        if value is None and dict(self._gis.properties).get(
            "hasMemberCategorySchema", False
        ):
            url = f"{self._gis._portal.resturl}portals/self/deleteMemberCategorySchema"
            params = {"f": "json"}
            res = self._gis._con.post(url, params)
            if res.get("success", False) == False:
                raise Exception(res)
            else:
                self._gis._properties = None
        elif isinstance(value, (tuple, list)):
            url = f"{self._gis._portal.resturl}portals/self/assignMemberCategorySchema"
            cat_param = []
            for category in value:
                if isinstance(value, str):
                    cat_param.append({"title": category})
                else:
                    cat_param.append(category)
            if self._gis.properties.hasMemberCategorySchema:
                for category in self._gis.users.categories[0]["categories"]:
                    cat_param.append(category)
            params = {
                "f": "json",
                "memberCategorySchema": {
                    "memberCategorySchema": [
                        {
                            "title": "Categories",
                            "categories": cat_param,
                        }
                    ]
                },
            }
            res = self._gis._con.post(url, params)
            if res.get("success", False) == False:
                raise Exception(res)
            else:
                self._gis._properties = None
        elif isinstance(value, dict) and "memberCategorySchema" in value:
            url = f"{self._gis._portal.resturl}portals/self/assignMemberCategorySchema"
            if self._gis.properties.hasMemberCategorySchema:
                for category in self._gis.users.categories[0]["categories"]:
                    value["memberCategorySchema"][0]["categories"].append(category)
            params = {"f": "json", "memberCategorySchema": value}
            res = self._gis._con.post(url, params)
            if res.get("success", False) == False:
                raise Exception(res)
            else:
                self._gis._properties = None
        else:
            raise ValueError("A list or tuple must be given to set the categories.")

    # ----------------------------------------------------------------------
    def advanced_search(
        self,
        query: str,
        return_count: bool = False,
        max_users: int = 10,
        start: int = 1,
        sort_field: str = "username",
        sort_order: str = "asc",
        as_dict: bool = False,
    ):
        """
        The ``advanced_search`` method allows for the full control of the query operations
        by any given user.  The searches are performed against a high performance
        index that indexes the most popular fields of an user. See the
        `Search reference page <https://developers.arcgis.com/web-scene-specification/objects/search/>`_ for information
        on the fields and the syntax of the query. The ``advanced_search`` method is
        quite similar to the :attr:`~arcgis.gis.UserManager.search` method, which is less refined.

        The search index is updated whenever users is added, updated, or deleted. There
        can be a lag between the time that the user is updated and the time when it's
        reflected in the search results.

        .. note::
            The results of a search only contain items that the user has permission to access.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        query                  Required String.  The search query. When the search filters
                               contain two or more clauses, the recommended schema is to have
                               clauses separated by blank, or `AND`, e.g.

                               :Usage Example:


                               gis.users.advanced_search(query='owner:USERNAME type:map')
                               # or
                               gis.users.advanced_search(query='type:map AND owner:USERNAME')

                               .. warning::
                               When the clauses are separated by comma, the filtering condition
                               for `owner` should not be placed at the first position, e.g.
                               `gis.users.advanced_search(query='type:map, owner:USERNAME')`
                               is allowed, while
                               `gis.users.advanced_search(query='owner:USERNAME, type:map')`
                               is not. For more, please check
                               https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm
        ------------------     --------------------------------------------------------------------
        return_count           Optional Boolean.  If True, the number of users found by the query
                               string is returned.
        ------------------     --------------------------------------------------------------------
        max_users              Optional Integer. Limits the total number of users returned in a
                               a query.  The default is `10` users.  If all users is needed, `-1`
                               should be used.
        ------------------     --------------------------------------------------------------------
        start                  Optional Int. The starting position to search from.  This is
                               only required if paging is needed.
        ------------------     --------------------------------------------------------------------
        sort_field             Optional String. Responses from the `search` operation can be
                               sorted on various fields. `username` is the default.
        ------------------     --------------------------------------------------------------------
        sort_order             Optional String. The sequence into which a collection of
                               records are arranged after they have been sorted. The allowed
                               values are: asc for ascending and desc for descending.
        ------------------     --------------------------------------------------------------------
        as_dict                Required Boolean. If True, the results comes back as a dictionary.
                               The result of the method will always be a dictionary but the
                               `results` key in the dictionary will be changed if set to False.
        ==================     ====================================================================

        :return:
            A dictionary if `return_count` is False, else an integer

        .. code-block:: python

            # Usage Example

            >>> gis.users.advanced_search(query ="1234", sort_field = "username", max_users=20, as_dict=False)
        """
        from arcgis.gis._impl import _search

        stype = "users"
        max_items = max_users
        group_id = None
        if max_items == -1:
            max_items = _search(
                gis=self._gis,
                query=query,
                stype=stype,
                max_items=0,
                start=start,
                sort_field=sort_field,
                sort_order=sort_order,
                group_id=group_id,
                as_dict=as_dict,
            )["total"]
        so = {
            "DESC": "DESC",
            "ASC": "ASC",
            "asc": "asc".upper(),
            "desc": "desc".upper(),
            "ascending": "asc".upper(),
            "descending": "desc".upper(),
        }
        if sort_order:
            sort_order = so[sort_order].upper()

        if return_count:
            max_items = 0
        if max_items <= 10:
            res = _search(
                gis=self._gis,
                query=query,
                stype=stype,
                max_items=max_items,
                start=start,
                sort_field=sort_field,
                sort_order=sort_order,
                group_id=group_id,
                as_dict=as_dict,
            )
            if "total" in res and return_count:
                return res["total"]
            elif "aggregations" in res:
                return res["aggregations"]
            return res
        else:
            allowed_keys = [
                "query",
                "return_count",
                "max_users",
                "bbox",
                "categories",
                "category_filter",
                "start",
                "sort_field",
                "sort_order",
                "count_fields",
                "count_size",
                "as_dict",
            ]
            inputs = locals()
            kwargs = {}
            for k, v in inputs.items():
                if k in allowed_keys:
                    kwargs[k] = v
            import concurrent.futures
            import math, copy

            num = 10
            steps = range(math.ceil(max_items / num))
            params = []
            for step in steps:
                new_start = start + num * step
                kwargs["max_users"] = num
                kwargs["start"] = new_start
                params.append(copy.deepcopy(kwargs))
            items = {
                "results": [],
                "start": start,
                "num": 10,
                "total": max_items,
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {
                    executor.submit(self.advanced_search, **param): param
                    for param in params
                }
                for future in future_to_url.keys():  # preserves order of query
                    result = future_to_url[future]
                    data = future.result()
                    if "results" in data:
                        items["results"].extend(data["results"])
            if len(items["results"]) > max_items:
                items["results"] = items["results"][:max_items]
            return items
        return None

    def org_search(
        self,
        query: str = None,
        sort_field: str = None,
        sort_order: str = None,
        as_dict: bool = False,
        exclude: bool = False,
    ) -> tuple[User] | tuple[dict[str, Any]]:
        """
        The `org_search` method allows users to find users within the organization only.
        Users can search for details such as `provider`, `fullName` and other user properties
        where the other user searches are limited.  This operation will not show any user outside
        the organization.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        query             Optional string. The query string.  See notes above. Pass None
                          to get list of all users in the organization.
        ----------------  --------------------------------------------------------
        sort_field        Optional string. Valid values can be username (the default) or created.
        ----------------  --------------------------------------------------------
        sort_order        Optional string. Valid values are asc (the default) or desc.
        ----------------  --------------------------------------------------------
        as_dict           Optional Boolean. Returns the raw response for each user as a dictionary
        ----------------  --------------------------------------------------------
        exclude           Optional Boolean. If `True`, the system accounts will be excluded from the query.
        ================  ========================================================

        :returns: Tuple[User] | Tuple[dict[str,Any]]
        """
        results = []

        if query is None:
            query = "*"
        if exclude:
            query = f"-username:esri_livingatlas -username:esri_boundaries -username:esri_demographics -username:esri_nav ({query})"
        count = self.advanced_search(query, return_count=True)

        url = f"{self._gis._portal.resturl}/portals/self/users/search"
        num = 100
        params = {
            "num": num,
            "f": "json",
            "q": query,
            "start": 1,
            "sortField": sort_field or "",
            "sortOrder": sort_order or "",
        }
        if count <= num:
            resp = self._gis._con.get(url, params)
            results.extend(resp.get("results", []))
            while resp.get("nextStart", -1) > 0:
                params["start"] = resp["nextStart"]
                resp = self._gis._con.get(url, params)
                users = resp.get("results", [])
                results.extend(users)
                if len(users) == 0:
                    break
        else:  # use multiple threads to capture the data.
            iterations = (count // num) + int((count % num > 0))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_users = {}
                for i in range(iterations):
                    params["start"] = 1 + i * params["num"]

                    future_users[
                        executor.submit(
                            self._gis._con.get,
                            **{"path": url, "params": params},
                        )
                    ] = i
                for future in concurrent.futures.as_completed(future_users):
                    users = future.result().get("results", [])
                    results.extend(users)

        if as_dict:
            return tuple(results)
        else:
            return tuple(
                User(gis=self._gis, username=user["username"], userdict=user)
                for user in results
            )

    # ----------------------------------------------------------------------
    def search(
        self,
        query: Optional[str] = None,
        sort_field: str = "username",
        sort_order: str = "asc",
        max_users: int = 100,
        outside_org: bool = False,
        exclude_system: bool = False,
        user_type: Optional[str] = None,
        role: Optional[str] = None,
    ):
        """
        The ``search`` method searches portal users, returning a list of users matching the specified query.
        The ``search`` method is quite similar to the :attr:`~arcgis.gis.UserManager.advanced_search` method,
        which is more refined.

        .. note::
            A few things that will be helpful to know.

            1. The query syntax has quite a few features that can't
               be adequately described here.  Please refer to the ArcGIS REST
               API `Search Reference <https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm>`_
               for details on the search engine used with this method.

            2. Searching without specifying a query parameter returns
               a list of all users in your organization.

            3. Most of the time when searching users you want to
               search within your organization in ArcGIS Online
               or within your Portal.  As a convenience, the method
               automatically appends your organization id to the query by
               default.  If you don't want the API to append to your query
               set outside_org to True.  If you use this feature with an
               OR clause such as field=x or field=y you should put this
               into parenthesis when using outside_org.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        query             Optional string. The query string.  See notes above. Pass None
                          to get list of all users in the organization.
        ----------------  --------------------------------------------------------
        sort_field        Optional string. Valid values can be username (the default) or created.
        ----------------  --------------------------------------------------------
        sort_order        Optional string. Valid values are asc (the default) or desc.
        ----------------  --------------------------------------------------------
        max_users         Optional integer. The maximum number of users to be returned. The default is 100.
        ----------------  --------------------------------------------------------
        outside_org       Optional boolean. This controls whether to search outside
                          your organization. The default is False (search only
                          within your organization).
        ----------------  --------------------------------------------------------
        exclude_system    Optional boolean. Controls if built-in system accounts are
                          returned or not.  True means built-in account are not
                          returned, where as False means that they are.
        ----------------  --------------------------------------------------------
        user_type         Optional String. This parameters allows for the filtering
                          of the users by their assigned type.
        ----------------  --------------------------------------------------------
        role              Optional String.  Specify the roleId. This parameter
                          allows for the filting of the users based on a roleId.
        ================  ========================================================

        :return:
            A list of :class:`~arcgis.gis.User` objects that fit the query parameters.

        .. code-block:: python

            # Usage Example

            >>> gis.users.search(query ="1234", sort_field = "username", max_users=20)
        """
        ut = {"creator": "creatorUT", "viewer": "viewerUT"}
        if user_type and user_type.lower() in ut:
            user_type = ut[user_type.lower()]
        if query is None:
            users = self._portal.get_org_users(
                max_users,
                exclude_system=json.dumps(exclude_system),
                user_type=user_type,
                role=role,
            )
            gis = self._gis
            user_storage = []
            for u in users:
                if "id" in u and (u["id"] is None or u["id"] == "null"):
                    un = u["username"]
                elif "id" not in u:
                    un = u["username"]
                else:
                    un = u["id"]
                if not "roleId" in u:
                    u["roleId"] = u.pop("role", None)
                user_storage.append(User(gis, un, u))
            return user_storage
        else:
            userlist = []
            users = self._portal.search_users(
                query,
                sort_field,
                sort_order,
                max_users,
                outside_org,
                json.dumps(exclude_system),
                user_type=user_type,
                role=role,
            )
            for user in users:
                if "id" in user and (user["id"] is None or user["id"] == "null"):
                    un = user["username"]
                elif self._gis.version <= [6, 4]:
                    un = user["username"]
                elif "id" not in user:
                    un = user["username"]
                else:
                    un = user["id"]
                userlist.append(User(self._gis, un))
            return userlist

    # ----------------------------------------------------------------------
    @property
    def me(self):
        """
        The ``me`` property retrieves the information of the logged in :class:`~arcgis.gis.User` object.
        """

        if self._me is None:
            meuser = self._portal.logged_in_user()
            if meuser is not None:
                self._me = User(self._gis, meuser["username"], meuser)
            else:
                self._me = None
        return self._me

    # ----------------------------------------------------------------------
    @_lazy_property
    def roles(self):
        """
        The ``roles`` property is a helper object to manage custom roles for users.

        :return:
            An instance of the :class:`~arcgis.gis.RoleManager`
        """
        return RoleManager(self._gis)

    # ----------------------------------------------------------------------
    def user_groups(self, users: Union[list[str], list[User]], max_results: int = -1):
        """
        Givens a List of Users, the ``user_groups`` method will report back all group ids
        that each :class:`~arcgis.gis.User` belongs to. ``user_groups`` is designed to be a reporting
        tool for administrators so they can easily manage a user or users groups.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        users             Required List. An array of User objects or usernames.
        ----------------  --------------------------------------------------------
        max_results       Optional Integer. A limiter on the number of groups
                          returned for each user.
        ================  ========================================================

        :return:
            List of dictionaries with each :class:`~arcgis.gis.User` object's group ids.

        .. code-block:: python

            # Usage Example

            >>> gis.users.user_groups([user1,user1234,user123, user1234], 50)

        """
        if max_results == -1 or max_results is None:
            max_results = None
        else:
            max_results = int(max_results)

        us = []
        for user in users:
            if isinstance(user, User):
                us.append(user.username)
            else:
                us.append(user)
        params = {"f": "json", "users": ",".join(us), "limit": max_results}
        url = "{base}/portals/self/usersGroups".format(base=self._portal.resturl)
        res = res = self._portal.con.get(url, params)
        if "results" in res:
            return res["results"]
        return res


class RoleManager(object):

    """
        The ``RoleManager`` class is a helper class to manage custom :class:`roles <arcgis.gis.Role>` for
        :class:`~arcgis.gis.User` in a GIS. It is available as the :attr:`~arcgis.gis.UserManager.roles`
        property of the :class:`~arcgis.gis.UserManager`

        .. note::
            Users don't create this class directly.

    .. code-block:: python

         # Usage Example

         >>> role_mgr = gis.users.roles
         >>> type(role_mgr)

         <class 'arcgis.gis.RoleManager'>
    """

    def __init__(self, gis):
        """Creates helper object to manage custom roles in the GIS"""
        self._gis = gis
        self._portal = gis._portal

    def clone(self, roles: list[Role]) -> list[_cloner.CloningJob]:
        """
        Clones a list of Roles from one organization to another

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        roles                  Required list[Role]. An array of roles from the source GIS.
        ==================     ====================================================================

        :returns: list[Future]
        """
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as tp:
            for role in roles:
                role: Role
                future: concurrent.futures.Future = tp.submit(
                    self.create,
                    **{
                        "name": role.name,
                        "description": role.description,
                        "privileges": role.privileges,
                    },
                )
                jobs.append(_cloner.CloningJob(future, "Role"))
            tp.shutdown(wait=True)
        return jobs

    def create(self, name: str, description: str, privileges: Optional[str] = None):
        """
            The ``create`` method creates a custom :class:`~arcgis.gis.Role` with the specified parameters.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The custom role's name.
        ------------------     --------------------------------------------------------------------
        description            Required string. The custom role's description.
        ------------------     --------------------------------------------------------------------
        privileges             Optional string. An array of strings with predefined permissions within
                               each privilege.  For supported privileges see the
                               `Privileges <https://developers.arcgis.com/rest/users-groups-and-items/privileges.htm>`_
                               page in the ArcGIS REST API documentation.
        ==================     ====================================================================

        :return:
           The custom :class:`role <arcgis.gis.Role>` object if successfully created, None if unsuccessful.

        .. code-block:: python

            # Usage Example
            >>> role_mgr = gis.users.roles
            >>> role_mgr.create("name_role", "role_description")
        """
        if self.exists(role_name=name) == False:
            role_id = self._portal.create_role(name, description)
            if role_id is not None:
                role_data = {
                    "id": role_id,
                    "name": name,
                    "description": description,
                }
                role = Role(self._gis, role_id, role_data)
                role.privileges = privileges
                return role
            else:
                return None
        else:
            n = str(name.lower())
            roles = [r for r in self.all() if r.name.lower() == n]
            return roles[0]
        return None

    def exists(self, role_name: str):
        """
        The ``exists`` method checks to see if a :class:`~arcgis.gis.Role` object exists given the declared role name.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        role_name              Required string. The name of the role to determine if it exists or not.
        ==================     ====================================================================

        :return:
           True if the :class:`role <arcgis.gis.Role>` exists, and False if it does not.

        .. code-block:: python

            # Usage Example

            >>> role_mgr = gis.users.roles
            >>> role_mgr.exists("name_role")
        """
        for role in self.all():
            if role.name.lower() == role_name.lower():
                return True
        return False

    def all(self, max_roles: int = 1000):
        """
        The ``all`` method provides a list containing the default ``Viewer`` and ``Data Editor`` roles, plus any
        custom roles defined in the :class:`~arcgis.gis.GIS`. (The ``org_admin``, ``org_user``,
        and ``org_publisher`` default roles are not returned. See `Default roles <https://enterprise.arcgis.com/en/portal/latest/administer/windows/roles.htm#ESRI_SECTION2_CB9BF0951AC647529EBB7CB09B8B3EDA>`_
        for detailed descriptions of each role.)

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        max_roles              Required integer. The maximum number of roles to be returned, defaults to 1000.
        ==================     ====================================================================

        :return:
           The list of all custom :class:`roles <arcgis.gis.Role>`, plus the default ``Viewer``
           and ``Data Editor`` roles defined in the GIS.

        .. code-block:: python
           :emphasize-lines: 6

            # Usage Example

            >>> primary_default_roles = ['org_admin', 'org_publisher', 'org_user']

            >>> role_mgr = gis.users.roles
            >>> org_roles = role_mgr.all()

            >>> for role in org_roles:
            >>>     print(f"{role.name:25}{role.role_id}")

                Viewer                   iAAAAAAAAAAAAAAA
                Data Editor              iBBBBBBBBBBBBBBB
                Analyzer                 8KqWobO1p1vDLZ2O
                Sharing_analyst          ZllNulU2kqaFwsaH
                Group_creator            uT3334C4LtnQ99Cj

            >>> all_org_roles = primary_default_roles + [r.name for r in org_roles]
            >>> print(all_org_roles)

                ['org_admin', 'org_publisher', 'org_user', 'Viewer', 'Data Editor', 'Analyzer', 'Sharing_analyst', 'Group_creator']

        See the :attr:`~arcgis.gis.UserManager.create` method of :class:`~arcgis.gis.UserManager` for using
        role information when creating users.
        """
        roles = self._portal.get_org_roles(max_roles)
        return [Role(self._gis, role["id"], role) for role in roles]

    def get_role(self, role_id: str):
        """
        The ``get_role`` method retrieves the :class:`~arcgis.gis.Role` object with the specified custom roleId.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        role_id                Required string. The role ID of the custom role to get.
        ==================     ====================================================================

        :return:
           The :class:`Role <arcgis.gis.Role>` object associated with the specified role ID
        """
        role = self._portal.con.post(
            "portals/self/roles/" + role_id, self._portal._postdata()
        )
        return Role(self._gis, role["id"], role)


class Role(object):
    """The ``Role`` class is used to represent a role in a GIS, either ArcGIS Online or ArcGIS Enterprise."""

    def __init__(self, gis, role_id, role):
        """Create a custom role"""
        self._gis = gis
        self._portal = gis._portal
        self.role_id = role_id
        if role is not None:
            self._name = role["name"]
            self._description = role["description"]

    def __repr__(self):
        return "<Role name: " + self.name + ", description: " + self.description + ">"

    def ___str___(self):
        return "Custom Role name: " + self.name + ", description: " + self.description

    @property
    def name(self):
        """The ``name`` method retrieves and sets the name of the custom role."""
        return self._name

    @name.setter
    def name(self, value):
        """Name of the custom role"""
        self._name = value
        self._update_role()

    @property
    def description(self):
        """The ``description`` method retrieves and sets the description of the custom role."""
        return self._description

    @description.setter
    def description(self, value: str):
        """Description of the custom role"""
        self._description = value
        self._update_role()

    def _update_role(self):
        """Updates the name or description of this role"""
        postdata = self._portal._postdata()
        postdata["name"] = self._name
        postdata["description"] = self._description

        resp = self._portal.con.post(
            "portals/self/roles/" + self.role_id + "/update", postdata
        )
        if resp:
            return resp.get("success")

    @property
    def privileges(self):
        """
        The ``privileges`` method retrieves and sets the privileges for the custom role as a list of strings.

        **Administrator Privileges**:

        *Members*

        =======================================      ========================================================================
        **Privilege**                                **Description**
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:viewUsers                       Grants the ability to view full member account information within
                                                     organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:updateUsers                     Grants the ability to update member account information within organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:deleteUsers                     Grants the ability to delete member accounts within organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:inviteUsers                     Grants the ability to invite members to organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:disableUsers                    Grants the ability to enable and disable member accounts within organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:changeUserRoles                 Grants the ability to change the role a member is assigned within
                                                     the organization. However, it does not grant the ability to promote
                                                     or demote a member from the Administrator role. That privilege is reserved
                                                     for the Administrator role alone.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:manageLicenses                  Grants the ability to assign licenses to members of organization.
        ---------------------------------------      ------------------------------------------------------------------------
        portal:admin:updateMemberCategorySchema      Grants the ability to configure categories for members.
        =======================================      ========================================================================

        *Groups*

        =====================================       ========================================================================
        **Privilege**                               **Description**
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:viewGroups                     Grants the ability to view all groups within organization.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:updateGroups                   Grants the ability to update groups within organization.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:deleteGroups                   Grants the ability to delete groups within organization.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:reassignGroups                 Grants the ability to reassign groups to other members within organization.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:assignToGroups                 Grants the ability to assign members to, and remove members from
                                                    groups within organization.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:manageEnterpriseGroups         Grants the ability to link group membership to an enterprise group.
        -------------------------------------       ------------------------------------------------------------------------
        portal:admin:createUpdateCapableGroup       Grants the ability to create a group with update capabilities.
        =====================================       ========================================================================


        *Content*

        =====================================     ========================================================================
        **Privilege**                             **Description**
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:viewItems                    Grants the ability to view all content within organization.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:updateItems                  Grants the ability to update content within organization.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:deleteItems                  Grants the ability to delete content within organization.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:reassignItems                Grants the ability to reassign content to other members within organization.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:shareToGroup                 Grants the ability to share other member's content to groups the user belongs to.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:shareToOrg                   Grants the ability to share other member's content to organization.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:shareToPublic                Grants the ability to share other member's content to all users of the portal.
        -------------------------------------     ------------------------------------------------------------------------
        portal:admin:updateItemCategorySchema     Grants the ability to create and update content categories in the organization.
        =====================================     ========================================================================

        *Webhooks* (*ArcGIS Enterprise* only)

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        portal:admin:manageWebhooks         Grant the ability to create, edit, delete and manage all webhooks within
                                            the organization.
        ===============================     ========================================================================

        *ArcGIS Marketplace Subscriptions* (*ArcGIS Online* only)

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        marketplace:admin:purchase          Grants the ability to request purchase information about apps and data
                                            in ArcGIS Marketplace.
        -------------------------------     ------------------------------------------------------------------------
        marketplace:admin:startTrial        Grants the ability to start trial subscriptions in ArcGIS Marketplace.
        -------------------------------     ------------------------------------------------------------------------
        marketplace:admin:manage            Grants the ability to create listings, list items and manage
                                            subscriptions in ArcGIS Marketplace.
        ===============================     ========================================================================

        *Organization Settings*

        ===================================     ========================================================================
        **Privilege**                           **Description**
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageSecurity             Grants ability to manage security and infrastructure settings
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageWebsite              Grants the ability to manage the website settings.
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageCollaborations       Grants the ability to administer the organization's collaborations.
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageCredits              Grants the ability to manage the organization's credit budget settings.
                                                (*ArcGIS Online* only)
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageServers              Grants the ability to manage the servers federated with the
                                                organization. (*ArcGIS Enterprise* only)
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageUtilityServices      Grants the ability to manage the utility services configured with the
                                                organization.
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:manageRoles                Grants the ability to manage the organization's roles.
        -----------------------------------     ------------------------------------------------------------------------
        portal:admin:createGPWebhook            Grants the ability to create, edit and delete their own
                                                geoprocessing webhook. (*ArcGIS Enterprise* only)
        ===================================     ========================================================================


        **Publisher Privileges:**

        *Content*

        ==========================================    =========================================================================
        **Privilege**                                 **Description**
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishFeatures              Grants the ability to publish hosted feature layers.
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishTiles                 Grants the ability to publish hosted tile layers.
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishScenes                Grants the ability to publish hosted scene layers.
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishServerServices        Grants the ability to publish non-hosted server services.
                                                      (*ArcGIS Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishServerGPServices      Grants the ability to publish non-hosted geoprocessing services
                                                      (*ArcGIS Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishKnowledgeGraph        Grants the ability to create and publish knowledge graphs.
                                                      (*ArcGIS Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:bulkPublishFromDataStores    Grants the ability to publish web layers from a registered data store.
                                                      (*ArcGIS Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:enumerateDataStores          Grants the ability to get list of datasets from a registered data store.
                                                      (*ArcGIS Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:pulisher:registerDataStores            Grants the ability to register data stores to the Enterprise. (*ArcGIS
                                                      Enterprise* only)
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishTiledImagery          Grants the ability to publish hosted tiled imagery layers from a single
                                                      image or collection of images.
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:publishDynamicImagery        Grants the ability to publish hosted dynamic imagery layers from a single
                                                      image or collection of images.
        ------------------------------------------    -------------------------------------------------------------------------
        premium:publisher:createNotebooks             Grants the ability to create and edit interactive notebooks items.
        ------------------------------------------    -------------------------------------------------------------------------
        premium:publisher:scheduleNotebooks           Grants the ability to schedule future automated runs of a notebook.
        ------------------------------------------    -------------------------------------------------------------------------
        portal:publisher:createDataPipelines          Grants the ability to create, edit and run data pipelines.
        ==========================================    =========================================================================

        *Premium Content*

        =========================================    =================================================================================
        **Privilege**                                **Description**
        -----------------------------------------    ---------------------------------------------------------------------------------
        premium:publisher:geoanalytics               Grants the ability to use big data analytics. ()
        -----------------------------------------    ---------------------------------------------------------------------------------
        premium:publisher:rasteranalysis             Grants the ability to use raster analystics.
        -----------------------------------------    ---------------------------------------------------------------------------------
        premium:publisher:createAdvancedNotebooks    Grants the ability to publish a notebook as a geoprocessing service.
                                                     (*ArcGIS Enterprise* only)
        =========================================    =================================================================================


        **User Privileges:**

        *Members*

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        portal:user:viewOrgUsers            Grants members to view other organization members.
        ===============================     ========================================================================

        *Groups*

        ===========================================    ========================================================================
        **Privilege**                                  **Description**
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:createGroup                        Grants the ability for a member to create, edit, and delete their own groups.
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:joinGroup                          Grants the ability to join groups within organization.
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:joinNonOrgGroup                    Grants the ability to join groups external to the organization.
                                                       (*ArcGIS Online* only)
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:viewOrgGroups                      Grants members the ability to view groups shared to the organization.
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:addExternalMembersToGroup          Grants the abitlity to create groups that allow external members,
                                                       as well as invite external members to groups (*ArcGIS Online* only)
        -------------------------------------------    ------------------------------------------------------------------------
        portal:user:manageCollaborationGroupMembers    Grants the ability to manage members in partnered collaboration groups.
                                                       (*ArcGIS Online* only)
        ===========================================    ========================================================================

        *Content*

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        portal:user:createItem              Grants the ability for a member to create, edit, and delete their own
                                            content.
        -------------------------------     ------------------------------------------------------------------------
        portal:user:viewTracks              Grants the ability to to view members' location tracks via shared track
                                            views when location sharing is enabled. (*ArcGIS Enterprise* only)
        -------------------------------     ------------------------------------------------------------------------
        portal:user:reassignItems           Grants users the ability to reassign their own content to other
                                            organization members with the receive items privilege.
        -------------------------------     ------------------------------------------------------------------------
        portal:user:receiveItems            Grants users the ability to receive items reassigned to them by other
                                            organization members with the reassign items privilege.
        -------------------------------     ------------------------------------------------------------------------
        portal:user:viewOrgItems            Grants members the ability to view content shared with the organization.
        ===============================     ========================================================================

        *Sharing*

        =================================     ========================================================================
        **Privilege**                         **Description**
        ---------------------------------     ------------------------------------------------------------------------
        portal:user:shareToGroup              Grants the ability to share content to groups.
        ---------------------------------     ------------------------------------------------------------------------
        portal:user:shareToOrg                Grants the ability to share content to organization.
        ---------------------------------     ------------------------------------------------------------------------
        portal:user:shareToPublic             Grants the ability to share content to all users of portal.
        ---------------------------------     ------------------------------------------------------------------------
        portal:user:shareGroupToOrg           Grants the ability to make groups discoverable by the organization.
        ---------------------------------     ------------------------------------------------------------------------
        portal:user:shareGroupToPublic        Grants the ability to make groups discoverable by all users of portal.
        =================================     ========================================================================

        *Premium Content*

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        premium:user:geocode                Grants the ability to perform large-volume geocoding tasks with the
                                            Esri World Geocoder such as publishing a CSV of addresses as hosted
                                            feature layer.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:networkanalysis        Grants the ability to perform network analysis tasks such as routing
                                            and drive-time areas.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:geoenrichment          Grants the ability to geoenrich features.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:demographics           Grants the ability to make use of premium demographic data.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:spatialanalysis        Grants the ability to perform spatial analysis tasks.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:elevation              Grants the ability to perform analytical tasks on elevation data.
        -------------------------------     ------------------------------------------------------------------------
        premium:user:featurereport          Grants the ability to create feature reports. (*ArcGIS Online* only)
        ===============================     ========================================================================

        *Features*

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        features:user:edit                  Grants the ability to edit features in editable layers, according to the
                                            edit options enabled on the layer.
        -------------------------------     ------------------------------------------------------------------------
        features:user:fullEdit              Grants the ability to add, delete, and update features in a hosted
                                            feature layer regardless of the editing options enabled on the layer.
        ===============================     ========================================================================

        *Version Management* (*ArcGIS Enterprise* only)

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        features:user:manageVersions        Grant the ability to manage version locks and view, alter, delete, edit,
                                            reconcile, and post to all branch versions accessed through
                                            ArcGIS Server feature layers.
        ===============================     ========================================================================

        *Open Data* (*ArcGIS Online* only)

        ===============================     ========================================================================
        **Privilege**                       **Description**
        -------------------------------     ------------------------------------------------------------------------
        opendata:user:openDataAdmin         Grants the ability to manage Open Data Sites for the organization.
        -------------------------------     ------------------------------------------------------------------------
        opendata:user:designateGroup        Grants the ability to designate groups within organization as being
                                            available for use in Open Data.
        ===============================     ========================================================================

        """
        resp = self._portal.con.post(
            "portals/self/roles/" + self.role_id + "/privileges",
            self._portal._postdata(),
        )
        if resp:
            return resp.get("privileges")
        else:
            return None

    @privileges.setter
    def privileges(self, value):
        """Privileges for the custom role as a list of strings"""
        postdata = self._portal._postdata()
        if isinstance(value, list):
            postdata["privileges"] = json.dumps({"privileges": value})
        elif value and isinstance(value, str) and len(value) == 0:
            postdata["privileges"] = {"privileges": []}
        elif value and isinstance(value, str):
            postdata["privileges"] = {"privileges": value}
        elif value is None:
            value = []
            postdata["privileges"] = json.dumps({"privileges": value})
        resp = self._portal.con.post(
            "portals/self/roles/" + self.role_id + "/setPrivileges", postdata
        )
        if len(self.privileges) != len(value):
            postdata["privileges"] = json.dumps(postdata["privileges"])
            resp = self._portal.con.post(
                "portals/self/roles/" + self.role_id + "/setPrivileges",
                postdata,
            )
        if resp:
            return resp.get("success")

    def delete(self):
        """
        The ``delete`` method is called to deletes the current role.

        .. code-block:: python

            # Usage Example

            >>> role.delete()

        :return:
           A boolean indicating success (True) or failure (False).
        """
        resp = self._portal.con.post(
            "portals/self/roles/" + self.role_id + "/delete",
            self._portal._postdata(),
        )
        if resp:
            return resp.get("success")


class GroupManager(object):
    """
    The ``GroupManager`` class is a helper class for managing GIS groups.
    An instance of this class, called :attr:`~arcgis.gis.GIS.groups`, is available
    as a property of the :class:`~arcgis.gis.GIS` object.

    .. note::
        This class is not created by users directly.
    """

    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal
        self._cloner = _cloner.GroupCloner(gis=self._gis)

    #  --------------------------------------------------------------------
    def load_offline_configuration(
        self, package: str
    ) -> list[concurrent.futures.Future]:
        """
        Loads the UX configuration file into the current active portal.

        ====================  =========================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------
        package               Required String. The GROUP_CLONER file that contains the offline information.
        ====================  =========================================================

        :returns: concurrent.futures.Future

        .. code-block:: python

            # Usage Example
            >>> package = r"/home/groups.GROUP_CLONER"
            >>> job = gis_destination.groups.load_offline_configuration(package)
            >>> job.result()
            [<Group>]

        """
        return self._cloner.load_offline_configuration(package)

    def clone(
        self,
        groups: list[Group],
        *,
        skip_existing: bool = True,
        offline: bool = False,
        save_folder: str | None = None,
        file_name: str | None = "GROUP_CLONER",
    ) -> list[_cloner.CloningJob]:
        """
        The group cloner will recreate groups from site A to site B.
        It will not clone the group's items.  This should be done using the `clone_items`
        or group item migrator tools.

        ====================  =========================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------
        groups                Required list[Group]. A list of Group objects to clone.
        --------------------  ---------------------------------------------------------
        skip_existing         Optional bool. If True, if a group exists, it will be skipped.
        --------------------  ---------------------------------------------------------
        offline               Optional bool. If True, a file will be saved locally that can be imported at a later date.
        --------------------  ---------------------------------------------------------
        save_folder           Optional str. The save path of the offline package.
        --------------------  ---------------------------------------------------------
        file_name             Optional str. The name of the file without an extension.
        ====================  =========================================================

        :returns: list[CloningJob]

        .. code-block:: python

            # Usage Example
            >>> group = gis_source.groups.create(title = "New Group",
                                  tags = "new, group, USA",
                                  description = "a new group in the USA",
                                  access = "public")
            >>> jobs = gis_destination.groups.clone([group])
            >>> [job.result() for job in jobs]
            [<Group>]

        .. code-block:: python

            # Usage Example 2
            >>> group = gis_source.groups.create(title = "New Group",
                                  tags = "new, group, USA",
                                  description = "a new group in the USA",
                                  access = "public")
            >>> job = gis_destination.groups.clone([group], offline=True, save_folder=r"c:\storage", file_name="groups)
            >>> job.result()
            c:\storage\groups.GROUP_CLONER

        """
        return self._cloner.clone(
            groups=groups,
            skip_existing=skip_existing,
            offline=offline,
            save_folder=save_folder,
            file_name=file_name,
        )

    def create(
        self,
        title: str,
        tags: Union[list[str], str],
        description: Optional[str] = None,
        snippet: Optional[str] = None,
        access: str = "public",
        thumbnail: Optional[str] = None,
        is_invitation_only: bool = False,
        sort_field: str = "avgRating",
        sort_order: str = "desc",
        is_view_only: bool = False,
        auto_join: bool = False,
        provider_group_name: Optional[str] = None,
        provider: Optional[str] = None,
        max_file_size: Optional[int] = None,
        users_update_items: bool = False,
        display_settings: Optional[str] = None,
        is_open_data: bool = False,
        leaving_disallowed: bool = False,
        hidden_members: bool = False,
        membership_access: Optional[str] = None,
        autojoin: bool = False,
    ):
        """
        The ``create`` method creates a group with the values for any particular
        arguments that are specified. The user who creates the group automatically
        becomes the owner of the group, and the owner automatically becomes an
        administrator. Use :attr:`~arcgis.gis.Group.reassign_to` to change the
        owner.

        .. note::
            Only title and tags are required.


        ====================  =========================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------
        title                 Required string. The name of the group.
        --------------------  ---------------------------------------------------------
        tags                  Required string. A comma-delimited list of tags, or
                              list of tags as strings.
        --------------------  ---------------------------------------------------------
        description           Optional string. A detailed description of the group.
        --------------------  ---------------------------------------------------------
        snippet               Optional string.  A short snippet (<250 characters)
                              that summarizes the group.
        --------------------  ---------------------------------------------------------
        access                Optional string. Choices are private, public, or org.
        --------------------  ---------------------------------------------------------
        thumbnail             Optional string. URL or file location to a group image.
        --------------------  ---------------------------------------------------------
        is_invitation_only    Optional boolean. Defines whether users can join by
                              request. Default is False meaning users can ask to join
                              by request or join by invitation.
        --------------------  ---------------------------------------------------------
        sort_field            Optional string. Specifies how shared items with
                              the group are sorted.
        --------------------  ---------------------------------------------------------
        sort_order            Optional string.  Choices are asc or desc for ascending
                              or descending, respectively.
        --------------------  ---------------------------------------------------------
        is_view_only          Optional boolean. Defines whether the group is searchable.
                              Default is False meaning the group is searchable.
        --------------------  ---------------------------------------------------------
        auto_join             Optional boolean. Only applies to org accounts. If True,
                              this group will allow joining without requesting
                              membership approval. Default is False.
        --------------------  ---------------------------------------------------------
        provider_group_name   Optional string. The name of the domain group.
                              Create an association between a Portal group and an
                              Active Directory or LDAP group.
        --------------------  ---------------------------------------------------------
        provider              Optional string. Name of the provider. Required if the
                              parameter `provider_group_name` is specified. Example of
                              use: provider_group_name = "groupNameTest", provider = "enterprise"
        --------------------  ---------------------------------------------------------
        max_file_size         Optional integer.  This is the maximum file size allowed
                              be uploaded/shared to a group. Default value is: 1024000
        --------------------  ---------------------------------------------------------
        users_update_items    Optional boolean.  Members can update all items in this
                              group.  Updates to an item can include changes to the
                              item's description, tags, metadata, as well as content.
                              This option can't be disabled once the group has
                              been created. Default is False.
        --------------------  ---------------------------------------------------------
        display_settings      Optional String. Defines the default display for the
                              group page to show a certain type of items. The allowed
                              values are: `apps, all, files, maps, layers, scenes, tools`.
                              The default value is `all`.
        --------------------  ---------------------------------------------------------
        is_open_data          Optional Boolean. Defines whether the group can be used
                              in the Open Data capabilities of ArcGIS Hub. The default
                              is False.
        --------------------  ---------------------------------------------------------
        leaving_disallowed    Optional boolean. Defines whether users are restricted
                              from choosing to leave the group. If True, only an
                              administrator can remove them from the group. The default
                              is False.
        --------------------  ---------------------------------------------------------
        hidden_members        Optional Boolean. Only applies to org accounts. If true,
                              only the group owner, group managers, and default
                              administrators can see all members of the group.
        --------------------  ---------------------------------------------------------
        membership_access     Optional String. Sets the membership access for the group.
                              Setting to `org` restricts group access to members of
                              your organization. Setting to `collaboration` restricts the
                              membership access to partnered collaboration and your
                              organization members. If `None` set, any organization
                              will have access. `None` is the default.

                              Values: `org`, `collaboration`, or `none`

                              .. note::
                                For Enterprise only "org" is accepted.
        --------------------  ---------------------------------------------------------
        autojoin              Optional Boolean. The default is `False`. Only applies to
                              org accounts. If `True`, this group will allow joined
                              without requesting membership approval.
        ====================  =========================================================

        :return:
            The :class:`~arcgis.gis.Group` if successfully created, None if unsuccessful.

        .. code-block:: python

            # Usage Example
            >>> gis.groups.create(title = "New Group",
                                  tags = "new, group, USA",
                                  description = "a new group in the USA",
                                  access = "public")
        """
        display_settings_lu = {
            "apps": {"itemTypes": "Application"},
            "all": {"itemTypes": ""},
            "files": {"itemTypes": "CSV"},
            None: {"itemTypes": ""},
            "none": {"itemTypes": ""},
            "maps": {"itemTypes": "Web Map"},
            "layers": {"itemTypes": "Layer"},
            "scenes": {"itemTypes": "Web Scene"},
            "tools": {"itemTypes": "Locator Package"},
        }
        if max_file_size is None:
            max_file_size = 1024000
        if users_update_items is None:
            users_update_items = False

        if type(tags) is list:
            tags = ",".join(tags)
        params = {
            "title": title,
            "tags": tags,
            "description": description,
            "snippet": snippet,
            "access": access,
            "sortField": sort_field,
            "sortOrder": sort_order,
            "isViewOnly": is_view_only,
            "isinvitationOnly": is_invitation_only,
            "autoJoin": auto_join,
            "leavingDisallowed": leaving_disallowed,
        }
        if provider_group_name:
            params["provider"] = provider
            params["providerGroupName"] = provider_group_name
        if users_update_items == True:
            params["capabilities"] = "updateitemcontrol"
        else:
            params["capabilities"] = ""
        params["isOpenData"] = is_open_data
        params["MAX_FILE_SIZE"] = max_file_size
        if hidden_members in [True, False]:
            params["hiddenMembers"] = hidden_members
        if membership_access in ["org", "collaboration", None, "none"]:
            if self._gis._is_agol is False:
                # Only value for Enterprise is org
                params["membershipAccess"] = "org"
            elif self._gis._is_agol and membership_access is None:
                membership_access = "none"
            params["membershipAccess"] = membership_access
        if autojoin in [True, False]:
            params["autoJoin"] = autojoin

        if (
            isinstance(display_settings, str)
            and display_settings.lower() in display_settings_lu
        ):
            params["displaySettings"] = display_settings_lu[display_settings.lower()]
        elif display_settings is None:
            params["displaySettings"] = display_settings_lu[display_settings]
        else:
            raise ValueError("Display settings must be set to a valid value.")
        # if self._gis.version >= [8,2] and display_settings:
        #    params["itemTypes"] = display_settings
        group = self._portal.create_group_from_dict(params, thumbnail)

        if group is not None:
            return Group(self._gis, group["id"], group)
        else:
            return None

    def create_from_dict(self, dict: dict[str, Any]):
        """
        The ``create_from_dict`` method creates a group via a dictionary with the values for any particular arguments
        that are specified.

        .. note::
            Only title and tags are required.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        dict                   Required dictionary. A dictionary of entries to create/define the
                               group.  See help of the :attr:`~arcgis.gis.GroupManager.create` method for parameters.
        ==================     ====================================================================


        :return:
            The :class:`~arcgis.gis.Group` if successfully created, None if unsuccessful.
        """
        thumbnail = dict.pop("thumbnail", None)

        if "tags" in dict:
            if type(dict["tags"]) is list:
                dict["tags"] = ",".join(dict["tags"])

        group = self._portal.create_group_from_dict(dict, thumbnail)
        if group is not None:
            return Group(self._gis, group["id"], group)
        else:
            return None

    def get(self, groupid: str):
        """
        The ``get`` method retrieves the :class:`~arcgis.gis.Group` object for the specified groupid.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        groupid                Required string. The group identifier.
        ==================     ====================================================================


        :return:
           The :class:`~arcgis.gis.Group` object if the group is found, None if it is not found.
        """
        try:
            group = self._portal.get_group(groupid)
        except RuntimeError as re:
            if re.args[0].__contains__("Group does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as re:
            if re.args[0].__contains__("Group does not exist or is inaccessible"):
                return None
            else:
                raise re

        if group is not None:
            return Group(self._gis, groupid, group)
        return None

    def search(
        self,
        query: str = "",
        sort_field: str = "title",
        sort_order: str = "asc",
        max_groups: int = 1000,
        outside_org: bool = False,
        categories: list[str] | str | None = None,
        filter: str | None = None,
    ):
        """
        The ``search`` method searches for portal groups.

        .. note::
            A few things that will be helpful to know.

            1.  The `group search <https://developers.arcgis.com/rest/users-groups-and-items/group-search.htm>`_ syntax has many features that can't
                be adequately described here. See the `Search Reference <https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm>`_
                page in the ArcGIS REST API for more information.

            2. Searching without specifying a query parameter returns
               a list of all groups in your organization.

            3.  Most of the time when searching for groups, you'll want to
                search within your organization in ArcGIS Online
                or within your Portal. As a convenience, the method
                automatically appends your organization id to the query by
                default. If you don't want the API to append to your query
                set outside_org to True.

        ================    ========================================================
        **Parameter**       **Description**
        ----------------    --------------------------------------------------------
        query               Optional string on Portal, or required string for ArcGIS Online.
                            If not specified, all groups will be searched. See notes above.
        ----------------    --------------------------------------------------------
        sort_field          Optional string. Valid values can be title, owner,
                            created.
        ----------------    --------------------------------------------------------
        sort_order          Optional string. Valid values are asc or desc.
        ----------------    --------------------------------------------------------
        max_groups          Optional integer. Maximum number of groups returned, default is 1,000.
        ----------------    --------------------------------------------------------
        outside_org         Optional boolean. Controls whether to search outside
                            your org. Default is False, do not search ourside your org.
        ----------------    --------------------------------------------------------
        categories          Optional string or list. A string of category values.
        ----------------    --------------------------------------------------------
        filter              Optional string. Structured filtering is accomplished
                            by specifying a field name followed by a colon and the
                            term you are searching for with double quotation marks.
                            It allows the passing in of application-level filters
                            based on the context. Use an exact keyword match of the expected
                            value for the specified field. Partially matching the filter keyword
                            will not return meaningful results.
        ================    ========================================================

        :return:
           A List of :class:`~arcgis.gis.Group` objects matching the specified query.

        .. code-block:: python

            # Usage Example

            >>> gis.groups.search(query ="Hurricane Trackers", categories = "Hurricanes, USA, Natural Disasters",
            >>>                   sort_field="title", sort_order = "asc")
        """
        grouplist = []
        groups = self._portal.search_groups(
            query,
            sort_field,
            sort_order,
            max_groups,
            outside_org,
            categories,
            filter,
        )
        for group in groups:
            grouplist.append(Group(self._gis, group["id"], group))
        return grouplist


def _is_shapefile(data):
    try:
        if zipfile.is_zipfile(data):
            zf = zipfile.ZipFile(data, "r")
            namelist = zf.namelist()
            for name in namelist:
                if name.endswith(".shp") or name.endswith(".SHP"):
                    return True
        return False
    except:
        return False


class ContentManager(object):
    """
    The ``ContentManager`` class is a helper class for managing content in ArcGIS Online or ArcGIS Enterprise.
    An instance of this class, called 'content', is available as a property of the GIS object. Users
    call methods on this 'content' object to manipulate (create, get, search,
    etc) items. See :attr:`~arcgis.gis.content` for more information.

    .. note::
        The class is not created by the user.
    """

    _depmgr = None
    _mrktplcmgr = None

    def __init__(self, gis):
        self._gis = gis
        self._portal = gis._portal

    # ----------------------------------------------------------------------
    def check_url(self, url: str) -> dict[str, Any]:
        """
        To verify a URL is accessible by the Organization, provide the `url` and
        the system will check if the location is valid and reachable.  This
        method is useful when checking service URLs or validating that URLs can
        be reached.

        :return: Dict[str, Any]

        """
        curl = f"{self._gis._portal.resturl}portals/checkUrl"
        params = {"f": "json", "url": url}
        return self._gis._con.get(curl, params, ignore_error_key=True)

    # ----------------------------------------------------------------------
    def can_reassign(self, items: list[Item], user: User) -> list[dict[str, Any]]:
        """
        Checks if a `list[Item]` can be reassigned to a particular user.
        The operation checks whether the items owned by one user can be successfully
        reassigned to a specified user before performing the `reassign` operation.
        Users assigned the default administrator role, or a custom role with
        administrative privileges, can perform this operation. The item owner can
        also use this operation; if the item owner that performs this operation
        is not a default administrator, or assigned a custom role with
        administrative privileges, they must have the portal:user:reassignItems
        privilege assigned to them to transfer content to another user.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        items                      Required list[Item]. A list of Items. The maximum number of items
                                   that can be transferred at one time is 100.
        ----------------------     --------------------------------------------------------------------
        user                       Required User. The user the items will be reassigned to. For a user
                                   to be eligible to receive transferred content, they must meet the
                                   following requirements:

                                   - The user must be assigned the portal:user:receiveItems privilege to receive the transferred content.
                                   - The user must have a user type that allows them to own content.
                                   - If the items being transferred to the user are shared with a group, the user receiving the items must be a member of the group. If the group is a view-only group, the user receiving the items must be the group owner or a group manager.

                                   If the above requirements are not met, an error response will be returned.
        ======================     ====================================================================

        :returns: `list[dict[str, Any]]`
        """
        if self._gis.version < [10, 1] and self._gis._portal.is_arcgisonline == False:
            return []
        urls = {}
        params = {}
        params["f"] = "json"
        params["targetUsername"] = user.username
        params["items"] = None
        for item in items:
            if item.owner in urls:
                urls[item.owner]["itemids"].append(item.itemid)
            else:
                urls[item.owner] = {
                    "itemids": [item.itemid],
                    "url": f"{self._gis._portal.resturl}content/users/{item.owner}/canReassignItems",
                }
        results = []
        for key, value in urls.items():
            params["items"] = ",".join(value["itemids"])
            results.append(self._gis._con.post(value["url"], params))
            #
        return results

    # ---------------------------------------------------------------------
    def cost(
        self,
        tile_storage: Optional[float] = None,
        file_storage: Optional[float] = None,
        feature_storage: Optional[float] = None,
        generate_tile_count: Optional[int] = None,
        loaded_tile_count: Optional[int] = None,
        enrich_variable_count: Optional[int] = None,
        enrich_report_count: Optional[int] = None,
        service_area_count: Optional[int] = None,
        geocode_count: Optional[int] = None,
    ) -> dict:
        """
        The `cost` allows for the estimation of amount of credits an
        operation will be required. For vector and tile storage, a user can
        estimate the cost to cook tile and the cost of storage.  This
        operation allows users to plan for future costs effeciently to best
        serve their organization and clients.

        .. note::
            This operation is only supported on ArcGIS Online.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        tile_storage               Optional Float.  The size of the uncompressed tiles in MBs.
        ----------------------     --------------------------------------------------------------------
        file_storage               Optional Float. Estimates the credit cost of MB file storage.
        ----------------------     --------------------------------------------------------------------
        feature_storage            Optional Float. Estimates the cost of feature storage per feature.
        ----------------------     --------------------------------------------------------------------
        generate_tile_count        Optional Int.  Estimates the credit cost per tile.
        ----------------------     --------------------------------------------------------------------
        loaded_tile_count          Optional Int. Estimates the credit cost of pregenerated tile storage.
        ----------------------     --------------------------------------------------------------------
        enrich_variable_count      Optional Int. Estimates the credit cost er geoenrichment variable.
        ----------------------     --------------------------------------------------------------------
        enrich_report_count        Optional Int. Estimates the credit cost of running reports.
        ----------------------     --------------------------------------------------------------------
        service_area_count         Optional Int. Estimates the credit cost per service area generation.
        ----------------------     --------------------------------------------------------------------
        geocode_count              Optional Int. Estimates the credit cost per record for geocoding.
        ======================     ====================================================================

        :returns: dict[str, float]

        """
        if self._gis._portal.is_arcgisonline == False:
            return {}
        url = f"{self._gis._portal.resturl}portals/self/cost"

        params = {
            "tileStorage": tile_storage,
            "fileStorage": file_storage,
            "featureStorage": feature_storage,
            "generatedTileCount": generate_tile_count,
            "loadedTileCount": loaded_tile_count,
            "enrichVariableCount": enrich_variable_count,
            "enrichReportCount": enrich_report_count,
            "serviceAreaCount": service_area_count,
            "geocodeCount": geocode_count,
            "f": "json",
        }
        return self._gis._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def dependency_manager(self) -> "DependencyManager":
        """
        Provides users the ability to manage the Enterprise's Item Dependencies Database.

        Available in ArcGIS Enterprise 10.9.1+

        :returns: :class:`~arcgis.gis.sharing.DependencyManager` or None for ArcGIS Online.
        """
        if self._depmgr is None and self._gis._portal.is_arcgisonline == False:
            from arcgis.gis.sharing._dependency import DependencyManager

            self._depmgr = DependencyManager(gis=self._gis)
        return self._depmgr

    # ----------------------------------------------------------------------
    @property
    def marketplace(self) -> "MarketPlaceManager":
        """
        Provides users the ability to manage the content's presence on the marketplace.

        :returns:
            :class:`~arcgis.gis.sharing.MarketPlaceManager` or None if not available
        """
        if self._mrktplcmgr is None and self._gis._portal.is_arcgisonline == False:
            from arcgis.gis.sharing._marketplace import MarketPlaceManager

            self._mrktplcmgr = MarketPlaceManager(gis=self._gis)
        return self._mrktplcmgr

    # ----------------------------------------------------------------------
    def _add_by_part(
        self,
        file_path,
        itemid,
        item_properties,
        size=1e7,
        owner=None,
        folder=None,
    ):
        """
        Performs a special add operation that chunks up a file and loads it piece by piece.
        This is an internal method used by `add`

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        file_path           Required String.  The path to the file to load into Portal.
        ---------------     --------------------------------------------------------------------
        itemid              Required String. The unique ID of the Item to load the data to.
        ---------------     --------------------------------------------------------------------
        item_properties     Required Dict.  The properties for the item.
        ---------------     --------------------------------------------------------------------
        size                Optional Integer.  The chunk size off the parts in bytes.  The
                            smallest size allowed is 5 MB or 5e6.
        ---------------     --------------------------------------------------------------------
        multipart           Optional Boolean.  Loads a file by chunks to the Enterprise. The
                            default is False.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder where placing item.
        ===============     ====================================================================


        """
        from typing import Union, Iterator, Tuple
        from io import BytesIO

        def chunk_by_file_size(
            fp: str,
            size: int = None,
            parameter_name: str = "file",
            upload_format: bool = False,
        ) -> Iterator[Union[Tuple[str, BytesIO, str], BytesIO]]:
            """Splits a File based on a specific bytes size"""
            if size is None:
                size = int(2.5e7)  # 25MB
            i = 1
            with open(fp, "rb") as reader:
                while True:
                    bio = BytesIO()
                    data = bio.write(reader.read(size))
                    bio.seek(0)
                    if not data:
                        break
                    if upload_format:
                        fpath = f"split{i}.split"
                        yield parameter_name, bio, fpath
                    else:
                        yield bio
                    i += 1
            return None

        size = int(size)
        if size < 2.5e7:
            size = int(2.5e7)
        size = int(self._calculate_upload_size(fp=file_path))

        owner_name = owner
        if isinstance(owner, User):
            owner_name = owner.username

        # If owner isn't specified, use the logged in user
        if not owner_name:
            owner_name = self._gis.users.me.username

        # Setup the item path, including the folder
        path = "content/users/" + owner_name
        if folder and folder != "/":
            folder_id = self._portal.get_folder_id(owner_name, folder)
            path += "/" + folder_id

        url = "{base}{path}/items/{itemid}/addPart".format(
            base=self._gis._portal.resturl, path=path, itemid=itemid
        )
        results = []
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="upld_"
        ) as tp:
            futures = {
                tp.submit(
                    self._gis._con.post_multipart,
                    **{
                        "path": url,
                        "params": {"f": "json", "partNum": f"{idx + 1}"},
                        "files": [part],
                    },
                ): idx
                + 1
                for idx, part in enumerate(
                    chunk_by_file_size(file_path, size=size, upload_format=True)
                )
            }
        messages = []
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            if "success" in r:
                results.append(r["success"])
            elif "status" in r and r["status"] == "success":
                results.append(True)
            else:
                results.append(False)
            messages.append(r)
        if all(results):  # Should be True/False list
            url = "{base}{path}/items/{itemid}/commit".format(
                base=self._gis._portal.resturl, path=path, itemid=itemid
            )
            params = {
                "f": "json",
                "id": itemid,
                "type": item_properties["type"],
                "async": True,
            }
            params.update(item_properties)
            res = self._gis._con.post(url, params)
            if "success" in res and res["success"]:
                url = "{base}{path}/items/{itemid}/status".format(
                    base=self._gis._portal.resturl, path=path, itemid=itemid
                )
                import time

                params = {"f": "json"}
                res = self._gis._portal.con.post(url, params)
                while res["status"] != "completed":
                    if "fail" in res["status"]:
                        return False
                    time.sleep(1)
                    res = self._gis._portal.con.post(url, {"f": "json"})

                return res["status"]
            else:
                return False

        return False

    # ----------------------------------------------------------------------
    def can_delete(self, item: Item):
        """
        The ``can_delete`` method indicates whether an :class:`~arcgis.gis.Item` can be erased or
        not. When the returned response from ``can_delete`` is true, the
        item can be safely removed. When the returned response is false,
        the item cannot be deleted due to a dependency or protection
        setting.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required :class:`~arcgis.gis.Item`. The `Item` to be erased.
        ===============     ====================================================================

        :return:
            A dictionary - see the table below for examples of a successful call of ``can_delete`` and a failed
            call of ``can_delete``.

        .. code-block:: python

            # Usage Example
            >>> gis.content.can_delete("9311d21a9a2047d19c0faaebd6f2cca6")



        ===============     ====================================================================
        **Status**          **Response**
        ---------------     --------------------------------------------------------------------
        success             {
                            "itemId": "e03f626be86946f997c29d6dfc7a9666",
                            "success": True
                            }

        ---------------     --------------------------------------------------------------------
        failure             {
                            "itemId": "a34c2e6711494e62b3b8d7452d4d6235",
                            "success": false,
                            "reason": {
                            "message": "Unable to delete item. Delete protection is turned on."
                            }
                            }

        ===============     ====================================================================

        """
        params = {"f": "json"}
        url = "{resturl}content/users/{username}/items/{itemid}/canDelete".format(
            resturl=self._portal.resturl,
            username=item.owner,
            itemid=item.itemid,
        )
        try:
            res = self._portal.con.post(url, params)
            return res
        except Exception as e:
            return {
                "itemId": item.itemid,
                "success": False,
                "reason": {"message": "{msg}".format(msg=e.args[0])},
            }
        return False

    # ----------------------------------------------------------------------
    def _calculate_upload_size(self, fp: str) -> int:
        """calculates the file MAX upload limit."""
        fd = os.open(fp, os.O_RDONLY)
        size: float = os.fstat(fd).st_size

        if size <= 5 * (1024 * 1024):
            return int(5 * (1024 * 1024))
        elif size > 5 * (1024 * 1024) and size <= 10 * (1024 * 1024):
            return int(7 * (1024 * 1024))
        elif size > 10 * (1024 * 1024) and size <= 15 * (1024 * 1024):
            return int(13 * (1024 * 1024))
        elif size > 15 * (1024 * 1024) and size <= 25 * (1024 * 1024):
            return int(25 * (1024 * 1024))
        elif size > 25 * (1024 * 1024) and size <= 35 * (1024 * 1024):
            return int(30 * (1024 * 1024))
        else:
            return int(35 * (1024 * 1024))

    # ----------------------------------------------------------------------
    def add(
        self,
        item_properties: dict[str, Any] | ItemProperties,
        data: Optional[str] = None,
        thumbnail: Optional[str] = None,
        metadata: Optional[str] = None,
        owner: Optional[str] = None,
        folder: Optional[str] = None,
        item_id: Optional[str] = None,
        **kwargs,
    ):
        """
        The ``add`` method adds content to the GIS by creating an :class:`~arcgis.gis.Item`.

        .. note::
            Content can be a file (such as a service definition, shapefile,
            CSV, layer package, file geodatabase, geoprocessing package,
            map package) or it can be a URL (to an ArcGIS Server service,
            WMS service, or an application).

            If you are uploading a package or other file, provide a path or
            URL to the file in the data argument.

            From a technical perspective, none of the item_properties (see
            table below *Key:Value Dictionary Options for Argument
            item_properties*) are required.  However, it is strongly
            recommended that arguments title, type, typeKeywords, tags,
            snippet, and description be provided.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        data                Optional string, io.StringIO, or io.BytesIO. Either a path or URL to
                            the data or an instance of `StringIO` or `BytesIO` objects.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Defaults to the logged in user.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of the folder where placing item.
        ---------------     --------------------------------------------------------------------
        item_id             Optional string. Available in ArcGIS Enterprise 10.8.1+. Not available in ArcGIS Online.
                            This parameter allows the desired item id to be specified during creation which
                            can be useful for cloning and automated content creation scenarios.
                            The specified id must be a 32 character GUID string without any special characters.

                            If the `item_id` is already being used, an error will be raised
                            during the `add` process.

                            Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        ===============     ====================================================================


        *Optional Input Parameters for the `add` method*

        ========================     ====================================================================
        **Optional Argument**        **Description**
        ------------------------     --------------------------------------------------------------------
        upload_size                  Optional float.  The default value is 1e7 bytes or ~10 MBs.  This the
                                     minimum default value for the size of the file when uploading by parts.
        ========================     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*


        ==========================  =====================================================================
        **Key**                     **Value**
        --------------------------  ---------------------------------------------------------------------
        type                        Optional string. Indicates type of item, see URL 1 below for valid values.
        --------------------------  ---------------------------------------------------------------------
        dataUrl                     Optional string. The Url of the data stored on cloud storage. If given, filename is required.
        --------------------------  ---------------------------------------------------------------------
        filename                    Optional string. The name of the file on cloud storage.  This is required is dataUrl is used.
        --------------------------  ---------------------------------------------------------------------
        typeKeywords                Optional string. Provide a lists all sub-types, see URL below for valid values.
        --------------------------  ---------------------------------------------------------------------
        description                 Optional string. Description of the item.
        --------------------------  ---------------------------------------------------------------------
        title                       Optional string. Name label of the item.
        --------------------------  ---------------------------------------------------------------------
        url                         Optional string. URL to item that are based on URLs.
        --------------------------  ---------------------------------------------------------------------
        text                        Optional string. For text based items such as Feature Collections & WebMaps
        --------------------------  ---------------------------------------------------------------------
        tags                        Optional string. Tags listed as comma-separated values, or a list of strings.
                                    Used for searches on items.
        --------------------------  ---------------------------------------------------------------------
        snippet                     Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        --------------------------  ---------------------------------------------------------------------
        extent                      Optional string. Provide comma-separated values for min x, min y, max x, max y.
        --------------------------  ---------------------------------------------------------------------
        spatialReference            Optional string. Coordinate system that the item is in.
        --------------------------  ---------------------------------------------------------------------
        accessInformation           Optional string. Information on the source of the content.
        --------------------------  ---------------------------------------------------------------------
        licenseInfo                 Optional string.  Any license information or restrictions regarding the content.
        --------------------------  ---------------------------------------------------------------------
        culture                     Optional string. Locale, country and language information.
        --------------------------  ---------------------------------------------------------------------
        commentsEnabled             Optional boolean. Default is true, controls whether comments are allowed (true)
                                    or not allowed (false).
        --------------------------  ---------------------------------------------------------------------
        access                      Optional string. Valid values are private, org, or public. Defaults to private.
        --------------------------  ---------------------------------------------------------------------
        overwrite                   Optional boolean. Default is `false`. Controls whether item can be overwritten.
        ==========================  =====================================================================


        See `Item and Item Types <https://developers.arcgis.com/rest/users-groups-and-items/items-and-item-types.htm>`_
        in the ArcGIS REST API for more information.

        :return:
           The :class:`~arcgis.gis.Item` if successfully added, None if unsuccessful.

        .. code-block:: python

            # Usage Example

            >>> gis.content.add(item_properties = {
            >>>                                         "type": "Feature Collection",
            >>>                                         "title" : "US Hurricane Data",
            >>>                                         "tags": "Hurricanes, Natural Disasters, USA",
            >>>                                         "description" : "Aggregated USA Hurricane Data for 2020",
            >>>                                         "commentsEnabled" : False
            >>>                                        } , owner = "User1234")
        """

        filetype = None

        if not isinstance(item_properties, (dict, ItemProperties)):
            raise ValueError(
                "`item_properties` must be  dictionary or `ItemProperties`."
            )
        elif isinstance(item_properties, ItemProperties):
            thumbnail = thumbnail or item_properties.thumbnail
            metadata = metadata or item_properties.metadata
            item_properties = item_properties.to_dict()
            item_properties.pop("thumbnail", None)
            item_properties.pop("metadata", None)
        if item_id and isinstance(item_id, str) and len(item_id) == 32:
            item_properties["itemIdToCreate"] = item_id
        if isinstance(data, arcgis.features.FeatureCollection):
            filetype = "Feature Collection"
            item_properties["text"] = {"layers": [data._lyr_dict]}
            data = None
        elif _is_geoenabled(data) and hasattr(data, "spatial"):
            filetype = "Feature Collection"
            item_properties["text"] = {
                "layers": [data.spatial.to_feature_collection()._lyr_dict]
            }
            data = None

        if data is not None and isinstance(data, (io.StringIO, io.BytesIO)):
            assert "type" in item_properties
            assert "title" in item_properties

        elif data is not None:
            title = os.path.splitext(os.path.basename(data))[0]
            extn = os.path.splitext(os.path.basename(data))[1].upper()

            filetype = None
            if extn == "GPKG":
                filetype = "GeoPackage"
            elif extn == ".CSV":
                filetype = "CSV"
            elif extn in [".XLSX", ".XLS"]:
                filetype = "Microsoft Excel"
            elif extn == ".SD":
                filetype = "Service Definition"
            elif title.upper().endswith(".GDB"):
                filetype = "File Geodatabase"
            elif extn in (".SLPK", ".SPK"):
                filetype = "Scene Package"
            elif extn in (".LPK", ".LPKX"):
                filetype = "Layer Package"
            elif extn in (".GPK", ".GPKX"):
                filetype = "Geoprocessing Package"
            elif extn == ".GCPK":
                filetype = "Locator Package"
            elif extn in (".TPK", ".TPKX"):
                filetype = "Tile Package"
            elif extn in (".MPK", ".MPKX"):
                filetype = "Map Package"
            elif extn == ".MMPK":
                filetype = "Mobile Map Package"
            elif extn == ".APTX":
                filetype = "Project Template"
            elif extn == ".VTPK":
                filetype = "Vector Tile Package"
            elif extn == ".PPKX":
                filetype = "Project Package"
            elif extn == ".RPK":
                filetype = "Rule Package"
            elif extn == ".MAPX":
                filetype = "Pro Map"

            if _is_shapefile(data):
                filetype = "Shapefile"

            if not "type" in item_properties:
                if filetype is not None:
                    item_properties["type"] = filetype
                else:
                    raise RuntimeError("Specify type in item_properties")
            if not "title" in item_properties:
                item_properties["title"] = title
        if (
            "type" in item_properties
            and item_properties["type"] == "WMTS"
            and "text" not in item_properties
        ):
            from arcgis.mapping.ogc import WMTSLayer

            item_properties["text"] = json.dumps(
                WMTSLayer(item_properties["url"], gis=self._gis).__text__
            )

        owner_name = owner
        if isinstance(owner, User):
            owner_name = owner.username

        if "tags" in item_properties:
            if type(item_properties["tags"]) is list:
                item_properties["tags"] = ",".join(item_properties["tags"])
        try:
            from arcgis._impl.common._utils import bytesto

            is_file = os.path.isfile(data)
            if is_file and bytesto(os.stat(data).st_size) < 7:
                multipart = False
                item_properties.pop("multipart", None)
            elif (
                is_file == False and hasattr(data, "tell") and bytesto(data.tell()) < 7
            ):
                multipart = False
                item_properties.pop("multipart", None)
            else:
                if "multipart" in item_properties:
                    item_properties["multipart"] = True
                multipart = True
        except:
            is_file = False
            multipart = False
            item_properties.pop("multipart", None)
        if multipart and is_file:
            item_properties["multipart"] = True
            params = {}
            params.update(item_properties)
            params["fileName"] = os.path.basename(data)
            # Create an empty Item
            itemid = self._portal.add_item(
                params, None, thumbnail, metadata, owner_name, folder
            )
            # check the status and commit the final result
            if kwargs.get("upload_size", 0) >= 1e7:
                upload_size = kwargs.get("upload_size")
            else:
                upload_size = self._calculate_upload_size(data)

            status = self._add_by_part(
                file_path=data,
                itemid=itemid,
                item_properties=item_properties,
                size=upload_size,
                owner=owner_name,
                folder=folder,
            )

            # Update the thumbnail
            item = Item(gis=self._gis, itemid=itemid)
            if item.type == "KML":
                item.update(
                    {
                        "url": f"{self._gis._portal.resturl}content/items/{item.itemid}/data"
                    }
                )
            item.update(thumbnail=thumbnail)

            # Update the access and return the item
            if item_properties and "access" in item_properties:
                if item_properties["access"] == "public":
                    item.share(everyone=True)
                elif item_properties["access"] == "org":
                    item.share(org=True)
                elif item_properties["access"] == "private":
                    item.share(everyone=False, org=False)
            return item
        else:
            if filetype:
                item_properties["fileName"] = os.path.basename(data)
            if "text" in kwargs:
                item_properties["text"] = kwargs.pop("text", None)
            itemid = self._portal.add_item(
                item_properties,
                data,
                thumbnail,
                metadata,
                owner_name,
                folder,
            )

        if itemid is not None:
            item = Item(self._gis, itemid)
            if item.type == "KML":
                item.update(
                    {
                        "url": f"{self._gis._portal.resturl}content/items/{item.itemid}/data"
                    }
                )

            # Update access
            if item_properties and "access" in item_properties:
                if item_properties["access"] == "public":
                    item.share(everyone=True)
                elif item_properties["access"] == "org":
                    item.share(org=True)
                elif item_properties["access"] == "private":
                    item.share(everyone=False, org=False)
            return item
        else:
            return None

    # ----------------------------------------------------------------------
    def analyze(
        self,
        url: Optional[str] = None,
        item: Optional[Union[str, Item]] = None,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        file_type: Optional[str] = None,
        source_locale: str = "en",
        geocoding_service: Optional[str] = None,
        location_type: Optional[str] = None,
        source_country: str = "world",
        country_hint: Optional[str] = None,
        enable_global_geocoding: Optional[bool] = None,
    ):
        """
        The ``analyze`` method helps a client analyze a CSV or Excel file (.xlsx, .xls) prior to publishing or
        generating features using the Publish or Generate operation, respectively.

        ``analyze`` returns information about the file including the fields present as well as sample records.
        ``analyze`` attempts to detect the presence of location fields that may be present as either X,Y fields or
        address fields.

        ``analyze`` packages its result so that publishParameters within the JSON response contains information that
        can be passed back to the server in a subsequent call to Publish or Generate. The publishParameters subobject
        contains properties that describe the resulting layer after publishing, including its fields, the desired
        renderer, and so on. ``analyze`` will suggest defaults for the renderer.

        In a typical workflow, the client will present portions of the ``analyze`` results to the user for editing
        before making the call to :attr:`~arcgis.gis.ContentManager.generate` or ``publish``.

        .. note::
            The maximum upload size for shapefiles is now 2 Mb and 10 Mb for all other supported file types.

        .. note::
            If the file to be analyzed currently exists in the portal as an item, callers can pass in its itemId.
            Callers can also directly post the file.
            In this case, the request must be a multipart post request pursuant to IETF RFC1867.
            The third option for text files is to pass the text in as the value of the text parameter.

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        url                        Optional string. The URL of the csv file.
        -----------------------    -------------------------------------------------------------
        item                       Optional string/:class:`~arcgis.gis.Item` . The ID or Item of the item to be
                                   analyzed.
        -----------------------    -------------------------------------------------------------
        file_path                  Optional string. The file to be analyzed.
        -----------------------    -------------------------------------------------------------
        text                       Optional string. The text in the file to be analyzed.
        -----------------------    -------------------------------------------------------------
        file_type                  Optional string. The type of the input file: shapefile, csv, excel,
                                   or geoPackage (Added ArcGIS API for Python 1.8.3+).
        -----------------------    -------------------------------------------------------------
        source_locale              Optional string. The locale used for the geocoding service source.
        -----------------------    -------------------------------------------------------------
        geocoding_service          Optional string/geocoder. The URL of the service.
        -----------------------    -------------------------------------------------------------
        location_type              Optional string. Indicates the type of spatial information stored in the dataset.

                                   Values for CSV: coordinates | address | lookup | none
                                   Values for Excel: coordinates | address | none
        -----------------------    -------------------------------------------------------------
        source_country             Optional string. The two character country code associated with the geocoding service, default is "world".
        -----------------------    -------------------------------------------------------------
        country_hint               Optional string. If first time analyzing, the hint is used. If source country is already specified than sourcecountry is used.
        -----------------------    -------------------------------------------------------------
        enable_global_geocoding    Optional boolean. Default is None. When True, the global geocoder is used.
        =======================    =============================================================

        :return: dictionary

        .. code-block:: python

            # Usage Example

            >>> gis.content.analyze(item = "9311d21a9a2047d19c0faaebd6f2cca6", file_type = "csv")

        """
        surl = "%s/sharing/rest/content/features/analyze" % self._gis._url
        params = {"f": "json", "analyzeParameters": {}}
        files = None
        if not (text or file_path or item or url):
            return Exception(
                "Must provide an itemid, file_path or text to analyze data."
            )
        if item:
            if isinstance(item, str):
                params["itemid"] = item
            elif isinstance(item, Item):
                params["itemid"] = item.itemid
        elif file_path and os.path.isfile(file_path):
            files = {"file": file_path}
        elif text:
            params["text"] = text
        elif url:
            params["sourceUrl"] = url

        params["analyzeParameters"]["sourcelocale"] = source_locale
        if geocoding_service:
            from arcgis.geocoding._functions import Geocoder

            if isinstance(geocoding_service, Geocoder):
                params["analyzeParameters"]["geocodeServiceUrl"] = geocoding_service.url
            else:
                params["analyzeParameters"]["geocodeServiceUrl"] = geocoding_service
        if location_type:
            params["analyzeParameters"]["locationType"] = location_type

        if file_type is None and (url or file_path):
            d = url or file_path
            if d:
                if str(d).lower().endswith(".csv"):
                    params["fileType"] = "csv"
                elif str(d).lower().endswith(".xls") or str(d).lower().endswith(
                    ".xlsx"
                ):
                    params["fileType"] = "excel"
                elif str(d).lower().endswith("gpkg"):
                    params["fileType"] = "geoPackage"

        elif str(file_type).lower() in ["excel", "csv"]:
            params["fileType"] = file_type
        elif str(file_type).lower() in ["filegeodatabase", "shapefile"]:
            params["fileType"] = file_type
            params["analyzeParameters"]["enableGlobalGeocoding"] = False
        if source_country:
            params["analyzeParameters"]["sourceCountry"] = source_country
        if country_hint:
            params["analyzeParameters"]["sourcecountryhint"] = country_hint
        if enable_global_geocoding in [True, False]:
            params["analyzeParameters"][
                "enableGlobalGeocoding"
            ] = enable_global_geocoding
        gis = self._gis
        params["analyzeParameters"] = json.dumps(params["analyzeParameters"])

        return gis._con.post(path=surl, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def create_empty_service(
        self,
        parameters: CreateServiceParameter,
        *,
        owner: User | None = None,
    ) -> Item:
        """
        Creates a blank or view based service.

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        parameters                 Required CreateServiceParameter. A dataclass that provides the
                                   create service parameters.
        -----------------------    -------------------------------------------------------------
        owner                      Optional User. The user to save the service to.
        =======================    =============================================================

        :returns: Item
        """
        regex = r"^[-a-zA-Z0-9_]*$"
        if len(re.findall(regex, parameters.name)) == 0:
            raise ValueError(
                "The service `name` cannot contain any spaces or special characters except underscores."
            )
        if owner:
            username = owner.username
        else:
            owner = self._gis.users.me
            username = owner.username
        self._gis._portal.resturl
        url = f"{self._gis._portal.resturl}content/users/{username}/createService"
        params = parameters.to_dict()
        params["f"] = "json"
        res = self._gis._con.post(url, params)
        itemid = res.get("itemId", None)

        if itemid:
            return self.get(itemid)
        else:
            return res

    # ----------------------------------------------------------------------
    def create_service(
        self,
        name: str,
        service_description: str = "",
        has_static_data: bool = False,
        max_record_count: int = 1000,
        supported_query_formats: str = "JSON",
        capabilities: Optional[str] = None,
        description: str = "",
        copyright_text: str = "",
        wkid: int = 102100,
        create_params: Optional[dict[str, Any]] = None,
        service_type: str = "featureService",
        owner: Optional[str] = None,
        folder: Optional[str] = None,
        item_properties: Optional[dict[str, Any]] = None,
        is_view: bool = False,
        tags: Optional[Union[list[str], str]] = None,
        snippet: Optional[Union[list[str], str]] = None,
        item_id: Optional[str] = None,
    ):
        """
        The ``create_service`` method creates a service in the Portal. See the table below for a list of arguments
        passed when calling ``create_service``.


        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        name                       Required string. The unique name of the service.
        -----------------------    -------------------------------------------------------------
        service_description        Optional string. Description of the service.
        -----------------------    -------------------------------------------------------------
        has_static_data            Optional boolean. Indicating whether the data can change.  Default is True, data is not allowed to change.
        -----------------------    -------------------------------------------------------------
        max_record_count           Optional integer. Maximum number of records in query operations.
        -----------------------    -------------------------------------------------------------
        supported_query_formats    Optional string. Formats in which query results are returned.
        -----------------------    -------------------------------------------------------------
        capabilities               Optional string. Specify service capabilities.
                                   If left unspecified, 'Image,Catalog,Metadata,Download,Pixels'
                                   are used for image services, and 'Query'
                                   is used for feature services, and 'Query' otherwise
        -----------------------    -------------------------------------------------------------
        description                Optional string. A user-friendly description for the published dataset.
        -----------------------    -------------------------------------------------------------
        copyright_text             Optional string. The copyright information associated with the dataset.
        -----------------------    -------------------------------------------------------------
        wkid                       Optional integer. The well known id (WKID) of the spatial reference for the service.
                                   All layers added to a hosted feature service need to have the same spatial
                                   reference defined for the feature service. When creating a new
                                   empty service without specifying its spatial reference, the spatial
                                   reference of the hosted feature service is set to the first layer added to that feature service.
        -----------------------    -------------------------------------------------------------
        create_params              Optional dictionary. Add all create_service parameters into a dictionary. If this parameter is used,
                                   all the parameters above are ignored.
        -----------------------    -------------------------------------------------------------
        service_type               Optional string. The type of service to be created.  Currently the options are imageService or featureService.
        -----------------------    -------------------------------------------------------------
        owner                      Optional string. The username of the owner of the service being created.
        -----------------------    -------------------------------------------------------------
        folder                     Optional string. The name of folder in which to create the service.
        -----------------------    -------------------------------------------------------------
        item_properties            Optional dictionary. See below for the keys and values
        -----------------------    -------------------------------------------------------------
        is_view                    Optional boolean. Indicating if the service is a hosted feature layer view
        -----------------------    -------------------------------------------------------------
        item_id                    Optional string. Available in ArcGIS Enterprise 10.8.1+. Not available in ArcGIS Online.
                                   This parameter allows the desired item id to be specified during creation which
                                   can be useful for cloning and automated content creation scenarios.
                                   The specified id must be a 32 character GUID string without any special characters.

                                   If the `item_id` is already being used, an error will be raised
                                   during the `add` process.

                                   Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        -----------------------    -------------------------------------------------------------
        tags                       Optional string. Tags listed as comma-separated values, or a list of strings.
                                   Used for searches on items.
        -----------------------    -------------------------------------------------------------
        snippet                    Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        =======================    =============================================================


        *Key:Value Dictionary Options for Argument item_properties*


        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        type               Optional string. Indicates type of item, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see URL 1 below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        url                Optional string. URL to item that are based on URLs.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        extent             Optional string. Provide comma-separated values for min x, min y, max x, max y.
        -----------------  ---------------------------------------------------------------------
        spatialReference   Optional string. Coordinate system that the item is in.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        -----------------  ---------------------------------------------------------------------
        commentsEnabled    Optional boolean. Default is true, controls whether comments are allowed (true)
                           or not allowed (false).
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Language and country information.
        =================  =====================================================================

        :return:
             The :class:`~arcgis.gis.Item` for the service if successfully created, None if unsuccessful.

        .. code-block:: python

            # Usage Example
            >>> gis.content.create_service("Hurricane Collection")
        """
        regex = r"^[-a-zA-Z0-9_]*$"
        if len(re.findall(regex, name)) == 0:
            raise ValueError(
                "The service `name` cannot contain any spaces or special characters except underscores."
            )
        if capabilities is None:
            if service_type == "imageService":
                capabilities = "Image,Catalog,Metadata,Download,Pixels"
            elif service_type == "featureService":
                capabilities = "Query"
            else:
                capabilities = "Query"
        if self._gis.version <= [7, 1] and item_id:
            item_id = None
            import warnings

            warnings.warn(
                "Item ID is not Support at this version. Please use version >=10.8.1 Enterprise."
            )
        itemid = self._portal.create_service(
            name,
            service_description,
            has_static_data,
            max_record_count,
            supported_query_formats,
            capabilities,
            description,
            copyright_text,
            wkid,
            service_type,
            create_params,
            owner,
            folder,
            item_properties,
            is_view,
            item_id,
            tags,
            snippet,
        )
        if itemid is not None:
            item = Item(self._gis, itemid)
            if item_properties is None:
                item_properties = {}
            else:
                item.update(item_properties=item_properties)
            if "access" in item_properties.keys():
                if item_properties["access"] == "public":
                    item.share(everyone=True)
                elif item_properties["access"] == "org":
                    item.share(org=True)
                elif item_properties["access"] == "private":
                    item.share(everyone=False, org=False)
                elif item_properties["access"] == "shared":
                    groups = item.shared_with["groups"]
                    item.share(groups=groups)
            return item
        else:
            return None

    # ----------------------------------------------------------------------
    @property
    def categories(self):
        """
        The ``categories`` property is category manager for an :class:`~arcgis.gis.Item` object.
        See :class:`~arcgis.gis.CategorySchemaManager`.
        """

        base_url = "{base}portals/self".format(base=self._gis._portal.resturl)
        return CategorySchemaManager(base_url=base_url, gis=self._gis)

    # ----------------------------------------------------------------------
    def get(self, itemid: str):
        """
        The ``get`` method returns the :class:`~arcgis.gis.Item` object for the specified itemid.

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        itemid                     Required string. The item identifier.
        =======================    =============================================================

        :return:
            The item object if the item is found, None if the item is not found.
        """
        try:
            item = self._portal.get_item(itemid)
        except RuntimeError as re:
            if re.args[0].__contains__("Item does not exist or is inaccessible"):
                return None
            else:
                raise re
        except Exception as e:
            if e.args[0].__contains__("Item does not exist or is inaccessible"):
                return None
            else:
                raise e

        if item is not None:
            return Item(self._gis, itemid, item)
        return None

    def advanced_search(
        self,
        query: str,
        return_count: bool = False,
        max_items: int = 100,
        bbox: Optional[Union[list[str], str]] = None,
        categories: Optional[str] = None,
        category_filter: Optional[str] = None,
        start: int = 1,
        sort_field: str = "title",
        sort_order: str = "asc",
        count_fields: Optional[str] = None,
        count_size: Optional[int] = None,
        as_dict: bool = False,
        enrich: bool = False,
    ):
        """
        The ``advanced_search`` method allows the ability to fully customize the search experience.
        The ``advanced_search`` method allows users to control of the finer grained parameters
        not exposed by the :attr:`~arcgis.gis.ContentManager` method.  Additionally, it allows for the manual paging of
        information and how the data is returned.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String.  The search query. When the search filters
                            contain two or more clauses, the recommended schema is to have
                            clauses separated by blank, or `AND`, e.g.

                            .. code-block:: python

                               #Usage Example:


                                >>> gis.content.advanced_search(query='owner:USERNAME type:map')
                                # or
                                >>> gis.content.advanced_search(query='type:map AND owner:USERNAME')


                            .. warning::
                                When the clauses are separated by comma, the filtering condition
                                for `owner` should not be placed at the first position, e.g.

                                .. code-block:: python

                                    >>> gis.content.advanced_search(query='type:map, owner:USERNAME')

                                is allowed, while

                                .. code-block:: python

                                    >>> gis.content.advanced_search(query='owner:USERNAME, type:map')

                                is not.  For more information, please check `Users, groups and items <https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm>`_.
        ----------------    ---------------------------------------------------------------
        return_count        Optional Boolean. When true, the total for the given search is
                            returned. It will ignore the `max_items` variable.
        ----------------    ---------------------------------------------------------------
        max_items           Optional Integer. The total number of items to return up to
                            10,000. When the value of -1 is given all aviable Items will be
                            returned up to 10,000.
        ----------------    ---------------------------------------------------------------
        bbox                Optional String/List. This is the xmin,ymin,xmax,ymax bounding
                            box to limit the search in.  Items like documents do not have
                            bounding boxes and will not be included in the search.
        ----------------    ---------------------------------------------------------------
        categories          Optional String. A comma separated list of up to 8 org content
                            categories to search items. Exact full path of each category is
                            required, OR relationship between the categories specified.

                            Each request allows a maximum of 8 categories parameters with
                            AND relationship between the different categories parameters
                            called.
        ----------------    ---------------------------------------------------------------
        category_filters    Optional String. A comma separated list of up to 3 category
                            terms to search items that have matching categories. Up to 2
                            `category_filters` parameter are allowed per request. It can
                            not be used together with categories to search in a request.
        ----------------    ---------------------------------------------------------------
        start               Optional Int. The starting position to search from.  This is
                            only required if paging is needed.
        ----------------    ---------------------------------------------------------------
        sort_field          Optional String. Responses from the `search` operation can be
                            sorted on various fields. `avgrating` is the default.
        ----------------    ---------------------------------------------------------------
        sort_order          Optional String. The sequence into which a collection of
                            records are arranged after they have been sorted. The allowed
                            values are: asc for ascending and desc for descending.
        ----------------    ---------------------------------------------------------------
        count_fields        Optional String. A comma separated list of fields to count.
                            Maximum count fields allowed per request is 3. Supported count
                            fields: `tags`, `type`, `access`, `contentstatus`, and
                            `categories`.
        ----------------    ---------------------------------------------------------------
        count_size          Optional Int. The maximum number of field values to count for
                            each `count_fields`. The default value is None, and maximum size
                            allowed is 200.
        ----------------    ---------------------------------------------------------------
        as_dict             Required Boolean. If True, the results comes back as a dictionary.
                            The result of the method will always be a dictionary but the
                            `results` key in the dictionary will be changed if set to False.
        ----------------    ---------------------------------------------------------------
        enrich              Optional Boolean. If True, search results will include both
                            literal and relevant matches. Without this parameter search
                            results will include only literal matches.
        ================    ===============================================================

        :return:
            Depends on the inputs:
                1. Dictionary for a standard search
                2. `return_count`=True an integer is returned
                3. `count_fields` is specified a list of dicts for each field specified

        .. code-block:: python

            # Usage Example

            >>> gis.content.advanced_search(query ="Hurricanes", categories = "Hurricanes, USA, Natural Disasters",
            >>>                                sort_order = "asc", count_fields = "tags, type", as_dict = True)


        """
        from arcgis.gis._impl import _search

        stype = "content"
        group_id = None
        if max_items == -1:
            max_items = _search(
                gis=self._gis,
                query=query,
                stype=stype,
                max_items=0,
                bbox=bbox,
                categories=categories,
                category_filter=category_filter,
                start=start,
                sort_field=sort_field,
                sort_order=sort_order,
                count_fields=count_fields,
                count_size=count_size,
                group_id=group_id,
                as_dict=as_dict,
                enrich=enrich,
            )["total"]
        so = {
            "asc": "asc",
            "desc": "desc",
            "ascending": "asc",
            "descending": "desc",
        }
        if sort_order:
            sort_order = so[sort_order]

        if count_fields or return_count:
            max_items = 0
        if max_items <= 100:
            res = _search(
                gis=self._gis,
                query=query,
                stype=stype,
                max_items=max_items,
                bbox=bbox,
                categories=categories,
                category_filter=category_filter,
                start=start,
                sort_field=sort_field,
                sort_order=sort_order,
                count_fields=count_fields,
                count_size=count_size,
                group_id=group_id,
                as_dict=as_dict,
                enrich=enrich,
            )
            if "total" in res and return_count:
                return res["total"]
            elif "aggregations" in res:
                return res["aggregations"]
            return res
        else:
            allowed_keys = [
                "query",
                "return_count",
                "max_items",
                "bbox",
                "categories",
                "category_filter",
                "start",
                "sort_field",
                "sort_order",
                "count_fields",
                "count_size",
                "as_dict",
                "enrich",
            ]
            inputs = locals()
            kwargs = {}
            for k, v in inputs.items():
                if k in allowed_keys:
                    kwargs[k] = v
            import concurrent.futures
            import math, copy

            num = 100
            steps = range(math.ceil(max_items / num))
            params = []
            for step in steps:
                new_start = start + num * step
                kwargs["max_items"] = num
                kwargs["start"] = new_start
                params.append(copy.deepcopy(kwargs))
            total_count = -999
            next_start_tracker = -1
            items = {
                "results": [],
                "start": start,
                "num": 100,
                "total": total_count,
                "query": query,
                "nextStart": next_start_tracker,
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {
                    executor.submit(self.advanced_search, **param): param
                    for param in params
                }
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future_to_url[future]
                    data = future.result()
                    if data.get("nextStart", -1) > next_start_tracker:
                        next_start_tracker = data.get("nextStart", -1)
                    if "results" in data:
                        items["results"].extend(data["results"])

            if len(items["results"]) > max_items:
                items["results"] = items["results"][:max_items]
                items["nextStart"] = max_items
            else:
                items["nextStart"] = next_start_tracker
            items["total"] = len(items["results"])
            return items

    def _market_listings(
        self,
        query: str,
        sort_field: str = None,
        sort_order: str = "asc",
        num: int = 10,
        start: int = 1,
        my_listings: bool = False,
    ) -> dict[str, Any]:
        """
        This operation searches for marketplace listings. The searches are
        performed against a high performance index that indexes the most
        popular fields of a listing. See the Search reference page for
        information on the fields and the syntax of the query.

        By default, this search spans all public listings in the
        marketplace. However, if you're logged in as a vendor org admin and
        you specify the ``mylistings=true`` parameter, it then searches all
        public and private listings in your organization.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String.  The search query.
        ----------------    ---------------------------------------------------------------
        sort_field          Optional String. The field to sort by. You can also sort by
                            multiple fields (comma separated) for listings, sort field
                            names are case-insensitive.

                            Supported sort field names are `title`, `created`,
                            `listingpublisheddate`, `type`, `owner`, `avgrating`,
                            `numratings`, `numcomments`, and `numviews`.
        ----------------    ---------------------------------------------------------------
        sort_order          Optional String. Describes whether the order returns in
                            ascending or descending order. Default is ascending.

                            Values: `asc` or `desc`
        ----------------    ---------------------------------------------------------------
        num                 Optional Integer. The maximum number of results to be included
                            in the result set response.

                            The default value is `10`, and the maximum allowed value is `100`.
        ----------------    ---------------------------------------------------------------
        start               Optional Integer. The number of the first entry in the result
                            set response. The index number is 1-based.
        ----------------    ---------------------------------------------------------------
        my_listings         Optional Boolean.  If `True` and you're logged in as a vendor
                            org admin, it searches all public and private listings in your
                            organization.

                            **Note** that if `my_listings=True`, the q parameter is optional.

                            Values: `False (default) | True`
        ================    ===============================================================


        :return: Dictionary[str, Any]
        """
        params = {
            "f": "json",
            "q": query,
            "sortField": sort_field,
            "sortOrder": sort_order,
            "mylistings": my_listings,
            "num": num,
            "start": start,
        }
        for key in list(params.keys()):
            if params[key] is None:
                del params[key]

        url = f"{self._gis._portal.resturl}content/listings"
        resp = self._gis._con.get(url, params)
        return resp

    def search(
        self,
        query: str,
        item_type: Optional[str] = None,
        sort_field: str = "avgRating",
        sort_order: str = "desc",
        max_items: int = 10,
        outside_org: bool = False,
        categories: Optional[Union[list[str], str]] = None,
        category_filters: Optional[Union[list[str], str]] = None,
        enrich: Optional[bool] = None,
    ):
        """
        The ``search`` method searches for portal items.


        .. note::
            A few things that will be helpful to know...

            1. The query syntax has many features that can't be adequately
               described here.  Please see the ArcGIS REST API `Search
               Reference <https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm>`_
               for full details on search engine used with this method.

            2. Most of the time when searching for items, you'll want to
               search within your organization in ArcGIS Online
               or within your Portal.  As a convenience, the method
               automatically appends your organization id to the query by
               default.  If you want content from outside your organization
               set outside_org to True.

        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        query             Required string. A query string.  See notes above. When the search filters
                          contain two or more clauses, the recommended schema is to have clauses
                          separated by blank, or `AND`, e.g.

                          :Usage Example:


                          gis.content.search(query='owner:USERNAME type:map')
                          # or
                          gis.content.search(query='type:map AND owner:USERNAME')

                          .. warning::
                          When the clauses are separated by comma, the filtering condition
                          for `owner` should not be placed at the first position, e.g.
                          `gis.content.search(query='type:map, owner:USERNAME')`
                          is recommended, while
                          `gis.content.search(query='owner:USERNAME, type:map')`
                          is not.
        ----------------  --------------------------------------------------------------------------
        item_type         Optional string. The type of item to search. See `Items and item types <https://developers.arcgis.com/rest/users-groups-and-items/items-and-item-types.htm>`_
                          for comprehensive list of values (the type column).
        ----------------  --------------------------------------------------------------------------
        sort_field        Optional string. Valid values can be title, uploaded, type, owner, modified,
                          avgRating, numRatings, numComments, and numViews.
        ----------------  --------------------------------------------------------------------------
        sort_order        Optional string. Valid values are asc or desc.
        ----------------  --------------------------------------------------------------------------
        max_items         Optional integer. Maximum number of items returned, default is 10.
        ----------------  --------------------------------------------------------------------------
        outside_org       Optional boolean. Controls whether to search outside your org (default is False, do not search ourside your org).
        ----------------  --------------------------------------------------------------------------
        categories        Optional string or list. A string of category values.
        ----------------  --------------------------------------------------------------------------
        category_filters  Optional string. A comma separated list of up to 3 category terms to
                          search items that have matching categories.

                          Up to 2 category_filters parameter are allowed per request. It can not be
                          used together with categories to search in a request.
        ----------------  --------------------------------------------------------------------------
        enrich            Optional Boolean. If True, search results will include both literal and
                          relevant matches. Without this parameter search results will include only
                          literal matches.
        ================  ==========================================================================

        :return:
            A list of :class:`items <arcgis.gis.Item>` objects matching the specified query.

        .. code-block:: python

            # Usage Example

            >>> gis.content.search(query ="Hurricanes", categories = "Hurricanes, USA, Natural Disasters",
            >>>                    item_type = "Feature Collection")

        """
        if max_items > 10000:
            raise Exception(
                (
                    "There is a limitation of 10,000 items that can be returned. Please use a smaller value for max_items. This is a limitation documented here: https://developers.arcgis.com/rest/users-groups-and-items/considerations-and-limitations.htm"
                )
            )
        itemlist = []
        if query is not None and query != "" and item_type is not None:
            query += " AND "

        if item_type is not None:
            item_type = item_type.lower()
            if item_type == "web map":
                query += ' (type:"web map" NOT type:"web mapping application")'
            elif item_type == "web scene":
                query += ' (type:"web scene" NOT type:"CityEngine Web Scene")'
            elif item_type == "feature layer":
                query += ' (type:"feature service")'
            elif item_type == "geoprocessing tool":
                query += ' (type:"geoprocessing service")'
            elif item_type == "geoprocessing toolbox":
                query += ' (type:"geoprocessing service")'
            elif item_type == "feature layer collection":
                query += ' (type:"feature service")'
            elif item_type == "image layer":
                query += ' (type:"image service")'
            elif item_type == "imagery layer":
                query += ' (type:"image service")'
            elif item_type == "map image layer":
                query += ' (type:"map service")'
            elif item_type == "vector tile layer":
                query += ' (type:"vector tile service")'
            elif item_type == "scene layer":
                query += ' (type:"scene service")'
            elif item_type == "layer":
                query += (
                    ' (type:"layer" NOT type:"layer package" NOT type:"Explorer Layer")'
                )
            elif item_type == "feature collection":
                query += ' (type:"feature collection" NOT type:"feature collection template")'
            elif item_type == "desktop application":
                query += ' (type:"desktop application" NOT type:"desktop application template")'
            else:
                query += ' (type:"' + item_type + '")'
        if isinstance(categories, str):
            categories = categories.split(",")

        if not outside_org:
            accountid = self._gis.properties.get("id")
            if accountid and query:
                query += " accountid:" + accountid
            elif accountid:
                query = "accountid:" + accountid
        itemlist = self.advanced_search(
            query=query,
            max_items=max_items,
            categories=categories,
            start=1,
            sort_field=sort_field,
            sort_order=sort_order,
            enrich=enrich,
        )["results"]
        return itemlist

    def create_folder(self, folder: str, owner: Optional[str] = None):
        """
        The ``create_folder`` method creates a folder with the given folder name, for the given owner.

        .. note::
            The ``create_folder`` method does nothing if the folder already exists.
            Additionally, if owner is not specified, owner is set as the logged in user.


        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        folder            Required string. The name of the folder to create for the owner.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user.
        ================  ==========================================================================

        :return:
            A json object like the following if the folder was created:
            {"username" : "portaladmin","id" : "bff13218991c4485a62c81db3512396f","title" : "testcreate"}; None otherwise.

        .. code-block:: python

            # Usage Example
            >>> gis.content.create_folder("Hurricane_Data", owner= "User1234")

        """
        if folder != "/":  # we don't create root folder
            if owner is None:
                owner = self._portal.logged_in_user()["username"]
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            if self._portal.get_folder_id(owner_name, folder) is None:
                return self._portal.create_folder(owner_name, folder)
            else:
                print("Folder already exists.")
        return None

    def _get_folder(self, folder_id: str, username: str = None) -> str:
        """
        Private method for when a folder name needs to be found from a folder id.
        Returns the folder name for the given folder id.

        This method will search your folders to find the folder name. If folder is not owned by you,
        then you must specify the username of the owner. Specifying the username requires
        administrator privileges.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        folder_id         Required string. The id of the folder.
        ----------------  ---------------------------------------------------------------
        username          Optional string. The username of the folder owner. Need admin
                          privileges to specify this parameter.
        ================  ===============================================================

        :return: String
        """
        # If username provided check privileges
        if username:
            if "portal:admin:viewUsers" not in self._gis.users.me.privileges:
                raise Exception(
                    "You do not have privileges to view other users folders."
                )
        else:
            username = self._gis.users.me.username
        # Get folder
        folders = self._gis.users.get(username).folders
        for folder in folders:
            if folder["id"] == folder_id:
                return folder["title"]
        return None

    def rename_folder(
        self, old_folder: str, new_folder: str, owner: Optional[str] = None
    ):
        """
        The ``rename_folder`` method renames an existing folder from it's existing name to a new name.

        .. note::
            If owner is not specified, owner is set as the logged in user.


        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        old_folder        Required string. The name of the folder to rename for the owner.
        ----------------  --------------------------------------------------------------------------
        new_folder        Required string. The new name of the folder.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user.
        ================  ==========================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example
            >>> gis.content.rename_folder("2020_Hurricane_Data", "2021_Hurricane_Data", "User1234")

        """
        params = {"f": "json", "newTitle": new_folder}
        if old_folder != "/":  # we don't rename the root folder
            if owner is None:
                owner = self._portal.logged_in_user()["username"]
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            folderid = self._portal.get_folder_id(owner_name, old_folder)
            if folderid is None:
                raise ValueError("Folder: %s does not exist." % old_folder)
            url = "{base}content/users/{user}/{folderid}/updateFolder".format(
                base=self._gis._portal.resturl,
                user=owner_name,
                folderid=folderid,
            )
            res = self._gis._con.post(url, params)
            if "success" in res:
                return res["success"]
        return False

    def delete_items(self, items: Union[list[Item], list[str]]):
        """
        The ``delete_items`` method deletes a collection of :class:`~arcgis.gis.Item` objects from a users content.
        All items must belong to the same user to delete.

        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        items             list of :class:`~arcgis.gis.Item` objects or Item Ids.  This is an array
                          of items to be deleted from the current user's content
        ================  ==========================================================================

        :return:
            A list of booleans indicating success if the items were deleted(True/False) or an empty list if nothing was deleted.
            (False)

        .. code-block:: python

            # Usage Example
            >>> gis.content.delete_items(items= ["item1", "item2", "item3", "item4", "item5"])

        """
        params = {"f": "json", "items": ""}
        items_dict = {}  # key will be ownner and value is list of their items
        for item in items:
            if isinstance(item, str):
                owner = self._gis.content.get(item).owner
                if owner in items_dict:
                    items_dict[owner].append(item)
                else:
                    items_dict[owner] = [item]
            elif isinstance(item, Item):
                owner = item.owner
                if owner in items_dict:
                    items_dict[owner].append(item.id)
                else:
                    items_dict[owner] = [item.id]
            del item

        # Now we have a dictionary to iterate through
        results = []
        for key, val in items_dict.items():
            owner = key  # owner username
            ditems = val  # list of item(s)

            # Check if admin or owner before deleting
            if (
                self._gis.users.me.username != owner
                and "portal:admin:deleteItems" not in self._gis.users.me.privileges
            ):
                return Exception(
                    "You are not the owner and you do not have the administrator privileges to perform this action."
                )

            # All items should be from same owner so we can set to first in list
            if self._gis._portal.con.baseurl.endswith("/"):
                url = "%s/%s/%s/deleteItems" % (
                    self._gis._portal.con.baseurl[:-1],
                    "content/users",
                    owner,
                )
            else:
                url = "%s/%s/%s/deleteItems" % (
                    self._gis._portal.con.baseurl,
                    "content/users",
                    owner,
                )

            if len(ditems) > 0:
                params["items"] = ",".join(ditems)
                res = self._gis._con.post(path=url, postdata=params)
                results.append(all([r["success"] for r in res["results"]]))
        return results

    def delete_folder(self, folder: str, owner: Optional[str] = None):
        """
        The ``delete_folder`` method deletes a folder for the given owner with
        the given folder name.

        .. note::
            If the an owner is note specified in the ``delete_folder`` call, the method defaults to the logged in user.


        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        folder            Required string. The name of the folder to delete.
        ----------------  --------------------------------------------------------------------------
        owner             Optional string. User, folder owner, None for logged in user is the default.
        ================  ==========================================================================

        :return:
            A boolean indicating success if the folder was deleted (True), or failure if the folder was not deleted
            (False)

        .. code-block:: python

            # Usage Example
            >>> gis.content.delete_folder("Hurricane_Data", owner= "User1234")

        """
        if folder != "/":
            if owner is None:
                owner = self._portal.logged_in_user()["username"]
                owner_name = owner
            elif isinstance(owner, User):
                owner_name = owner.username
            else:
                owner_name = owner
            return self._portal.delete_folder(owner_name, folder)

    # ----------------------------------------------------------------------
    def _generate(self, gurl, params, files, gis):
        """
        private async logic for `generate`.
        """
        res = gis._con.post(gurl, params, files=files)
        if "success" in res and res["success"] == False:
            raise Exception("Failed to Generate Features")

        if res["status"]:
            item = gis.content.get(res["outputItemId"])
            status = item.status(res["jobId"], "generateFeatures")
            while status["status"] != "completed":
                if status["status"] == "failed":
                    try:
                        item.delete()
                        return status
                    except:
                        return status
                status = item.status(res["jobId"], "generateFeatures")
            item.update(item_properties={"title": f"Generate Features: {res['jobId']}"})
            return item
        return res

    # ----------------------------------------------------------------------
    def generate(
        self,
        item: Optional[Item] = None,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        text: Optional[str] = None,
        publish_parameters: Optional[dict[str, Any]] = None,
        future: bool = False,
    ):
        """
        The ``generate`` method helps a client generate features from a CSV file, shapefile,
        GPX, or GeoJson file types.

        .. note::
            The maximum upload size for shapefiles is now 2 Mb and 10 Mb for all other supported file types.

        ===================  ==========================================================================
        **Parameter**         **Description**
        -------------------  --------------------------------------------------------------------------
        item                 Optional Item. An `Item` on the current portal.
        -------------------  --------------------------------------------------------------------------
        file_path            Optional String. The file resource location on local disk.
        -------------------  --------------------------------------------------------------------------
        url                  Optional String. A web resource of a 'shapefile', 'csv', 'gpx' or 'geojson' file.
        -------------------  --------------------------------------------------------------------------
        text                 Optional String. The source text.
        -------------------  --------------------------------------------------------------------------
        publish_parameters   Optional Dict. A Python dictionary describing the layer and service to be created
                             as part of the `publish` operation. The appropriate key-value pairs
                             depend on the file type being published. For a complete
                             description, see the `Publish Item <https://developers.arcgis.com/rest/users-groups-and-items/publish-item.htm>`_
                             documentation in the REST API. Also see the :class:`~arcgis.gis.Item`
                             :meth:`~arcgis.gis.Item.publish` method. ``CSV``, ``Shapefiles`` and
                             ``GeoJSON`` file types must have publish parameters provided.
        -------------------  --------------------------------------------------------------------------
        future               Optional Boolean.  This allows the operation to run asynchronously allowing
                             the user to not pause the thread and continue to perform multiple operations.
                             The default is `False`.  When `True` the result of the method will be a
                             concurrent `Future` object.  The `result` of the method can be obtained
                             using the `result()` on the `Future` object.  When `False`, and Item is
                             returned. Future == True is only supported for 'shapefiles' and 'gpx' files.
        ===================  ==========================================================================

        :return:
            The method has 3 potential returns:
                1. A `Future` object when `future==True`, Call ``results()`` to get the response.
                2. An :class:`~arcgis.gis.Item` object when `future==False`
                3. A dictionary of error messages when Exceptions are raised

        .. code-block:: python

            # Usage Example
            >>> gis.content.generate(item= item1, future=False)

        """
        if item is None and file_path is None and text is None and url is None:
            raise Exception("You must provide an item, file_path, text or url.")
        gurl = f"{self._gis._portal.resturl}content/features/generate"
        params = {
            "f": "json",
            "itemid": "",
            "sourceUrl": "",
            "text": "",
            "filetype": "",
            "publishParameters": publish_parameters or "",
            "async": True,
        }
        files = None
        file_types = {
            ".gpx": "gpx",
            ".csv": "csv",
            ".zip": "shapefile",
            ".json": "geojson",
        }
        if item and item.type.lower() in [
            "shapefile",
            "csv",
            "gpx",
            "geojson",
        ]:
            params["itemid"] = item.itemid
            if item.type.lower() == "shapefile":
                params["filetype"] = "shapefile"
                if (
                    publish_parameters in (None, "")
                    and self._gis._portal.is_arcgisonline == False
                ):
                    raise ValueError(
                        "A publish parameter is needed for this data type."
                    )
            elif item.type.lower() == "gpx":
                params["filetype"] = "gpx"
            elif item.type.lower() == "csv":
                params["filetype"] = "csv"
                if publish_parameters is None:
                    raise ValueError(
                        "A publish parameter is needed for this data type."
                    )
            elif item.type.lower() == "geojson":
                params["filetype"] = "geojson"
                if publish_parameters is None:
                    raise ValueError(
                        "A publish parameter is needed for this data type."
                    )
            else:
                raise Exception(f"Invalid Item Type {item.type}")

        elif url:
            params["sourceUrl"] = url
            part = os.path.splitext(url)[-1]
            if part in file_types:
                params["filetype"] = file_types[part]
            else:
                raise Exception(f"Invalid file extension: {part}")
        elif file_path and os.path.isfile(file_path):
            files = []
            part = os.path.splitext(file_path)[-1]
            if part in file_types:
                params["filetype"] = file_types[part]
            else:
                raise Exception(f"Invalid file extension: {part}")
            if (
                params["filetype"] in ["shapefile", "csv", "geojson"]
                and self._gis._portal.is_arcgisonline == False
                and params["publishParameters"] in (None, "")
            ):
                raise ValueError("A publish parameter is needed for this data type.")
            files.append(("file", file_path, os.path.basename(file_path)))
        elif text:
            params["text"] = text
            params["fileType"] = "csv"

        if future == True:
            executor = concurrent.futures.ThreadPoolExecutor(1)
            futureobj = executor.submit(
                self._generate,
                **{
                    "gurl": gurl,
                    "params": params,
                    "files": files,
                    "gis": self._gis,
                },
            )
            executor.shutdown(False)
            return futureobj
        else:
            params["async"] = False
            res = self._gis._con.post(gurl, params, files=files)
            return res

    # ----------------------------------------------------------------------
    def import_table(
        self,
        df: pd.DataFrame,
        *,
        service_name: str | None = None,
        title: str | None = None,
        publish_parameters: dict[str, Any] = None,
    ) -> Item:
        """
        The `import_table` function takes a Pandas' DataFrame and publishes it
        as a Hosted Table on a WebGIS.

        ===================  ==========================================================================
        **Parameter**         **Description**
        -------------------  --------------------------------------------------------------------------
        df                   Required DataFrame. A Pandas dataframe containing the tabular information.
        -------------------  --------------------------------------------------------------------------
        service_name         Optional String. The name of the service.
        -------------------  --------------------------------------------------------------------------
        title                Optional String. The name of the title of the created Item.
        -------------------  --------------------------------------------------------------------------
        publish_parameters   Optional dict[str,Any]. The publish parameters.  If given, the user is
                             responsible for passing all the publish parameters defined
                             `here <https://developers.arcgis.com/rest/users-groups-and-items/publish-item.htm>`_.
        ===================  ==========================================================================

        returns: Published Hosted Table Item

        """
        assert isinstance(
            df, pd.DataFrame
        ), f"The df parameter must be a Pandas' DataFrame, not {type(df).__name__}"
        fname: str = tempfile.mkstemp(suffix=".csv")[1]

        df.to_csv(fname)
        if title is None:
            now: datetime = datetime.now()
            title: str = f"Import Table created on: {now.strftime('%m/%d/%Y')}"
        if service_name is None:
            service_name = f"import_table_{uuid.uuid4().hex[:3]}"
        pp: dict[str, Any] = {
            "type": "CSV",
            "title": title,
        }
        csv_item: Item = self.add(item_properties=pp, data=fname)
        try:
            os.remove(fname)
        except:
            pass
        if publish_parameters is None:
            publish_parameters: dict[str, Any] = self.analyze(
                item=csv_item, file_type="csv"
            )["publishParameters"]
            publish_parameters["name"] = service_name
            publish_parameters["locationType"] = "none"
        return csv_item.publish(publish_parameters)

    # ----------------------------------------------------------------------
    def import_data(
        self,
        df,
        address_fields: Optional[dict[str, Any]] = None,
        folder: Optional[str] = None,
        item_id: Optional[str] = None,
        **kwargs,
    ):
        """
        The ``import_data`` method imports a Pandas `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        (that has an address column), or an arcgis spatial
        :class:`~arcgis.gis.features._data.geodataset.geodataframe.DataFrame` into the GIS.

        Spatial dataframes are imported into the GIS and published as feature
        layers. Pandas dataframes that have an address column are imported as
        an in-memory feature collection.

        .. note::
            By default, there is a limit of 1,000 rows/features for Pandas
            dataframes. This limit isn't there for spatial dataframes.

        ================  ==========================================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------------------------
        df                Required DataFrame. Pandas dataframe
        ----------------  --------------------------------------------------------------------------
        address_fields    Optional dictionary. Dictionary containing mapping of df columns to address fields, eg: { "CountryCode" : "Country"} or { "Address" : "Address" }.
        ----------------  --------------------------------------------------------------------------
        folder            Optional string. Name of the folder where imported data would be stored.
        ----------------  --------------------------------------------------------------------------
        title             Optional string. Title of the item. This is used for spatial dataframe objects.
        ----------------  --------------------------------------------------------------------------
        tags              Optional string. Tags listed as comma-separated values, or a list of strings. Provide tags when publishing a spatial dataframe to the the GIS.
        ----------------  --------------------------------------------------------------------------
        item_id           Optional string. Available in ArcGIS Enterprise 10.8.1+. Not available in ArcGIS Online.
                          This parameter allows the desired item id to be specified during creation which
                          can be useful for cloning and automated content creation scenarios.
                          The specified id must be a 32 character GUID string without any special characters.

                          If the `item_id` is already being used, an error will be raised
                          during the `add` process.

                          Example: item_id=9311d21a9a2047d19c0faaebd6f2cca6
        ================  ==========================================================================

        In addition to the parameters above, you can specify additional information to help publish CSV
        data.

        =====================  ==========================================================================
        **Optional Argument**  **Description**
        ---------------------  --------------------------------------------------------------------------
        location_type          Optional string. Indicates the type of spatial information stored in the
                               dataset.

                               Values for CSV:

                                  + coordinates
                                  + address (default)
                                  + lookup
                                  + none

                               Values for Excel:

                                  + coordinates
                                  + address (default)
                                  + none

                               When location_type = coordinates, the CSV or Excel data contains x,y
                               information.
                               When location_type = address, the CSV or Excel data contains address
                               fields that will be geocoded to a single point.
                               When location_type = lookup, the CSV or Excel data contains fields that
                               can be mapped to well-known sets of geographies.
                               When location_type = none, the CSV or Excel data contains no spatial
                               content and data will be loaded and subsequently queried as tabular data.

                               Based on this parameter, additional parameters will be required, for
                               example, when specifying location_type = coordinates, the latitude and
                               longitude field names must be specified.
        ---------------------  --------------------------------------------------------------------------
        latitude_field         Optional string. If location_type = coordinates, the name of the field that
                               contains the y coordinate.
        ---------------------  --------------------------------------------------------------------------
        longitude_field        Optional string. If location_type = coordinates, the name of the field that
                               contains the x coordinate.
        ---------------------  --------------------------------------------------------------------------
        coordinate_field_type  Optional string. Specify the type of coordinates that contain location
                               information. Values: LatitudeAndLongitude (default), MGRS, USNG
        ---------------------  --------------------------------------------------------------------------
        coordinate_field_name  Optional string. The name of the field that contains the coordinates
                               specified in coordinate_field_type
        ---------------------  --------------------------------------------------------------------------
        lookup_type            Optional string. The type of place to look up.
        ---------------------  --------------------------------------------------------------------------
        lookup_fields          Optional string. A JSON object with name value pairs that define the
                               fields used to look up the location.
        ---------------------  --------------------------------------------------------------------------
        geocode_url            Optional string. The URL of the geocoding service that supports batch
                               geocoding.
        ---------------------  --------------------------------------------------------------------------
        source_locale          Optional string. The locale used for the geocoding service source.
        ---------------------  --------------------------------------------------------------------------
        source_country         Optional string. The two character country code associated with the
                               geocoding service, default is 'world'.
        ---------------------  --------------------------------------------------------------------------
        country_hint           Optional string. If first time analyzing, the hint is used. If source
                               country is already specified than source_country is used.
        =====================  ==========================================================================


        When publishing a Spatial Dataframe, additional options can be given:

        =====================  ==========================================================================
        **Optional Argument**  **Description**
        ---------------------  --------------------------------------------------------------------------
        target_sr              optional integer.  WKID of the output data.  This is used when publishing
                               Spatial Dataframes to Hosted Feature Layers. The default is: 102100
        ---------------------  --------------------------------------------------------------------------
        title                  optional string. Name of the layer. The default is a random string.
        ---------------------  --------------------------------------------------------------------------
        tags                   optional string. Comma seperated strings that provide metadata for the
                               items. The default is FGDB.
        ---------------------  --------------------------------------------------------------------------
        capabilities           optional string. specifies the operations that can be performed on the
                               feature layer service. The default is Query.
        ---------------------  --------------------------------------------------------------------------
        sanitize_columns       Optional boolean. The default is False.  When true, the column name will
                               modified in order to allow for successful publishing.
        ---------------------  --------------------------------------------------------------------------
        service_name           Optional String. The name for the service that will be added to the Item.
        ---------------------  --------------------------------------------------------------------------
        overwrite              Optional boolean. If True, the specified feature layer for the specified
                               feature service will be overwritten.
        ---------------------  --------------------------------------------------------------------------
        append                 Optional boolean. If True, the SeDF will be appended to the specified
                               feature service.
        ---------------------  --------------------------------------------------------------------------
        service                Dictionary that is required if `overwrite = True` or `append = True`.
                               Dictionary with two keys: "FeatureServiceId" and "layers".
                               "featureServiceId" value is a string of the feature service id that the layer
                               belongs to.
                               "layer" value is an integer depicting the index value of the layer to
                               overwrite. For append, None can be passed as value.
        =====================  ==========================================================================


        :return:
           A :class:`feature collection <arcgis.features.FeatureCollection>` or :class:`feature layer <arcgis.features.FeatureLayer>`
           that can be used for analysis, visualization, or published to the GIS as an :class:`~arcgis.gis.Item`.
           If geoenabled DataFrame is passed in then an :class:`~arcgis.gis.Item` is directly returned.
        """
        # Get parameters right
        if item_id and self._gis.version <= [7, 1]:
            kwargs["item_id"] = None

            warnings.warn(
                "`item_id` is not allowed at this version of Portal, please use Enterprise 10.8.1+"
            )
        else:
            kwargs["item_id"] = item_id
        kwargs["folder"] = folder
        kwargs["address_fields"] = address_fields

        # Check which workflow to do
        overwrite = kwargs.get("overwrite", False)
        insert = kwargs.get("append", False)
        if _is_geoenabled(df) or (overwrite or insert):
            # Item Workflow
            return _cm_helper.import_as_item(self._gis, df, **kwargs)
        else:
            # Feature Collection Workflow
            return _cm_helper.import_as_fc(self._gis, df, **kwargs)

    # ----------------------------------------------------------------------
    def is_service_name_available(self, service_name: str, service_type: str):
        """
            The ``is_service_name_available`` method determines if that service name is
            available for use or not, for the specified service type.

        ================  ======================================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------------------
        service_name      Required string. A desired service name.
        ----------------  ----------------------------------------------------------------------
        service_type      Required string. The type of service to be created.  Currently the options are imageService or featureService.
        ================  ======================================================================

        :return:
            True if the specified service_name is available for the
            specified service_type, False if the service_name is unavailable.

        .. code-block:: python

                # Usage Example
                >>> gis.content.is_service_name_available("Hurricane Collection", "featureService")


        """
        path = "portals/self/isServiceNameAvailable"

        postdata = {"f": "pjson", "name": service_name, "type": service_type}

        res = self._portal.con.post(path, postdata)
        return res["available"]

    def clone_items(
        self,
        items: list[Item],
        folder: Optional[str] = None,
        item_extent: Optional[dict[str, Any]] = None,
        use_org_basemap: bool = False,
        copy_data: bool = True,
        copy_global_ids: bool = False,
        search_existing_items: bool = True,
        item_mapping: Optional[dict[str, str]] = None,
        group_mapping: Optional[dict[str, str]] = None,
        owner: Optional[str] = None,
        preserve_item_id: bool = False,
        **kwargs,
    ):
        """
        The ``clone_items`` method is used to clone content to the GIS by creating new :class:`~arcgis.gis.Item`
        objects.

        .. note::
            Cloning an item will create a copy of the item and for certain
            item types a copy of the item dependencies in the :class:`~arcgis.gis.GIS`.

        For example, a web application created using Web AppBuilder
        or a Configurable App Template which is built from a web map
        that references one or more hosted feature layers. This function
        will clone all of these items to the GIS and swizzle the paths
        in the web map and web application to point to the new layers.

        .. note::
            The actions in the example above create an exact copy of the application, map, and layers
            in the :class:`~arcgis.gis.GIS`.

        =====================     ====================================================================
        **Parameter**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required list. Collection of :class:`~arcgis.gis.Item` objects to clone.
        ---------------------     --------------------------------------------------------------------
        folder                    Optional string. Name of the folder where placing item.
        ---------------------     --------------------------------------------------------------------
        item_extent               Optional Envelope. Extent set for any cloned items. Default is None,
                                  extent will remain unchanged. Spatial reference of the envelope will be
                                  used for any cloned feature layers.
        ---------------------     --------------------------------------------------------------------
        use_org_basemap           Optional boolean. Indicating whether the basemap in any cloned web maps
                                  should be updated to the organizations default basemap. Default is False,
                                  basemap will not change.
        ---------------------     --------------------------------------------------------------------
        copy_data                 Optional boolean. If False, the data is put by reference rather than
                                  by copy. Default is True, data will be copied. This creates a Hosted
                                  Feature Collection or Feature Layer.
        ---------------------     --------------------------------------------------------------------
        copy_global_ids           Optional boolean. Assumes previous parameter is set to True. If True,
                                  features copied will preserve their global IDs. Default is False
        ---------------------     --------------------------------------------------------------------
        search_existing_items     Optional boolean. Indicating whether items that have already been cloned
                                  should be searched for in the GIS and reused rather than cloned again.
        ---------------------     --------------------------------------------------------------------
        item_mapping              Optional dictionary. Can be used to associate an item id in the source
                                  GIS (key) to an item id in the target GIS (value). The target item will
                                  be used rather than cloning the source item.
        ---------------------     --------------------------------------------------------------------
        group_mapping             Optional dictionary. Can be used to associate a group id in the source
                                  GIS (key) to a group id in the target GIS (value). The target group will
                                  be used rather than cloning the source group.
        ---------------------     --------------------------------------------------------------------
        owner                     Optional string. Defaults to the logged in user.
        ---------------------     --------------------------------------------------------------------
        preserve_item_id          Optional Boolean.  When true and the destination `GIS` is not ArcGIS
                                  Online, the clone item will attempt to keep the same item ids for the
                                  items if available.  ArcGIS Enterprise must be 10.9+.
        =====================     ====================================================================

        **keyword arguments**

        =====================     ====================================================================
        copy_code_attachment      Option Boolean.  Determines whether a *code_attachment* item should
                                  be created when cloning a Web App Builder item. Default values is
                                  *True*. Set to *False* to prevent item from being created.
        =====================     ====================================================================


        :return:
           A list of :class:`~arcgis.gis.Item` objects created during the clone.

        .. code-block:: python

            # Usage Example
            >>> gis.content.clone_items(items= ["item1", "item2", "item3", "item4", "item5"],
                                        folder ="/", owner = 'User1234')


        """

        import arcgis._impl.common._clone as clone

        wgs84_extent = None
        service_extent = item_extent
        if service_extent:
            wgs84_extent = clone._wgs84_envelope(service_extent)
        owner_name = owner
        if owner_name is None:
            owner_name = self._gis.users.me.username
        if isinstance(owner, User):
            owner_name = owner.username
        if (preserve_item_id and self._gis.version < [8, 2]) or (
            preserve_item_id and self._gis._portal.is_arcgisonline
        ):
            print(
                "Cannot preserve ItemIds on ArcGIS Enterprise "
                "older than v10.9 or to ArcGIS Online organizations. \n"
                "`preserve_item_id` will be ignored."
            )
            preserve_item_id = False

        deep_cloner = clone._DeepCloner(
            self._gis,
            items,
            folder,
            wgs84_extent,
            service_extent,
            use_org_basemap,
            copy_data,
            copy_global_ids,
            search_existing_items,
            item_mapping,
            group_mapping,
            owner_name,
            preserve_item_id=preserve_item_id,
            from_dash=kwargs.pop("from_dash", False),
            wab_code_attach=kwargs.pop("copy_code_attachment", True),
        )
        return deep_cloner.clone()

    def bulk_update(
        self,
        itemids: Union[list[str], list[Item]],
        properties: dict[str, Any],
    ):
        """
        The ``bulk_update`` method updates a collection of items' properties.

        .. note::
            bulk_update only works with content categories at this time.

        ================  ======================================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------------------
        itemids           Required list of string or Item. The collection of Items to update.
        ----------------  ----------------------------------------------------------------------
        properties        Required dictionary. The Item's properties to update.
        ================  ======================================================================

        :return:
            A List of results

        .. code-block:: python

            # Usage Example
            >>> itemsids = gis.content.search("owner: TestUser12399")
            >>> properties = {'categories' : ["clothes","formal_wear/socks"]}
            >>> gis.content.bulk_update(itemids, properties)
            [{'results' : [{'itemid' : 'id', 'success' : "True/False" }]}]

        """
        path = "content/updateItems"
        params = {"f": "json", "items": []}
        updates = []
        results = []
        for item in itemids:
            if isinstance(item, Item):
                updates.append({item.itemid: properties})
            elif isinstance(item, str):
                updates.append({item: properties})
            else:
                raise ValueError("Invalid Item or ItemID, must be string or Item")

        def _chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i : i + n]

        for i in _common_utils.chunks(l=updates, n=100):
            params["items"] = i

            res = self._gis._con.post(path=path, postdata=params)
            results.append(res)
            del i
        return results

    # ----------------------------------------------------------------------
    def replace_service(
        self,
        replace_item: Union[str, Item],
        new_item: Union[str, Item],
        replaced_service_name: Optional[str] = None,
        replace_metadata: bool = False,
    ):
        """
        The ``replace_service`` operation allows you to replace your production vector tile layers with staging ones.
        This operation allows you to perform quality control on a staging tile layer and to then replace the production
        tile layer with the staging with minimal downtime. This operation has the option to keep a backup of the
        production tile layer.

        .. note::
            This functionality is only available for hosted vector tile layers, hosted tile layers and hosted scene
            layers based on packages.  If you are looking to clone services, use the
            :attr:`~arcgis.gis.ContentManager.clone_items` method instead.

        The workflow for the ``replace_service`` method is as follows:

        1. Publish the staging service to the same system as the production service. Both services are active at
        the same time. Share the staging service with a smaller set of users and QA the staging service.

        2. The item properties (ex: thumbnail, iteminfo, metadata) of the production item will be preserved.
        If you need to update them use the `Item.update()` method.

        3. Call the replace_service operation. The service running on the hosting server gets replaced
        (for example, its cache).

        .. note::
            It is the responsibility of the user to ensure both services are functionally equivalent for clients
            consuming them. For example, when replacing a hosted feature service, ensure the new service is constructed
            with the anticipated layers and fields for its client application.

        If you want to retain the replaced production service, for example, to keep an archive of the evolution of the
        service you can do so by omitting a value for "Replaced Service Name" . If replaced service name is not provided,
        the production service being replaced will be archived with a time stamp when replace service was executed.
        You can provide any name for the replaced service as long as it is not pre-existing on your portal content.

        ======================  ======================================================================
        **Parameter**            **Description**
        ----------------------  ----------------------------------------------------------------------
        replace_item            Required Item or Item's Id as string. The service to be replaced
        ----------------------  ----------------------------------------------------------------------
        new_item                Required Item or Item's Id as string. The replacement service.
        ----------------------  ----------------------------------------------------------------------
        replaced_service_name   Optional string. The name of the replacement service.
        ----------------------  ----------------------------------------------------------------------
        replace_metadata        Optional Boolean. When set to `True`, the item info {"thumbnail", "tag",
                                "description", "summary"} of the current service is updated to that of
                                the replacement service. The Credits, Terms of use, and Created from
                                details will not be replaced. This option is set to `False` by default.

        ======================  ======================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example
            >>> gis.content.replace_service(replace_item="9311d21a9a2047d19c0faaebd6f2cca6",
                                            new_item = "420554d21a9a2047d19c0faaebd6f2cca4")
        """
        user = self._gis.users.me
        if "id" in user:
            user = user.username
        else:
            user = user.username
        url = "%s/content/users/%s/replaceService" % (
            self._portal.resturl,
            user,
        )

        if isinstance(replace_item, Item):
            replace_item = replace_item.itemid

        if isinstance(new_item, Item):
            new_item = new_item.itemid

        create_new_item = False
        if replaced_service_name:
            create_new_item = True

        params = {
            "toReplaceItemId": replace_item,
            "replacementItemId": new_item,
            "replaceMetadata": replace_metadata,
            "createNewItem": create_new_item,
            "f": "json",
        }
        if replaced_service_name is not None:
            params["replacedServiceName"] = replaced_service_name
        res = self._gis._con.post(path=url, postdata=params)
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def share_items(
        self,
        items: Union[list[str], list[Item]],
        everyone: bool = False,
        org: bool = False,
        groups: Optional[Union[list[str], list[Group]]] = None,
        allow_members_to_edit: bool = False,
    ):
        """
        The ``shares_items`` method shares a batch of items with everyone, members of the organization, or
        specified list of :class:`~arcgis.gis.Group`. A :class:`~arcgis.gis.User` can only share items with
        groups to which they belong. This method is quite similar to the
        :attr:`~arcgis.gis.ContentManager.unshare_items` method, which achieves the exact opposite of ``share_items``.

        =====================     ====================================================================
        **Parameter**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required List. A list of Item or item ids to modify sharing on.
        ---------------------     --------------------------------------------------------------------
        everyone                  Optional boolean. Default is False, don't share with everyone.
        ---------------------     --------------------------------------------------------------------
        org                       Optional boolean. Default is False, don't share with the
                                  organization.
        ---------------------     --------------------------------------------------------------------
        groups                    Optional list of group names as strings, or a list of
                                  arcgis.gis.Group objects, or a comma-separated list of group IDs.
        ---------------------     --------------------------------------------------------------------
        allow_members_to_edit     Optional boolean. Default is False, to allow item to be shared with
                                  groups that allow shared update
        =====================     ====================================================================

        :return:
            A dictionary of shared :class:`~arcgis.gis.Item` objects

        .. code-block:: python

            # Usage Example

            >>> gis.content.share_items(items=[item1, item2, item3], everyone=True, org=True)

        """
        url = "{base}content/users/{username}/shareItems".format(
            base=self._portal.resturl, username=self._gis.users.me.username
        )
        params = {"f": "json"}
        if groups is None:
            groups = []
        elif isinstance(groups, (tuple, list)):
            grps = []
            for grp in groups:
                if isinstance(grp, str):
                    grps.append(grp)
                elif isinstance(grp, Group):
                    grps.append(grp.groupid)
            groups = grps
        if isinstance(items, Item):
            sitems = [items.itemid]
            items = [items]
        elif isinstance(items, str):
            sitems = [items]
            items = [Item(gis=self._gis, itemid=items)]
        else:
            sitems = []
            for i in items:
                if isinstance(i, Item):
                    sitems.append(i.itemid)
                else:
                    sitems.append(i)
            if not isinstance(sitems[0], Item):
                items = [Item(gis=self._gis, itemid=i) for i in sitems]
        params["items"] = ",".join(sitems)
        params["everyone"] = everyone
        params["org"] = org
        params["confirmItemControl"] = allow_members_to_edit
        params["groups"] = ",".join(groups)
        res = self._gis._con.post(url, params)
        for i in items:
            i._hydrated = False
        return res

    # ----------------------------------------------------------------------
    def unshare_items(
        self,
        items: Union[list[str], list[Item]],
        groups: Optional[Union[list[str], list[Group]]] = None,
        everyone: Optional[bool] = None,
        org: Optional[bool] = None,
    ):
        """
        The ``unshare_items`` methodUnshares a batch of items with the specified list of groups, everyone, or
        organization. This method is quite similar to the
        :attr:`~arcgis.gis.ContentManager.share_items` method, which achieves the exact opposite of ``unshare_items``.

        .. note::
            Each item's current sharing will be overwritten with this method.

        =====================     ====================================================================
        **Parameter**              **Description**
        ---------------------     --------------------------------------------------------------------
        items                     Required List. A list of Item or item ids to modify sharing on.
        ---------------------     --------------------------------------------------------------------
        everyone                  Required Boolean. If provided, the everyone sharing property will be
                                  updated.  True means it will share the items with anyone. False means
                                  the item will not be shared with all users.
        ---------------------     --------------------------------------------------------------------
        org                       Required Boolean. A true value means that the items will be shared
                                  with all members of the organization.  A false value means that the
                                  item will not be shared with all organization users.
        ---------------------     --------------------------------------------------------------------
        groups                    Required list of group names as strings, or a list of
                                  :class:`~arcgis.gis.Group` objects, or a list of group IDs.
        =====================     ====================================================================

        :return:
            A dictionary of unshared :class:`~arcgis.gis.Item` objects

        .. code-block:: python

            # Usage Example

            >>> gis.content.share_items(items=[item1, item2, item3], everyone=True, org=True,
                                        groups = ["Developers", "Engineers", "GIS_Analysts"])

        """
        res = True
        if groups is None and everyone is None and org is None:
            return True
        if groups:
            url = "{base}content/users/{username}/unshareItems".format(
                base=self._portal.resturl,
                username=self._gis.users.me.username,
            )
            params = {"f": "json"}
            if isinstance(groups, (list, tuple)) == False:
                groups = [groups]
            if isinstance(items, (list, tuple)) == False:
                items = [items]
            if isinstance(groups, (tuple, list)):
                grps = []
                for grp in groups:
                    if isinstance(grp, str):
                        grps.append(grp)
                    elif isinstance(grp, Group):
                        grps.append(grp.groupid)
                groups = grps
            if isinstance(items, (tuple, list)):
                sitems = []
                for i in items:
                    if isinstance(i, str):
                        sitems.append(i)
                    elif isinstance(i, Item):
                        sitems.append(i.itemid)
                    # items = sitems
            params["groups"] = ",".join(groups)
            params["items"] = ",".join(sitems)
            res = self._gis._con.post(url, params)
        if everyone is not None and org is not None:
            for item in items:
                if isinstance(item, Item):
                    item.share(everyone=everyone, org=org)
                elif isinstance(item, str):
                    Item(gis=self._gis, itemid=item).share(everyone=everyone, org=org)
        elif everyone is not None and org is None:
            for item in items:
                if isinstance(item, Item):
                    org = item.shared_with["org"]
                    item.share(everyone=everyone, org=org)
                if isinstance(item, str):
                    usitem = Item(gis=self._gis, itemid=item)
                    org = usitem.shared_with["org"]
                    usitem.share(everyone=everyone, org=org)
        elif everyone is None and org is not None:
            for item in items:
                if isinstance(item, Item):
                    everyone = item.shared_with["everyone"]
                    item.share(everyone=everyone, org=org)
                if isinstance(item, str):
                    usitem = Item(gis=self._gis, itemid=item)
                    everyone = usitem.shared_with["everyone"]
                    usitem.share(everyone=everyone, org=org)
        for item in items:
            if isinstance(item, Item):
                item._hydrated = False
            if isinstance(item, str):
                Item(gis=self._gis, itemid=item)._hydrated = False
        return res


########################################################################
class CategorySchemaManager(object):
    """
    The ``CategorySchemaHelper`` class is for managing category schemas. An instance of this class, called `categories`,
    is available as a property in the :class:`~arcgis.gis.ContentManager` and the :class:`~arcgis.gis.Group` class. See
    :attr:`~gis.ContentManager.categories` and :attr:`~arcgis.gis.Group.categories` for more information.

    .. note::
        This class is not created by users directly.
    """

    _gis = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, base_url, gis=None):
        """Constructor"""
        self._url = base_url
        if gis is None:
            gis = arcgis_env.active_gis
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< CategorySchemaManager at {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """The ``properties`` method retrieves the properties of the schema."""

        return _mixins.PropertyMap(self.schema)

    # ----------------------------------------------------------------------
    @property
    def schema(self):
        """
        The ``schema`` property allows group owners/managers to manage the content
        categories for a group. These content categories are a hierarchical
        set of classes to help organize and browse group content.

        .. note::
            Each group can have a maximum of 5 category trees with each
            category schema can have up to 4 hierarchical levels. The maximum
            number of categories a group can have in total is 200 with each
            category of less than 100 characters title and 300 characters
            description.

        When getting ``schema``, returns the content category schema
        set on a group.

        When setting ``schema``, will update the group category schema
        based on the `dict` this property is set to. See below.

        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        categories          Required Dict. A category schema object consists of an
                            array of dict objects representing top level categories.
                            Each object has title, description and categories
                            properties where categories consists of an array of
                            objects with each having the same properties and
                            represents the descendant categories or subcategories and
                            so on.
        ==================  =========================================================
        """
        params = {"f": "json"}
        url = "{base}/categorySchema".format(base=self._url)
        res = self._gis._con.get(url, params)
        if "categorySchema" in res:
            return res["categorySchema"]
        return res

    # ----------------------------------------------------------------------
    @schema.setter
    def schema(self, categories):
        """See main `schema` property docstring"""
        if categories is None:
            return self.delete()
        elif len(categories) == 0:
            return self.delete()
        params = {
            "f": "json",
            "categorySchema": {"categorySchema": categories},
        }
        url = "{base}/assignCategorySchema".format(base=self._url)
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return

    # ----------------------------------------------------------------------
    def delete(self):
        """
        The ``delete`` function allows group owner or managers to remove the
        category schema set on a group.

        .. code-block:: python

            # Usage Example

            >>> gis.content.categories.delete()

        :return:
            A boolean indicating success (True), or failure (False)
        """
        params = {"f": "json"}
        url = "{base}/deleteCategorySchema".format(base=self._url)
        try:
            if self.schema == []:
                return True
            res = self._gis._con.post(url, params)
            if "success" in res:
                return res["success"]
            return False
        except:
            return False

    # ----------------------------------------------------------------------
    def assign_to_items(
        self, items: Union[list[str], list[Item], list[dict[str, Any]]]
    ):
        """
        The ``assign_to_items`` function adds group content categories to the portal items
        specified in the `items` argument (see below). For assigning categories
        to items in a group, you must be the group owner/manager. For assigning
        organization content categories on items, you must be the item owner
        or an administrator who has the `portal:admin:updateItems` privilege.

        .. note::
            A maximum of 100 items can be bulk updated per request.

        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        items               Required List. A JSON array of item objects. Each is
                            specified with the item ID that consists of a categories
                            object. categories is specified with an array that lists
                            all content categories to update on the item, each with
                            full hierarchical path prefixed with /.

                            Each item can be categorized to a maximum of 20
                            categories.

        ==================  =========================================================

        :return:
            A `dict` of `item_id` : `status`, with `status` being
            whether the content categories were successfully added. If the `status` is
            unsuccessfully updated, a message will provide information to help you debug
            the issue.

        .. code-block:: python

            # Usage Example

            >>> gis.content.categories.assign_to_items(items = [{"2678d3002eea4e4a825e3bdf10016e61": {
                                                                 "categories": ["/Categories/Geology",
                                                                                "/Categories/Elevation"]}},
                                                                {"c3ad4ed8bcf04d619537cfe252a1760d": {
                                                                 "categories": ["/Categories/Geology",
                                                                                "/Categories/Land cover/Forest/Deciduous Forest"]}},
                                                                 {"9ced00fdce3e4b20bb4b05155acbe817": {
                                                                 "categories": []}}])

        """
        params = {
            "f": "json",
        }
        if self._url.lower().find("/portals/") == -1:
            # If this SchemaManager is attached to a GroupManager
            cats = []
            for val in items:
                for key in val.keys():
                    cats.append({key: val[key]["categories"]})
            params["items"] = json.dumps(cats)
            group = os.path.basename(self._url)
            url = f"{self._gis._portal.resturl}content/groups/{group}/updateCategories"
        else:
            # else this SchemaManager is attached to a ContentManager
            params["items"] = json.dumps(items)
            url = "{base}content/updateItems".format(base=self._gis._portal.resturl)
        response = self._gis._con.post(url, params)
        output = {}
        if "results" in response:
            return response["results"]
            # for res in response['results']:
            # if 'success' in res and 'itemId' in res:
            # output[res['itemId']] = res['success']
        return response


class ResourceManager(object):
    """
    The ``ResourceManager`` class is a helper class for managing resource files of an item.
    An instance of this class is available as a property of the :class:`~arcgis.gis.Item` object
    (See :attr:`~arcgis.gis.Item.resources` for more information on this property).
    Users call methods on this :attr:`~arcgis.gis.Item.resources` object to manage
    (add, remove, update, list, get) item resources.

    .. note::
        Users do not create this class directly.
    """

    _user_id = None

    def __init__(self, item, gis):
        self._gis = gis
        self._portal = gis._portal
        self._item = item

        owner = self._item.owner
        user = gis.users.get(owner)
        if (hasattr(user, "id")) and (user.id != "null"):
            self._user_id = user.username
            # self._user_id = user.id
        else:
            self._user_id = user.username

    def export(
        self,
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        """
        The ``export`` method export's the data's resources as a zip file

        .. code-block:: python

            # Usage Example

            >>> Item.resources.export("file_name")

        :return:
            A .zip file containing the data's resources
        """
        url = (
            "content/users/"
            + self._user_id
            + "/items/"
            + self._item.itemid
            + "/resources/export"
        )

        if save_path is None:
            save_path = tempfile.gettempdir()
        if file_name is None:
            import uuid

            file_name = f"{uuid.uuid4().hex[:6]}.zip"
        params = {"f": "zip"}
        # from arcgis.gis._impl._con import Connection
        con = self._portal.con
        # isinstance(con, Connection)
        resources = con.get(
            url,
            params=params,
            out_folder=save_path,
            file_name=file_name,
            try_json=False,
        )
        return resources

    def add(
        self,
        file: Optional[str] = None,
        folder_name: Optional[str] = None,
        file_name: Optional[str] = None,
        text: Optional[str] = None,
        archive: bool = False,
        access: Optional[str] = None,
        properties: Optional[dict] = None,
    ):
        """
        The ``add`` operation adds new file resources to an existing item. For example, an image that is
        used as custom logo for Report Template. All the files are added to 'resources' folder of the item. File
        resources use storage space from your quota and are scanned for viruses. The item size is updated to
        include the size of added resource files.

        .. note::
            Each file added should be no more than 25 Mb.

        Supported item types that allow adding file resources are: Vector Tile Service, Vector Tile Package,
        Style, Code Attachment, Report Template, Web Mapping Application, Feature Service, Web Map,
        Statistical Data Collection, Scene Service, and Web Scene.

        Supported file formats are: JSON, XML, TXT, PNG, JPEG, GIF, BMP, PDF, MP3, MP4, and ZIP.
        This operation is only available to the item owner and the organization administrator.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Optional string. The path to the file that needs to be added.
        ----------------  ---------------------------------------------------------------
        folder_name       Optional string. Provide a folder name if the file has to be
                          added to a folder under resources.
        ----------------  ---------------------------------------------------------------
        file_name         Optional string. The file name used to rename an existing file
                          resource uploaded, or to be used together with text as file name for it.
        ----------------  ---------------------------------------------------------------
        text              Optional string. Text input to be added as a file resource,
                          used together with file_name. If this resource is used, then
                          file_name becomes required.
        ----------------  ---------------------------------------------------------------
        archive           Optional boolean. Default is False.  If True, file resources
                          added are extracted and files are uploaded to respective folders.
        ----------------  ---------------------------------------------------------------
        access            Optional String. Set file resource to be private regardless of
                          the item access level, or revert it by setting it to `inherit`
                          which makes the item resource have the same access as the item.

                          Supported values: `private` or `inherit`.
        ----------------  ---------------------------------------------------------------
        properties        Optional Dictionary. Set the properties for the resources such
                          as the `editInfo`.
        ================  ===============================================================

        :return:
            Python dictionary in the following format (if successful):
            {
                "success": True,
                "itemId": "<item id>",
                "owner": "<owner username>",
                "folder": "<folder id>"}

            else like the following if it failed:
            {"error": {
                        "code": 400,
                        "messageCode": "CONT_0093",
                        "message": "File type not allowed for addResources",
                        "details": []
                        }}

         .. code-block:: python

            # Usage Example

            >>> Item.resources.add("file_path", "folder_name", "file_name", access = "private")

        """
        if not file and (not text or not file_name):
            raise ValueError("Please provide a valid file or text/file_name.")
        query_url = (
            "content/users/"
            + self._user_id
            + "/items/"
            + self._item.itemid
            + "/addResources"
        )

        files = []  # create a list of named tuples to hold list of files
        if file and os.path.isfile(os.path.abspath(file)):
            files.append(("file", file, os.path.basename(file)))
        elif file and os.path.isfile(os.path.abspath(file)) == False:
            raise RuntimeError("File(" + file + ") not found.")

        params = {}
        params["f"] = "json"

        if folder_name is not None:
            params["resourcesPrefix"] = folder_name
        if file_name is not None:
            params["fileName"] = file_name
        if text is not None:
            params["text"] = text
        params["archive"] = "true" if archive else "false"
        if isinstance(properties, dict):
            params["properties"] = properties
        if access and str(access) in ["inherit", "private"]:
            params["access"] = access
        # IF properties passed in, add them to params
        resp = self._portal.con.post(query_url, params, files=files, compress=False)
        return resp

    def update(
        self,
        file: Optional[str] = None,
        folder_name: Optional[str] = None,
        file_name: Optional[str] = None,
        text: Optional[str] = None,
        properties: Optional[dict[Any, Any]] = None,
    ):
        """The ``update`` operation allows you to update existing file resources of an item.
        File resources use storage space from your quota and are scanned for viruses. The item size
        is updated to include the size of updated resource files.

        Supported file formats are: JSON, XML, TXT, PNG, JPEG, GIF, BMP, PDF, and ZIP.
        This operation is only available to the item owner and the organization administrator.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Required string. The path to the file on disk to be used for
                          overwriting an existing file resource.
        ----------------  ---------------------------------------------------------------
        folder_name       Optional string. Provide a folder name if the file resource
                          being updated resides in a folder.
        ----------------  ---------------------------------------------------------------
        file_name         Optional string. The destination name for the file used to update
                          an existing resource, or to be used together with the text parameter
                          as file name for it.

                          For example, you can use fileName=banner.png to update an existing
                          resource banner.png with a file called billboard.png without
                          renaming the file locally.
        ----------------  ---------------------------------------------------------------
        text              Optional string. Text input to be added as a file resource,
                          used together with file_name.
        ----------------  ---------------------------------------------------------------
        properties        Optional Dictionary. Set the properties for the resources such
                          as the `editInfo`.
        ================  ===============================================================

        :return:
            If successful, a dictionary with  will be returned in the following format:
            {
                "success": True,
                "itemId": "<item id>",
                "owner": "<owner username>",
                "folder": "<folder id>" }

            else a dictionary with error information will be returned in the following format:
            {"error": {
                        "code": 404,
                        "message": "Resource does not exist or is inaccessible.",
                        "details": []
                        } }

        .. code-block:: python

            # Usage Example

            >>> Item.resources.add("file_path", "folder_name", "file_name")

        """

        query_url = (
            "content/users/"
            + self._user_id
            + "/items/"
            + self._item.itemid
            + "/updateResources"
        )

        files = []  # create a list of named tuples to hold list of files
        if file:
            if not os.path.isfile(os.path.abspath(file)):
                raise RuntimeError("File(" + file + ") not found.")
            files.append(("file", file, os.path.basename(file)))

        params = {}
        params["f"] = "json"

        if folder_name is not None:
            params["resourcesPrefix"] = folder_name
        if file_name is not None:
            params["fileName"] = file_name
        if text is not None:
            params["text"] = text
        if isinstance(properties, dict):
            params["properties"] = properties
        resp = self._portal.con.post(query_url, params, files=files)
        return resp

    def list(self):
        """
        The ``list`` method provides a lists all file resources of an existing item.

        .. note::
            This resource is only available to
            the item owner and the organization administrator.

        :return:
            A Python list of dictionaries of the form:
            [
                {
                  "resource": "<resource1>"
                },
                {
                  "resource": "<resource2>"
                },
                {
                  "resource": "<resource3>"
                }
            ]
        """
        query_url = "content/items/" + self._item.itemid + "/resources"
        params = {"f": "json", "num": 1000}
        resp = self._portal.con.get(query_url, params)
        resp_resources = resp.get("resources")
        count = int(resp.get("num"))
        next_start = int(
            resp.get("nextStart", -999)
        )  # added for back support for portal (10.4.1)

        # loop through pages
        while next_start > 0:
            params2 = {"f": "json", "num": 1000, "start": next_start + 1}

            resp2 = self._portal.con.get(query_url, params2)
            resp_resources.extend(resp2.get("resources"))
            count += int(resp2.get("num"))
            next_start = int(
                resp2.get("nextStart", -999)
            )  # added for back support for portal (10.4.1)
            if next_start == -999:
                break

        return resp_resources

    def get(
        self,
        file: str,
        try_json: bool = True,
        out_folder: Optional[str] = None,
        out_file_name: Optional[str] = None,
    ):
        """
        The ``get`` method retrieves a specific file resource of an existing item.

        .. note::
            This operation is only available to the item owner and the organization administrator.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Required string. The path to the file to be downloaded.
                          For files in the root, just specify the file name. For files in
                          folders (prefixes), specify using the format
                          <foldername>/<foldername>./../<filename>
        ----------------  ---------------------------------------------------------------
        try_json          Optional boolean. If True, will attempt to convert JSON files to
                          Python dictionary objects. Default is True.
        ----------------  ---------------------------------------------------------------
        out_folder        Optional string. Specify the folder into which the file has to
                          be saved. Default is user's temporary directory.
        ----------------  ---------------------------------------------------------------
        out_file_name     Optional string. Specify the name to use when downloading the
                          file. Default is the resource file's name.
        ================  ===============================================================

        :return:
           Path to the downloaded file if getting a binary file (like a jpeg or png file) or if
           try_jon = False when getting a JSON file.

           If file is a JSON, returns as a Python dictionary.

        .. code-block:: python

            # Usage Example

            >>> Item.resources.get("file_path", try_json=True, out_folder="out_folder_name")

        """
        out_folder: str = out_folder or tempfile.gettempdir()
        safe_file_format: str = file.replace(r"\\", "/")
        safe_file_format: str = safe_file_format.replace("//", "/")

        query_url: str = (
            "content/items/" + self._item.itemid + "/resources/" + safe_file_format
        )

        resp: requests.Response = self._portal.con.get(
            query_url,
            try_json=try_json,
            out_folder=out_folder,
            file_name=out_file_name,
            return_raw_response=True,
        )
        if resp.status_code == 200:
            if (
                resp.headers["Content-Type"].lower().find("json") > -1
                and resp.text.find("Resource does not exist or is inaccessible.") > -1
            ):
                raise Exception(resp.text)
            elif resp.headers["Content-Type"].lower().find("json") > -1 and try_json:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return resp.text
            else:
                from arcgis.gis._impl._con._helpers import (
                    _filename_from_headers,
                    _filename_from_url,
                )
                from requests_toolbelt.downloadutils import stream

                if out_file_name is None:
                    out_file_name = _filename_from_headers(
                        resp.headers
                    ) or _filename_from_url(resp.url)
                file_name: str = os.path.join(out_folder, out_file_name)
                if os.path.isfile(file_name):
                    os.remove(file_name)
                stream_size: int = 512 * 2
                if "Content-Length" in resp.headers:
                    max_length: int = int(resp.headers["Content-Length"])
                    if max_length > stream_size * 2 and max_length < 1024 * 1024:
                        stream_size = 1024 * 2
                    elif max_length > 5 * (1024 * 1024):
                        stream_size = 5 * (1024 * 1024)  # 5 mb
                    elif max_length > (1024 * 1024):
                        stream_size = 1024 * 1024  # 1 mb
                    else:
                        stream_size = 512 * 2

                fp: str = stream.stream_response_to_file(
                    response=resp, path=file_name, chunksize=stream_size
                )
                return fp
        else:
            raise Exception("Resource does not exist or is inaccessible.")

    def remove(self, file: Optional[str] = None):
        """
        The ``remove`` method removes a single resource file or all resources. The item size is updated once
        resource files are deleted.

        .. note::
            This operation is only available to the item owner
            and the organization administrator.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        file              Optional string. The path to the file to be removed.
                          For files in the root, just specify the file name. For files in
                          folders (prefixes), specify using the format
                          <foldername>/<foldername>./../<filename>

                          If not specified, all resource files will be removed.
        ================  ===============================================================

        :return:
            If successful, a boolean of True will be returned.

            Else, a dictionary with error information will be returned in the following format:
            {"error": {"code": 404,
                        "message": "Resource does not exist or is inaccessible.",
                        "details": []
                      }
            }

        .. code-block:: python

            # Usage Example

            >>> Item.resources.remove("file_path")
        """
        safe_file_format = ""
        delete_all = "false"
        if file:
            safe_file_format = file.replace(r"\\", "/")
            safe_file_format = safe_file_format.replace("//", "/")
        else:
            delete_all = "true"

        query_url = (
            "content/users/"
            + self._user_id
            + "/items/"
            + self._item.itemid
            + "/removeResources"
        )
        params = {
            "f": "json",
            "resource": safe_file_format if safe_file_format else "",
            "deleteAll": delete_all,
        }
        res = self._portal.con.post(query_url, postdata=params)
        if "success" in res:
            return res["success"]
        return res


class Group(dict):
    """
    The ``Group`` class is an object that represents a group within the GIS, either ArcGIS Online or ArcGIS Enterprise.
    """

    def __init__(self, gis, groupid, groupdict=None):
        dict.__init__(self)
        self._gis = gis
        self._migrate = None
        self._portal = gis._portal
        self.groupid = groupid
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        # groupdict = self._portal.get_group(self.groupid)
        self._hydrated = False
        if groupdict:
            groupdict.update(self.__dict__)
            super(Group, self).update(groupdict)

    def _hydrate(self):
        groupdict = self._portal.get_group(self.groupid)
        self._hydrated = True
        super(Group, self).update(groupdict)
        self.__dict__.update(groupdict)

    def __getattr__(
        self, name
    ):  # support group attributes as group.access, group.owner, group.phone etc
        if not self._hydrated and not name.startswith("_"):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def __getitem__(
        self, k
    ):  # support group attributes as dictionary keys on this object, eg. group['owner']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith("_"):
                self._hydrate()
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    def __repr__(self):
        return '<%s title:"%s" owner:%s>' % (
            type(self).__name__,
            self.title,
            self.owner,
        )

    def get_thumbnail_link(self):
        """
        The ``get_thumbnail_link`` method retrieves the URL to the thumbnail image.

        :return:
            A URL linked to the thumbnail image.
        """

        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            return self._gis.url + "/home/images/group-no-image.png"
        else:
            thumbnail_url_path = (
                self._gis._public_rest_url
                + "community/groups/"
                + self.groupid
                + "/info/"
                + thumbnail_file
            )
            return thumbnail_url_path

    def search(
        self,
        query: str,
        return_count: bool = False,
        max_items: int = 100,
        bbox: Optional[Union[list[str], str]] = None,
        categories: Optional[str] = None,
        category_filter: Optional[str] = None,
        start: int = 1,
        sort_field: str = "title",
        sort_order: str = "ASC",
        as_dict: True = False,
    ):
        """
        The ``search`` operation allows users to find content within the specific group.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        query               Required String.  The search query. When the search filters
                            contain two or more clauses, the recommended schema is to have
                            clauses separated by blank, or `AND`, e.g.

                            :Usage Example:


                            group.search(query='owner:USERNAME type:map')
                            # or
                            group.search(query='type:map AND owner:USERNAME')

                            .. warning::
                            When the clauses are separated by comma, the filtering condition
                            for `owner` should not be placed at the first position, e.g.
                            `group.search(query='type:map, owner:USERNAME')`
                            is allowed, while
                            `group.search(query='owner:USERNAME, type:map')`
                            is not. For more, please check
                            https://developers.arcgis.com/rest/users-groups-and-items/search-reference.htm
        ----------------    ---------------------------------------------------------------
        bbox                Optional String/List. This is the xmin,ymin,xmax,ymax bounding
                            box to limit the search in.  Items like documents do not have
                            bounding boxes and will not be included in the search.
        ----------------    ---------------------------------------------------------------
        categories          Optional String. A comma separated list of up to 8 org content
                            categories to search items. Exact full path of each category is
                            required, OR relationship between the categories specified.

                            Each request allows a maximum of 8 categories parameters with
                            AND relationship between the different categories parameters
                            called.
        ----------------    ---------------------------------------------------------------
        category_filters    Optional String. A comma separated list of up to 3 category
                            terms to search items that have matching categories. Up to 2
                            `category_filters` parameter are allowed per request. It can
                            not be used together with categories to search in a request.
        ----------------    ---------------------------------------------------------------
        start               Optional Int. The starting position to search from.  This is
                            only required if paging is needed.
        ----------------    ---------------------------------------------------------------
        sort_field          Optional String. Responses from the `search` operation can be
                            sorted on various fields. `avgrating` is the default.
        ----------------    ---------------------------------------------------------------
        sort_order          Optional String. The sequence into which a collection of
                            records are arranged after they have been sorted. The allowed
                            values are: asc for ascending and desc for descending.
        ----------------    ---------------------------------------------------------------
        as_dict             Required Boolean. If True, the results comes back as a dictionary.
                            The result of the method will always be a dictionary but the
                            `results` key in the dictionary will be changed if set to False.
        ================    ===============================================================

        :return: List of :class:`~arcgis.gis.Item` objects

        .. code-block:: python

            # Usage Example

            >>> group.search("Hurricane Data", category_filters =["Natural_Disasters", "Hurricanes", "USA"])

        """
        from ._impl._search import _search

        if return_count:
            return _search(
                gis=self._gis,
                query=query,
                stype="group_content",
                max_items=max_items,
                bbox=bbox,
                categories=categories,
                category_filter=category_filter,
                start=start,
                sort_field=sort_field,
                sort_order=sort_order,
                group_id=self.id,
                as_dict=True,
            )["total"]
        return _search(
            gis=self._gis,
            query=query,
            stype="group_content",
            max_items=max_items,
            bbox=bbox,
            categories=categories,
            category_filter=category_filter,
            start=start,
            sort_field=sort_field,
            sort_order=sort_order,
            group_id=self.id,
            as_dict=as_dict,
        )

    # ----------------------------------------------------------------------
    @property
    def categories(self):
        """
        The ``categories`` property serves as the category manager for groups.
        See :class:`~arcgis.gis.CategorySchemaManager` for more information on category managers.
        """
        base_url = "{base}community/groups/{groupid}".format(
            base=self._gis._portal.resturl, groupid=self.groupid
        )
        return CategorySchemaManager(base_url=base_url, gis=self._gis)

    @property
    def homepage(self):
        """
        The ``homepage`` method retrieves the URL to the HTML page for the group.

        :return:
            A URL linking to the group HTML page.
        """
        return "{}{}{}".format(self._gis.url, "/home/group.html?id=", self.groupid)

    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        else:
            b64 = base64.b64encode(self.get_thumbnail())
            thumbnail = "data:image/png;base64," + str(b64, "utf-8") + "' "

        title = "Not Provided"
        snippet = "Not Provided"
        description = "Not Provided"
        owner = "Not Provided"
        try:
            title = self.title
        except:
            title = "Not Provided"

        try:
            description = self.description
        except:
            description = "Not Provided"

        try:
            snippet = self.snippet
        except:
            snippet = "Not Provided"

        try:
            owner = self.owner
        except:
            owner = "Not available"

        url = self.homepage

        return (
            """<div class="9item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='"""
            + str(url)
            + """' target='_blank'>
                        <img src='"""
            + str(thumbnail)
            + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right" style="float: none; width: auto; overflow: hidden;">
                        <a href='"""
            + str(url)
            + """' target='_blank'><b>"""
            + str(title)
            + """</b>
                        </a>
                        <br/>
                        <br/><b>Summary</b>: """
            + str(snippet)
            + """
                        <br/><b>Description</b>: """
            + str(description)
            + """
                        <br/><b>Owner</b>: """
            + str(owner)
            + """
                        <br/><b>Created</b>: """
            + str(datetime.fromtimestamp(self.created / 1000).strftime("%B %d, %Y"))
            + """

                    </div>
                </div>
                """
        )

    def content(self, max_items: int = 1000):
        """
        The ``content`` method retrieves the list of items shared with this group.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        max_items              Required integer. The maximum number of items to be returned, defaults to 1000.
        ==================     ====================================================================


        :return:
           The list of items that are shared.

        """
        itemlist = []
        items = self._portal.search(
            "group:" + self.groupid, max_results=max_items, outside_org=True
        )
        for item in items:
            itemlist.append(Item(self._gis, item["id"], item))
        return itemlist

    def delete(self):
        """
        The ``delete`` method deletes this group permanently.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        return self._portal.delete_group(self.groupid)

    def get_thumbnail(self):
        """
        The ``get_thumbnail`` method retrieves the bytes that make up the thumbnail for this group.


        :return:
            Bytes that represent the image.

        Example

        .. code-block:: python

            response = group.get_thumbnail()
            f = open(filename, 'wb')
            f.write(response)
        """
        return self._portal.get_group_thumbnail(self.groupid)

    @property
    def migration(self):
        """
        The ``migration`` property accesses a :class:`~arcgis.gis.GroupMigrationManager`
        object which has methods for exporting and importing supported ``group`` content
        between ArcGIS Enterprise organizations.

        .. note::
            Functionality only available for ArcGIS Enterprise.

        .. code-block:: python

            #Usage Example: Initializing a ``GroupMigrationManager`` object:

            >>> from arcgis.gis import GIS

            >>> gis = GIS(profile="your_enterprise_admin_profile")

            >>> source_grp = gis.groups.get("<group_id>")

            >>> grp_mig_mgr = source_grp.migration
            >>> type(grp_mig_mgr)

            arcgis.gis.GroupMigrationManager
        """
        if self._gis.version > [7, 3] and self._gis._portal.is_arcgisonline == False:
            self._migrate = GroupMigrationManager(group=self)
        return self._migrate

    def download_thumbnail(self, save_folder: Optional[str] = None):
        """
        The ``download_thumbnail`` method downloads the item thumbnail for this user and saves it in the folder that
        is passed when ``download_thumbnail`` is called.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        save_folder            Optional string. The file path to where the group thumbnail will be downloaded.
        ==================     ====================================================================


        :return:
           The file path to which the group thumbnail is downloaded.
        """
        if self.thumbnail is None:
            self._hydrate()
        thumbnail_file = self.thumbnail
        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = (
                "community/groups/" + self.groupid + "/info/" + thumbnail_file
            )
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50:  # If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                self._portal.con.get(
                    path=thumbnail_url_path,
                    try_json=False,
                    out_folder=save_folder,
                    file_name=file_name,
                )
                return file_path

        else:
            return None

    def add_users(
        self,
        usernames: Optional[Union[list[str], str]] = None,
        admins: Optional[Union[list[str], str]] = None,
    ):
        """
        The ``adds_users`` method adds users to this group.

        .. note::
            The ``add_users`` method will only work if the user for the
            Portal object is either an administrator for the entire
            Portal or the owner of the group.

        =============   =====================================
        **Parameter**    **Description**
        -------------   -------------------------------------
        usernames       Optional list of strings or single string.
                        The list of usernames or single username
                        to be added.
        -------------   -------------------------------------
        admins          Optional List of String, or Single String.
                        This is a list of users to be an administrator
                        of the group.
        =============   =====================================

        :return:
           A dictionary containing the users that were not added to the group.

        .. code-block:: python

            # Usage Example

            >>> group.add_users(usernames=["User1234","User5678"], admin="Admin9012")
        """
        if usernames is None and admins is None:
            return {"notAdded": []}

        users = None
        ladmins = None

        if isinstance(usernames, (list, tuple)) == False:
            usernames = [usernames]

        if admins and isinstance(admins, (list, tuple)) == False:
            admins = [admins]

        if admins:
            ladmins = []
            for u in admins:
                if isinstance(u, str):
                    ladmins.append(u)
                elif isinstance(u, User):
                    ladmins.append(u.username)
        if usernames:
            users = []
            for u in usernames:
                if isinstance(u, str):
                    users.append(u)
                elif isinstance(u, User):
                    users.append(u.username)
        n = 25
        results = {"notAdded": []}
        if users:
            users_added = [
                self._portal.add_group_users(
                    users[i * n : (i + 1) * n], self.groupid, []
                )["notAdded"]
                for i in range((len(users) + n - 1) // n)
            ]
            [results["notAdded"].extend(a) for a in users_added]
        if ladmins:
            admins_added = [
                self._portal.add_group_users(
                    [], self.groupid, ladmins[i * n : (i + 1) * n]
                )["notAdded"]
                for i in range((len(ladmins) + n - 1) // n)
            ]
            [results["notAdded"].extend(a) for a in admins_added]
        return results

    def delete_group_thumbnail(self):
        """
        The ``delete_group_thumbnail`` method deletes the group's thumbnail.

        :return:
            A boolean indicating success (True) or failure (False).

        """
        return self._portal.delete_group_thumbnail(self.groupid)

    def remove_users(self, usernames: Union[list[str], str]):
        """
        The ``remove_users`` method is used to remove users from this group.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        usernames         Required list of strings. A comman-separated list of
                          users to be removed.
        ================  ========================================================

        :return:
            A dictionary with a key notRemoved that is a list of users not removed.
        """
        users = []
        if isinstance(usernames, (list, tuple)) == False:
            usernames = [usernames]
        for u in usernames:
            if isinstance(u, str):
                users.append(u)
            elif isinstance(u, User):
                users.append(u.username)

        return self._portal.remove_group_users(users, self.groupid)

    # ----------------------------------------------------------------------
    def update_users_roles(
        self,
        managers: Optional[list[User]] = None,
        users: Optional[list[User]] = None,
    ) -> list:
        """
        The ``update_users_roles`` upgrades a set of users to become either Group Members or Group Managers.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        managers          Required List.  A comma-separated array of User objects to upgrade to a group manager role.
        ----------------  --------------------------------------------------------
        users             Required List.  A comma-separated array of User objects to make group member roles.
        ================  ========================================================

        :return: List[dictionary]

        """
        params = {
            "admins": managers or [],
            "users": users or [],
            "f": "json",
        }
        if managers is None and users is None:
            return {"results": []}
        if isinstance(params["admins"], (list, tuple)):
            managers = params["admins"]
            for idx, a in enumerate(managers):
                if isinstance(a, str):
                    managers[idx] = a
                elif isinstance(a, User):
                    managers[idx] = a.username
                else:
                    raise ValueError(
                        "'admins' must be a list of strings or User objects"
                    )
            params["admins"] = ",".join(managers)
        else:
            raise ValueError("'admins' must be a list of strings or User objects")

        if isinstance(params["users"], (list, tuple)):
            users = params["users"]
            for idx, a in enumerate(users):
                if isinstance(a, str):
                    users[idx] = a
                elif isinstance(a, User):
                    users[idx] = a.username
                else:
                    raise ValueError(
                        "'admins' must be a list of strings or User objects"
                    )
            params["users"] = ",".join(users)
        else:
            raise ValueError("'admins' must be a list of strings or User objects")

        url = "community/groups/" + self.groupid + "/updateUsers"
        return self._portal.con.post(url, params)

    def invite_users(
        self,
        usernames: list[str],
        role: str = "group_member",
        expiration: int = 10080,
    ):
        """
        The ``invite_users`` method invites existing users to this group.

        .. note::
            The user executing the ``invite_users`` command must be the owner of the group.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        usernames         Required list of strings. The users to invite.
        ----------------  --------------------------------------------------------
        role              Optional string. Either group_member (the default) or group_admin.
        ----------------  --------------------------------------------------------
        expiration        Optional integer. Specifies how long the invitation is
                          valid for in minutes.  Default is 10,080 minutes (7 days).
        ================  ========================================================

        .. note::
            A user who is invited to this group will see a list of invitations
            in the "Groups" tab of Portal listing invitations. The user
            can either accept or reject the invitation.

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> group.invite_users(usernames=["User1234","User5678"], role="group_admin")
        """
        return self._portal.invite_group_users(
            usernames, self.groupid, role, expiration
        )

    # ----------------------------------------------------------------------
    @_common_deprecated.deprecated(
        deprecated_in="v1.5.1",
        removed_in=None,
        current_version=None,
        details="Use `Group.invite` instead.",
    )
    def invite_by_email(
        self,
        email: str,
        message: str,
        role: str = "member",
        expiration: str = "1 Day",
    ):
        """
        .. Warning::
            Deprecated: The ``invite_by_email`` function is no longer supported.

        The ``invite_by_email`` method invites a user by email to the existing group.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        email             Required string. The user to send join email to.
        ----------------  --------------------------------------------------------
        message           Required string. The message to send to the user.
        ----------------  --------------------------------------------------------
        role              Optional string. Either member (the default) or admin.
        ----------------  --------------------------------------------------------
        expiration        Optional string.  The is the time out of the invite.
                          The values are: 1 Day (default), 3 Days, 1 Week, or
                          2 Weeks.
        ================  ========================================================

        :return: A boolean indicating success (True) or failure (False)
        """

        if self._gis.version >= [6, 4]:
            return False

        time_lookup = {
            "1 Day".upper(): 1440,
            "3 Days".upper(): 4320,
            "1 Week".upper(): 10080,
            "2 Weeks".upper(): 20160,
        }
        role_lookup = {"member": "group_member", "admin": "group_admin"}
        url = "community/groups/" + self.groupid + "/inviteByEmail"
        params = {
            "f": "json",
            "emails": email,
            "message": message,
            "role": role_lookup[role.lower()],
            "expiration": time_lookup[expiration.upper()],
        }
        return self._portal.con.post(url, params)

    def reassign_to(self, target_owner: Union[str, User]):
        """
        The ``reassign_to`` method reassigns this group from its current owner to another owner.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        target_owner      Required string or User.  The username of the new group owner.
        ================  ========================================================

        :return:
            A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json"}
        if isinstance(target_owner, User):
            params["targetUsername"] = target_owner.username
        else:
            params["targetUsername"] = target_owner
        res = self._gis._con.post(
            "community/groups/" + self.groupid + "/reassign", params
        )
        if res:
            self._hydrated = False
            self._hydrate()
            return res.get("success")
        return False

    # ----------------------------------------------------------------------
    def notify(
        self,
        users: Union[list[str], list[User]],
        subject: str,
        message: str,
        method: str = "email",
        client_id: Optional[str] = None,
    ):
        """
        The ``notify`` method creates a group notification that sends a message to all users within
        the group.

        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        users               Required List. A list of users or user names.
        ------------------  ---------------------------------------------------------
        subject             Required String. The subject of the notification.
        ------------------  ---------------------------------------------------------
        message             Required String. The message body that will be sent to
                            the group's users.
        ------------------  ---------------------------------------------------------
        method              Optional String. This is the form for which users will be
                            contacted.  The allowed values are: email, push, and
                            builtin.

                            + email - sent a message via smtp.
                            + push - pushes a message out.
                            + builtin - creates a user notification.


        ------------------  ---------------------------------------------------------
        client_id           Optional String. The client id of the application for the
                            push operation.
        ==================  =========================================================

        :return: A boolean indicating success (True), or failure (False).

        .. code-block:: python

            # Usage Example

            >>> group.notify(users=["User1234"], subject= "Test Message", message="Testing the notification system",
            >>>              method="email"

        """
        from arcgis.gis import User

        cusers = []
        for user in users:
            if isinstance(user, User):
                cusers.append(user.username)
            else:
                cusers.append(user)
            del user
        url = "community/groups/{groupid}/createNotification".format(
            groupid=self.groupid
        )
        params = {
            "notificationChannelType": method,
            "subject": subject,
            "message": message,
            "users": ",".join(cusers),
            "clientId": client_id,
            "f": "json",
        }
        return self._gis._con.post(url, params)

    def get_members(self):
        """
        The ``get_members`` method retrieves the members of this group.


            ================  ========================================================
            **Key**           **Value**
            ----------------  --------------------------------------------------------
            owner             The group's owner (string).
            ----------------  --------------------------------------------------------
            admins            The group's admins (list of strings). Typically this is the same as the owner.
            ----------------  --------------------------------------------------------
            users             The members of the group (list of strings).
            ================  ========================================================


        :return:
            A dictionary with keys: owner, admins, and users.


        .. code-block:: python

            # Usage Example: To print users in a group

            response = group.get_members()
            for user in response['users'] :
                print(user)

        """
        url = "%s/community/groups/%s/users" % (
            self._gis._portal.resturl,
            self.groupid,
        )
        params = {"f": "json"}
        return self._gis._con.post(url, params)

    def user_list(self) -> dict:
        """
        The ``user_list`` method returns a dictionary listing users and owners for the group.

        .. note::
            The ``user_list`` method is only available on ArcGIS Online and ArcGIS Enterprise 10.9+.

        :return: A dictionary of users and owners for the group

        """
        if self._gis.version >= [8, 4]:
            url = "%s/community/groups/%s/userList" % (
                self._gis._portal.resturl,
                self.groupid,
            )
            params = {"f": "json", "start": 1, "num": 100}
            res = self._gis._con.get(url, params)
            users = res["users"]
            owner = res["owner"]
            while res["nextStart"] > -1:
                params["start"] = res["nextStart"]
                res = self._gis._con.get(url, params)
                users.extend(res["users"])
                if res["nextStart"] == -1:
                    break
            value = {"owner": owner, "users": users}
            return value
        return None

    def update(
        self,
        title: Optional[str] = None,
        tags: Optional[Union[list[str], str]] = None,
        description: Optional[str] = None,
        snippet: Optional[str] = None,
        access: Optional[str] = None,
        is_invitation_only: Optional[bool] = None,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
        is_view_only: Optional[bool] = None,
        thumbnail: Optional[str] = None,
        max_file_size: Optional[int] = None,
        users_update_items: bool = False,
        clear_empty_fields: bool = False,
        display_settings: Optional[str] = None,
        is_open_data: bool = False,
        leaving_disallowed: bool = False,
        member_access: bool = None,
        hidden_members: bool = False,
        membership_access: Optional[str] = None,
        autojoin: bool = False,
    ):
        """
        The ``update`` method updates the group's properties with the values supplied for particular arguments.

        .. note::
            If a value is not supplied for a particular argument, the corresponding property will not be updated.


        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        title               Optional string. The new name of the group.
        ------------------  ---------------------------------------------------------
        tags                Optional string. A comma-delimited list of new tags, or
                            a list of tags as strings.
        ------------------  ---------------------------------------------------------
        description         Optional string. The new description for the group.
        ------------------  ---------------------------------------------------------
        snippet             Optional string. A new short snippet (<250 characters)
                            that summarizes the group.
        ------------------  ---------------------------------------------------------
        access              Optional string. Choices are private, public, or org.
        ------------------  ---------------------------------------------------------
        is_invitation_only  Optional boolean. Defines whether users can join by
                            request. True means an invitation is required.
        ------------------  ---------------------------------------------------------
        sort_field          Optional string. Specifies how shared items with the
                            group are sorted.
        ------------------  ---------------------------------------------------------
        sort_order          Optional string. Choices are asc or desc for ascending
                            or descending, respectively.
        ------------------  ---------------------------------------------------------
        is_view_only        Optional boolean. Defines whether the group is searchable.
                            True means the group is searchable.
        ------------------  ---------------------------------------------------------
        thumbnail           Optional string. URL or file location to a new group image.
        ------------------  ---------------------------------------------------------
        max_file_size       Optional integer.  This is the maximum file size allowed
                            be uploaded/shared to a group. Default value is: 1024000
        ------------------  ---------------------------------------------------------
        users_update_items  Optional boolean.  Members can update all items in this
                            group.  Updates to an item can include changes to the
                            item's description, tags, metadata, as well as content.
                            This option can't be disabled once the group has
                            been created. Default is False.
        ------------------  ---------------------------------------------------------
        clear_empty_fields  Optional Boolean. If True, the user can set values to
                            empty string, else, None values will be ignored.
        ------------------  ---------------------------------------------------------
        display_settings    Optional String. Defines the default display for the
                            group page to show a certain type of items. The allowed
                            values are: `apps, all, files, maps, layers, scenes, tools`.
                            The default value is `all`.
        ------------------  ---------------------------------------------------------
        is_open_data        Optional Boolean. Defines whether the group can be used
                            in the Open Data capabilities of ArcGIS Hub. The default
                            is False.
        ------------------  ---------------------------------------------------------
        leaving_disallowed  Optional boolean. Defines whether users are restricted
                            from choosing to leave the group. If True, only an
                            administrator can remove them from the group. The default
                            is False.
        ------------------  ---------------------------------------------------------
        hidden_members      Optional Boolean. Only applies to org accounts. If true,
                            only the group owner, group managers, and default
                            administrators can see all members of the group.
        ------------------  ---------------------------------------------------------
        membership_access   Optional String. Sets the membership access for the group.
                            Setting to `org` restricts group access to members of
                            your organization. Setting to `collaboration` restricts the
                            membership access to partnered collaboration and your
                            organization members. If `None` set, any organization
                            will have access. `None` is the default.

                            Values: `org`, `collaboration`, or `None`
        ------------------  ---------------------------------------------------------
        autojoin            Optional Boolean. The default is `False`. Only applies to
                            org accounts. If `True`, this group will allow joined
                            without requesting membership approval.
        ==================  =========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> user.update(description="Aggregated US Hurricane Data", tags = "Hurricanes,USA, 2020")
        """
        display_settings_lu = {
            "apps": {"itemTypes": "Application"},
            "all": {"itemTypes": ""},
            "files": {"itemTypes": "CSV"},
            None: {"itemTypes": ""},
            "maps": {"itemTypes": "Web Map"},
            "layers": {"itemTypes": "Layer"},
            "scenes": {"itemTypes": "Web Scene"},
            "tools": {"itemTypes": "Locator Package"},
        }
        if max_file_size is None:
            max_file_size = 1024000
        if users_update_items is None:
            users_update_items = False
        if tags is not None:
            if type(tags) is list:
                tags = ",".join(tags)
        if (
            isinstance(display_settings, str)
            and display_settings.lower() in display_settings_lu
        ):
            display_settings = display_settings_lu[display_settings.lower()]
        elif display_settings is None:
            display_settings = display_settings_lu[display_settings]
        else:
            raise ValueError("Display settings must be set to a valid value.")
        resp = self._portal.update_group(
            self.groupid,
            title,
            tags,
            description,
            snippet,
            access,
            is_invitation_only,
            sort_field,
            sort_order,
            is_view_only,
            thumbnail,
            max_file_size,
            users_update_items,
            clear_empty_fields=clear_empty_fields,
            display_settings=display_settings,
            is_open_data=is_open_data,
            leaving_disallowed=leaving_disallowed,
            hidden_members=hidden_members,
            membership_access=membership_access,
            autojoin=autojoin,
        )
        if resp:
            self._hydrate()
        return resp

    def leave(self):
        """
        The ``leave`` method removes the logged in user from this group.

        .. note::
            The user must be logged in to use the ``leave`` command.


        :return:
           A boolean indicating success (True) or failure (False).
        """
        return self._portal.leave_group(self.groupid)

    def join(self):
        """
        Users apply to join a group using the ``join`` operation. This
        creates a new group application, which the group administrators
        accept or decline. This operation also creates a notification for
        the user indicating that they have applied to join this group.

        .. note::
            Available only to authenticated users. Users can only apply to join groups to which they have access - if
            the group is private, users will not be able to find it to ask to
            join it.
        Information pertaining to the applying user, such as their full
        name and username, can be sent as part of the group application.

        :return:
             A boolean indicating success (True) or failure (False).
        """
        url = "community/groups/%s/join" % (self.groupid)
        params = {"f": "json"}
        res = self._portal.con.post(url, params)
        if "success" in res:
            return res["success"] == True
        return res

    # ----------------------------------------------------------------------
    @property
    def applications(self):
        """
        The ``applications`` property retrieves the group applications for the given group as a list.

        .. note::
            The ``applications`` method is available to administrators of the group or administrators of an organization
            if the group is part of one.
        """
        apps = []
        try:
            path = "%scommunity/groups/%s/applications" % (
                self._portal.resturl,
                self.groupid,
            )
            params = {"f": "json"}
            res = self._portal.con.post(path, params)
            if "applications" in res:
                for app in res["applications"]:
                    url = "%s/%s" % (path, app["username"])
                    apps.append(GroupApplication(url=url, gis=self._gis))
        except:
            print()
        return apps

    # ----------------------------------------------------------------------
    def application(self, user: str):
        """
        The ``application`` method retrieves one group application for the given group.

        ==================  ====================================
        **Parameter**        **Description**
        ------------------  ------------------------------------
        user                Required String. The username of
                            the user applying to join the group.
        ==================  ====================================

        .. note::
            The ``application`` method is available to administrators of the group or administrators of an organization
            if the group is part of one.
        """
        try:
            path = "%scommunity/groups/%s/applications/%s" % (
                self._portal.resturl,
                self.groupid,
                user,
            )
            params = {"f": "json"}
            res = self._portal.con.post(path, params)
            return GroupApplication(url=path, gis=self._gis)
        except:
            print()

    # ----------------------------------------------------------------------
    @property
    def protected(self):
        """
        Indicates if the group is protected from deletion.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        value                   Required bool.
                                Values: True (protect group) | False (unprotect)
        ==================      ====================================================================

        :return: True if group currently protected, False if unprotected
        """
        return self["protected"]

    # ----------------------------------------------------------------------
    @protected.setter
    def protected(self, value: bool):
        """
        See main ``protected`` property docstring
        """
        params = {"f": "json"}
        if value == True and self.protected == False:
            url = "%s/community/groups/%s/protect" % (
                self._portal.resturl,
                self.groupid,
            )
            res = self._portal.con.post(url, params)
            self._hydrated = False
            self._hydrate()
        elif value == False and self.protected == True:
            url = "%s/community/groups/%s/unprotect" % (
                self._portal.resturl,
                self.groupid,
            )
            res = self._portal.con.post(url, params)
            self._hydrated = False
            self._hydrate()


class GroupApplication(object):
    """
    The ``GroupApplication`` class represents a single group application on the GIS, either ArcGIS Online or
    ArcGIS Enterprise.
    """

    _con = None
    _portal = None
    _gis = None
    _url = None
    _properties = None

    def __init__(self, url, gis, **kwargs):
        initialize = kwargs.pop("initialize", False)
        self._url = url
        self._gis = gis
        self._portal = gis._portal
        self._con = self._portal.con
        if initialize:
            self._init()

    def _init(self):
        """Loads the properties."""
        try:
            res = self._con.get(self._url, {"f": "json"})
            self._properties = _mixins.PropertyMap(res)
            self._json_dict = res
        except:
            self._properties = _mixins.PropertyMap({})
            self._json_dict = {}

    @property
    def properties(self):
        """The ``properties`` operation retrevies the properties of the GroupApplication."""
        if self._properties is None:
            self._init()
        return self._properties

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s for %s>" % (
            type(self).__name__,
            self.properties.username,
        )

    def accept(self):
        """
        The ``accept`` method is used to manage a :class:`~arcgis.gis.User` application. When a
        :class:`~arcgis.gis.User` applies to join a :class:`~arcgis.gis.Group`, a
        ``GroupApplication`` object is created. Group administrators choose to accept this application
        using the ``accept`` operation. This operation adds the applying user to the group then deletes the application.
        This operation also creates a notification for the user indicating that the user's group application was
        accepted. This method is very similar to the :attr:`~arcgis.gis.GroupApplication.decline` method, which declines
        rather than accepts the application to join a group.

        .. note::
            The ``accept`` method is only available to group owners and administrators.

        .. code-block:: python

            # Usage Example

            >>> group1 = gis.groups.get('name')
            >>> group_app = group1.applications[0]
            >>> group_app.accept()

        :return:
           A boolean indicating success (True) or failure (False).
        """
        url = "%s/accept" % self._url
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"] == True
        return res

    def decline(self):
        """
        The ``decline`` method is used to manage a :class:`~arcgis.gis.User` application. When a
        :class:`~arcgis.gis.User` asks to join a :class:`~arcgis.gis.Group`, a
        ``GroupApplication`` object is created. Group administrators choose to decline this application
        using the ``decline`` operation. This operation deletes the application and creates a notification for the user
        indicating that the user's group application was declined. This method is very similar to the
        :attr:`~arcgis.gis.GroupApplication.accept` method, which accepts rather than declines the application to
        join a group.

        .. note::
            The ``decline`` method is only available to group owners and administrators.

        .. code-block:: python

            # Usage Example

            >>> group_app = group.applications[0]
            >>> groupapplication.decline()

        :return:
           A boolean indicating success (True) or failure (False).
        """
        url = "%s/decline" % self._url
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"] == True
        return res


class User(dict):
    """
    The ``User`` class represents a registered user of the GIS system, either ArcGIS Online or ArcGIS Enterprise. The
    User class has a myriad of properties that are specific to a particular user - these properties are
    enumerated in the table below.

    =====================    =========================================================
    **Property**             **Details**
    ---------------------    ---------------------------------------------------------
    username                 The username of the user.
    ---------------------    ---------------------------------------------------------
    fullName                 The user's full name
    ---------------------    ---------------------------------------------------------
    availableCredits         The number of credits available to the user.
    ---------------------    ---------------------------------------------------------
    assignedCredits          The number of credits allocated to the user.
    ---------------------    ---------------------------------------------------------
    firstName                The user's first name.
    ---------------------    ---------------------------------------------------------
    lastName                 The user's last name.
    ---------------------    ---------------------------------------------------------
    preferredView            The user's preferred view for content, either web or GIS.
    ---------------------    ---------------------------------------------------------
    description              A description of the user.
    ---------------------    ---------------------------------------------------------
    email                    The user's e-mail address.
    ---------------------    ---------------------------------------------------------
    idpUsername              The original username if using enterprise logins.
    ---------------------    ---------------------------------------------------------
    favGroupId               The user's favorites group and is created automatically for each user.
    ---------------------    ---------------------------------------------------------
    lastLogin                The last login date of the user in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    mfaEnabled               Indicates if the user's account has multifactor authentication set up.
    ---------------------    ---------------------------------------------------------
    access                   Indicates the level of access of the user: private, org, or public. If private, the user descriptive information will not be available to others nor will the username be searchable.
    ---------------------    ---------------------------------------------------------
    storageUsage             | The amount of storage used for the entire organization.

                             **NOTE:** This value is an estimate for the organization, not the specific user.
                             For storage estimate of a user's items, see code example in the :attr:`items` method.
    ---------------------    ---------------------------------------------------------
    storageQuota             Applicable to public users as it sets the total amount of storage available for a subscription. The maximum quota is 2GB.
    ---------------------    ---------------------------------------------------------
    orgId                    The ID of the organization the user belongs to.
    ---------------------    ---------------------------------------------------------
    role                     | Defines the user's role in the organization.
                             Values:
                               * ``org_admin`` - administrator or custom role with administrative privileges
                               * ``org_publisher`` - publisher or custom role with publisher privileges
                               * ``org_user`` - user or custom role with user privileges)
    ---------------------    ---------------------------------------------------------
    privileges               A JSON array of strings with predefined permissions in each. For a complete listing, see Privileges.
    ---------------------    ---------------------------------------------------------
    roleId                   (Optional) The ID of the user's role if it is a custom one.
    ---------------------    ---------------------------------------------------------
    level                    The level of the user.
    ---------------------    ---------------------------------------------------------
    disabled                 The login access to the organization for the user.
    ---------------------    ---------------------------------------------------------
    units                    User-defined units for measurement.
    ---------------------    ---------------------------------------------------------
    tags                     User-defined tags that describe the user.
    ---------------------    ---------------------------------------------------------
    culture                  The user locale information (language and country).
    ---------------------    ---------------------------------------------------------
    cultureFormat            The user preferred number and date format defined in CLDR (only applicable for English,
                             Spanish, French, German, and italian: i.e. when culture is en, es, fr, de, or it).

                             .. note::
                                See `Languages <https://developers.arcgis.com/rest/users-groups-and-items/languages.htm>`_
                                for supported formats. It will inherit from
                                `organization <https://developers.arcgis.com/rest/users-groups-and-items/portal-self.htm>`_
                                cultureFormat if undefined.
    ---------------------    ---------------------------------------------------------
    region                   The user preferred region, used to set the featured maps on the home page, content in the gallery, and the default extent of new maps in the Viewer.
    ---------------------    ---------------------------------------------------------
    thumbnail                The file name of the thumbnail used for the user.
    ---------------------    ---------------------------------------------------------
    created                  The date the user was created. Shown in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    modified                 The date the user was last modified. Shown in milliseconds since the Unix epoch.
    ---------------------    ---------------------------------------------------------
    groups                   A JSON array of groups the user belongs to. See Group for properties of a group.
    ---------------------    ---------------------------------------------------------
    provider                 The identity provider for the organization.<br>Values: arcgis (for built-in users) ,enterprise (for external users managed by an enterprise identity store), facebook (for public accounts in ArcGIS Online), google (for public accounts in ArcGIS Online)
    ---------------------    ---------------------------------------------------------
    id                       (optional) The unique identifier of the user used in ArcGIS Online or ArcGIS Enterprise 10.7+
    =====================    =========================================================



    """

    def __init__(self, gis, username, userdict=None):
        dict.__init__(self)
        self._gis = gis
        self._portal = gis._portal
        self._user_id = username
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        self._invitemgr = None
        # userdict = self._portal.get_user(self.username)
        self._hydrated = False
        if userdict:
            if (
                "groups" in userdict and len(userdict["groups"]) == 0
            ):  # groups aren't set unless hydrated
                del userdict["groups"]
            if "role" in userdict and "roleId" not in userdict:
                userdict["roleId"] = userdict["role"]
            elif "roleId" in userdict and "role" not in userdict:
                # try getting role name - only needed for custom roles
                try:
                    role_obj = self._gis.users.roles.get_role(userdict["roleId"])
                    userdict["role"] = role_obj.name
                except Exception as ex:
                    userdict["role"] = userdict["roleId"]
            self.__dict__.update(userdict)
            super(User, self).update(userdict)
        if hasattr(self, "id") and self.id != "null":
            # self._user_id = self.id
            self._user_id = self.username
        else:
            self._user_id = self.username

    # Using http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991

    def _hydrate(self):
        userdict = self._portal.get_user(self._user_id)
        if not "roleId" in userdict and "role" in userdict:
            userdict["roleId"] = userdict["role"]
        self._hydrated = True
        super(User, self).update(userdict)
        self.__dict__.update(userdict)

    def __getattr__(
        self, name
    ):  # support user attributes as user.access, user.email, user.role etc
        if not self._hydrated and not name.startswith("_"):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def __getitem__(
        self, k
    ):  # support user attributes as dictionary keys on this object, eg. user['role']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith("_"):
                self._hydrate()
            return dict.__getitem__(self, k)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<%s username:%s>" % (type(self).__name__, self.username)

    # ----------------------------------------------------------------------
    @property
    def recyclebin(self) -> "RecycleBin":
        """Provides access to the user's recyclebin.

        .. note::
            This functionality is **only** available to organizations participating
            in the Recycle Bin private beta.

        :Returns: :class:`~arcgis.gis._impl._content_manager.RecycleBin` object

        .. code-block:: python

            # Usage Example:
            >>> gis = GIS(profile="your_online_user")

            >>> my_user_obj = gis.users.me
            >>> my_recy_bin = my_user_obj.recyclebin
            >>> type(my_recy_bin)

            <class 'arcgis.gis._impl._content_manager._recyclebin.RecycleBin'>
        """
        gis: GIS = self._gis
        if gis._is_arcgisonline or (
            gis._is_arcgisonline == False and gis.version > [11, 2]
        ):
            from ._impl._content_manager._recyclebin import RecycleBin

            return RecycleBin(gis=self._gis, user=self.username)
        return None

    # ----------------------------------------------------------------------
    def user_types(self):
        """
        The ``user_types`` method is used to retrieve the user type and any assigned applications of the user.

        .. note::
            The ``user_types`` method is available in Portal 10.7+.
        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError(
                "`user_types` is not implemented at version %s"
                % ".".join([str(i) for i in self._gis.version])
            )

        url = "%s/community/users/%s/userLicenseType" % (
            self._portal.resturl,
            self.username,
        )
        params = {"f": "json"}
        return self._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def invitations(self) -> "UserInvitationManager":
        """
        Provides a list of invitations for a given user

        :returns: UserInvitationManager
        """
        if self._invitemgr is None:
            from arcgis.gis.sharing import UserInvitationManager

            self._invitemgr = UserInvitationManager(self)
        return self._invitemgr

    # ----------------------------------------------------------------------
    def report(
        self,
        report_type: str,
        start_time: Optional[datetime],
        *,
        duration: Optional[str] = "weekly",
    ) -> Item:
        """

        The reports operation is to generate the reports of the overall
        usage of the organizations. Reports define organization usage
        metrics in one place for the day, week, or month. Administrators
        can monitor who is using which services, consuming how much credits
        and storage within certain time period. Reports also include
        current state of the organization such as number of items, groups,
        users, level 1s vs level 2s, App license assignments and public
        items.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        report_type       Required String. The type of organizational report to
                          generated. The allowed report types are: `credits`,
                          `content`, `users`, and `activity`.
        ----------------  --------------------------------------------------------
        start_time        Required Datetime. The day on which the report is
                          generated. Each report must start on a Sunday or Monday
                          for the start date for weekly and monthly reports. All
                          datetimes must be in GMT timezone. Passing in `None` for
                          the `start_time` will use the closest Sunday to the date
                          for weekly and monthly reports.  For daily reports, the
                          current day/time will be used in GMT.
        ----------------  --------------------------------------------------------
        duration          Optional String. The time frame on which the reports are
                          ran.  The allowed values are: `monthly`, `weekly`,
                          `daily`. For `activity` and `credits` a `start_time`
                          is required.
        ================  ========================================================


        .. code-block:: python

            # Usage Example

            import datetime as _dt
            seven_days_ago = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=7)
            item = user.report("content",
                               seven_days_ago,
                               duration="weekly")


        :return: Item

        """

        import datetime as _dt

        assert report_type in ["users", "credits", "activity", "content"]
        assert duration in ["monthly", "weekly", "daily"]

        def weeknumber(dayname):
            if dayname == "Monday":
                return -1
            if dayname == "Tuesday":
                return -2
            if dayname == "Wednesday":
                return -3
            if dayname == "Thursday":
                return -4
            if dayname == "Friday":
                return -5
            if dayname == "Saturday":
                return -6
            if dayname == "Sunday":
                return 0

        if report_type.lower() != "activity" and duration == "daily":
            raise ValueError("Daily only applies to activity report type.")
        if (
            start_time
            and isinstance(start_time, _dt.datetime)
            and start_time.date().today().strftime("%A") in ["Monday", "Sunday"]
            and duration in ["weekly", "monthly"]
        ):
            raise ValueError(
                "Invalid start_time. Weekly report must start from Sunday or Monday."
            )
        elif start_time and isinstance(start_time, _dt.datetime):
            start_time = int(start_time.timestamp() * 1000)
        elif isinstance(start_time, (int, float)):
            start_time = int(start_time)
        elif start_time is None and duration in ["weekly", "monthly"]:
            now = _dt.datetime.now(_dt.timezone.utc)
            dow = weeknumber(now.strftime("%A"))  # day of the week
            start_time = now + _dt.timedelta(days=dow)
            start_time = int(start_time.timestamp() * 1000)
        elif start_time is None and duration in ["daily"]:
            start_time = _dt.datetime.now(_dt.timezone.utc)
        params = {
            "f": "json",
            "reportType": "org",
            "reportSubType": report_type,
            "timeDuration": duration,
            "startTime": start_time,
        }

        url = "%s/sharing/rest/community/users/%s/report" % (
            self._gis._url,
            self._user_id,
        )
        res = self._gis._con.post(url, params)
        time.sleep(10)
        try:
            count = 0
            item = None
            while count < 10:
                try:
                    item = Item(self._gis, res["itemId"])
                except:
                    ...
                if item:
                    break
                count += 1
                time.sleep(count)

            if item is None:
                raise Exception(f"Cannot find Item: {res['itemID']}")

            status = item.status()
            counter = 1
            while not status["status"] in ["completed", "failed"]:
                status = item.status()
                time.sleep(counter)
                counter += 1
                if counter > 5:
                    counter = 5
            return item
        except Exception as e:
            raise e

        return item

    # ----------------------------------------------------------------------
    @property
    def tasks(self):
        """
        The ``tasks`` property retrieves the users tasks, effectively serving as sesource manager for user's tasks.
        See :class:`~arcgis.gis.tasks.TaskManager` for more information on task managers.
        """
        if str(self.role).lower() == "org_admin" or self._gis.properties["user"]:
            url = f"{self._gis._portal.resturl}community/users/{self.username}/tasks"
            from .tasks import TaskManager

            return TaskManager(url=url, user=self, gis=self._gis)
        return None

    def transfer_content(
        self, target_user: str | User, folder: str | None = None
    ) -> concurrent.futures.Future:
        """
        This operation transfers all the current user's content to a new user.
        This is an asynchronous operation that can take up to 15 minutes to complete.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        target_user       Required str or User. The user who will received the current user's content.
        ----------------  --------------------------------------------------------
        folder            Optional str. The folder where the content is stored.
        ================  ========================================================

        :returns: concurrent.futures.Future

        """
        if isinstance(target_user, User):
            username: str = target_user.username
        elif isinstance(target_user, str):
            username: str = target_user
            target_user: User = self._gis.users.get(username)
        else:
            raise ValueError("target_user must be a string or User object")
        target_user: User = target_user
        target_user.folders
        if folder is None:
            folder_dest: str = username
        else:
            folder_dest: str = None
            for f in target_user.folders:
                if folder.lower() == f["id"].lower():
                    folder_dest = f["title"]
                    break
                elif folder.lower() == f["title"].lower():
                    folder_dest = f["title"]
                    break
            if folder_dest is None:
                cm: ContentManager = self._gis.content
                cm.create_folder(folder=folder, owner=target_user)
                folder_dest = folder
        params: dict[str, Any] = {
            "f": "json",
            "reassign": json.dumps(
                {
                    "targetUser": username,  # destination user
                    "reassignedUsers": [self.username],  # content source
                    "targetFolderName": folder_dest,  # folder location, default is the root
                    "createSubFolderPerReassignedUser": False,
                }
            ),
        }
        url: str = f"{self._portal.resturl}portals/self/reassignUsersContent"
        resp: requests.Response = self._gis._con._session.post(url=url, data=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        job_url: str = f"{self._portal.resturl}portals/self/jobs/{data['jobId']}"

        tp = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future: concurrent.futures.Future = tp.submit(
            self._transfer_content_status,
            **{"session": self._gis._con._session, "url": job_url},
        )
        tp.shutdown(cancel_futures=False)
        return future

    def _transfer_content_status(self, session: requests.Session, url: str) -> dict:
        """checks the job status for transfer content operation"""

        resp: requests.Response = session.get(
            url=url,
            params={
                "f": "json",
            },
        )
        data: dict[str, Any] = resp.json()
        status: str = data.get("status", "submitted")
        while "status" in data:
            if status == "submitted":
                time.sleep(10)
            elif status in ["success", "failed", "succeeded"]:
                return data
            resp: requests.Response = session.get(
                url=url,
                params={
                    "f": "json",
                },
            )
            data: dict[str, Any] = resp.json()
            status: str = data.get("status", "submitted")
        return data

    # ----------------------------------------------------------------------
    def generate_direct_access_url(
        self,
        store_type: str,
        expiration: int | None = None,
        subfolder: str | None = None,
    ) -> dict | None:
        """
        The ``generate_direct_access_url`` method creates a direct access URL that is ideal
        for uploading large files to datafile share, notebook workspaces or raster stores.

        =====================  =========================================================
        **Parameter**           **Description**
        ---------------------  ---------------------------------------------------------
        store_type             Optional String. The type of upload URL to generate.
                               Types: `big_data_file`, 'notebook', or 'raster`.
        ---------------------  ---------------------------------------------------------
        expiration             Optional Int. The expiration of the link in minutes.  The default is 1440.
        ---------------------  ---------------------------------------------------------
        subfolder              Optional String. The folder to upload to. The default is `None`.
        =====================  =========================================================

        :return: A dictionary containing the direct access URL

        .. code-block:: python

            # Usage Example

            >>> user.generate_direct_access_url(store_type="notebook")


        """
        if self._gis._portal.is_arcgisonline == False:
            return None
        _lu = {
            "big_data_file": "bigDataFileShare",
            "notebook": "notebookWorkspace",
            "raster": "rasterStore",
        }
        url = f"{self._gis._portal.resturl}content/users/{self.username}/generateDirectAccessUrl"
        params = {
            "f": "json",
            "expiration": expiration or 1440,
            "storeType": _lu[store_type.lower()],
        }
        if subfolder:
            params["subPath"] = subfolder

        return self._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def provisions(self):
        """
        The ``provisions`` property returns a list of all provisioned licenses for the current user.

        .. note::
            The ``provisions method`` is only available in ArcGIS Enterprise 10.7+.

        :return:
            A list contaning provisional licenses

        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError(
                "Provisions is not implemented at version %s"
                % ".".join([str(i) for i in self._gis.version])
            )

        provs = []
        url = "%s/community/users/%s/provisionedListings" % (
            self._portal.resturl,
            self.username,
        )
        params = {
            "f": "json",
            "start": 1,
            "num": 100,
            "includeExpired": True,
        }
        res = self._portal.con.post(url, params)
        provs = [
            Item(gis=self._gis, itemid=i["itemId"]) for i in res["provisionedListings"]
        ]
        while res["nextStart"] > -1:
            params["start"] = res["nextStart"]
            res = self._portal.con.post(url, params)
            provs.extend(
                [
                    Item(gis=self._gis, itemid=i["itemId"])
                    for i in res["provisionedListings"]
                ]
            )
        return provs

    # ----------------------------------------------------------------------
    @property
    def bundles(self):
        """

        The ``bundles`` method provides the current user's assigned application bundles.

        .. note::
            The ``bundles`` method is available in ArcGIS Online and Portal 10.7+.

        :return: A List of :class:`~arcgis.gis.admin._license.Bundle` objects
        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError(
                "`bundles` is not implemented at version %s"
                % ".".join([str(i) for i in self._gis.version])
            )

        from arcgis.gis.admin._license import Bundle

        url = "%s/community/users/%s/appBundles" % (
            self._portal.resturl,
            self.username,
        )
        params = {"f": "json", "start": 1, "num": 10}
        bundles = []
        res = self._portal.con.post(url, params)
        bundles = res["appBundles"]
        while res["nextStart"] > -1:
            params["start"] = res["nextStart"]
            res = self._portal.con.post(url, params)
            bundles += res["appBundles"]
        return [
            Bundle(
                url="{base}content/listings/{id}".format(
                    base=self._gis._portal.resturl, id=b["id"]
                ),
                properties=b,
                gis=self._gis,
            )
            for b in bundles
        ]

    # ----------------------------------------------------------------------
    def get_thumbnail_link(self):
        """
        ``The get_thumbnail_link`` method retrieves the URL to the thumbnail image.

        :return:
           The thumbnail's URL.
        """
        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            return self._gis.url + "/home/js/arcgisonline/css/images/no-user-thumb.jpg"
        else:
            thumbnail_url_path = (
                self._gis._public_rest_url
                + "community/users/"
                + self._user_id
                + "/info/"
                + thumbnail_file
            )
            return thumbnail_url_path

    @property
    def homepage(self):
        """
        The ``homepage`` property retrieves the URL to the HTML page for the user.
        """
        return "{}{}{}".format(self._gis.url, "/home/user.html?user=", self._user_id)

    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        elif self.get_thumbnail():
            b64 = base64.b64encode(self.get_thumbnail())
            thumbnail = (
                "data:image/png;base64,"
                + str(b64, "utf-8")
                + "' width='200' height='133"
            )
        else:
            thumbnail = (
                "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgEAYABgAAD/4QgwRXhpZgAATU0AKgAAAAgABwESAAMAAAABAAEAAAEaAAUAA"
                "AABAAAAYgEbAAUAAAABAAAAagEoAAMAAAABAAIAAAExAAIAAAAcAAAAcgEyAAIAAAAUAAAAjodpAAQAAAABAAAApAAAANAADqYAA"
                "AAnEAAOpgAAACcQQWRvYmUgUGhvdG9zaG9wIENTMyBXaW5kb3dzADIwMTE6MDI6MjUgMjM6NDc6NTcAAAAAA6ABAAMAAAAB//8AA"
                "KACAAQAAAABAAAAlqADAAQAAAABAAAAlgAAAAAAAAAGAQMAAwAAAAEABgAAARoABQAAAAEAAAEeARsABQAAAAEAAAEmASgAAwAAA"
                "AEAAgAAAgEABAAAAAEAAAEuAgIABAAAAAEAAAb6AAAAAAAAAEgAAAABAAAASAAAAAH/2P/gABBKRklGAAECAABIAEgAAP/tAAxBZ"
                "G9iZV9DTQAC/+4ADkFkb2JlAGSAAAAAAf/bAIQADAgICAkIDAkJDBELCgsRFQ8MDA8VGBMTFRMTGBEMDAwMDAwRDAwMDAwMDAwMD"
                "AwMDAwMDAwMDAwMDAwMDAwMDAENCwsNDg0QDg4QFA4ODhQUDg4ODhQRDAwMDAwREQwMDAwMDBEMDAwMDAwMDAwMDAwMDAwMDAwMD"
                "AwMDAwMDAwM/8AAEQgAlgCWAwEiAAIRAQMRAf/dAAQACv/EAT8AAAEFAQEBAQEBAAAAAAAAAAMAAQIEBQYHCAkKCwEAAQUBAQEBA"
                "QEAAAAAAAAAAQACAwQFBgcICQoLEAABBAEDAgQCBQcGCAUDDDMBAAIRAwQhEjEFQVFhEyJxgTIGFJGhsUIjJBVSwWIzNHKC0UMHJ"
                "ZJT8OHxY3M1FqKygyZEk1RkRcKjdDYX0lXiZfKzhMPTdePzRieUpIW0lcTU5PSltcXV5fVWZnaGlqa2xtbm9jdHV2d3h5ent8fX5"
                "/cRAAICAQIEBAMEBQYHBwYFNQEAAhEDITESBEFRYXEiEwUygZEUobFCI8FS0fAzJGLhcoKSQ1MVY3M08SUGFqKygwcmNcLSRJNUo"
                "xdkRVU2dGXi8rOEw9N14/NGlKSFtJXE1OT0pbXF1eX1VmZ2hpamtsbW5vYnN0dXZ3eHl6e3x//aAAwDAQACEQMRAD8A9FYxha07R"
                "wOw8Pgn2M/dH3D+5Jn0G/BSQSx2M/dH4f3JbGfuj8P7lJJJTHYz90fh/clsZ+6Pw/uUkklMdjP3R+H9yWxn7o/D+5SSSUx2M/dH4"
                "f3JbGfuj8P7lJJJTHYz90fh/clsZ+6Pw/uUkklMdjP3R+H9yWxn7o/D+5SSSUx2M/dH4f3JbGfuj8P7lJJJTHYz90fh/clsZ+6Pw"
                "/uUkjwkpgWN3NEDny4j4JJ3fSb8f4FJFD//0PRmfQb8FJRZ9BvwUkEqSSSSUpJJO1rnGAkpYa8I1dPd33KTKQ3XuiIoYelX4Jekz"
                "w/E/wB6mkkph6NfghWVFuo1CsJGD5pKaaSnbXtM9lBBKkkkklKSPCSR4SUxd9Jvx/gkk76Tfj/BJFD/AP/R9GZ9BvwUlFn0G/BSQ"
                "SpJJJJSkXHHJKEj4/0SkpI5waJUfUceGpnHc+B+anRQtusPgEiHn84/JOkkpjt8SSmjYdzfmFNJJS7gHN+SqxCsVmCWH5fBBsEPI"
                "SUxSSSQSpI8JJHhJTF30m/H+CSTvpN+P8EkUP8A/9L0Zn0G/BSUWfQb8FJBKkkkklKR6Pon4oCnXYWaQkpI3knxKkos+j8ZUkUKS"
                "SSSUpJJJJSw/nG+YQ7x75UrTEEIRJJkpKWSSSQSpI8JJHhJTF30m/H+CSTvpN+P8EkUP//T9GZ9BvwUlFn0G/BSQSpJJJJSkkkkl"
                "JqtW/BTQaXdkZFCkkkklKSSS4SUitPCGne6SopKXSSSQSpI8JJHhJTF30m/H+CSTvpN+P8ABJFD/9T0Zn0G/BSUWfQb8FJBKkkkk"
                "lKSSSRQppLSjtO4SgIrG6SDqkpIko7/AN4Qn3DkFJS6Ha+PaE/ud9HQdyUJ4hySlkkkkEqSSSSUpI8JJHhJTF30m/H+CSTvpN+P8"
                "EkUP//V9GZ9BvwUlFn0G/BSQSpKCdAkrFLABuPJSQibS8qYx/EoydFSA0ACQnbG0IqFG15HY6j4pKZJoE8apJ0lKUWsD5ce/CTpM"
                "NHdFAgADskpEcdp4Kicd3bVHTpKabmOadUyuEAiCq9tewyOCkpGkeEkjwgli76Tfj/BJJ30m/H+CSKH/9b0Zn0G/BSUWfQb8FJBK"
                "hqQPFXAIACrUtl4VlFCk6SSSlKFjZbPcKaSSkQMie6dNG15b2OoSeZEeKSlMBJLj8kRM0QAE6Sl0kkklLKFwlnwRExEiElNLunPC"
                "ciCQmPCCmLvpN+P8EknfSb8f4JIqf/X9GZ9BvwUlFn0G/BSSUmxxrPyR0Gge0/FFSUukkkkpSSSZJTCxst05Cav3HcfkiJRHCSlJ"
                "0ydJSkkkklKSSSSU1bRDyhnhFv+khJKWd9Jvx/gkk76Tfj/AASSU//Q9FYTtEA8eSfc790/gvmRJJT9QVPuA9tZInxH96J6l/eo/"
                "eP718tpJKfqX1Lf9Efvb/el6lv+iP3t/vXy0kkp+pfUt/0Tvvb/AOST+pZ/onfe3/yS+WUklP1N6ln+id97f/JJvUs/0Tvvb/5Jf"
                "LSSSn6l9Sz/AETvvb/5JL1Lf9Efvb/evlpJJT9S+pd/oj94/vUTZd/oj94/vXy4kkp+o99/+j/6n/ySiXZH7h+W3/yS+XkklP069"
                "1k+9hn4hRJdH0T+C+ZEklP00S7c32mZ0GngkvmVJJT/AP/Z/+0NTFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAHHAIAAAIAAAA4Q"
                "klNBCUAAAAAABDo8VzzL8EYoaJ7Z63FZNW6OEJJTQQvAAAAAABKWOMBAEgAAABIAAAAAAAAAAAAAADQAgAAQAIAAAAAAAAAAAAAG"
                "AMAAGQCAAAAAcADAACwBAAAAQAPJwEAagBwAGcAAAAuAGoAcAA4QklNA+0AAAAAABAAYAAAAAEAAQBgAAAAAQABOEJJTQQmAAAAA"
                "AAOAAAAAAAAAAAAAD+AAAA4QklNBA0AAAAAAAQAAAAeOEJJTQQZAAAAAAAEAAAAHjhCSU0D8wAAAAAACQAAAAAAAAAAAQA4QklNB"
                "AoAAAAAAAEAADhCSU0nEAAAAAAACgABAAAAAAAAAAI4QklNA/UAAAAAAEgAL2ZmAAEAbGZmAAYAAAAAAAEAL2ZmAAEAoZmaAAYAA"
                "AAAAAEAMgAAAAEAWgAAAAYAAAAAAAEANQAAAAEALQAAAAYAAAAAAAE4QklNA/gAAAAAAHAAAP///////////////////////////"
                "/8D6AAAAAD/////////////////////////////A+gAAAAA/////////////////////////////wPoAAAAAP///////////////"
                "/////////////8D6AAAOEJJTQQIAAAAAAAQAAAAAQAAAkAAAAJAAAAAADhCSU0EHgAAAAAABAAAAAA4QklNBBoAAAAAA08AAAAGA"
                "AAAAAAAAAAAAACWAAAAlgAAAA0AbgBvAC0AdQBzAGUAcgAtAHQAaAB1AG0AYgAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAA"
                "AAAAAAAlgAAAJYAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAQAAAAAAAG51bGwAAAACAAAABmJvdW5kc"
                "09iamMAAAABAAAAAAAAUmN0MQAAAAQAAAAAVG9wIGxvbmcAAAAAAAAAAExlZnRsb25nAAAAAAAAAABCdG9tbG9uZwAAAJYAAAAAU"
                "mdodGxvbmcAAACWAAAABnNsaWNlc1ZsTHMAAAABT2JqYwAAAAEAAAAAAAVzbGljZQAAABIAAAAHc2xpY2VJRGxvbmcAAAAAAAAAB"
                "2dyb3VwSURsb25nAAAAAAAAAAZvcmlnaW5lbnVtAAAADEVTbGljZU9yaWdpbgAAAA1hdXRvR2VuZXJhdGVkAAAAAFR5cGVlbnVtA"
                "AAACkVTbGljZVR5cGUAAAAASW1nIAAAAAZib3VuZHNPYmpjAAAAAQAAAAAAAFJjdDEAAAAEAAAAAFRvcCBsb25nAAAAAAAAAABMZ"
                "WZ0bG9uZwAAAAAAAAAAQnRvbWxvbmcAAACWAAAAAFJnaHRsb25nAAAAlgAAAAN1cmxURVhUAAAAAQAAAAAAAG51bGxURVhUAAAAA"
                "QAAAAAAAE1zZ2VURVhUAAAAAQAAAAAABmFsdFRhZ1RFWFQAAAABAAAAAAAOY2VsbFRleHRJc0hUTUxib29sAQAAAAhjZWxsVGV4d"
                "FRFWFQAAAABAAAAAAAJaG9yekFsaWduZW51bQAAAA9FU2xpY2VIb3J6QWxpZ24AAAAHZGVmYXVsdAAAAAl2ZXJ0QWxpZ25lbnVtA"
                "AAAD0VTbGljZVZlcnRBbGlnbgAAAAdkZWZhdWx0AAAAC2JnQ29sb3JUeXBlZW51bQAAABFFU2xpY2VCR0NvbG9yVHlwZQAAAABOb"
                "25lAAAACXRvcE91dHNldGxvbmcAAAAAAAAACmxlZnRPdXRzZXRsb25nAAAAAAAAAAxib3R0b21PdXRzZXRsb25nAAAAAAAAAAtya"
                "WdodE91dHNldGxvbmcAAAAAADhCSU0EKAAAAAAADAAAAAE/8AAAAAAAADhCSU0EEQAAAAAAAQEAOEJJTQQUAAAAAAAEAAAAAThCS"
                "U0EDAAAAAAHFgAAAAEAAACWAAAAlgAAAcQAAQjYAAAG+gAYAAH/2P/gABBKRklGAAECAABIAEgAAP/tAAxBZG9iZV9DTQAC/+4AD"
                "kFkb2JlAGSAAAAAAf/bAIQADAgICAkIDAkJDBELCgsRFQ8MDA8VGBMTFRMTGBEMDAwMDAwRDAwMDAwMDAwMDAwMDAwMDAwMDAwMD"
                "AwMDAwMDAENCwsNDg0QDg4QFA4ODhQUDg4ODhQRDAwMDAwREQwMDAwMDBEMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM/8AAE"
                "QgAlgCWAwEiAAIRAQMRAf/dAAQACv/EAT8AAAEFAQEBAQEBAAAAAAAAAAMAAQIEBQYHCAkKCwEAAQUBAQEBAQEAAAAAAAAAAQACA"
                "wQFBgcICQoLEAABBAEDAgQCBQcGCAUDDDMBAAIRAwQhEjEFQVFhEyJxgTIGFJGhsUIjJBVSwWIzNHKC0UMHJZJT8OHxY3M1FqKyg"
                "yZEk1RkRcKjdDYX0lXiZfKzhMPTdePzRieUpIW0lcTU5PSltcXV5fVWZnaGlqa2xtbm9jdHV2d3h5ent8fX5/cRAAICAQIEBAMEB"
                "QYHBwYFNQEAAhEDITESBEFRYXEiEwUygZEUobFCI8FS0fAzJGLhcoKSQ1MVY3M08SUGFqKygwcmNcLSRJNUoxdkRVU2dGXi8rOEw"
                "9N14/NGlKSFtJXE1OT0pbXF1eX1VmZ2hpamtsbW5vYnN0dXZ3eHl6e3x//aAAwDAQACEQMRAD8A9FYxha07RwOw8Pgn2M/dH3D+5"
                "Jn0G/BSQSx2M/dH4f3JbGfuj8P7lJJJTHYz90fh/clsZ+6Pw/uUkklMdjP3R+H9yWxn7o/D+5SSSUx2M/dH4f3JbGfuj8P7lJJJT"
                "HYz90fh/clsZ+6Pw/uUkklMdjP3R+H9yWxn7o/D+5SSSUx2M/dH4f3JbGfuj8P7lJJJTHYz90fh/clsZ+6Pw/uUkjwkpgWN3NEDn"
                "y4j4JJ3fSb8f4FJFD//0PRmfQb8FJRZ9BvwUkEqSSSSUpJJO1rnGAkpYa8I1dPd33KTKQ3XuiIoYelX4Jekzw/E/wB6mkkph6Nfg"
                "hWVFuo1CsJGD5pKaaSnbXtM9lBBKkkkklKSPCSR4SUxd9Jvx/gkk76Tfj/BJFD/AP/R9GZ9BvwUlFn0G/BSQSpJJJJSkXHHJKEj4"
                "/0SkpI5waJUfUceGpnHc+B+anRQtusPgEiHn84/JOkkpjt8SSmjYdzfmFNJJS7gHN+SqxCsVmCWH5fBBsEPISUxSSSQSpI8JJHhJ"
                "TF30m/H+CSTvpN+P8EkUP8A/9L0Zn0G/BSUWfQb8FJBKkkkklKR6Pon4oCnXYWaQkpI3knxKkos+j8ZUkUKSSSSUpJJJJSw/nG+Y"
                "Q7x75UrTEEIRJJkpKWSSSQSpI8JJHhJTF30m/H+CSTvpN+P8EkUP//T9GZ9BvwUlFn0G/BSQSpJJJJSkkkklJqtW/BTQaXdkZFCk"
                "kkklKSSS4SUitPCGne6SopKXSSSQSpI8JJHhJTF30m/H+CSTvpN+P8ABJFD/9T0Zn0G/BSUWfQb8FJBKkkkklKSSSRQppLSjtO4S"
                "gIrG6SDqkpIko7/AN4Qn3DkFJS6Ha+PaE/ud9HQdyUJ4hySlkkkkEqSSSSUpI8JJHhJTF30m/H+CSTvpN+P8EkUP//V9GZ9BvwUl"
                "Fn0G/BSQSpKCdAkrFLABuPJSQibS8qYx/EoydFSA0ACQnbG0IqFG15HY6j4pKZJoE8apJ0lKUWsD5ce/CTpMNHdFAgADskpEcdp4"
                "Kicd3bVHTpKabmOadUyuEAiCq9tewyOCkpGkeEkjwgli76Tfj/BJJ30m/H+CSKH/9b0Zn0G/BSUWfQb8FJBKhqQPFXAIACrUtl4V"
                "lFCk6SSSlKFjZbPcKaSSkQMie6dNG15b2OoSeZEeKSlMBJLj8kRM0QAE6Sl0kkklLKFwlnwRExEiElNLunPCciCQmPCCmLvpN+P8"
                "EknfSb8f4JIqf/X9GZ9BvwUlFn0G/BSSUmxxrPyR0Gge0/FFSUukkkkpSSSZJTCxst05Cav3HcfkiJRHCSlJ0ydJSkkkklKSSSSU"
                "1bRDyhnhFv+khJKWd9Jvx/gkk76Tfj/AASSU//Q9FYTtEA8eSfc790/gvmRJJT9QVPuA9tZInxH96J6l/eo/eP718tpJKfqX1Lf9"
                "Efvb/el6lv+iP3t/vXy0kkp+pfUt/0Tvvb/AOST+pZ/onfe3/yS+WUklP1N6ln+id97f/JJvUs/0Tvvb/5JfLSSSn6l9Sz/AETvv"
                "b/5JL1Lf9Efvb/evlpJJT9S+pd/oj94/vUTZd/oj94/vXy4kkp+o99/+j/6n/ySiXZH7h+W3/yS+XkklP0691k+9hn4hRJdH0T+C"
                "+ZEklP00S7c32mZ0GngkvmVJJT/AP/ZOEJJTQQhAAAAAABVAAAAAQEAAAAPAEEAZABvAGIAZQAgAFAAaABvAHQAbwBzAGgAbwBwA"
                "AAAEwBBAGQAbwBiAGUAIABQAGgAbwB0AG8AcwBoAG8AcAAgAEMAUwAzAAAAAQA4QklNBAYAAAAAAAcABgABAAEBAP/hDpdodHRwO"
                "i8vbnMuYWRvYmUuY29tL3hhcC8xLjAvADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkI"
                "j8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDQuMS1jMDM2IDQ2L"
                "jI3NjcyMCwgTW9uIEZlYiAxOSAyMDA3IDIyOjQwOjA4ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cud"
                "zMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4YXA9I"
                "mh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIge"
                "G1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4YXBNTT0iaHR0cDovL25zL"
                "mFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iIHhtbG5zOmV4a"
                "WY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIiB4YXA6Q3JlYXRlRGF0ZT0iMjAxMC0wNC0wOFQxMDozNTo0OC0wNzowM"
                "CIgeGFwOk1vZGlmeURhdGU9IjIwMTEtMDItMjVUMjM6NDc6NTctMDg6MDAiIHhhcDpNZXRhZGF0YURhdGU9IjIwMTEtMDItMjVUM"
                "jM6NDc6NTctMDg6MDAiIHhhcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENTMyBXaW5kb3dzIiBkYzpmb3JtYXQ9ImltY"
                "WdlL2pwZWciIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiIHBob3Rvc2hvcDpIaXN0b3J5PSIiIHhhcE1NOkluc3RhbmNlSUQ9InV1a"
                "WQ6Q0JFN0I3NkY3QzQxRTAxMUJFOTQ5NzcwOUI0NjMzQkMiIHRpZmY6T3JpZW50YXRpb249IjEiIHRpZmY6WFJlc29sdXRpb249I"
                "jk2MDAwMC8xMDAwMCIgdGlmZjpZUmVzb2x1dGlvbj0iOTYwMDAwLzEwMDAwIiB0aWZmOlJlc29sdXRpb25Vbml0PSIyIiB0aWZmO"
                "k5hdGl2ZURpZ2VzdD0iMjU2LDI1NywyNTgsMjU5LDI2MiwyNzQsMjc3LDI4NCw1MzAsNTMxLDI4MiwyODMsMjk2LDMwMSwzMTgsM"
                "zE5LDUyOSw1MzIsMzA2LDI3MCwyNzEsMjcyLDMwNSwzMTUsMzM0MzI7NDE2QzZBQjk2Qjg2MTJBNERGMENFM0Q1MjM5RTc3RDAiI"
                "GV4aWY6UGl4ZWxYRGltZW5zaW9uPSIxNTAiIGV4aWY6UGl4ZWxZRGltZW5zaW9uPSIxNTAiIGV4aWY6Q29sb3JTcGFjZT0iLTEiI"
                "GV4aWY6TmF0aXZlRGlnZXN0PSIzNjg2NCw0MDk2MCw0MDk2MSwzNzEyMSwzNzEyMiw0MDk2Miw0MDk2MywzNzUxMCw0MDk2NCwzN"
                "jg2NywzNjg2OCwzMzQzNCwzMzQzNywzNDg1MCwzNDg1MiwzNDg1NSwzNDg1NiwzNzM3NywzNzM3OCwzNzM3OSwzNzM4MCwzNzM4M"
                "SwzNzM4MiwzNzM4MywzNzM4NCwzNzM4NSwzNzM4NiwzNzM5Niw0MTQ4Myw0MTQ4NCw0MTQ4Niw0MTQ4Nyw0MTQ4OCw0MTQ5Miw0M"
                "TQ5Myw0MTQ5NSw0MTcyOCw0MTcyOSw0MTczMCw0MTk4NSw0MTk4Niw0MTk4Nyw0MTk4OCw0MTk4OSw0MTk5MCw0MTk5MSw0MTk5M"
                "iw0MTk5Myw0MTk5NCw0MTk5NSw0MTk5Niw0MjAxNiwwLDIsNCw1LDYsNyw4LDksMTAsMTEsMTIsMTMsMTQsMTUsMTYsMTcsMTgsM"
                "jAsMjIsMjMsMjQsMjUsMjYsMjcsMjgsMzA7QUZCRTBGRkExQzU1RTU2Mzc1NUQ1OTlDRjYxMzEyNEIiLz4gPC9yZGY6UkRGPiA8L"
                "3g6eG1wbWV0YT4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
                "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8P3hwYWNrZXQgZW5kPSJ3Ij8+/+4ADkFkb2JlAGRAAAAAAf/bAIQAA"
                "gICAgICAgICAgMCAgIDBAMCAgMEBQQEBAQEBQYFBQUFBQUGBgcHCAcHBgkJCgoJCQwMDAwMDAwMDAwMDAwMDAEDAwMFBAUJBgYJD"
                "QoJCg0PDg4ODg8PDAwMDAwPDwwMDAwMDA8MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwM/8AAEQgAlgCWAwERAAIRAQMRAf/dA"
                "AQAE//EAIUAAAEFAQEBAAAAAAAAAAAAAAABAgMEBQYJBwEBAQEAAAAAAAAAAAAAAAAAAAECEAABAwICAwsKBAQHAQAAAAABAAIDE"
                "QQhMUESBVGxIjJysjNzsxQ0YXGBkaFCktITVMEj0xVSg1UGYkNTY5MkNiURAQEBAAAAAAAAAAAAAAAAAAABEf/aAAwDAQACEQMRA"
                "D8A9nLSztX2trI60ic58THOc5jKuIjaD7mhYaWu52mixh+BnyIE7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7"
                "pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iA7pa/Y"
                "w/DH8iA7pa/Yw/DH8iA7pa/Yw/DH8iANpa0dWyhAIIJDWYAjkKxKgltbcTW0YgjDX3BBfqsxZ9Fx/gWkf/Q9prLwViNP0hzWrDS0"
                "gEAgEAgFcTRXEDGp0UTDS0IxIoN0phpMDgCCdwJhpaHcp50w00OBNFMCooQCAQCBr+I/kneViVXm8RZn/ed2L1pH//R9pbPwlj1Q"
                "5rVhpbQCAQCAJAFTgFYlKyr3AMBJOVBVaR0NhscupLdt4OhhOfqQav7TY6IaHdDnDeIPtQJ+02elhI3C+Q77yPYgX9psP8AQ9p/F"
                "Bz9/sx9sXPjrJEMTo1VKRkVGB3clloIBAIBA2TiP5J3lYlQTdNadc7sXLSP/9L2ls/CWPVDmtWGltAIBAIJ4Lea4fqQtDnUrjSlF"
                "YldXY7JjtdSR1TPnX3RhuLSNihpjmMkDThicPOgWmkYoAZ7iBjtV4NSJAMHNUo47amz+6ymZvROxZ6VMXWVmpgEUIBA2TiP5J3lY"
                "lQTdNadc7sXLSP/0/aWz8JY9UOa1YaW0AgEAcj5ig6HYDGkySOz1Q0HyqxK6C5uWW7dcjWoaagzJWkUjtCd4rFaEjQ4uA/FA3698"
                "8f5EPlo5zh5sKIInNu3A618/ksAaEDO7aw4c0rznRzsD56IIyzub23UALWnxEZcTQZVQat1FHd2rgOEHM1mO82KDgzHqEVPC1iCF"
                "KQqy0EAgbJxH8k7ysSoJumtOud2LlpH/9T2ls/CWPVDmtWGltAIBAhyQdZsENEEhpU1pRWJUk8gnvAxg4No3Wruudwae1aRLju0G"
                "4gEAgEBQOIaRUO4JHnwQNsJDHJLZvNQKmI7rMt9Bze0mNjvJI25DFSkUVloIBA2TiP5J3lYlQTdNadc7sXLSP/V9pbPwlj1Q5rVh"
                "pbQCAQI7L0jfQdXsI/kPpmZK+hWJRb8J1xLmXyuA5LcN9aRaQCAQCBRmDuEH1IIWcG+tDkXxua7z11t4IMfbbKXuuMQ8ZqUjIWWg"
                "gEDZOI/kneViVBN01p1zuxctI//1vaWz8JY9UOa1YaW0AgEBWmiqDQsNoyWI1QzXa5w1juCuKsStqyP/XDjnK55Pkq6oWkWkAgEA"
                "gUIMvaUjojBIw0c2tD5xRBgyyyTPD5DUjJSkMWWggEDZOI/kneViVBN01p1zuxctI//1/aWz8JY9UOa1YaW0AgEAgN1WJXR7NcX2"
                "4r7ma0jRQCAQCAoTgM9CDC2q6n02n3c0GOHAmgKlIVZaCAQNk4j+Sd5WJUE3TWnXO7Fy0j/0PaWz8JY9UOa1YaW0AgEAgDkaKxK1"
                "dkTEazCaArSOgphXQgRAIBAE6tScKAoORvJ/rSuaDWhU0xVa0ggkJaqRZUIBA2TiP5J3lYlQTdNadc7sXLSP//R9pbPwlj1Q5rVh"
                "pbQCAQCBDWhorEp8Er4Htc0YDNaR1VtK2eMPDxUZjLfQWEAgDUEA4E5A4EoMrad4Wn6EZFSOEQclKRgDDWw42lZaCAQCAQNk4j+S"
                "d5WJUE3TWnXO7Fy0j//0vaWz8JY9UOa1YaW0AgECawrTSriaVJA00IpVaR0FpbkQtfDINc8UOrRBbbdmurcMcxzcnjFvsqgl7zGG"
                "67ZBUZafYgrF1xO6kAMTDi+WUguwx4NK0QYV5GIrhxdUF41qnSPwUorazf4h61MXQXNGkYphoBBTDSqYBFNk4j+Sd5WJUE3TWnXO"
                "7Fy0j//0/aWz8JY9UOa1YaW0AgXVc4hra6xyoKqxK0INj3c2qSKNObsAtI049gEU+rJhpQLNsSKON8jHF7m4hu+gsWzWd3Zq46ua"
                "CbhZcEt0tIQIIoi4O+i3XGRGCB+DQXAU1MXDyDNBTt7SK+Mt3LUCV2rEANAx0oFl2DbyA6kjmu0BwFPYgpzbBnaKxFr6bmB9qDGn"
                "tLi3f8AmNcGtzwJHrQQ1ClIVZaNk4j+Sd5WJUE3TWnXO7Fy0j//1PaWz8JY9UOa1YaW0CE0FdxEddsizjbC26e3WkmzadCsG8CMA"
                "FpAckDaHcr5EGE5ndr2WKtIbissR/x6W+TDFBOCCaaUDh56eVBWuC52pbx494NAfIMT7Ag2YomxRsja2jWDDzoJQDVApyQRvYyRp"
                "a8Va7NpQcdtWxFpIHt4j1KRkrLRsnEfyTvKxKgm6a0653YuWkf/1faWz8JY9UOa1YaWiaCpRDo2/Vkjjbi57gAPSiPocUYiijjHu"
                "inpVgmANVoOQIa0NM0GdfW5kg+o0fmxcJvoxPsQU4pRLGyd3Be8cJm4EEpyrkBjXzYoG2LXTSS3Tm8EmkP4oNoHJAqBDkgbSmeSD"
                "N2tC2W0c73o6EesKUjhG62sa5VwKy0dJxH8k7ysSoJumtOud2LlpH//1vaWz8JY9UOa1YaWziCiNHZEBku4tYYtOt6FcR3ABqa7t"
                "R6kkD1oCAQIcvwQYOoLa7kt/cmcZYid2mLfUgW6eXtbC3B0po30Yn2BBrQx6kUbGtoGhBMAaoHIBAhyQRSxiSN7HDBwUo+eSs+nI"
                "9pFKONFlUL+I/kneVghm6a0653YuWkf/9f2ls/CWPVDmtWGlo5GmZwCsSuk2A0az5XZhuoD5dK0jqKhAqAQCBDkgy9oW5kga5nSw"
                "EOa7SRXH2IIbAd6ebt4wZwY2n1FBs4VFDloQOQCAQCBDkpRwe1WCK9l0NOLVkZb3AseAfdO8rBHN01p1zuxctD/0PaWz8JZYg0iF"
                "SDX3WrOLqySDShFajT5UkHXbCa3u8jq1P1DlitI3RnkUD0CVG6gWqBDkgbSuBGBwKBgjawAMbqNbjQaUDxmgdUbqBUBWiBK+f1IA"
                "5KUcVt4DvJNRUjAafUpgwDk7kneSQJMR9a0653YuWh//9H2ctJZRb2witpntEQ1iDGGngt3XILQmnqNazkpUV4UW7ykG9s282syN"
                "/ddkSzR65r+bA3R5XhBqt2htyo19gSlukCeCvaIJRtHatf/AD1wP58H4yIHfuG1f6Bceie2r2iAG0Np1Fdg3Xpmtv1EEn7htH+hX"
                "P8AzW36qA/cNo/0K5/5rb9VA120No0P/wAG5Pk+tbfqoGDaO0tH9v3VdFZrb9VAp2jtWn/n7g/z7f8ACRA39x2scv7fnHlM8FOeg"
                "rvv9sk4bDe07pmgPsMg30Effduni7IcDoIdbfroK0t1/cNDTZ8tdxhtg70ETE+xBiXVxtAyDvlhOyTQXPi1vY5BWdNNqv8A+pcDg"
                "n3o9zlIK80s5ntHOtbhsjZnfTj1o+F+S6uOtTJB/9k="
            )

            thumbnail += "' width='200' height='133"

        firstName = "Not Provided"
        lastName = "Not Provided"
        fullName = "Not Provided"
        description = "This user has not provided any personal information."

        try:
            firstName = self.firstName
        except:
            firstName = "Not Provided"

        try:
            lastName = self.lastName
        except:
            firstName = "Not Provided"

        try:
            fullName = self.fullName
        except:
            fullName = "Not Provided"

        try:
            description = self.description
        except:
            description = "This user has not provided any personal information."

        url = self.homepage

        return (
            """<div class="9item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='"""
            + str(url)
            + """' target='_blank'>
                        <img src='"""
            + str(thumbnail)
            + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right" style="float: none; width: auto; overflow: hidden;">
                        <a href='"""
            + str(url)
            + """' target='_blank'><b>"""
            + str(fullName)
            + """</b>
                        </a>
                        <br/><br/><b>Bio</b>: """
            + str(description)
            + """
                        <br/><b>First Name</b>: """
            + str(firstName)
            + """
                        <br/><b>Last Name</b>: """
            + str(lastName)
            + """
                        <br/><b>Username</b>: """
            + str(self.username)
            + """
                        <br/><b>Joined</b>: """
            + str(datetime.fromtimestamp(self.created / 1000).strftime("%B %d, %Y"))
            + """

                    </div>
                </div>
                """
        )

    @property
    def groups(self):
        """The ``groups`` property retrieves a List of :class:`~arcgis.gis.Group` objects the current user belongs to."""
        return [Group(self._gis, group["id"]) for group in self["groups"]]

    def update_license_type(self, user_type: str):
        """

        The ``update_license_type`` method is primarily used to update the user's licensing type. This allows
        administrators to change a user from a creator to a viewer or any
        other custom user license type.

        .. note::
            The ``update_license_type`` method is available in ArcGIS Online and Portal 10.7+.

        =====================  =========================================================
        **Parameter**           **Description**
        ---------------------  ---------------------------------------------------------
        user_type              Required string. The user license type to assign a user.

                               Built-in Types: creator or viewer
        =====================  =========================================================

        :return: A boolean indicating success (True) or failure (False).

        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError(
                "`update_license_type` is not implemented at version %s"
                % ".".join([str(i) for i in self._gis.version])
            )

        builtin = {"creator": "creatorUT", "viewer": "viewerUT"}
        if user_type.lower() in builtin:
            user_type = builtin[user_type.lower()]

        url = "%s/portals/self/updateUserLicenseType" % self._portal.resturl
        params = {
            "users": [self.username],
            "userLicenseType": user_type,
            "f": "json",
        }
        res = self._gis._con.post(url, params)
        status = [r["status"] for r in res["results"]]
        self._hydrated = False
        self._hydrate()
        if all(status):
            return all(status)
        return res

    # ----------------------------------------------------------------------
    def delete_thumbnail(self):
        """
        The ``delete_thumbnail`` removes the thumbnail from the user's profile.

        :return:
            A boolean indicating success (True) or failure (False).

        """
        if self._gis.version >= [7, 3]:
            url = (
                self._gis._portal.resturl
                + "community/users/%s/deleteThumbnail" % self.username
            )
            params = {"f": "json"}
            res = self._gis._con.post(url, params)
            if "success" in res:
                return res["success"]
            return res
        else:
            raise Exception(
                "The operation delete_thumbnail is not supported on this portal."
            )

    def expire_password(self, temporary_password: Optional[str] = None) -> bool:
        """
        Expires the current user's Password.

        =====================  ==========================================================
        **Parameter**           **Description**
        ---------------------  ----------------------------------------------------------
        temporary_password     Optional String. Allows the administrator to set a new
                               temporary password for a given user. This is available on
                               ArcGIS Enterprise Only.
        =====================  ==========================================================

        :returns: Boolean
        """
        if temporary_password and self._gis._portal.is_arcgisonline == False:
            url = f"{self._gis._portal.resturl}community/users/{self.username}/update"
            params = {"f": "json", "password": temporary_password}
            resp = self._gis._con.post(url, params)
        url = (
            f"{self._gis._portal.resturl}community/users/{self.username}/expirePassword"
        )
        if self._gis._portal.is_arcgisonline:
            params = {"f": "json", "expiration": -1}
        else:
            params = {"f": "json", "expiration": 1}
        resp = self._gis._con.post(url, params)
        return resp.get("success", False)

    def reset(
        self,
        password: Optional[str] = None,
        new_password: Optional[str] = None,
        new_security_question: Optional[str] = None,
        new_security_answer: Optional[str] = None,
        reset_by_email: bool = False,
    ):
        """
        The ``reset`` method resets a user's password, security question, and/or security answer.
        If a new security question is specified, a new security answer should be provided.


        .. note::
            This function does not apply to those using enterprise accounts
            that come from an enterprise such as ActiveDirectory, LDAP, or SAML.
            It only has an effect on built-in users.

        .. note::
            To reset the password by email, set `reset_by_email` to True and `password`
            to `None`.

        =====================  =========================================================
        **Parameter**           **Description**
        ---------------------  ---------------------------------------------------------
        password               Required string. The current password.
        ---------------------  ---------------------------------------------------------
        new_password           Optional string. The new password if resetting password.
        ---------------------  ---------------------------------------------------------
        new_security_question  Optional string. The new security question if desired.
        ---------------------  ---------------------------------------------------------
        new_security_answer    Optional string. The new security question answer if desired.
        ---------------------  ---------------------------------------------------------
        reset_by_email         | Optional Boolean.  If True, the `user` will be reset by email. The default is False.

                               **NOTE:** Not available with ArcGIS on Kubernetes.
        =====================  =========================================================

        .. Warning::
            This function does not apply to those using enterprise accounts
            that come from an enterprise such as ActiveDirectory, LDAP, or SAML.
            It only has an effect on built-in users.

        :return:
            A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> user.reset("password123",new_password="passWORD1234",reset_by_email=True)

        """
        postdata = {"f": "json"}
        if password:
            postdata["password"] = password
        if new_password:
            postdata["newPassword"] = new_password
        if new_security_question:
            postdata["newSecurityQuestionIdx"] = new_security_question
        if new_security_answer:
            postdata["newSecurityAnswer"] = new_security_answer
        if reset_by_email:
            postdata["email"] = reset_by_email
        url = self._gis._portal.resturl + "community/users/" + self.username + "/reset"
        resp = self._gis._con.post(url, postdata, ssl=True)
        if resp:
            return resp.get("success")
        return False

    def update(
        self,
        access: Optional[str] = None,
        preferred_view: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Union[list[str], str]] = None,
        thumbnail: Optional[str] = None,
        fullname: Optional[str] = None,
        email: Optional[str] = None,
        culture: Optional[str] = None,
        region: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        security_question: Optional[int] = None,
        security_answer: Optional[str] = None,
        culture_format: Optional[str] = None,
        categories: Optional[list] = None,
    ):
        """
        The ``update`` method updates this user's properties based on the arguments passed when calling ``update``.

        .. note::
            Only pass in arguments for properties you want to update.
            All other properties will be left as they are.  If you
            want to update the description, then only provide
            the description argument.

           **When updating the security question, you must provide a security_answer as well.**

        ==================  ==========================================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------------------------
        access              Optional string. The access level for the user, values
                            allowed are private, org, public.
        ------------------  ----------------------------------------------------------
        preferred_view      Optional string. The preferred view for the user, values allowed are Web, GIS, null.
        ------------------  ----------------------------------------------------------
        description         Optional string. A description of the user.
        ------------------  ----------------------------------------------------------
        tags                Optional string. Tags listed as comma-separated values, or a list of strings.
        ------------------  ----------------------------------------------------------
        thumbnail           Optional string. The path or url to a file of type PNG, GIF,
                            or JPEG. Maximum allowed size is 1 MB.
        ------------------  ----------------------------------------------------------
        fullname            Optional string. The full name of this user, only for built-in users.
        ------------------  ----------------------------------------------------------
        email               Optional string. The e-mail address of this user, only for built-in users.
        ------------------  ----------------------------------------------------------
        culture             Optional string. The two-letter language code, fr for example.
        ------------------  ----------------------------------------------------------
        region              Optional string. The two-letter country code, FR for example.
        ------------------  ----------------------------------------------------------
        first_name          Optional string. User's first name.
        ------------------  ----------------------------------------------------------
        last_name           Optional string. User's first name.
        ------------------  ----------------------------------------------------------
        security_question   Optional integer.  The is a number from 1-14.  The
                            questions are as follows:

                            1. What city were you born in?
                            2. What was your high school mascot?
                            3. What is your mother's maden name?
                            4. What was the make of your first car?
                            5. What high school did you got to?
                            6. What is the last name of your best friend?
                            7. What is the middle name of your youngest sibling?
                            8. What is the name of the street on which your grew up?
                            9. What is the name of your favorite fictional character?
                            10. What is the name of your favorite pet?
                            11. What is the name of your favorite restaurant?
                            12. What is the title of your facorite book?
                            13. What is your dream job?
                            14. Where did you go on your first date?

                            Usage Example:

                            security_question=13
        ------------------  ----------------------------------------------------------
        security_answer     Optional string.  This is the answer to security question.
                            If you are changing a user's question, an answer must be
                            provided.

                            Usage example:

                            security_answer="Working on the Python API"
        ------------------  ----------------------------------------------------------
        culture_format      Optional String. Specifies user-preferred number and date format
        ------------------  ----------------------------------------------------------
        categories          Optional List[str]. A list of category names.

                            example: ```categories = ["category11", "category12"]```
        ==================  ==========================================================

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> user.update(description="Aggregated US Hurricane Data", tags = "Hurricanes,USA, 2020")
        """
        culture_check = [
            lang["culture"].lower() for lang in self._gis.languages if lang
        ]
        if culture and not culture.lower() in culture_check:
            raise ValueError(
                f"Invalid culture provided. Allowed cultures: {''.join(culture_check)}"
            )
        if region and not region.upper() in [g["region"] for g in self._gis.regions]:
            raise ValueError(
                f"Invalid region provided. Allowed regions: {''.join([g['region'] for g in self._gis.regions])}"
            )
        user_type = None
        if tags is not None and isinstance(tags, list):
            tags = ",".join(tags)
        import copy

        params = {
            "f": "json",
            "access": access,
            "preferredView": preferred_view,
            "description": description,
            "tags": tags,
            "password": None,
            "fullname": fullname,
            "email": email,
            "securityQuestionIdx": None,
            "securityAnswer": None,
            "culture": culture,
            "region": region,
            "firstName": first_name,
            "lastName": last_name,
            "clearEmptyFields": True,
            "cultureFormat": culture_format,
            "region": region,
        }
        if categories:
            params["categories"] = categories
        if security_answer and not security_question is None:
            params["securityQuestionIdx"] = security_question
            params["securityAnswer"] = security_answer
        for k, v in copy.copy(params).items():
            if v is None:
                del params[k]

        if thumbnail:
            files = {"thumbnail": thumbnail}
        else:
            files = None
        url = "%scommunity/users/%s/update" % (
            self._gis._portal.resturl,
            self._user_id,
        )
        ret = self._gis._con.post(path=url, postdata=params, files=files)
        if ret["success"] == True:
            self._hydrate()
        return ret["success"]

    @property
    def landing_page(self) -> str:
        """
        Gets or sets the User's login page,

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required string. The values are `home`, `gallery`, `map`,
                          `scene`, `groups`, `content`, or `organization`
        ================  ==========================================================

        :return: str

        .. code-block:: python

           # Usage example: Setting login page

           >>> user1 = gis.users.get("org_data_viewer") # approach 1: set via landing_page

           >>> user1.landing_page = "map"

           >>> us = user.user_settings # approach 2: set via user_settings

           >>> us['landingPage']['url'] = "webmap/viewer.html"

           >>> user1.user_settings = us

        """
        value = self.user_settings.get("landingPage", {}).get("url", "")
        lu = {
            "index.html": "home",
            "gallery.html": "gallery",
            "webmap/viewer.html": "map",
            "webscene/viewer.html": "scene",
            "groups.html": "groups",
            "content.html": "content",
            "organization.html": "organization",
        }

        if value == "":
            return None
        return lu[value.lower()]

    @landing_page.setter
    def landing_page(self, value: str):
        """
        Returns the User's login page

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required string. The values are `home`, `gallery`, `map`,
                          `scene`, `groups`, `content`, or `organization`
        ================  ==========================================================

        :return: str
        """
        value = value.lower()
        landing_pages_lu = {
            "home": "index.html",
            "gallery": "gallery.html",
            "map": "webmap/viewer.html",
            "scene": "webscene/viewer.html",
            "groups": "groups.html",
            "content": "content.html",
            "organization": "organization.html",
        }
        if value in landing_pages_lu:
            value = landing_pages_lu[value]
            us = self.user_settings
            us["landingPage"] = {"url": f"{value}"}
            url = "%s/sharing/rest/community/users/%s/setProperties" % (
                self._gis._url,
                self.username,
            )
            params = {"f": "json", "properties": us}
            res = self._gis._con.post(url, params)

        else:
            raise ValueError("The ")

    # ----------------------------------------------------------------------
    @property
    def user_settings(self) -> dict:
        """
        Get/set the current user's settings that are defined in the user profile.

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required dict. The `landingPage` and `appLauncher` settings.
        ================  ==========================================================

        :return: dict

        .. code-block:: python

           # Usage example: Getting the current user settings

           >>> us = user.user_settings # similar to setting the landing_page property

           >>> us['landingPage']['url'] = "webmap/viewer.html"

           >>> user1.user_settings = us

        """
        url = "%s/sharing/rest/community/users/%s/properties" % (
            self._gis._url,
            self.username,
        )
        params = {"f": "json"}
        return self._gis._con.get(url, params).get("properties", {})

    # ----------------------------------------------------------------------
    @user_settings.setter
    def user_settings(self, value: dict):
        """
        Get/set the current user's settings that are defined in the user profile.

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required dict. The `landingPage` and `appLauncher` settings.
        ================  ==========================================================

        :return: dict
        """
        url = "%s/sharing/rest/community/users/%s/setProperties" % (
            self._gis._url,
            self.username,
        )
        params = {"f": "json", "properties": value}
        self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def disable(self):
        """
        The ``disable`` method disables login access for the user.

        .. note::
            The ``disable`` method is only available to the administrator of the organization.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {"f": "json"}
        url = "%s/sharing/rest/community/users/%s/disable" % (
            self._gis._url,
            self._user_id,
        )
        res = self._gis._con.post(url, params)
        if "status" in res:
            self._hydrate()
            return res["status"] == "success"
        elif "success" in res:
            self._hydrate()
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def enable(self):
        """
        The ``enable`` method enables login access for the user.

        .. note::
            ``enable`` is only available to the administrator of the organization.
        """
        params = {"f": "json"}
        url = "%s/sharing/rest/community/users/%s/enable" % (
            self._gis._url,
            self._user_id,
        )
        res = self._gis._con.post(url, params)
        if "status" in res:
            self._hydrate()
            return res["status"] == "success"
        elif "success" in res:
            self._hydrate()
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    @property
    def esri_access(self):
        """
        The ``esri_access`` property will return a string describing the current user's Esri access.
        When setting, supply a ``boolean`` to enable or disable ``esri_access`` for that :class:`~arcgis.gis.User`
        object.

        .. note::
            Administrator privileges are required to enable or disable Esri access

        A member whose account has Esri access enabled can use My Esri and
        Community and Forums (GeoNet), access e-Learning on the Training
        website, and manage email communications from Esri. The member
        cannot enable or disable their own access to these Esri resources.

        .. warning::
            Trial accounts cannot modify esri_access property.

        Please see the `Enable Esri access <https://doc.arcgis.com/en/arcgis-online/administer/manage-members.htm#ESRI_SECTION1_7CE845E428034AE8A40EF8C1085E2A23>`_
        section in the Manage members page in ArcGIS Online Resources for more information.

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        value             Required boolean. The current user will be allowed to use
                          the username for other Esri/ArcGIS logins when the value
                          is set to True. If false, the account can only be used to
                          access a given individual's organization.
        ================  ==========================================================

        """
        if self._portal.is_arcgisonline:
            self._hydrate()
            return self["userType"]
        else:
            return False

    # ----------------------------------------------------------------------
    @esri_access.setter
    def esri_access(self, value: bool):
        """
        See main ``esri_access`` property docstring
        """
        if self._portal.is_arcgisonline:
            if value == True:
                ret = self._portal.update_user(self._user_id, user_type="both")
            else:
                ret = self._portal.update_user(self._user_id, user_type="arcgisonly")
            self._hydrate()

    # ----------------------------------------------------------------------
    @property
    def linked_accounts(self):
        """The ``linked_accounts`` method retrieves all linked accounts for the current user as
        :class:`~arcgis.gis.User` objects

        :return:
            A list of :class:`~arcgis.gis.User` objects
        """
        if self._gis._portal.is_arcgisonline == False:
            return []
        url = "%s/sharing/rest/community/users/%s/linkedUsers" % (
            self._gis._url,
            self._user_id,
        )
        start = 1
        params = {"f": "json", "num": 10, "start": start}
        users = []
        res = self._gis._con.get(url, params)
        users = res["linkedUsers"]
        if len(users) == 0:
            return users
        else:
            while res["nextStart"] > -1:
                start += 10
                params["start"] = start
                res = self._gis._con.get(url, params)
                users += res["linkedUsers"]
        users = [self._gis.users.get(user["username"]) for user in users]
        return users

    # ----------------------------------------------------------------------
    def link_account(self, username: Union[str, User], user_gis: GIS):
        """
        The ``link_account`` method allows a user to link several accounts to gether and share information between them.
        For example, if you use multiple accounts for ArcGIS Online and Esri websites,
        you can link them so you can switch between accounts and share your
        Esri customer information with My Esri, e-Learning, and GeoNet. You
        can link your organizational, public, enterprise, and social login
        accounts. Your content and privileges are unique to each account.
        From Esri websites, only Esri access-enabled accounts appear in
        your list of linked accounts. See the :attr:`~arcgis.gis.User.unlink_account` method for more information on how
        to unlink linked accounts that have been produced using the ``link_account`` method.

        See the `Sign in <http://doc.arcgis.com/en/arcgis-online/reference/sign-in.htm>`_ page in ArcGIS Online Resources
        for addtional information.

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        username          required string/User. This is the username or User object
                          that a user wants to link to.
        ----------------  ----------------------------------------------------------
        user_gis          required GIS.  This is the GIS object for the username.
                          In order to link an account, a user must be able to login
                          to that account.  The GIS object is the entry into that
                          account.
        ================  ==========================================================

        .. code-block:: python

            # Usage Example

            >>> gis = GIS("https://www.arcgis.com", "username1", "password123")
            >>> user = gis.users.get('username')
            >>> user.link_account("User1234", gis)

        returns: A boolean indicating success (True) or failure (False).

        """
        userToken = user_gis._con.token
        if isinstance(username, User):
            username = username.username
        params = {"f": "json", "user": username, "userToken": userToken}
        url = "%s/sharing/rest/community/users/%s/linkUser" % (
            self._gis._url,
            self._user_id,
        )
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def unlink_account(self, username: Union[str, User]):
        """
        The ``unlink_account`` method allows for the removal of linked accounts when a user wishes to no longer have
        a linked account. See the :attr:`~arcgis.gis.User.link_account` method for more information on how accounts are
        linked together.

        See the `Sign in <http://doc.arcgis.com/en/arcgis-online/reference/sign-in.htm>`_ page in ArcGIS Online Resources
        for addtional information.

        ================  ==========================================================
        **Parameter**      **Description**
        ----------------  ----------------------------------------------------------
        username          required string/User. This is the username or User object
                          that a user wants to unlink.
        ================  ==========================================================

        .. code-block:: python

            # Usage Example

            >>> user.unlink_account("User1234")

        returns: A boolean indicating success (True) or failure (False).
        """
        if isinstance(username, User):
            username = username.username
        params = {"f": "json", "user": username}
        url = "%s/sharing/rest/community/users/%s/unlinkUser" % (
            self._gis._url,
            self._user_id,
        )
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def update_level(self, level: int):
        """
        The ``update_level`` allows administrators
        of an organization to update the level of a user. Administrators can
        leverage two levels of membership when assigning roles and
        privileges to members. Membership levels allow organizations to
        control access to some ArcGIS capabilities for some members while
        granting more complete access to other members.

        .. note::
            Level 1 membership is designed for members who need privileges to view and interact
            with existing content, while Level 2 membership is for those who
            contribute, create, and share content and groups, in addition to
            other tasks.

        Maximum user quota of an organization at the given level is checked
        before allowing the update.

        Built-in roles including organization administrator, publisher, and
        user are assigned as Level 2, members with custom roles can be
        assigned as Level 1, 1PlusEdit, or Level 2.

        Level 1 membership allows for limited capabilities given through a
        maximum of 8 privileges: `portal:user:joinGroup,
        portal:user:viewOrgGroups, portal:user:viewOrgItems,
        portal:user:viewOrgUsers, premium:user:geocode,
        premium:user:networkanalysis, premium:user:demographics, and
        premium:user:elevation`. If updating the role of a Level 1 user with
        a custom role that has more privileges than the eight, additional
        privileges will be disabled for the user to ensure restriction.

        .. note::
            Level 1 users are not allowed to own any content or group which can
            be reassigned to other users through the Reassign Item and Reassign
            Group operations before downgrading them. The operation will also
            fail if the user being updated has got licenses assigned to premium
            apps that are not allowed at the targeting level.

        =====================  =========================================================
        **Parameter**           **Description**
        ---------------------  ---------------------------------------------------------
        level                  Required int. The values of 1 or 2. This
                               is the user level for the given user.


                                    + 1 - View only
                                    + 2 - Content creator


        =====================  =========================================================

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> user.update_level(2)
        """
        if self._gis.version >= [6, 4]:
            raise NotImplementedError(
                "`update_level` is not applicable at version %s"
                % ".".join([str(i) for i in self._gis.version])
            )
        if "roleId" in self and self["roleId"] != "iAAAAAAAAAAAAAAA":
            self.update_role("iAAAAAAAAAAAAAAA")
            self._hydrated = False
            self._hydrate()
        elif not ("roleId" in self) and level == 1:
            self.update_role("iAAAAAAAAAAAAAAA")
            self._hydrated = False
            self._hydrate()

        allowed_roles = {"1", "2", "11"}

        if level not in allowed_roles:
            raise ValueError("level must be in %s" % ",".join(allowed_roles))

        url = "%s/portals/self/updateUserLevel" % self._portal.resturl
        params = {"user": self.username, "level": level, "f": "json"}
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def update_role(self, role: str):
        """
        The ``update_role`` method updates this user's role to org_user, org_publisher, org_admin, viewer, view_only,
        viewplusedit, or a custom role.

        .. note::
            There are four types of roles in Portal - `user`, `publisher`, `administrator` and `custom roles`.
            A user can share items, create maps, create groups, etc.  A publisher can
            do everything a user can do and additionally create hosted services.  An administrator can
            do everything that is possible in Portal. A custom roles privileges can be customized.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        role              Required string. Value must be either org_user,
                          org_publisher, org_admin, viewer, view_only, viewplusedit
                          or a custom role object (from gis.users.roles).
        ================  ========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        lookup = {
            "admin": "org_admin",
            "user": "org_user",
            "publisher": "org_publisher",
            "view_only": "tLST9emLCNfFcejK",
            "viewer": "iAAAAAAAAAAAAAAA",
            "viewplusedit": "iBBBBBBBBBBBBBBB",
        }

        if isinstance(role, Role):
            role = role.role_id
        elif isinstance(role, str):
            if role.lower() in lookup:
                role = lookup[role.lower()]
        passed = self._portal.update_user_role(self.username, role)
        if passed:
            self._hydrated = False
            self._hydrate()

        return passed

    def delete(self, reassign_to: Optional[str] = None):
        """
        The ``delete`` method deletes this user from the portal, optionally deleting or reassigning groups and items.

        .. note::
            You can not delete a user in Portal if that user owns groups or items and/or is
            assigned an application bundle.  If you specify a user in the reassign_to
            argument, then items and groups will be transferred to that user.  If that
            argument is not set, the method will fail provided the user has items or groups
            that need to be reassigned. Additionally, see the :attr:`~arcgis.gis.User.reassign_to` method for more
            information on reassignment.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        reassign_to       Optional string. The new owner of the items and groups
                          that belong to the user being deleted.
        ================  ========================================================

        .. code-block:: python

            # Usage Example

            user.delete(reassign_to="User1234")

        :return:
            A boolean indicating success (True) or failure (False).

        """
        if isinstance(reassign_to, User):
            reassign_to = reassign_to.username

        for l in self._gis.admin.license.all():
            try:
                entitle = l.check(user=self.username)
            except Exception as e:
                entitle = []
            if len(entitle) > 0:
                l.revoke(
                    username=self.username,
                    entitlements="*",
                    suppress_email=True,
                )
        for bundle in self._gis.admin.license.bundles:
            bundle.revoke(users=self.username)
        if reassign_to:
            # reassigns the group owner to the reassigned_to user.
            [
                grp.reassign_to(User(gis=self._gis, username=reassign_to))
                for grp in self.groups
                if grp.owner == self.username
            ]
        else:
            # delete the groups owned by the user
            [grp.delete() for grp in self.groups if grp.owner == self.username]
        if self._gis._portal.is_arcgisonline:
            self.esri_access = "arcgisonly"
        return self._portal.delete_user(self._user_id, reassign_to)

    def reassign_to(self, target_username: str):
        """
        The ``reassign_to`` method reassigns all of this user's items and groups to another user.

        Items are transferred to the target user into a folder named
        <user>_<folder> where user corresponds to the user whose items were
        moved and folder corresponds to the folder that was moved.

        .. note::
            This method must be executed as an administrator.  This method also
            can not be undone.  The changes are immediately made and permanent.

        ================  ===========================================================
        **Parameter**      **Description**
        ----------------  -----------------------------------------------------------
        target_username   Required string. The user who will be the new owner of the
                          items and groups from which these are being reassigned from.
        ================  ===========================================================

        :return:
            A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> user.reassign_to(target_username="User1234")


        """
        if isinstance(target_username, User):
            target_username = target_username.username
        # currently issue with REST API method, so we try/except for it
        try:
            return self._portal.reassign_user(self._user_id, target_username)
        except:
            # variables to ensure that every item & group is assigned
            # issue with dependencies for these methods too, so try/except/pass
            items_success = True
            group_success = True
            for item in self.items():
                try:
                    if not item.reassign_to(target_username):
                        items_success = False
                except:
                    pass
            for group in self.groups:
                try:
                    if not group.reassign_to(target_username):
                        group_success = False
                except:
                    pass
            return items_success and group_success

    def get_thumbnail(self):
        """
        The ``get_thumbnail`` method returns the bytes that make up the thumbnail for this user.

        :return:
            Bytes that represent the image.

        .. code-block:: python

            Usage Example:

            response = user.get_thumbnail()
            f = open(filename, 'wb')
            f.write(response)

        """
        thumbnail_file = self.thumbnail
        if thumbnail_file:
            thumbnail_url_path = (
                "community/users/" + self._user_id + "/info/" + thumbnail_file
            )
            if thumbnail_url_path:
                return self._portal.con.get(
                    thumbnail_url_path, try_json=False, force_bytes=True
                )

    def download_thumbnail(self, save_folder: Optional[str] = None):
        """
        The ``download_thumbnail`` method downloads the item thumbnail for this user and saves it in the folder that
        is passed when ``download_thumbnail`` is called.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        save_folder            Optional string. The desired folder name to download the thumbnail to.
        ==================     ====================================================================


        :return:
           The file path of the downloaded thumbnail.
        """
        thumbnail_file = self.thumbnail

        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = (
                "community/users/" + self._user_id + "/info/" + thumbnail_file
            )
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50:  # If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                return self._portal.con.get(
                    path=thumbnail_url_path,
                    try_json=False,
                    out_folder=save_folder,
                    file_name=file_name,
                )
        else:
            return None

    @property
    def folders(self):
        """
        The ``folders`` property, when called, retrieves the list of the user's folders.

        :return:
            List of folders represented as dictionaries.
            Dictionary keys include: username, folder id (id), title, and date created (created)

         .. code-block:: python

            # Example to get name of all folders

            user = User(gis, username)
            folders = user.folders
            for folder in folders:
                print(folder["title"])

            # Example to get id of all folders

            user = User(gis, username)
            folders = user.folders
            for folder in folders:
                print(folder["id"])

        """
        return self._portal.user_folders(self._user_id)

    def items(self, folder: Optional[str] = None, max_items: int = 100):
        """
        The ``item`` method provides a list of :class:`~arcgis.gis.Item` objects in the specified folder.
        For content in the root folder, use the default value of None for the folder argument.
        For other folders, pass in the folder name as a string, or as a dictionary containing
        the folder ID, such as the dictionary obtained from the folders property.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. The specifc folder (as a string or dictionary)
                               to get a list of items in.
        ------------------     --------------------------------------------------------------------
        max_items              Optional integer. The maximum number of items to be returned. The default is 100.
        ==================     ====================================================================


        :return:
           The list of :class:`~arcgis.gis.Item` objects in the specified folder.

        .. code-block:: python

            # Example to **estimate** storage for a user's items

            storage = 0
            for item in user.items():
                storage += item.size
            try:
                for f in user.folders:
                    for f_item in user.folders(folder=f):
                        storage += f_item.size
                print(f"{user.username} using {storage} bytes")
            except Exception as e:
                print(f"{user.username} using {storage} bytes")

        .. code-block:: python

            # Example get items in each folder that is not root

            user = User(gis, username)
            folders = user.folders
            for folder in folders:
                items = user.items(folder=folder["title"])
                for item in items:
                    print(item, folder)

        """

        items = []
        folder_id = None
        if folder is not None:
            if isinstance(folder, str):
                folder_id = self._portal.get_folder_id(self._user_id, folder)
                if folder_id is None:
                    msg = "Could not locate the folder: %s" % folder
                    raise ValueError(
                        "%s. Please verify that this folder exists and try again." % msg
                    )
            elif isinstance(folder, dict):
                folder_id = folder["id"]
            else:
                print(
                    "folder should be folder name as a string"
                    "or a dict containing the folder 'id'"
                )

        resp = self._portal.user_items(self._user_id, folder_id, max_items)
        for item in resp:
            items.append(Item(self._gis, item["id"], item))

        return items

    # ----------------------------------------------------------------------
    @property
    def notifications(self):
        """
        The ``notifications`` property retrieves the list of notifications available for the given user.

        :return:
            A list containing available notifications
        """
        from .._impl.notification import Notification

        result = []
        url = "%s/community/users/%s/notifications" % (
            self._portal.resturl,
            self._user_id,
        )
        params = {"f": "json"}
        ns = self._portal.con.get(url, params)
        if "notifications" in ns:
            for n in ns["notifications"]:
                result.append(
                    Notification(
                        url="%s/%s" % (url, n["id"]),
                        user=self,
                        data=n,
                        initialize=False,
                    )
                )
                del n
            return result
        return result


class Item(dict):
    """
    The ``Item`` class represents an item  in the GIS, where an item is simply considered a unit of content in the GIS.
    Each item has a unique identifier and a well-known URL that is independent of the user owning the item.
    For a comprehensive list of properties of an item please see the REST API documentation `here <https://developers.arcgis.com/rest/users-groups-and-items/item.htm>`_ .
    Additionally, each item can have associated binary or textual data that's available via the item data resource.
    For example, an item of type `Map Package` returns the actual bits corresponding to the
    map package via the item data resource.

    Items that have layers (eg FeatureLayerCollection items and ImageryLayer items) and tables have
    the dynamic ``layers`` and ``tables`` properties to get to the individual layers/tables in this item.

    """

    _uid = None
    _snapeshots = None

    def __init__(self, gis, itemid, itemdict=None):
        dict.__init__(self)
        self._portal = gis._portal
        self._gis = gis
        self.itemid = itemid
        self.thumbnail = None
        self._workdir = tempfile.gettempdir()
        if itemdict:
            self._hydrated = False
            if "size" in itemdict and itemdict["size"] == -1:
                del itemdict["size"]  # remove nonsensical size
            self.__dict__.update(itemdict)
            super(Item, self).update(itemdict)
        else:
            self._hydrated = False
        try:
            self._depend = ItemDependency(item=self)
        except:
            pass

        if self._has_layers():
            self.layers = None
            self.tables = None
            self["layers"] = None
            self["tables"] = None

    def __hash__(self):
        return hash(tuple(frozenset(self)))

    # ----------------------------------------------------------------------
    @property
    def favorite(self) -> bool:
        """
        Gets/Sets if the Item is in the user's favorites
        """
        try:
            user: User = self._gis.users.get(self.owner)
            query = f'(group:"{user.favGroupId}" AND id:"{self.itemid}")'
            if len(self._gis.content.search(query)) > 0:
                return True
            return False
        except Exception as ex:
            raise Exception(f"Could not get the user's favorites. {str(ex)}")

    # ----------------------------------------------------------------------
    @favorite.setter
    def favorite(self, value: bool):
        """
        Gets/Sets if the Item is in the user's favorites
        """
        user: User = self._gis.users.get(self.owner)
        if value == True:
            url: str = f"{self._gis._portal.resturl}content/items/{self.itemid}/share"
        elif value == False:
            url: str = f"{self._gis._portal.resturl}content/items/{self.itemid}/unshare"
        else:
            raise ValueError("'value' must be a boolean.")

        params = {
            "f": "json",
            "everyone": self.shared_with["everyone"],
            "org": self.shared_with["org"],
            "items": self.itemid,
            "groups": user.favGroupId,
        }

        res = self._gis._con.post(url, params=params)
        assert self.shared_with
        if "error" in res:
            raise Exception(f"An error has occurred: {str(res)}")
        else:
            self._hydrated = False
            self._hydrate()

    # ----------------------------------------------------------------------
    @property
    def snapshots(self) -> list:
        """
        The ``snapshots`` property provides access to the Notebook Item's Snapshots. If the user is not
        the owner of the `Item`, the snapshots will be an empty list.

        :return: List[SnapShot]
        """
        if (
            self._is_notebook
            and self._gis.notebook_server
            and self.owner == self._gis.users.me.username
            and len(self._gis.notebook_server) > 0
        ):
            nbs = self._gis.notebook_server[0]
            if self._gis._portal.is_arcgisonline == False:
                return nbs.notebooks.snapshots.list(self)
            elif self._gis._portal.is_arcgisonline:
                sm = nbs.snaphots
                return sm.list(self)
        return []

    # ----------------------------------------------------------------------
    @_lazy_property
    def _is_notebook(self) -> bool:
        return self.type.lower() == "notebook"

    # ----------------------------------------------------------------------
    @_lazy_property
    def _get_nbs_server(self):
        urls = self._gis._registered_servers()
        if self._gis._portal.is_arcgisonline:
            return urls
        else:
            return urls

    # ----------------------------------------------------------------------
    @_lazy_property
    def resources(self):
        """
        The ``resources`` property returns the Item's Resource Manager

        :return: A :class:`~arcgis.gis.ResourceManager` object
        """
        return ResourceManager(self, self._gis)

    # ----------------------------------------------------------------------
    @property
    def _user_id(self):
        """gets/sets the _user_id property"""
        if self._uid is None:
            user = self._gis.users.get(self.owner)
            if hasattr(user, "id") and getattr(user, "id") != "null":
                # self._uid = user.id
                self._uid = user.username
            else:
                self._uid = user.username
        return self._uid

    # ----------------------------------------------------------------------
    @_user_id.setter
    def _user_id(self, value):
        """gets/sets the user id property"""
        self._uid = value

    # ----------------------------------------------------------------------
    def _has_layers(self):
        return (
            self.type == "Feature Collection"
            or self.type == "Feature Service"
            or self.type == "Big Data File Share"
            or self.type == "Image Service"
            or self.type == "Map Service"
            or self.type == "Globe Service"
            or self.type == "Scene Service"
            or self.type == "Network Analysis Service"
            or self.type == "Vector Tile Service"
        )

    def _populate_layers(self):
        from arcgis.features import (
            FeatureLayer,
            FeatureCollection,
            FeatureLayerCollection,
            Table,
        )
        from arcgis.mapping import VectorTileLayer, MapImageLayer, SceneLayer
        from arcgis.network import NetworkDataset
        from arcgis.raster import ImageryLayer

        if self._has_layers():
            layers = []
            tables = []

            params = {"f": "json"}

            if self.type == "Image Service":  # service that is itself a layer
                lyr = ImageryLayer(self.url, self._gis)
                try:
                    item_data = self.get_data()
                    rendering_rule = item_data.get("renderingRule", None)
                    if rendering_rule:
                        lyr._fn = rendering_rule
                        lyr._fnra = rendering_rule
                        lyr._rendering_rule_from_item = True
                    if lyr._mosaic_rule is None:
                        lyr._mosaic_rule = item_data.get("mosaicRule", None)
                except:
                    pass
                layers.append(lyr)

            elif self.type == "Feature Collection":
                lyrs = self.get_data()["layers"]
                for layer in lyrs:
                    layers.append(FeatureCollection(layer))

            elif self.type == "Big Data File Share":
                serviceinfo = self._portal.con.post(self.url, params)
                for lyr in serviceinfo["children"]:
                    lyrurl = self.url + "/" + lyr["name"]
                    layers.append(Layer(lyrurl, self._gis))

            elif self.type == "Vector Tile Service":
                layers.append(VectorTileLayer(self.url, self._gis))

            elif self.type == "Network Analysis Service":
                svc = NetworkDataset.fromitem(self)

                # route laters, service area layers, closest facility layers
                for lyr in svc.route_layers:
                    layers.append(lyr)
                for lyr in svc.service_area_layers:
                    layers.append(lyr)
                for lyr in svc.closest_facility_layers:
                    layers.append(lyr)

            elif self.type == "Feature Service":
                m = re.search(r"[0-9]+$", self.url)
                if (
                    m is not None
                ):  # ends in digit - it's a single layer from a Feature Service
                    layers.append(FeatureLayer(self.url, self._gis))
                else:
                    svc = FeatureLayerCollection.fromitem(self)
                    data = self.get_data()
                    for idx, lyr in enumerate(svc.layers):
                        if (
                            "layers" in data
                            and idx < len(data["layers"])
                            and "layerDefinition" in data["layers"][idx]
                            and "drawingInfo" in data["layers"][idx]["layerDefinition"]
                            and "renderer"
                            in data["layers"][idx]["layerDefinition"]["drawingInfo"]
                        ):
                            lyr.renderer = data["layers"][idx]["layerDefinition"][
                                "drawingInfo"
                            ]["renderer"]
                        layers.append(lyr)
                    for tbl in svc.tables:
                        tables.append(tbl)

            elif self.type == "Map Service":
                svc = MapImageLayer.fromitem(self)
                for lyr in svc.layers:
                    layers.append(lyr)
            else:
                m = re.search(r"[0-9]+$", self.url)
                if m is not None:  # ends in digit
                    layers.append(FeatureLayer(self.url, self._gis))
                else:
                    svc = _GISResource(self.url, self._gis)
                    for lyr in svc.properties.layers:
                        if self.type == "Scene Service":
                            lyr_url = svc.url + "/layers/" + str(lyr.id)
                            lyr = SceneLayer(lyr_url, self._gis)
                        else:
                            lyr_url = svc.url + "/" + str(lyr.id)
                            lyr = Layer(lyr_url, self._gis)
                        layers.append(lyr)
                    try:
                        for lyr in svc.properties.tables:
                            lyr = Table(svc.url + "/" + str(lyr.id), self._gis)
                            tables.append(lyr)
                    except:
                        pass

            self.layers = layers
            self.tables = tables
            self["layers"] = layers
            self["tables"] = tables

    def _hydrate(self):
        itemdict = self._portal.get_item(self.itemid)
        self._hydrated = True
        super(Item, self).update(itemdict)
        self.__dict__.update(itemdict)
        try:
            with _common_utils._DisableLogger():
                self._populate_layers()
        except:
            pass
        user = self._gis.users.get(self.owner)
        if hasattr(user, "id") and user.id != "null":
            self._user_id = user.username
            # self._user_id = user.id
        else:
            self._user_id = user.username

    def __getattribute__(self, name):
        if name == "layers":
            if self["layers"] == None or self["layers"] == []:
                try:
                    with _common_utils._DisableLogger():
                        self._populate_layers()
                except Exception as e:
                    if (
                        str(e).lower().find("token required") > -1
                        and self._gis._con._auth.lower() == "pki"
                    ):
                        with _common_utils._DisableLogger():
                            self._populate_layers()
                    else:
                        print(e)
                    pass
                return self["layers"]
        elif name == "tables":
            if self["tables"] == None or self["tables"] == []:
                try:
                    with _common_utils._DisableLogger():
                        self._populate_layers()
                except:
                    pass
                return self["tables"]
        elif name == "url" and self._gis._validate_item_url:
            url = dict.__getitem__(self, "url")
            return self._validate_url(url)
        return super(Item, self).__getattribute__(name)

    def __getattr__(self, name):  # support item attributes
        if not self._hydrated and not name.startswith("_"):
            self._hydrate()
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def __getitem__(
        self, k
    ):  # support item attributes as dictionary keys on this object
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            if not self._hydrated and not k.startswith("_"):
                self._hydrate()
            return dict.__getitem__(self, k)

    # ----------------------------------------------------------------------
    @property
    def can_delete(self) -> bool:
        """
        Checks if the Item can be removed from the system.

        :return: bool
        """
        url = f"{self._portal.resturl}content/users/{self.owner}/items/{self.itemid}/canDelete"
        params = {"f": "json"}
        try:
            return self._gis._con.get(url, params).get("success", False)
        except Exception as e:
            _log.warning(e)
            return False

    # ----------------------------------------------------------------------
    @property
    def content_status(self):
        """
        The content_status property states if an Item is authoritative or deprecated.  This
        givens owners and administrators of Item the ability to warn users that they
        should be either this information or not.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Optional string or None.  Defines if an Item is deprecated or
                               authoritative.
                               If a value of None is given, then the value will be reset.

                               Allowed Values: authoritative, deprecated, or None
        ==================     ====================================================================
        """
        try:
            return self.contentStatus
        except:
            return ""

    # ----------------------------------------------------------------------
    @content_status.setter
    def content_status(self, value: Optional[str]):
        """
        See main ``content_status`` property docstring
        """
        status_values = [
            "authoritative",
            "org_authoritative",
            "public_authoritative",
            "deprecated",
        ]

        if value is None:
            pass
        elif str(value).lower() not in status_values:
            raise ValueError(
                "%s is not valid value of: authoritative or deprecated" % value
            )

        if str(value).lower() == "authoritative":
            value = "org_authoritative"

        params = {"f": "json", "status": value}
        url = "content/items/" + self.itemid + "/setContentStatus"

        if value is None:
            value = ""
            params["status"] = ""
            params["clearEmptyFields"] = True
        else:
            params["clearEmptyFields"] = False
        res = self._portal.con.get(url, params)
        if "success" in res:
            self.contentStatus = value
            self._hydrate()

    # ----------------------------------------------------------------------
    @property
    def homepage(self):
        """The ``homepage`` property gets the URL to the HTML page for the item."""
        return "{}{}{}".format(self._gis.url, "/home/item.html?id=", self.itemid)

    # ----------------------------------------------------------------------
    def copy_feature_layer_collection(
        self,
        service_name: str,
        layers: Optional[Union[list[int], str]] = None,
        tables: Optional[Union[list[int], str]] = None,
        folder: Optional[str] = None,
        description: Optional[str] = None,
        snippet: Optional[str] = None,
        owner: Optional[Union[str, User]] = None,
    ):
        """
        The ``copy_feature_layer_collection`` method allows users to copy existing Feature Layer Collections and select
        the layers/tables that the user wants in the service. It is quite similar to the ``copy`` method, but only
        copies the selected Feature Layer Collections.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        service_name           Required string. It is the name of the service.
        ------------------     --------------------------------------------------------------------
        layers                 Optional list/string.  This is a either a list of integers or a comma
                               seperated list of integers as a string.  Each index value represents
                               a layer in the feature layer collection.
        ------------------     --------------------------------------------------------------------
        tables                 Optional list/string. This is a either a list of integers or a comma
                               seperated list of integers as a string.  Each index value represents
                               a table in the feature layer collection.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. This is the name of the folder to place in.  The
                               default is None, which means the root folder.
        ------------------     --------------------------------------------------------------------
        description            Optional string. This is the Item description of the service.
        ------------------     --------------------------------------------------------------------
        snippet                Optional string. This is the Item's snippet of the service. It is
                               no longer than 250 characters.
        ------------------     --------------------------------------------------------------------
        owner                  Optional string/User. The default is the current user, but if you
                               want the service to be owned by another user, pass in this value.
        ==================     ====================================================================

        :return:
           If successful, returns an :class:`~arcgis.gis.Item` object. Otherwise, returns None on failure.

        .. code-block:: python

            # Usage Example

            >>> item.copy_feature_layer_collection(service_name="service_name", layers="1,4,5,8")

        """
        from ..features import FeatureLayerCollection

        if self.type != "Feature Service" and self.type != "Feature Layer Collection":
            return
        if layers is None and tables is None:
            raise ValueError("An index of layers or tables must be provided")
        content = self._gis.content
        if isinstance(owner, User):
            owner = owner.username
        idx_layers = []
        idx_tables = []
        params = {}
        allowed = [
            "description",
            "allowGeometryUpdates",
            "units",
            "syncEnabled",
            "serviceDescription",
            "capabilities",
            "serviceItemId",
            "supportsDisconnectedEditing",
            "maxRecordCount",
            "supportsApplyEditsWithGlobalIds",
            "name",
            "supportedQueryFormats",
            "xssPreventionInfo",
            "copyrightText",
            "currentVersion",
            "syncCapabilities",
            "_ssl",
            "hasStaticData",
            "hasVersionedData",
            "editorTrackingInfo",
            "name",
        ]
        parent = None
        if description is None:
            description = self.description
        if snippet is None:
            snippet = self.snippet
        i = 1
        is_free = content.is_service_name_available(
            service_name=service_name, service_type="Feature Service"
        )
        if is_free == False:
            while is_free == False:
                i += 1
                s = service_name + "_%s" % i
                is_free = content.is_service_name_available(
                    service_name=s, service_type="Feature Service"
                )
                if is_free:
                    service_name = s
                    break
        if len(self.tables) > 0 or len(self.layers) > 0:
            parent = FeatureLayerCollection(url=self.url, gis=self._gis)
        else:
            raise Exception("No tables or layers found in service, cannot copy it.")
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                for idx in layers:
                    idx_layers.append(self.layers[idx])
                    del idx
            elif isinstance(layers, (str)):
                for idx in layers.split(","):
                    idx_layers.append(self.layers[idx])
                    del idx
            else:
                raise ValueError(
                    "layers must be a comma seperated list of integers or a list"
                )
        if tables is not None:
            if isinstance(tables, (list, tuple)):
                for idx in tables:
                    idx_tables.append(self.tables[idx])
                    del idx
            elif isinstance(tables, (str)):
                for idx in tables.split(","):
                    idx_tables.append(self.tables[idx])
                    del idx
            else:
                raise ValueError(
                    "tables must be a comma seperated list of integers or a list"
                )
        for k, v in dict(parent.properties).items():
            if k in allowed:
                if k.lower() == "name":
                    params[k] = service_name
                if k.lower() == "_ssl":
                    params["_ssl"] = False
                params[k] = v
            del k, v
        if "name" not in params.keys():
            params["name"] = service_name
        params["_ssl"] = False
        copied_item = content.create_service(
            name=service_name,
            create_params=params,
            folder=folder,
            owner=owner,
            item_properties={
                "description": description,
                "snippet": snippet,
                "tags": self.tags,
                "title": service_name,
            },
        )

        fs = FeatureLayerCollection(url=copied_item.url, gis=self._gis)
        fs_manager = fs.manager
        add_defs = {"layers": [], "tables": []}
        for l in idx_layers:
            v = dict(l.manager.properties)
            if "indexes" in v:
                del v["indexes"]
            if "adminLayerInfo" in v:
                del v["adminLayerInfo"]
            add_defs["layers"].append(v)
            del l
        for l in idx_tables:
            v = dict(l.manager.properties)
            if "indexes" in v:
                del v["indexes"]
            if "adminLayerInfo" in v:
                del v["adminLayerInfo"]
            add_defs["tables"].append(v)
            del l
        res = fs_manager.add_to_definition(json_dict=add_defs)
        if res["success"] == True:
            return copied_item
        else:
            try:
                copied_item.delete()
            except:
                pass
        return None

    # ----------------------------------------------------------------------
    def download(
        self,
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        """
        The ``download`` method downloads the data to the specified folder or a temporary folder, if a folder is not provided.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        save_path           Optional string. Folder location to download the file to.
        ---------------     --------------------------------------------------------------------
        file_name           Optional string. The name of the file.
        ===============     ====================================================================

        :return:
           The download path if data was available, otherwise None.

        .. code-block:: python

            # Usage Example

            >>> item.download("C:\ARCGIS\Projects\", "hurricane_data")

        """
        data_path = "content/items/" + self.itemid + f"/data"
        if file_name is None:
            if "name" in self or "title" in self:
                file_name = self.name or self.title
        if not save_path:
            save_path = self._workdir
        try:
            url = self._gis._portal.resturl + data_path
            con = self._gis._con
            resp = con.get(
                path=url,
                file_name=file_name,
                out_folder=save_path,
                try_json=False,
                force_bytes=False,
                allow_redirects=False,
                return_raw_response=True,
            )
            if resp.status_code >= 300 and resp.status_code < 400:
                url = resp.headers["location"]
                resp = con.get(
                    path=url,
                    file_name=file_name,
                    out_folder=save_path,
                    try_json=False,
                    force_bytes=False,
                    allow_redirects=False,
                    return_raw_response=True,
                    drop_auth=True,
                )
                download_path = con._handle_response(
                    resp,
                    file_name=file_name,
                    out_path=save_path,
                    try_json=False,
                )
            else:
                download_path = con._handle_response(
                    resp,
                    file_name=file_name,
                    out_path=save_path,
                    try_json=False,
                )
        except Exception as e:
            _log.debug(msg=str(e))
            _log.debug(
                msg="Retrying download parsing name from title or name property."
            )
            if file_name is None:
                import re

                file_name = self.name or self.title
                file_name = re.sub(r"[^a-zA-Z0-9 \n\.]", "", file_name) or self.itemid
            if save_path is None:
                save_path = tempfile.gettempdir()
            download_path = self._portal.con.get(
                path=data_path,
                file_name=file_name,
                out_folder=save_path,
                try_json=False,
                force_bytes=False,
            )
        if download_path == "":
            return None
        else:
            return download_path

    # ----------------------------------------------------------------------
    def export(
        self,
        title: str,
        export_format: str,
        parameters: Optional[str] = None,
        wait: bool = True,
        enforce_fld_vis: Optional[bool] = None,
        tags: Optional[Union[list[str], str]] = None,
        snippet: Optional[str] = None,
    ):
        """
        The ``export`` method is used to export a service item to the specified export format.
        However, it is available only to users with an organizational subscription and can only be invoked by the
        service item owner or an administrator, unless a Location Tracking Service or Location Tracking View is used.
        The ``export`` method is useful for long running exports that could hold up a script.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The desired name of the exported service item.
        ---------------     --------------------------------------------------------------------
        export_format       Required string. The format to export the data to. Allowed types: `Shapefile`,
                            `CSV`, `File Geodatabase`, `Feature Collection`, `GeoJson`, `Scene Package`, `KML`,
                             `Excel`, `geoPackage`, or `Vector Tile Package`.
        ---------------     --------------------------------------------------------------------
        parameters          Optional string. A JSON object describing the layers to be exported
                            and the export parameters for each layer.and the export parameters for each layer. See
                            `Export Item <https://developers.arcgis.com/rest/users-groups-and-items/export-item.htm>`_
                            in the REST API for guidance.
        ---------------     --------------------------------------------------------------------
        wait                Optional boolean. Default is True, which forces a wait for the
                            export to complete; use False for when it is okay to proceed while
                            the export continues to completion.
        ---------------     --------------------------------------------------------------------
        enforce_fld_vis     Optional boolean. Be default when you are the owner of an item and
                            the `export` operation is called, the data provides all the columns.
                            If the export is being perform on a view, to ensure the view's
                            column definition is honor, then set the value to True. When the
                            owner of the service and the value is set to False, all data and
                            columns will be exported.
        ---------------     --------------------------------------------------------------------
        tags                Optional String.  A comma seperated value of item descriptors.
        ---------------     --------------------------------------------------------------------
        snippet             Optional String. A short descriptive piece of text.
        ===============     ====================================================================

        :return:
           An :class:`~arcgis.gis.Item` object or a dictionary.  Item is returned when `wait=True`.
           A dictionary describing the status of the item is returned when `wait=False`. See the
           :attr:`~arcgis.gis.Item.status` method for more information.

        .. code-block:: python

            # Usage Example

            >>> item.export("hurricane_data", "CSV", wait =True, tags= "Hurricane, Natural Disasters")
        """
        import time

        formats = [
            "Shapefile",
            "CSV",
            "File Geodatabase",
            "Feature Collection",
            "GeoJson",
            "GeoPackage",  # geoPackage
            "geoPackage",
            "Scene Package",
            "KML",
            "Excel",
            "Vector Tile Package",
        ]
        if export_format not in formats:
            raise Error("Unsupported export format: " + export_format)
        if export_format == "GeoPackage":
            export_format = "geoPackage"
        user_id = self._user_id
        # allow exporting of LTS / LTV even if not owner for ArcGIS Online
        if (
            not self._gis.properties["isPortal"]
            and "Location Tracking Service" in self.typeKeywords
        ):
            user_id = self._gis.users.me.username
        data_path = "content/users/%s/export" % user_id
        params = {
            "f": "json",
            "itemId": self.itemid,
            "exportFormat": export_format,
            "title": title,
        }
        if tags and isinstance(tags, (list, tuple)):
            tags = ",".join([str(t) for t in tags])
        if tags and isinstance(tags, str):
            params["tags"] = tags
        if snippet:
            params["snippet"] = snippet
        if parameters:
            params.update({"exportParameters": parameters})
        if not enforce_fld_vis is None and "View Service" in self.typeKeywords:
            if "exportParameters" in params:
                params["exportParameters"]["enforceFieldVisibility"] = enforce_fld_vis
            else:
                params["exportParameters"] = {"enforceFieldVisibility": enforce_fld_vis}
        try:
            res = self._portal.con.post(data_path, params)
        except Exception as e:
            if e.args[0].find("You do not have permissions") > -1:
                data_path = "content/users/%s/export" % self._gis.users.me.username
                res = self._portal.con.post(data_path, params)
            else:
                raise
        if "success" in res and res["success"] == False:
            raise Exception("Could not export item.")
        elif not "exportItemId" in res:
            raise Exception("Could not export item.")
        export_item = Item(gis=self._gis, itemid=res["exportItemId"])
        if wait == True:
            status = "partial"
            while status != "completed":
                status = export_item.status(job_id=res["jobId"], job_type="export")
                if status["status"] == "failed":
                    raise Exception("Could not export item: %s" % self.itemid)
                elif status["status"].lower() == "completed":
                    return export_item
                time.sleep(2)
        return res

    # ----------------------------------------------------------------------
    def status(self, job_id: Optional[str] = None, job_type: Optional[str] = None):
        """
        The ``status`` method provides the status of an :class:`~arcgis.gis.Item` in the following situations:
            1. Publishing an :class:`~arcgis.gis.Item`
            2. Adding an :class:`~arcgis.gis.Item` in async mode
            3. Adding with a multipart upload. `Partial` is available for ``Add Item Multipart`` when only a part is
            uploaded and the :class:`~arcgis.gis.Item` object is not committed.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Optional string. The job ID returned during publish, generateFeatures,
                            export, and createService calls.
        ---------------     --------------------------------------------------------------------
        job_type            Optional string. The type of asynchronous job for which the status
                            has to be checked. Default is none, which checks the item's status.
                            This parameter is optional unless used with the operations listed
                            below. Values: `publish`, `generateFeatures`, `export`, and `createService`
        ===============     ====================================================================

        :return:
           The status of a publishing :class:`~arcgis.gis.Item` object.

        .. code-block:: python

            # Usage Example

            >>> item.status(job_type="generateFeatures")
        """
        params = {"f": "json"}
        data_path = "content/users/%s/items/%s/status" % (
            self._user_id,
            self.itemid,
        )
        if job_type is not None:
            params["jobType"] = job_type
        if job_id is not None:
            params["jobId"] = job_id
        return self._portal.con.get(data_path, params)

    # ----------------------------------------------------------------------
    def get_thumbnail(self):
        """
        The ``get_thumbnail`` method retrieves the bytes that make up the thumbnail for this item.

        :return:
           Bytes that represent the item's thumbnail.

        Example

        .. code-block:: python

            >>> response = item.get_thumbnail()
            >>> f = open(filename, 'wb')
            >>> f.write(response)

        """
        thumbnail_file = self.thumbnail
        if thumbnail_file:
            thumbnail_url_path = (
                "content/items/" + self.itemid + "/info/" + thumbnail_file
            )
            if thumbnail_url_path:
                return self._portal.con.get(
                    thumbnail_url_path, try_json=False, force_bytes=True
                )

    # ----------------------------------------------------------------------
    def download_thumbnail(self, save_folder: Optional[str] = None):
        """
        The ``download_thumbnail`` method is similar to the ``download`` method but only downloads the item thumbnail.


         ===============     ====================================================================
         **Parameter**        **Description**
         ---------------     --------------------------------------------------------------------
         save_folder          Optional string. Folder location to download the item's thumbnail to.
         ===============     ====================================================================


         :return:
           A file path, If the download was successful. None if the item does not have a thumbnail.
        """
        if self.thumbnail is None:
            self._hydrate()
        thumbnail_file = self.thumbnail

        # Only proceed if a thumbnail exists
        if thumbnail_file:
            thumbnail_url_path = (
                "content/items/" + self.itemid + "/info/" + thumbnail_file
            )
            if thumbnail_url_path:
                if not save_folder:
                    save_folder = self._workdir
                file_name = os.path.split(thumbnail_file)[1]
                if len(file_name) > 50:  # If > 50 chars, truncate to last 30 chars
                    file_name = file_name[-30:]

                file_path = os.path.join(save_folder, file_name)
                self._portal.con.get(
                    path=thumbnail_url_path,
                    try_json=False,
                    out_folder=save_folder,
                    file_name=file_name,
                )
                return file_path
        else:
            return None

    # ----------------------------------------------------------------------
    def get_thumbnail_link(self):
        """
        The ``get_thumbnail_link`` method is similar to the ``get_thumbnail`` method, but retrieves the link to the
        item's thumbnail rather than the bytes that make up the thumbnail for this item.

        :return:
           The link to the item's thumbnail."""

        thumbnail_file = self.thumbnail
        if thumbnail_file is None:
            if self._gis.properties.portalName == "ArcGIS Online":
                return "http://static.arcgis.com/images/desktopapp.png"
            else:
                return self._gis.url + "/portalimages/desktopapp.png"
        else:
            thumbnail_url_path = (
                self._gis._public_rest_url
                + "/content/items/"
                + self.itemid
                + "/info/"
                + thumbnail_file
            )
            return thumbnail_url_path

    # ----------------------------------------------------------------------
    def create_thumbnail(self, update: bool = True):
        """
        The ``create_thumbnail`` method creates a Thumbnail for a feature service portal item using the service's
        symbology and the print service registered for the enterprise.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        update              Optional boolean. When set to True, the item will be updated with the
                            thumbnail generated in this call, else it will not update the item.
                            The default is True.
        ===============     ====================================================================

        :return: A :class:`~arcgis.geoprocessing._types.DataFile` object

        """
        from arcgis.geoprocessing._tool import Toolbox
        from arcgis.features import FeatureLayer, FeatureLayerCollection
        from arcgis.gis.server._service import Service

        props = self._gis.properties
        gp_url = os.path.dirname(self._gis.properties.helperServices.printTask.url)

        if self.type == "Feature Service":
            layers = []
            container = self.layers[0].container
            extent = container.properties.initialExtent
            for lyr in self.layers:
                layers.append(
                    {
                        "id": "%s_%s"
                        % (lyr.properties.serviceItemId, lyr.properties.id),
                        "title": lyr.properties.name,
                        "opacity": 1,
                        "minScale": lyr.properties.minScale,
                        "maxScale": lyr.properties.maxScale,
                        "layerDefinition": {
                            "drawingInfo": dict(lyr.properties.drawingInfo)
                        },
                        "token": self._gis._con.token,
                        "url": lyr._url,
                    }
                )
                del lyr
            wmjs = {
                "mapOptions": {
                    "showAttribution": False,
                    "extent": dict(extent),
                    "spatialReference": dict(container.properties.spatialReference),
                },
                "operationalLayers": layers,
                "exportOptions": {"outputSize": [600, 400], "dpi": 96},
            }

        elif self.type == "Web Map":
            import json

            layers = []
            mapjson = self.get_data()
            container = None
            for lyr in mapjson["baseMap"]["baseMapLayers"]:
                del lyr["layerType"]
                layers.append(lyr)
            for lyr in mapjson["operationalLayers"]:
                flyr = Service(url=lyr["url"], server=self._gis._con)
                if container is None and isinstance(flyr, FeatureLayer):
                    container = FeatureLayerCollection(
                        url=os.path.dirname(flyr._url), gis=self._gis
                    )
                layers.append(
                    {
                        "id": "%s" % lyr["id"],
                        "title": lyr["title"],
                        "opacity": lyr["opacity"] if "opacity" in lyr else None,
                        "minScale": flyr.properties.minScale,
                        "maxScale": flyr.properties.maxScale,
                        "layerDefinition": {
                            "drawingInfo": dict(flyr.properties.drawingInfo)
                        },
                        "token": self._gis._con.token,
                        "url": lyr["url"],
                    }
                )
                del lyr

            wmjs = {
                "mapOptions": {
                    "showAttribution": False,
                    "extent": dict(container.properties.initialExtent)
                    if container
                    else dict(self._gis.properties.defaultExtent),
                    "spatialReference": dict(container.properties.spatialReference)
                    if container
                    else dict(self._gis.properties.defaultExtent.spatialReference),
                },
                "operationalLayers": layers,
                "exportOptions": {"outputSize": [600, 400], "dpi": 96},
            }
        else:
            return None
        if (
            isinstance(self._gis._portal, _portalpy.Portal)
            and self._gis._portal.is_arcgisonline
        ):
            tbx = Toolbox(url=gp_url)
        else:
            tbx = Toolbox(url=gp_url, gis=self._gis)
        res = tbx.export_web_map_task(web_map_as_json=wmjs, format="png32")
        if update:
            self.update(item_properties={"thumbnailUrl": res.url})
        return res

    # ----------------------------------------------------------------------
    def delete_thumbnail(self) -> bool:
        """
        Deletes the item's thumbnail

        :returns: bool
        """
        url = f"{self._gis._portal.resturl}content/users/{self.owner}/items/{self.itemid}/deleteThumbnail"
        params = {"f": "json"}
        res = self._gis._con.post(url, params)
        if res.get("success", False):
            self._hydrated = False
            self._hydrate()
            return True
        return res

    # ----------------------------------------------------------------------
    def update_thumbnail(
        self,
        file_path: str | None = None,
        encoded_image: str | None = None,
        file_name: str | None = None,
        url: str | None = None,
    ) -> bool:
        """
        The `update_thumbnail` updates the thumbnail of any ArcGIS item in your organization. The updated thumbnail
        can be provided in a variety of formats, as either a file to be uploaded as part of a multipart request,
        a direct URL to the thumbnail file, or as a Base64 encoded image.

        ================  =========================================================================================
        **Parameter**      **Description**
        ----------------  -----------------------------------------------------------------------------------------
        file_path         Optional String. The local path to the thumbnail.
        ----------------  -----------------------------------------------------------------------------------------
        encoded_image     Optional String. A base64 encoded image as a string.
        ----------------  -----------------------------------------------------------------------------------------
        file_name         Optional String. This is required with `encoded_image` is used. It is the name of the file
                          with extension.  Example thumbnail.png
        ----------------  -----------------------------------------------------------------------------------------
        url               Optional String. A URL location of a thumbnail.
        ================  =========================================================================================


        .. code-block:: python

            # Usage Example 1: Using a base64 encoded image

            gis = GIS(profile='your_profile')
            item = gis.content.get(<item id>)
            base64_img = (
                'data:image/png;base64,'
                'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAA'
                'LEwEAmpwYAAAB1klEQVQ4jY2TTUhUURTHf+fy/HrjhNEX2KRGiyIXg8xgSURuokX'
                'LxFW0qDTaSQupkHirthK0qF0WQQQR0UCbwCQyw8KCiDbShEYLJQdmpsk3895p4aS'
                'v92ass7pcfv/zP+fcc4U6kXKe2pTY3tjSUHjtnFgB0VqchC/SY8/293S23f+6VEj'
                '9KKwCoPDNIJdmr598GOZNJKNWTic7tqb27WwNuuwGvVWrAit84fsmMzE1P1+1TiK'
                'MVKvYUjdBvzPZXCwXzyhyWNBgVYkgrIow09VJMznpyebWE+Tdn9cEroBSc1JVPS+'
                '6moh5Xyjj65vEgBxafGzWetTh+rr1eE/c/TMYg8hlAOvI6JP4KmwLgJ4qD0TIbli'
                'TB+sunjkbeLekKsZ6Zc8V027aBRoBRHVoduDiSypmGFG7CrcBEyDHA0ZNfNphC0D'
                '6amYa6ANw3YbWD4Pn3oIc+EdL36V3od0A+MaMAXmA8x2Zyn+IQeQeBDfRcUw3B+2'
                'PxwZ/EdtTDpCPQLMh9TKx0k3pXipEVlknsf5KoNzGyOe1sz8nvYtTQT6yyvTjIax'
                'smHGB9pFx4n3jIEfDePQvCIrnn0J4B/gA5J4XcRfu4JZuRAw3C51OtOjM3l2bMb8'
                'Br5eXCsT/w/EAAAAASUVORK5CYII='
            )
            res = item.update_thumbnail(encoded_image=base64_img, file_name="thumbnail.png")

        .. code-block:: python

            # Usage Example 2: URL image

            gis = GIS(profile='your_profile')
            item = gis.content.get(<item id>)
            img_url = "https://www.esri.com/content/dam/esrisites/en-us/common/icons/product-logos/ArcGIS-Pro.png"
            res = item.update_thumbnail(url=img_url)

        .. code-block:: python

            # Usage Example 3: Using a local file

            gis = GIS(profile='your_profile')
            item = gis.content.get(<item id>)
            fp = "c:/images/ArcGIS-Pro.png"
            res = item.update_thumbnail(file_path=fp)

        :returns: bool
        """
        if file_path is None and encoded_image is None and url is None:
            return False
        files = None
        rest_url = f"{self._gis._portal.resturl}content/users/{self.owner}/items/{self.itemid}/updateThumbnail"
        params = {
            "f": "json",
        }
        if file_path and os.path.isfile(file_path):
            files = []
            files.append(("file", file_path, os.path.basename(file_path)))
        elif encoded_image:
            params["data"] = encoded_image
        elif url:
            params["url"] = url
        if encoded_image and file_name is None:
            params["filename"] = "thumbnail.png"
        elif file_name:
            params["filename"] = file_name
        resp = self._gis._con.post(rest_url, params, files=files)
        if resp.get("success", False):
            self._hydrated = False
            self._hydrate()
            return True
        return False

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """The ``metadata`` property gets and sets the item metadata for the specified item.
        ``metadata`` returns None if the item does not have metadata.

        .. note::
            Items with metadata have 'Metadata' in their typeKeywords.

        """
        metadataurlpath = "content/items/" + self.itemid + "/info/metadata/metadata.xml"
        try:
            return self._portal.con.get(metadataurlpath, try_json=False)

        # If the get operation returns a 400 HTTP Error then the metadata simply
        # doesn't exist, let's just return None in this case
        except HTTPError as e:
            if e.code == 400 or e.code == 500:
                return None
            else:
                raise e

    # ----------------------------------------------------------------------
    @metadata.setter
    def metadata(self, value):
        """
        See main ``metadata`` property docstring
        """
        with tempfile.TemporaryDirectory(suffix="metadata") as tempdir:
            xml_file = os.path.join(tempdir, "metadata.xml")
            if os.path.isfile(xml_file) == True:
                os.remove(xml_file)
            if (
                str(value).lower().endswith(".xml")
                and len(value) <= 32767
                and os.path.isfile(value) == True
            ):
                if os.path.basename(value).lower() != "metadata.xml":
                    shutil.copy(value, xml_file)
                else:
                    xml_file = value
            elif isinstance(value, str):
                with open(xml_file, mode="w") as writer:
                    writer.write(value)
                    writer.close()
            else:
                raise ValueError("Input must be XML path file or XML Text")
            self.update(metadata=xml_file)

    # ----------------------------------------------------------------------
    def download_metadata(self, save_folder: Optional[str] = None):
        """
        The ``download_metadata`` method is similar to the ``download`` method but only downloads the item metadata for
        the specified item id. Items with metadata have 'Metadata' in their typeKeywords.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        save_folder          Optional string. Folder location to download the item's metadata to.
        ===============     ====================================================================


        :return:
           A file path, if the metadata download was successful. None if the item does not have metadata.
        """
        metadataurlpath = "content/items/" + self.itemid + "/info/metadata/metadata.xml"
        if not save_folder:
            with tempfile.TemporaryDirectory(
                suffix=f"meta{uuid.uuid4().hex[:2]}",
                dir=tempfile.gettempdir(),
            ) as save_folder:
                file_name = "metadata.xml"
                file_path = os.path.join(save_folder, file_name)
                self._portal.con.get(
                    path=metadataurlpath,
                    out_folder=save_folder,
                    file_name=file_name,
                    try_json=False,
                )
                return file_path
        try:
            file_name = "metadata.xml"
            file_path = os.path.join(save_folder, file_name)
            self._portal.con.get(
                path=metadataurlpath,
                out_folder=save_folder,
                file_name=file_name,
                try_json=False,
            )
            return file_path

        # If the get operation returns a 400 HTTP/IO Error then the metadata
        # simply doesn't exist, let's just return None in this case
        except HTTPError as e:
            if e.code == 400 or e.code == 500:
                return None
            else:
                raise e

    def _get_icon(self):
        icon = "layers16.png"
        if self.type.lower() == "web map":
            icon = "maps16.png"
        elif self.type.lower() == "web scene":
            icon = "websceneglobal16.png"
        elif self.type.lower() == "cityengine web scene":
            icon = "websceneglobal16.png"
        elif self.type.lower() == "pro map":
            icon = "mapsgray16.png"
        elif self.type.lower() == "feature service" and "Table" in self.typeKeywords:
            icon = "table16.png"
        elif self.type.lower() == "feature service":
            icon = "featureshosted16.png"
        elif self.type.lower() == "map service":
            icon = "mapimages16.png"
        elif self.type.lower() == "image service":
            icon = "imagery16.png"
        elif self.type.lower() == "kml":
            icon = "features16.png"
        elif self.type.lower() == "wms":
            icon = "mapimages16.png"
        elif self.type.lower() == "feature collection":
            icon = "features16.png"
        elif self.type.lower() == "feature collection template":
            icon = "maps16.png"
        elif self.type.lower() == "geodata service":
            icon = "layers16.png"
        elif self.type.lower() == "globe service":
            icon = "layers16.png"
        elif self.type.lower() == "shapefile":
            icon = "datafiles16.png"
        elif self.type.lower() == "web map application":
            icon = "apps16.png"
        elif self.type.lower() == "map package":
            icon = "mapsgray16.png"
        elif self.type.lower() == "feature layer":
            icon = "featureshosted16.png"
        elif self.type.lower() == "map service":
            icon = "maptiles16.png"
        elif self.type.lower() == "map document":
            icon = "mapsgray16.png"
        elif self.type.lower() == "csv":
            return (
                f"{self._gis.url}/home/js/arcgisonline/img/item-types/datafiles16.svg"
            )
        elif self.type.lower() == "notebook":
            return f"{self._gis.url}/home/js/arcgisonline/img/item-types/notebook16.svg"
        elif self.type.lower() == "shapefile":
            return (
                f"{self._gis.url}/home/js/arcgisonline/img/item-types/datafiles16.svg"
            )
        elif self.type.lower() == "notebook code snippet library":
            return (
                f"{self._gis.url}/home/js/arcgisonline/img/item-types/codeSnippet16.svg"
            )
        elif self.type.lower() == "web mapping application":
            return f"{self._gis.url}/home/js/arcgisonline/img/item-types/apps16.svg"
        elif self.type.lower() == "geoprocessing service":
            return f"{self._gis.url}/home/js/arcgisonline/img/item-types/layers16.svg"
        else:
            icon = "layers16.png"

        icon = self._gis.url + "/home/js/jsapi/esri/css/images/item_type_icons/" + icon
        return icon

    # ----------------------------------------------------------------------
    def _ux_item_type(self):
        item_type = self.type
        if self.type == "Geoprocessing Service":
            item_type = "Geoprocessing Toolbox"
        elif self.type.lower() == "feature service" and "Table" in self.typeKeywords:
            item_type = "Table Layer"
        elif self.type.lower() == "feature service":
            item_type = "Feature Layer Collection"
        elif self.type.lower() == "map service":
            item_type = "Map Image Layer"
        elif self.type.lower() == "image service":
            item_type = "Imagery Layer"
        elif self.type.lower().endswith("service"):
            item_type = self.type.replace("Service", "Layer")
        return item_type

    # ----------------------------------------------------------------------
    def _repr_html_(self):
        thumbnail = self.thumbnail
        if self.thumbnail is None or not self._portal.is_logged_in:
            thumbnail = self.get_thumbnail_link()
        else:
            try:
                b64 = base64.b64encode(self.get_thumbnail())
                thumbnail = (
                    "data:image/png;base64,"
                    + str(b64, "utf-8")
                    + "' width='200' height='133"
                )
            except:
                if self._gis.properties.portalName == "ArcGIS Online":
                    thumbnail = "http://static.arcgis.com/images/desktopapp.png"
                else:
                    thumbnail = self._gis.url + "/portalimages/desktopapp.png"

        snippet = self.snippet
        if snippet is None:
            snippet = ""

        portalurl = self.homepage

        # locale.setlocale(locale.LC_ALL, "")
        numViews = locale.format("%d", self.numViews, grouping=True)
        return (
            """<div class="item_container" style="height: auto; overflow: hidden; border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; line-height: 1.21429em; padding: 10px;">
                    <div class="item_left" style="width: 210px; float: left;">
                       <a href='"""
            + portalurl
            + """' target='_blank'>
                        <img src='"""
            + thumbnail
            + """' class="itemThumbnail">
                       </a>
                    </div>

                    <div class="item_right"     style="float: none; width: auto; overflow: hidden;">
                        <a href='"""
            + portalurl
            + """' target='_blank'><b>"""
            + self.title
            + """</b>
                        </a>
                        <br/>"""
            + snippet
            + """<img src='"""
            + self._get_icon()
            + """' style="vertical-align:middle;" width=16 height=16>"""
            + self._ux_item_type()
            + """ by """
            + self.owner
            + """
                        <br/>Last Modified: """
            + datetime.fromtimestamp(self.modified / 1000).strftime("%B %d, %Y")
            + """
                        <br/>"""
            + str(self.numComments)
            + """ comments, """
            + str(numViews)
            + """ views
                    </div>
                </div>
                """
        )

    # ----------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()
        # state = ["   %s=%r" % (attribute, value) for (attribute, value) in self.__dict__.items()]
        # return '\n'.join(state)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return '<%s title:"%s" type:%s owner:%s>' % (
            type(self).__name__,
            self.title,
            self._ux_item_type(),
            self.owner,
        )

    # ----------------------------------------------------------------------
    def reassign_to(
        self, target_owner: str | User, target_folder: Optional[str] = None
    ):
        """
        The ``reassign_to`` method allows the administrator to reassign a single item from one user to another.

        .. note::
            If you wish to move all of a user's items (and groups) to another user then use the
            user.reassign_to() method.  The ``item.reassign_to`` method (this method) only moves one item at a time.

        ================  ========================================================
        **Parameter**      **Description**
        ----------------  --------------------------------------------------------
        target_owner      Required string or :class:`~arcgis.gis.User`. The string
                          must be a *username* value.
        ----------------  --------------------------------------------------------
        target_folder     Optional string. The folder title to move the item to.
        ================  ========================================================

        :return:
            A boolean indicating success (True) with the ID of the reassigned item, or failure (False).

        .. code-block:: python

            # Usage Example

            >>> item.reassign_to("User1234")

        """
        try:
            current_folder = self.ownerFolder
        except:
            current_folder = None
        if isinstance(target_owner, User):
            target_owner = target_owner.username
        resp = self._portal.reassign_item(
            self.itemid,
            self._user_id,
            target_owner,
            current_folder,
            target_folder,
        )
        if resp is True:
            self._hydrate()  # refresh
            return resp

    # ----------------------------------------------------------------------
    @property
    def shared_with(self):
        """
        The ``shared_with`` property reveals the privacy or sharing status of the current item. An item can be private
        or shared with one or more of the following:
            1. A specified list of groups
            2. All members in the organization
            3. Everyone (including anonymous users).

        .. note::
            If the return is False for `org`, `everyone` and contains an empty list of `groups`, then the
            item is private and visible only to the owner.

        :return:
            A Dictionary in the following format:
            {
            'groups': [],  # one or more Group objects
            'everyone': True | False,
            'org': True | False
            }
        """
        if not self._hydrated:
            self._hydrate()  # hydrated properties needed below

        # find if portal is ArcGIS Online
        try:
            ig_url = f"{self._gis._portal.resturl}content/itemsgroups"
            params = {"f": "json", "items": self.itemid}
            ig_groups = list(self._portal.con.get(ig_url, params).keys())
        except:
            ig_groups = []

        if self._gis._portal.is_arcgisonline:
            # Call with owner info
            if self._user_id != self._gis.users.me.username:
                url = "{resturl}content/items/{itemid}/groups".format(
                    resturl=self._gis._portal.resturl, itemid=self.itemid
                )
                resp = self._portal.con.post(url, {"f": "json"})
                ret_dict = {
                    "everyone": self.access == "public",
                    "org": (self.access == "public" or self.access == "org"),
                    "groups": [],
                }
                for grpid in (
                    resp.get("admin", [])
                    + resp.get("other", [])
                    + resp.get("member", [])
                    + ig_groups
                ):
                    try:
                        grp = Group(gis=self._gis, groupid=grpid["id"])
                        ret_dict["groups"].append(grp)
                    except:
                        pass
                return ret_dict
            else:
                resp = self._portal.con.get(
                    "content/users/" + self._user_id + "/items/" + self.itemid
                )

        else:  # gis is a portal, find if item resides in a folder
            if self._user_id != self._gis.users.me.username:
                url = "{resturl}content/items/{itemid}/groups".format(
                    resturl=self._gis._portal.resturl, itemid=self.itemid
                )
                resp = self._portal.con.post(url, {"f": "json"})
                ret_dict = {
                    "everyone": self.access == "public",
                    "org": (self.access == "public" or self.access == "org"),
                    "groups": [],
                }
                for grpid in (
                    resp.get("admin", [])
                    + resp.get("other", [])
                    + resp.get("member", [])
                    + ig_groups
                ):
                    try:
                        grp = Group(gis=self._gis, groupid=grpid["id"])
                        ret_dict["groups"].append(grp)
                    except:
                        pass
                return ret_dict
            if self.ownerFolder is not None:
                resp = self._portal.con.get(
                    "content/users/"
                    + self._user_id
                    + "/"
                    + self.ownerFolder
                    + "/items/"
                    + self.itemid
                )
            else:
                resp = self._portal.con.get(
                    "content/users/" + self._user_id + "/items/" + self.itemid
                )

        # Get the sharing info
        sharing_info = resp["sharing"]
        ret_dict = {"everyone": False, "org": False, "groups": []}

        if sharing_info["access"] == "public":
            ret_dict["everyone"] = True
            ret_dict["org"] = True

        if sharing_info["access"] == "org":
            ret_dict["org"] = True

        if len(sharing_info["groups"]) > 0:
            grps = []
            for g in sharing_info["groups"]:
                try:
                    grps.append(Group(self._gis, g))
                except:  # ignore groups you can't access
                    pass
            ret_dict["groups"] = grps

        return ret_dict

    # ----------------------------------------------------------------------
    def share(
        self,
        everyone: bool = False,
        org: bool = False,
        groups: Optional[Union[list[Group], list[str]]] = None,
        allow_members_to_edit: bool = False,
    ):
        """
        The ``share`` method allows you to set the list of groups the Item will be shared with.
        You can also set the Item to be shared with your org or with everyone(public including org).

        ======================      ========================================================
        **Parameter**               **Description**
        ----------------------      --------------------------------------------------------
        everyone                    Optional boolean. If True, this item will be shared with
                                    everyone, meaning that it will be publicly accessible and
                                    available to users outside of the organization.
                                    If set to False (default), the item will not be shared
                                    with the public.
        ----------------------      --------------------------------------------------------
        org                         Optional boolean. If True, this item will be shared with
                                    the organization. If set to False (default),
                                    the item will not be shared with the organization.
        ----------------------      --------------------------------------------------------
        groups                      Optional list of group ids as strings, or a list of
                                    arcgis.gis.Group objects. Default is None, don't share
                                    with any specific groups.
        ----------------------      --------------------------------------------------------
        allow_members_to_edit       Optional boolean. Set to True when the item will be shared
                                    with groups with item update capability so that any member
                                    of such groups can update the item that is shared with them.
        ======================      ========================================================

        :return:
            A dictionary with a key titled "`notSharedWith`",containing array of groups with which the item could not be
            shared as well as "itemId" key containing the item id.

        .. code-block:: python

            # Usage Example

            >>> item.share(org = True, allow_members_to_edit = True)


        """
        # Check that the values passed in are valid, groups is checked later if passed in
        if (
            not isinstance(everyone, bool)
            or not isinstance(org, bool)
            or not isinstance(allow_members_to_edit, bool)
        ):
            raise ValueError(
                "everyone, org, and allow_members_to_edit must be boolean values"
            )

        # If everyone is True, set org to True
        if everyone is True:
            org = True

        # If group is passed in, handle it
        group_ids = ""
        if groups is not None:
            if isinstance(groups, list):
                for group in groups:
                    if isinstance(group, Group):
                        # create string list of group ids
                        if len(group_ids) == 0:
                            group_ids = group.id
                        else:
                            group_ids = group_ids + "," + group.id

                    elif isinstance(group, str):
                        # search for group using id to make sure exists
                        search_result = self._gis.groups.search(
                            query="id:" + group, max_groups=1
                        )
                        if len(search_result) > 0:
                            group_ids = group_ids + "," + search_result[0].id
                        else:
                            raise Exception("Cannot find group with id: " + group)
                    else:
                        raise Exception(
                            "Invalid group(s). Must be a list of group ids or group objects."
                        )
            elif isinstance(groups, Group):
                # Only one group provided
                group_ids = groups.id
            elif isinstance(groups, str):
                # old API - groups sent as comma separated group ids
                # could be one group or already made string list of many group ids
                group_ids = groups

        # Check privileges for sharing:
        can_share = False
        if self.owner == self._gis.users.me.username:
            can_share = True
        else:
            privileges = self._gis.users.me.privileges
            if len(group_ids) > 0 and "portal:admin:shareToGroup" in privileges:
                can_share = True
            if everyone is True and "portal:admin:shareToPublic" in privileges:
                can_share = True
            if org is True and "portal:admin:shareToOrg" in privileges:
                can_share = True

        # Create url and params
        if can_share:
            url = "{resturl}content/items/{itemid}/share".format(
                resturl=self._gis._portal.resturl, itemid=self.itemid
            )
            params = {
                "f": "json",
                "groups": group_ids,
                "everyone": everyone,
                "org": org,
                "confirmItemControl": allow_members_to_edit,
            }
            res = self._portal.con.post(url, params)
            self._hydrated = False
            self._hydrate()
            return res
        else:
            raise Exception(
                "User does not own the item or have the privileges to share this item."
            )

    # ----------------------------------------------------------------------
    def unshare(self, groups: Union[list[str], list[Group]]):
        """
        The ``unshare`` method stops sharing of the Item with the specified list of groups.


        ================  =========================================================================================
        **Parameter**      **Description**
        ----------------  -----------------------------------------------------------------------------------------
        groups            Optional list of group names as strings, or a list of :class:`~arcgis.gis.Group` objects,
                          or a comma-separated list of group IDs.
        ================  =========================================================================================


        :return:
            A Dictionary containing the key `notUnsharedFrom` containing array of groups from which the item
            could not be unshared.
        """
        try:
            folder = self.ownerFolder
        except:
            folder = None

        # get list of group IDs
        group_ids = ""
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, Group):
                    group_ids = group_ids + "," + group.id

                elif isinstance(group, str):
                    # search for group using id
                    search_result = self._gis.groups.search(
                        query="id:" + group, max_groups=1
                    )
                    if len(search_result) > 0:
                        group_ids = group_ids + "," + search_result[0].id
                    else:
                        raise Exception("Cannot find group with id: " + group)
                else:
                    raise Exception("Invalid group(s)")

        elif isinstance(groups, str):
            # old API - groups sent as comma separated group ids
            group_ids = groups

        if self.access == "public":
            return self._portal.unshare_item_as_group_admin(self.itemid, group_ids)
        else:
            owner = self._user_id
            return self._portal.unshare_item(self.itemid, owner, folder, group_ids)

    # ----------------------------------------------------------------------
    def delete(self, force: bool = False, dry_run: bool = False):
        """
        The ``delete`` method deletes the item. If the item is unable to be deleted , a RuntimeException is raised.
        To know if you can safely delete the item, use the optional parameter 'dry_run' in order to test the operation
        without actually deleting the item.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        force               Optional boolean. Available in ArcGIS Enterprise 10.6.1 and higher.
                            Force deletion is applicable only to items that were orphaned when
                            a server federated to the ArcGIS Enterprise was removed accidentally
                            before properly unfederating it. When called on other items, it has
                            no effect.
        ---------------     --------------------------------------------------------------------
        dry_run             Optional boolean. Available in ArcGIS Enterprise 10.6.1 and higher.If
                            True, checks if the item can be safely deleted and gives you back
                            either a dictionary with details. If dependent items are preventing
                            deletion, a list of such Item objects are provided.
        ===============     ====================================================================

        :return:
            A boolean indicating success (True), or failure (False). When ``dry_run`` is used, a dictionary containing
            details of the item is returned.

        .. code-block:: python

            USAGE EXAMPLE: Successful deletion of an item

            item1 = gis.content.get('itemId12345')
            item1.delete()

            >> True

        .. code-block:: python

            USAGE EXAMPLE: Failed deletion of an item

            item1 = gis.content.get('itemId12345')
            item1.delete()

            >> RuntimeError: Unable to delete item. This service item has a related Service item
            >> (Error Code: 400)

        .. code-block:: python

            USAGE EXAMPLE: Dry run to check deletion of an item

            item1 = gis.content.get('itemId12345abcde')
            item1.delete(dry_run=True)

            >> {'can_delete': False,
            >> 'details': {'code': 400,
            >> 'message': 'Unable to delete item. This service item has a related Service item',
            >> 'offending_items': [<Item title:"Chicago_accidents_WFS" type:WFS owner:sharing1>]}}

        .. note::
            During the `dry run`, if you receive a list of offending items, attempt to delete them first before deleting
            the current item. You can in turn call ``dry_run`` on those items to ensure they can be deleted safely.
        """

        try:
            folder = self.ownerFolder
        except:
            folder = None

        if dry_run:
            can_delete_resp = self._portal.can_delete(
                self.itemid, self._user_id, folder
            )
            if can_delete_resp[0]:
                return {"can_delete": True}
            else:
                error_dict = {
                    "code": can_delete_resp[1].get("code"),
                    "message": can_delete_resp[1].get("message"),
                    "offending_items": [
                        Item(self._gis, e["itemId"])
                        for e in can_delete_resp[1].get("offendingItems")
                    ],
                }

                return {"can_delete": False, "details": error_dict}
        else:
            return self._portal.delete_item(self.itemid, self._user_id, folder, force)

    # ----------------------------------------------------------------------
    def update(
        self,
        item_properties: Optional[dict[str, Any]] | ItemProperties = None,
        data: Optional[str] = None,
        thumbnail: Optional[str] = None,
        metadata: Optional[str] = None,
    ):
        """
        The ``update`` method updates an item in a Portal.

        .. note::
            The content can be a file (such as a layer package, geoprocessing package,
            map package) or a URL (to an ArcGIS Server service, WMS service,
            or an application).

            To upload a package or other type of file,  a path or URL
            to the file must be provided in the data argument.

            For item_properties, pass in arguments for only the properties you want to be updated.
            All other properties will be untouched.  For example, if you want to update only the
            item's description, then only provide the description argument in item_properties.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_properties     Required dictionary. See table below for the keys and values.
        ---------------     --------------------------------------------------------------------
        data                Optional string, io.StringIO, or io.BytesIO. Either a path or URL to
                            the data or an instance of `StringIO` or `BytesIO` objects.
        ---------------     --------------------------------------------------------------------
        thumbnail           Optional string. Either a path or URL to a thumbnail image.
        ---------------     --------------------------------------------------------------------
        metadata            Optional string. Either a path or URL to the metadata.
        ===============     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        type               Optional string. Indicates type of item, see the link below for valid values.
        -----------------  ---------------------------------------------------------------------
        typeKeywords       Optional string. Provide a lists all sub-types, see the link below for valid values.
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        title              Optional string. Name label of the item.
        -----------------  ---------------------------------------------------------------------
        url                Optional string. URL to item that are based on URLs.
        -----------------  ---------------------------------------------------------------------
        tags               Optional string. Tags listed as comma-separated values, or a list of strings.
                           Used for searches on items.
        -----------------  ---------------------------------------------------------------------
        text               Optional string. For text based items such as Feature Collections & WebMaps
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        extent             Optional string. Provide comma-separated values for min x, min y, max x, max y.
        -----------------  ---------------------------------------------------------------------
        spatialReference   Optional string. Coordinate system that the item is in.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        -----------------  ---------------------------------------------------------------------
        access             Optional string. Valid values are private, shared, org, or public.
        =================  =====================================================================


        .. note::
             See `Items and Item Types
             <https://developers.arcgis.com/rest/users-groups-and-items/items-and-item-types.htm>`_
             in the ArcGIS REST API documentation for more details.

        :return:
            A boolean indicating success (True) or failure (False).

        .. code-block:: python

            # Usage Example

            >>> item.update(item_properties = {"description":"Boundaries and infrastructure for Warren County",
                                                "title":"Warren County Feature Layer",
                                                "tags":"local government, administration, Warren County"
                                               })
        """
        if isinstance(item_properties, ItemProperties):
            if (
                thumbnail is None
                and item_properties.thumbnail
                and (
                    os.path.isfile(item_properties.thumbnail)
                    or item_properties.thumbnail_url
                )
            ):
                thumbnail = item_properties.thumbnail or item_properties.thumbnail_url
            if (
                metadata is None
                and item_properties.metadata
                and os.path.isfile(item_properties.metadata)
            ):
                metadata = item_properties.metadata

            if "access" in item_properties:
                access = item_properties.pop("access")
                if access == "private":
                    self.share(everyone=False, org=False)
                if access == "org":
                    self.share(everyone=False, org=True)
                if access == "public":
                    self.share(everyone=True)
                if access == "shared":
                    groups = self.shared_with["groups"]
                    self.share(groups=groups)

            item_properties = item_properties.to_dict()
            item_properties.pop("metadata", None)
            item_properties.pop("thumbnail", None)
        if (
            data
            and isinstance(data, str)
            and os.path.isfile(data)
            and os.stat(data).st_size > int(2.5e7)
        ):
            owner = self._user_id

            try:
                folder = self.ownerFolder
            except:
                folder = None

            if item_properties:
                large_thumbnail = item_properties.pop("largeThumbnail", None)
            else:
                large_thumbnail = None

            if item_properties is not None:
                if "tags" in item_properties:
                    if type(item_properties["tags"]) is list:
                        item_properties["tags"] = ",".join(item_properties["tags"])

            if data is not None and isinstance(data, (io.StringIO, io.BytesIO)):
                if item_properties is None:
                    item_properties = {}
                if not "type" in item_properties:
                    item_properties["type"] = self.type
                if not "fileName" in item_properties:
                    fileName = self.name
                    item_properties["fileName"] = fileName
            # update everything but the data
            ret = self._portal.update_item(
                self.itemid,
                item_properties,
                None,
                thumbnail,
                metadata,
                owner,
                folder,
                large_thumbnail,
            )
            # update the data by part:
            params = {
                "f": "json",
                "multipart": True,
                "async": True,
                "filename": os.path.basename(data),
            }
            url = f"{self._gis._portal.resturl}content/users/{self.owner}"
            if folder:
                url += "/" + folder
            url += "/items/" + self.itemid + "/update"
            res = self._gis._con.post(url, params)
            if item_properties is None:
                item_properties = {"type": self.type}
            elif not "type" in item_properties:
                item_properties["type"] = self.type
            status = self._gis.content._add_by_part(
                file_path=data,
                itemid=self.itemid,
                item_properties=item_properties,
                size=int(3.5e7),
                owner=self.owner,
                folder=folder,
            )
            if status == "completed":
                self._hydrate()
            elif ret:
                self._hydrate()
            return ret
        else:
            owner = self._user_id

            try:
                folder = self.ownerFolder
            except:
                folder = None

            if item_properties:
                large_thumbnail = item_properties.pop("largeThumbnail", None)
            else:
                large_thumbnail = None

            if item_properties is not None:
                if "tags" in item_properties:
                    if type(item_properties["tags"]) is list:
                        item_properties["tags"] = ",".join(item_properties["tags"])
                if "access" in item_properties:
                    access = item_properties.pop("access")
                    if access == "private":
                        self.share(everyone=False, org=False)
                    if access == "org":
                        self.share(everyone=False, org=True)
                    if access == "public":
                        self.share(everyone=True)
                    if access == "shared":
                        groups = self.shared_with["groups"]
                        self.share(groups=groups)

            if data is not None and isinstance(data, (io.StringIO, io.BytesIO)):
                if item_properties is None:
                    item_properties = {}
                if not "type" in item_properties:
                    item_properties["type"] = self.type
                if not "fileName" in item_properties:
                    fileName = self.name
                    item_properties["fileName"] = fileName

            ret = self._portal.update_item(
                self.itemid,
                item_properties,
                data,
                thumbnail,
                metadata,
                owner,
                folder,
                large_thumbnail,
            )
            if ret:
                self._hydrate()
            return ret

    # ----------------------------------------------------------------------
    def update_info(self, file: str, folder_name: Optional[str] = None):
        """
        You can upload JSON, XML, CFG, TXT, PBF, and PNG files only. The file size limit is 100K.
        The uploaded file is also available through the https://item-url/info/filename resource.

        Must be the owner of the item to update this.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        file                Required String. The path to the file that will be uploaded.
        ---------------     --------------------------------------------------------------------
        folder_name         Optional String. The name of the subfolder for added information.
        ===============     ====================================================================

        :return: Success or Failure
        """
        url = "{resturl}content/users/{owner}/items/{itemid}/updateInfo".format(
            resturl=self._gis._portal.resturl,
            owner=self.owner,
            itemid=self.itemid,
        )
        params = {
            "f": "json",
            "file": file,
            "folderName": folder_name,
        }
        res = self._portal.con.post(url, params)
        self._hydrated = False
        self._hydrate()
        return res

    # ----------------------------------------------------------------------
    def delete_info(self, file: str):
        """
        This is available for all items and allows you to delete an individual file from an item's esriinfo folder.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        file                Required String. The file to be deleted.
        ===============     ====================================================================

        """
        url = "{resturl}content/users/{owner}/items/{itemid}/deleteInfo".format(
            resturl=self._gis._portal.resturl,
            owner=self.owner,
            itemid=self.itemid,
        )
        params = {
            "f": "json",
            "file": file,
        }
        res = self._portal.con.post(url, params)
        self._hydrated = False
        self._hydrate()
        return res

    # ----------------------------------------------------------------------
    @property
    def view_manager(self) -> ViewManager | None:
        """
        If the `Item` is a `Feature Service` and a `Hosted Service`, the `Item`
        can have views.  This Manager allows users to work with `Feature Service`
        to create **views**

        :returns:
            A  :class:`~arcgis.gis.ViewManager` object to administer hosted
            feature layer view :class:`items <arcgis.gis.Item>` created from
            the hosted feature layer.
        """
        if self.type == "Feature Service" and "Hosted Service" in self.typeKeywords:
            return ViewManager(item=self)
        return None

    # ----------------------------------------------------------------------
    def _interval_times(self, sd, ed, params, as_df, dr_type):
        """
        Method for the usage method. If a date range is greater than 5 months
        it needs to be sent in multiple posts.
        """
        if dr_type == "6m":
            ranges = {
                "1": [sd, sd + timedelta(days=60)],
                "2": [sd + timedelta(days=61), sd + timedelta(days=120)],
                "3": [sd + timedelta(days=121), sd + timedelta(days=180)],
                "4": [
                    sd + timedelta(days=181),
                    ed + timedelta(days=1),
                ],
            }
        elif dr_type == "12m":
            ranges = {
                "1": [sd, sd + timedelta(days=60)],
                "2": [sd + timedelta(days=61), sd + timedelta(days=120)],
                "3": [sd + timedelta(days=121), sd + timedelta(days=180)],
                "4": [sd + timedelta(days=181), sd + timedelta(days=240)],
                "5": [sd + timedelta(days=241), sd + timedelta(days=320)],
                "6": [sd + timedelta(days=321), sd + timedelta(days=366)],
            }
        else:
            # custom date range
            ranges = {
                "1": [sd, sd + timedelta(days=60)],
                "2": [sd + timedelta(days=61), sd + timedelta(days=120)],
                "3": [sd + timedelta(days=121), sd + timedelta(days=180)],
            }
            # since over 5 months we know that there are at least 4 ranges and up to 6 for 1 year.
            stop = False
            range_add = 4
            days = 181
            # need to check if time delta will surpass our end_date or not
            while stop is False:
                next_time = sd + timedelta(days=days + 59)
                if next_time >= ed:
                    # we reached the end time specified by user
                    ranges[str(range_add)] = [
                        sd + timedelta(days=days),
                        ed + timedelta(days=1),
                    ]
                    stop = True
                else:
                    # add a range
                    ranges[str(range_add)] = [
                        sd + timedelta(days=days),
                        sd + timedelta(days=days + 59),
                    ]
                range_add = range_add + 1
                days = days + 60

        # set the url
        if self._gis._portal.is_logged_in:
            url = "%s/portals/%s/usage" % (
                self._portal.resturl,
                self._gis.properties.id,
            )
        else:
            url = "%s/portals/%s/usage" % (self._portal.resturl, "self")

        # Go through ranges and gather results from each post request
        results = []
        for k, v in ranges.items():
            sd = int(v[0].timestamp() * 1000)
            ed = int(v[1].timestamp() * 1000)
            params["startTime"] = sd
            params["endTime"] = ed
            res = self._portal.con.post(url, params)
            if as_df:
                import pandas as pd

                if "data" not in res or len(res["data"]) == 0:
                    res = pd.DataFrame([], columns=["Date", "Usage"])
                elif len(res["data"]):
                    res = pd.DataFrame(res["data"][0]["num"], columns=["Date", "Usage"])
                    res.Date = pd.to_datetime(res["Date"], unit="ms")
                    res.Usage = res.Usage.astype(int)

            results.append(res)
            del k, v
        if as_df:
            if len(results):
                return (
                    pd.concat(results)
                    .reset_index(drop=True)
                    .drop_duplicates(keep="first", inplace=False)
                )
            else:
                return pd.DataFrame([], columns=["Date", "Usage"])
        else:
            return results

    # ----------------------------------------------------------------------
    @cached(cache=TTLCache(maxsize=255, ttl=60))
    def usage(self, date_range: str = "7D", as_df: bool = True):
        """

        .. note::
            The ``usage`` method is available for ArcGIS Online Only.

        For item owners and administrators, the ``usage`` method provides usage details about an item that help you
        gauge its popularity. Usage details show how many times the item has been used for the time
        period you select. Historical usage information is available for the past year. Depending on
        the item type, usage details can include the number of views, requests, or downloads, and
        the average number of views, requests, or downloads per day.

        Views refers to the number of times the item has been viewed or opened. For maps, scenes,
        non-hosted layers, and web apps, the view count is increased by one when you open the item
        page or open the item in Map Viewer. For example, if you opened the item page for a map
        image layer and clicked Open in Map Viewer, the count would increase by two. For other items
        such as mobile apps, KML, and so on, the view count is increased by one when you open the
        item; the count does not increase when you open the item details page.

        For hosted web layers (feature, tile, and scene), the number of requests is provided instead
        of views. Requests refers to the number of times a request is made for the data within the
        layer. For example, someone might open an app that contains a hosted feature layer. Opening
        the app counts as one view for the application, but multiple requests may be necessary to
        draw all the features in the hosted layer and are counted as such.

        For downloadable file item types such as CSV, SHP, and so on, the number of downloads is
        displayed. For registered apps, the Usage tab also displays the number of times users have
        logged in to the app. Apps that allow access to subscriber content through the organization
        subscription show usage by credits. Additionally, the time frame for the credit usage
        reporting period can be changed.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        date_range          Optional string.  The default is 7d.  This is the period to query
                            usage for a given item.

                            =============           =========================
                            24H                     Past 24 hours
                            -------------           -------------------------
                            7D                      Past 7 days (default)
                            -------------           -------------------------
                            14D                     Past 14 days
                            -------------           -------------------------
                            30D                     Past 30 days
                            -------------           -------------------------
                            60D                     Past 60 days
                            -------------           -------------------------
                            6M                      Past 6 months
                            -------------           -------------------------
                            1Y                      Past 12 months
                            -------------           -------------------------
                            (date1,date2)           Tuple of 2 datetime
                                                    objects defining custom
                                                    date range
                            =============           =========================
        ---------------     --------------------------------------------------------------------
        as_df               Optional boolean.  Returns a Pandas DataFrame when True, returns data
                            as a dictionary when False
        ===============     ====================================================================

        :return: Pandas `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ or Dictionary

        .. code-block:: python

            # Usage Example #1: Standard date_range

            >>> flyr_item = gis.content.get("8961540a52da402876e0168fa29bb82d")
            >>> result = flyr_item.usage(date_range = "7D")
                Date  Usage
            0 2022-08-05      0
            1 2022-08-06      0
            2 2022-08-07      0
            3 2022-08-08      0
            4 2022-08-09      0
            5 2022-08-10      0
            6 2022-08-11      0
            7 2022-08-12      8

            # Usage Example #2: Custom date_range

            >>> import datetime as dt

            >>> flyr_item = gis.content.get("8961540a52da402876e0168fa29bb82d")
            >>> date_1 = dt.datetime(2022,7,31)
            >>> date_2 = dt.datetime.now(2022,8,12)
                # Early value, later value
            >>> result = flyr_item.usage(date_range = (date_1, date_2))
            >>> result
                     Date  Usage
            0  2022-07-31      0
            1  2022-08-01      0
            2  2022-08-02      0
            3  2022-08-03      0
            4  2022-08-04      0
            5  2022-08-05      0
            6  2022-08-06      0
            7  2022-08-07      0
            8  2022-08-08      0
            9  2022-08-09      0
            10 2022-08-10      0
            11 2022-08-11      0
            12 2022-08-12     10
        """
        if not self._portal.is_arcgisonline:
            raise ValueError("Usage() only supported for ArcGIS Online items.")

        # Set end date and params dict
        end_date = datetime.now()
        params = {
            "f": "json",
            "startTime": None,
            "endTime": int(end_date.timestamp() * 1000),
            "period": "",
            "vars": "num",
            "groupby": "name",
            "etype": "svcusg",
            "name": self.itemid,
        }

        # Handle Feature Service
        if self.type == "Feature Service":
            params["stype"] = "features"
            if len(self.layers) > 0 and not self.layers[0].container:
                params["name"] = os.path.basename(
                    os.path.abspath(
                        os.path.join(self.layers[0]._url, ".." + os.sep + "..")
                    )
                )
            else:  # hasattr(self, "url") and self.url and len(self.url) > 0:
                params["name"] = os.path.basename(os.path.dirname(self.url))

        # Handle Vector Tile Service
        if self.type == "Vector Tile Service":
            params["name"] = self.title.replace(" ", "_")

        # Date Range Handling
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            sd = date_range[0]
            end_date = date_range[1]
            params["period"] = "1d"
            if (end_date - sd).days <= 152:  # 5 months in days
                # normal workflow
                params["startTime"] = int(sd.timestamp() * 1000)
                params["endTime"] = int(end_date.timestamp() * 1000)
            else:
                # need to send as more than one post
                results = self._interval_times(sd, end_date, params, as_df, "custom")
                return results
        elif date_range.lower() in ["24h", "1d"]:
            params["period"] = "1h"
            params["startTime"] = int((end_date - timedelta(days=1)).timestamp() * 1000)
        elif date_range.lower() == "7d":
            params["period"] = "1d"
            params["startTime"] = int((end_date - timedelta(days=7)).timestamp() * 1000)
        elif date_range.lower() == "14d":
            params["period"] = "1d"
            params["startTime"] = int(
                (end_date - timedelta(days=14)).timestamp() * 1000
            )
        elif date_range.lower() == "30d":
            params["period"] = "1d"
            params["startTime"] = int(
                (end_date - timedelta(days=30)).timestamp() * 1000
            )
        elif date_range.lower() == "60d":
            params["period"] = "1d"
            params["startTime"] = int(
                (end_date - timedelta(days=60)).timestamp() * 1000
            )
        elif date_range.lower() == "6m":
            params["period"] = "1d"
            sd = end_date - timedelta(days=int(365 / 2))
            results = self._interval_times(sd, end_date, params, as_df, "6m")
            return results
        elif date_range.lower() in ["12m", "1y"]:
            sd = end_date - timedelta(days=int(365))
            params["period"] = "1d"
            results = self._interval_times(sd, end_date, params, as_df, "12m")
            return results
        else:
            raise ValueError("Invalid date range.")

        # if date range is less than 5 months
        if self._gis._portal.is_logged_in:
            url = "%sportals/%s/usage" % (
                self._portal.resturl,
                self._gis.properties.id,
            )
        else:
            url = "%sportals/%s/usage" % (self._portal.resturl, "self")

        try:
            res = self._portal.con.post(url, params)
            if as_df:
                import pandas as pd

                if "data" not in res or len(res["data"]) == 0:
                    df = pd.DataFrame([], columns=["Date", "Usage"])
                elif len(res["data"]):
                    df = pd.DataFrame(res["data"][0]["num"], columns=["Date", "Usage"])
                    df.Date = pd.to_datetime(df["Date"], unit="ms")
                    df.Usage = df.Usage.astype(int)
                return df
            return res
        except:
            return None

    # ----------------------------------------------------------------------
    def get_data(self, try_json: bool = True):
        """
        The ``get_data`` method retrieves the data associated with an item.

        .. note::
            This call may return different results for different item types: some item types may even return *None*. See
            `Working with users, groups, and items
            <https://developers.arcgis.com/rest/users-groups-and-items/working-with-users-groups-and-items.htm>`_
            in the ArcGIS REST API for more information.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        try_json            Optional string. Default is True. For JSON/text files, if try_json
                            is True, the method tries to convert the data to a Python dictionary
                            (use json.dumps(data) to convert the dictionary to a string),
                            otherwise the data is returned as a string.
        ===============     ====================================================================

        :return:
           Dependent on the content type of the data.
           For non-JSON/text data, binary files are returned and the path to the downloaded file.
           For JSON/text files, a Python dictionary or a string.  All others will be a byte array,
           that can be converted to string using data.decode('utf-8'). Zero byte files will return None.
        """
        folder = None
        try:
            item_data = self._portal.get_item_data(self.itemid, try_json, folder)
        except:
            item_data = {}

        if item_data == "":
            return None
        elif type(item_data) == bytes:
            try:
                item_data_str = item_data.decode("utf-8")
                if item_data_str == "":
                    return None
                else:
                    return item_data
            except:
                return item_data
        else:
            return item_data

    # ----------------------------------------------------------------------
    def dependent_upon(self):
        """
        The ``dependent_upon`` method returns items, urls, etc that this item is dependent on.

        .. note::
            This method only available for items in an ArcGIS Enterprise organization.
        """
        return self._portal.get_item_dependencies(self.itemid)

    # ----------------------------------------------------------------------
    def dependent_to(self):
        """
        The ``dependent_to`` method returns items, urls, etc that are dependent to this item.

        .. note::
            This method only available for items in an ArcGIS Enterprise organization.
        """
        return self._portal.get_item_dependents_to(self.itemid)

    # ----------------------------------------------------------------------
    _RELATIONSHIP_TYPES = frozenset(
        [
            "Map2Service",
            "WMA2Code",
            "Map2FeatureCollection",
            "MobileApp2Code",
            "Service2Data",
            "Service2Service",
            "Map2AppConfig",
            "Item2Attachment",
            "Item2Report",
            "Listed2Provisioned",
            "Style2Style",
            "Service2Style",
            "Survey2Service",
            "Survey2Data",
            "Service2Route",
            "Area2Package",
            "Map2Area",
            "Service2Layer",
            "Area2CustomPackage",
            "TrackView2Map",
            "SurveyAddIn2Data",
            "WorkforceMap2FeatureService",
            "Theme2Story",
            "WebStyle2DesktopStyle",
            "Solution2Item",
            "APIKey2Item",
            "Mission2Item",
            "Map2FeatureCollectionMobileApp2Code",
            "Notebook2WebTool",
        ]
    )
    _RELATIONSHIP_DIRECTIONS = frozenset(["forward", "reverse"])

    # ----------------------------------------------------------------------
    def related_items(self, rel_type: str, direction: str = "forward"):
        """
        The ``related_items`` method retrieves the items related to this item. Relationships can be added and deleted
        using item.add_relationship() and item.delete_relationship(), respectively.

        .. note::
            With WebMaps items, relationships are only available on local enterprises.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_ in the REST API help
                            for more information on this parameter.
        ---------------     --------------------------------------------------------------------
        direction           Required string. One of ['forward', 'reverse']
        ===============     ====================================================================

        :return:
           The list of related items.

        .. code-block:: python

            # Usage Example

            >>> item.related_items("Service2Service", "forward")
        """

        if rel_type not in self._RELATIONSHIP_TYPES:
            raise Error("Unsupported relationship type: " + rel_type)
        if not direction in self._RELATIONSHIP_DIRECTIONS:
            raise Error("Unsupported direction: " + direction)

        related_items = []

        postdata = {"f": "json"}
        postdata["relationshipType"] = rel_type
        postdata["direction"] = direction
        resp = self._portal.con.post(
            "content/items/" + self.itemid + "/relatedItems", postdata
        )
        for related_item in resp["relatedItems"]:
            related_items.append(Item(self._gis, related_item["id"], related_item))
        return related_items

    # ----------------------------------------------------------------------
    def add_relationship(self, rel_item: Item, rel_type: str):
        """The ``add_relationship`` method adds a relationship from the current item to ``rel_item``.

        .. note::
            Note: Relationships are not tied to an item. Instead, they are directional links from an origin item
            to a destination item and have a type. The type of the relationship defines the valid origin and destination
            item types as well as some rules. See `Relationship types <https://bit.ly/2LAHNoK>`_ in REST API help for more information.
            Users don't have to own the items they relate unless so defined by the rules of the relationship
            type.

            Users can only delete relationships they create.

            Relationships are deleted automatically if one of the two items is deleted.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_item            Required Item object corresponding to the related item.
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_ in the REST API help
                            for more information on this parameter.
        ===============     ====================================================================

        :return:
           A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> item.add_relationship(reL_item=item2, rel_type='Map2FeatureCollection')
            <True>
        """
        if not rel_type in self._RELATIONSHIP_TYPES:
            raise Error("Unsupported relationship type: " + rel_type)

        postdata = {"f": "json"}
        postdata["originItemId"] = self.itemid
        postdata["destinationItemId"] = rel_item.itemid
        postdata["relationshipType"] = rel_type
        path = "content/users/{uid}/addRelationship".format(uid=self._user_id)

        resp = self._portal.con.post(path, postdata)
        if resp:
            return resp.get("success")

    # ----------------------------------------------------------------------
    def delete_relationship(self, rel_item: Item, rel_type: str):
        """
        The ``delete_relationship`` method  deletes a relationship between this item and the rel_item.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        rel_item            Required Item object corresponding to the related item.
        ---------------     --------------------------------------------------------------------
        rel_type            Required string.  The type of the related item; is one of
                            ['Map2Service', 'WMA2Code', 'Map2FeatureCollection', 'MobileApp2Code',
                            'Service2Data', 'Service2Service']. See
                            `Relationship Types <https://bit.ly/2LAHNoK>`_ in the REST API help
                            for more information on this parameter.
        ===============     ====================================================================

        :return:
           A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            item.delete_relationship(item2, 'Map2FeatureCollection')
        """
        if not rel_type in self._RELATIONSHIP_TYPES:
            raise Error("Unsupported relationship type: " + rel_type)
        postdata = {"f": "json"}
        postdata["originItemId"] = self.itemid
        postdata["destinationItemId"] = rel_item.itemid
        postdata["relationshipType"] = rel_type
        path = "content/users/{uid}/deleteRelationship".format(uid=self._user_id)

        resp = self._portal.con.post(path, postdata)
        if resp:
            return resp.get("success")

    # ----------------------------------------------------------------------
    def publish(
        self,
        publish_parameters: Optional[dict[str, Any]] = None,
        address_fields: Optional[dict[str, str]] = None,
        output_type: Optional[str] = None,
        overwrite: bool = False,
        file_type: Optional[str] = None,
        build_initial_cache: bool = False,
        item_id: Optional[str] = None,
        geocode_service=None,
    ):
        """
        The ``publishes`` method is used to publish a hosted service based on an existing source item (this item).
        Publishers can then create feature, tiled map, vector tile and scene services.
        Feature services can be created from  input files of various types, including
            1. csv files
            2. shapefiles
            3. service definition files
            4. feature collection files
            5. file geodatabase files
        CSV files that contain location fields (i.e. address fields or XY fields) are spatially enabled during the process of publishing.
        Shapefiles and file geodatabases should be packaged as *.zip files.

        Tiled map services can be created from service definition (*.sd) files, tile packages, and existing feature services.

        Vector tile services can be created from vector tile package (*.vtpk) files.

        Scene services can be created from scene layer package (*.spk, *.slpk) files.

        Service definitions are authored in ArcGIS Pro or ArcGIS Desktop and contain both the cartographic definition for a map
        as well as its packaged data together with the definition of the geo-service to be created.

        .. note::
            ArcGIS does not permit overwriting if you published multiple hosted feature layers from the same data item.

        .. note::
            ArcGIS for Enterprise for Kubernetes does not support publishing service definition file generated by ArcMap.

        ===================    ===============================================================
        **Parameter**           **Description**
        -------------------    ---------------------------------------------------------------
        publish_parameters     Optional dictionary. containing publish instructions and customizations.
                               Cannot be combined with overwrite.
                               See `Publish Item <https://developers.arcgis.com/rest/users-groups-and-items/publish-item.htm>`_
                               in the ArcGIS REST API for details.
        -------------------    ---------------------------------------------------------------
        address_fields         Optional dictionary. containing mapping of df columns to address fields,
        -------------------    ---------------------------------------------------------------
        output_type            Optional string.  Only used when a feature service is published as a tile service.
        -------------------    ---------------------------------------------------------------
        overwrite              Optional boolean.   If True, the hosted feature service is overwritten.
                               Only available in ArcGIS Enterprise 10.5+ and ArcGIS Online.
        -------------------    ---------------------------------------------------------------
        file_type              Optional string.  Some formats are not automatically detected,
                               when this occurs, the file_type can be specified:
                               serviceDefinition, shapefile, csv, excel, tilePackage,
                               featureService, featureCollection, fileGeodatabase, geojson,
                               scenepackage, vectortilepackage, imageCollection, mapService,
                               and sqliteGeodatabase are valid entries. This is an
                               optional parameter.
        -------------------    ---------------------------------------------------------------
        build_initial_cache    Optional boolean.  The boolean value (default False), if true
                               and applicable for the file_type, the value will built cache
                               for the service.
        -------------------    ---------------------------------------------------------------
        item_id                Optional string. Available in ArcGIS Enterprise 10.8.1+. Not available in ArcGIS Online.
                               This parameter allows the desired item id to be specified during creation which
                               can be useful for cloning and automated content creation scenarios.
                               The specified id must be a 32 character GUID string without any special characters.

                               If the `item_id` is already being used, an error will be raised
                               during the `publish` process.

        -------------------    ---------------------------------------------------------------
        geocode_service        Optional Geocoder. When publishing a table of data, an optional
                               `Geocoder` can be supplied in order to specify which service
                               geocodes the information. If no geocoder is given, the first
                               registered `Geocoder` is used.
        ===================    ===============================================================

        :return:
            An :class:`~arcgis.gis.Item` object corresponding to the published web layer.

        .. code-block:: python

            # Publishing a Hosted Table Example

            >>> csv_item = gis.content.get('<csv item id>')
            >>> analyzed = gis.content.analyze(item=csv_item)
            >>> publish_parameters = analyzed['publishParameters']
            >>> publish_parameters['name'] = 'AVeryUniqueName' # this needs to be updated
            >>> publish_parameters['locationType'] = None # this makes it a hosted table
            >>> published_item = csv_item.publish(publish_parameters)

        .. code-block:: python

            # Publishing a Tile Service Example

            >>> item.publish(address_fields= { "CountryCode" : "Country"},
            >>>               output_type="Tiles",
            >>>               file_type="CSV",
            >>>               item_id=9311d21a9a2047d19c0faaebd6f2cca6
            >>>             )

        .. note::
            For publish_parameters, see `Publish Item
            <https://developers.arcgis.com/rest/users-groups-and-items/publish-item.htm>`_
            in the ArcGIS REST API for more details.
        """

        import time

        if str(output_type).lower() in ["ogc", "ogcfeatureservice"]:
            output_type = "OGCFeatureService"
            file_type = "featureService"
            scrubbed = re.sub("[^a-zA-Z0-9_]+", "", self.title)
            if publish_parameters is None:
                publish_parameters = {}
            publish_parameters.update(
                {"name": publish_parameters.get("name", scrubbed)}
            )
            build_initial_cache = False
        buildInitialCache = build_initial_cache
        if file_type is None:
            if self["type"] == "GeoPackage":
                fileType = "geoPackage"
            elif self["type"].lower().find("excel") > -1:
                fileType = "excel"
            elif self["type"] == "Compact Tile Package":
                fileType = "compactTilePackage"
            elif self["type"] == "Service Definition":
                fileType = "serviceDefinition"
            elif self["type"] == "Microsoft Excel":
                fileType = "excel"
            elif self["type"] == "Feature Collection":
                fileType = "featureCollection"
            elif self["type"] == "CSV":
                fileType = "csv"
            elif self["type"] == "Shapefile":
                fileType = "shapefile"
            elif self["type"] == "File Geodatabase":
                fileType = "fileGeodatabase"
            elif self["type"] == "Vector Tile Package":
                fileType = "vectortilepackage"
                if output_type is None:
                    output_type = "VectorTiles"
            elif self["type"] == "Scene Package":
                fileType = "scenePackage"
            elif self["type"] == "Tile Package":
                fileType = "tilePackage"
            elif self["type"] == "SQLite Geodatabase":
                fileType = "sqliteGeodatabase"
            elif self["type"] in ["GeoJson", "geojson"]:
                fileType = "geojson"
            elif (
                self["type"] == "Feature Service"
                and "Spatiotemporal" in self["typeKeywords"]
            ):
                fileType = "featureService"
            else:
                raise ValueError(
                    "A file_type must be provide, data format not recognized"
                )
        else:
            fileType = file_type
        try:
            folder = self.ownerFolder
        except:
            folder = None

        if publish_parameters is None:
            if fileType == "shapefile" and not overwrite:
                publish_parameters = {
                    "hasStaticData": True,
                    "name": os.path.splitext(self["name"])[0].replace(" ", "_"),
                    "maxRecordCount": 2000,
                    "layerInfo": {"capabilities": "Query"},
                }

            elif fileType in ["csv", "excel"] and not overwrite:
                res = self._gis.content.analyze(item=self, file_type=fileType)
                publish_parameters = res["publishParameters"]
                service_name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters.update({"name": service_name})

            elif (
                fileType in ["csv", "shapefile", "fileGeodatabase", "excel"]
                and overwrite
            ):  # need to construct full publishParameters
                # find items with relationship 'Service2Data' in reverse direction - all feature services published using this data item
                related_items = self.related_items("Service2Data", "reverse")

                if (
                    len(related_items) == 1
                ):  # simple 1:1 relationship between data and service items
                    r_item = related_items[0]
                    # construct a FLC manager
                    from arcgis.features import FeatureLayerCollection

                    flc = FeatureLayerCollection.fromitem(r_item)
                    flc_mgr = flc.manager

                    # get the publish parameters from FLC manager
                    (
                        publish_parameters,
                        update_params,
                    ) = flc_mgr._gen_overwrite_publishParameters(r_item)
                    if (
                        update_params
                    ):  # when overwriting file on portals, need to update source item metadata
                        self.update(item_properties=update_params)

                    # if source file type is CSV or Excel, blend publish parameters with analysis results
                    if fileType in ["csv", "excel"]:
                        publish_parameters_orig = publish_parameters
                        path = "content/features/analyze"

                        postdata = {
                            "f": "pjson",
                            "itemid": self.itemid,
                            "filetype": fileType,
                            "analyzeParameters": {
                                "enableGlobalGeocoding": "true",
                                "sourceLocale": "en-us",
                                "sourceCountry": "",
                                "sourceCountryHint": "",
                            },
                        }

                        if address_fields is not None:
                            postdata["analyzeParameters"]["locationType"] = "address"

                        res = self._portal.con.post(path, postdata)
                        publish_parameters = res["publishParameters"]
                        publish_parameters.update(publish_parameters_orig)

                elif len(related_items) == 0:
                    # the CSV item was never published. Hence overwrite should work like first time publishing - analyze csv
                    path = "content/features/analyze"
                    postdata = {
                        "f": "pjson",
                        "itemid": self.itemid,
                        "filetype": "csv",
                        "analyzeParameters": {
                            "enableGlobalGeocoding": "true",
                            "sourceLocale": "en-us",
                            "sourceCountry": "",
                            "sourceCountryHint": "",
                        },
                    }

                    if address_fields is not None:
                        postdata["analyzeParameters"]["locationType"] = "address"

                    res = self._portal.con.post(path, postdata)
                    publish_parameters = res["publishParameters"]
                    if address_fields is not None:
                        publish_parameters.update({"addressFields": address_fields})

                    # use csv title for service name, after replacing non-alphanumeric characters with _
                    service_name = re.sub(r"[\W_]+", "_", self["title"])
                    publish_parameters.update({"name": service_name})

                elif len(related_items) > 1:
                    # length greater than 1, then 1:many relationship
                    raise RuntimeError(
                        "User cant overwrite this service, using this data, as this data is already referring to another service."
                    )

            elif fileType == "vectortilepackage":
                name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters = {"name": name, "maxRecordCount": 2000}
                output_type = "VectorTiles"
                buildInitialCache = True

            elif fileType == "scenePackage":
                name = re.sub(r"[\W_]+", "_", self["title"])
                buildInitialCache = True
                publish_parameters = {"name": name, "maxRecordCount": 2000}
                output_type = "sceneService"
            elif fileType == "featureService":
                name = re.sub(r"[\W_]+", "_", self["title"])
                c = self._gis.content
                is_avail = c.is_service_name_available(name, "featureService")
                i = 1
                while is_avail == False:
                    sname = name + "_%s" % i
                    is_avail = c.is_service_name_available(sname, "featureService")
                    if is_avail:
                        name = sname
                        break
                    i += 1
                ms = self.layers[0].container.manager
                publish_parameters = ms._generate_mapservice_definition()
                output_type = "bdsMapService"
                buildInitialCache = True
                if "serviceName" in publish_parameters:
                    publish_parameters["serviceName"] = name

            elif fileType == "tilePackage":
                name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters = {"name": name, "maxRecordCount": 2000}
                buildInitialCache = True
            elif fileType == "sqliteGeodatabase":
                name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters = {
                    "name": name,
                    "maxRecordCount": 2000,
                    "capabilities": "Query, Sync",
                }
            else:  # sd files
                name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters = {
                    "hasStaticData": True,
                    "name": name,
                    "maxRecordCount": 2000,
                    "layerInfo": {"capabilities": "Query"},
                }
        elif (
            fileType
            in [
                "csv",
                "excel",
            ]
            and overwrite is False
        ):  # merge users passed-in publish parameters with analyze results
            publish_parameters_orig = publish_parameters

            res = self._gis.content.analyze(item=self, file_type=fileType)
            publish_parameters = res["publishParameters"]
            # case for hosted tables
            if (
                "layerInfo" in publish_parameters
                and "layerInfo" in publish_parameters_orig
            ):
                # do general update
                publish_parameters.update(publish_parameters_orig)
            # case for hosted fl
            else:
                # check if layers key exist. If not, add empty array to avoid error in update
                if "layers" not in publish_parameters:
                    publish_parameters["layers"] = []
                    # csv analyze returns layerInfo rather than a layer
                    if "layerInfo" in publish_parameters:
                        publish_parameters["layers"].append(
                            publish_parameters["layerInfo"]
                        )

                # do general update and assign service name
                publish_parameters.update(publish_parameters_orig)
                service_name = re.sub(r"[\W_]+", "_", self["title"])
                publish_parameters.update({"name": service_name})
                if not self._gis.content.is_service_name_available(
                    publish_parameters["name"], "featureService"
                ):
                    raise Exception("Service name already exists in your org.")

        ret = self._portal.publish_item(
            self.itemid,
            None,
            None,
            fileType,
            publish_parameters,
            output_type,
            overwrite,
            self.owner,
            folder,
            buildInitialCache,
            item_id=item_id,
        )

        # Check publishing job status

        if (
            buildInitialCache
            and self._gis._portal.is_arcgisonline
            and fileType.lower() in ["tilepackage", "compacttilepackage"]
        ):
            from ..mapping._types import MapImageLayer
            from ..raster._layer import ImageryLayer

            if len(ret) > 0 and "success" in ret[0] and ret[0]["success"] == False:
                raise Exception(ret[0]["error"])
            ms_url = self._gis.content.get(ret[0]["serviceItemId"]).url
            if ms_url.lower().find("mapserver") > -1:
                ms = MapImageLayer(url=ms_url, gis=self._gis)
                manager = ms.manager
            elif ms_url.lower().find("imageserver") > -1:
                ms = ImageryLayer(url=ms_url, gis=self._gis)
                manager = ms.cache_manager
                if not self._gis._portal.is_arcgisonline:
                    return Item(self._gis, ret[0]["serviceItemId"])
            serviceitem_id = ret[0]["serviceItemId"]
            try:
                # first edit the tile service to set min, max scales
                if not ms.properties.minScale:
                    min_scale = ms.properties.tileInfo.lods[0]["scale"]
                    max_scale = ms.properties.tileInfo.lods[-1]["scale"]
                else:
                    min_scale = ms.properties.minScale
                    max_scale = ms.properties.maxScale

                edit_result = manager.edit_tile_service(
                    min_scale=min_scale, max_scale=max_scale
                )

                # Get LoD from Map Image Layer
                full_extent = dict(ms.properties.fullExtent)
                lod_dict = ms.properties.tileInfo["lods"]
                lod = [
                    current_lod["level"]
                    for current_lod in lod_dict
                    if (min_scale <= current_lod["scale"] <= max_scale)
                ]
                ret = manager.update_tiles(levels=lod, extent=full_extent)
            except Exception as tiles_ex:
                raise Exception("Error unpacking tiles :" + str(tiles_ex))
        elif (
            not buildInitialCache
            and output_type is not None
            and output_type.lower() in ["sceneservice"]
        ):
            return Item(self._gis, ret[0]["serviceItemId"])
        elif (
            "success" in ret[0]
            and ret[0]["success"] == False
            and ret[0].get("error", None)
        ):
            raise Exception(ret[0].get("error"))
        elif not buildInitialCache and ret[0]["type"].lower() == "image service":
            return Item(self._gis, ret[0]["serviceItemId"])
        else:
            serviceitem_id = self._check_publish_status(ret, folder)
        return Item(self._gis, serviceitem_id)

    def move(self, folder: str):
        """
        The ``move`` method moves the current item to the name of the folder passed when ``move`` is called.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        folder            Required string. The name of the folder to move the item to.
                          Use '/' for the root folder. For other folders, pass in the
                          folder name as a string, or a dictionary containing the folder ID,
                          such as the dictionary obtained from the folders property.
        ================  ===============================================================

        :return:
            A json object in the following format:
            {
            "success": true | false,
            "itemId": "<item id>",
            "owner": "<owner username>",
            "folder": "<folder id>"
            }

        .. code-block:: python

            # Usage Example

            >>> item.move("C:\Projects\ARCGIS\ArcGis_data\")

        """
        owner_name = self._user_id
        folder_id = None
        if folder is not None:
            if isinstance(folder, str):
                if folder == "/":
                    folder_id = "/"
                else:
                    folder_id = self._portal.get_folder_id(owner_name, folder)
            elif isinstance(folder, dict):
                folder_id = folder["id"]
            else:
                print("folder should be folder name as a string, or dict with id")

        if folder_id is not None:
            ret = self._portal.move_item(
                self.itemid, owner_name, self.ownerFolder, folder_id
            )
            self._hydrate()
            return ret
        else:
            print("Folder not found for given owner")
            return None

    # ----------------------------------------------------------------------
    def create_tile_service(
        self,
        title: str,
        min_scale: float,
        max_scale: float,
        cache_info: Optional[dict[str, Any]] = None,
        build_cache: bool = False,
    ):
        """
        The ``create_tile_service`` method allows publishers and administrators to publish hosted feature
        layers and hosted feature layer views as a tile service.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        title             Required string. The name of the new service.
        ----------------  ---------------------------------------------------------------
        min_scale         Required float. The smallest scale at which to view data.
        ----------------  ---------------------------------------------------------------
        max_scale         Required float. The largest scale at which to view data.
        ----------------  ---------------------------------------------------------------
        cache_info        Optional dictionary. If not none, administrator provides the
                          tile cache info for the service. The default is the ArcGIS Online scheme.
        ----------------  ---------------------------------------------------------------
        build_cache       Optional boolean. Default is False; if True, the cache will be
                          built at publishing time.  This will increase the time it takes
                          to publish the service.
        ================  ===============================================================

        :return:
           The :class:`~arcgis.gis.Item` object if successfully added, None if unsuccessful.

        .. code-block:: python

            # Usage Example

            >>> item.create_tile_service(title="SeasideHeightsNJTiles", min_scale= 70000.0,max_scale=80000.0)

        """

        if self.type.lower() == "Feature Service".lower():
            p = self.layers[0].container
            if cache_info is None:
                cache_info = {
                    "spatialReference": {"latestWkid": 3857, "wkid": 102100},
                    "rows": 256,
                    "preciseDpi": 96,
                    "cols": 256,
                    "dpi": 96,
                    "origin": {"y": 20037508.342787, "x": -20037508.342787},
                    "lods": [
                        {
                            "level": 0,
                            "scale": 591657527.591555,
                            "resolution": 156543.033928,
                        },
                        {
                            "level": 1,
                            "scale": 295828763.795777,
                            "resolution": 78271.5169639999,
                        },
                        {
                            "level": 2,
                            "scale": 147914381.897889,
                            "resolution": 39135.7584820001,
                        },
                        {
                            "level": 3,
                            "scale": 73957190.948944,
                            "resolution": 19567.8792409999,
                        },
                        {
                            "level": 4,
                            "scale": 36978595.474472,
                            "resolution": 9783.93962049996,
                        },
                        {
                            "level": 5,
                            "scale": 18489297.737236,
                            "resolution": 4891.96981024998,
                        },
                        {
                            "level": 6,
                            "scale": 9244648.868618,
                            "resolution": 2445.98490512499,
                        },
                        {
                            "level": 7,
                            "scale": 4622324.434309,
                            "resolution": 1222.99245256249,
                        },
                        {
                            "level": 8,
                            "scale": 2311162.217155,
                            "resolution": 611.49622628138,
                        },
                        {
                            "level": 9,
                            "scale": 1155581.108577,
                            "resolution": 305.748113140558,
                        },
                        {
                            "level": 10,
                            "scale": 577790.554289,
                            "resolution": 152.874056570411,
                        },
                        {
                            "level": 11,
                            "scale": 288895.277144,
                            "resolution": 76.4370282850732,
                        },
                        {
                            "level": 12,
                            "scale": 144447.638572,
                            "resolution": 38.2185141425366,
                        },
                        {
                            "level": 13,
                            "scale": 72223.819286,
                            "resolution": 19.1092570712683,
                        },
                        {
                            "level": 14,
                            "scale": 36111.909643,
                            "resolution": 9.55462853563415,
                        },
                        {
                            "level": 15,
                            "scale": 18055.954822,
                            "resolution": 4.77731426794937,
                        },
                        {
                            "level": 16,
                            "scale": 9027.977411,
                            "resolution": 2.38865713397468,
                        },
                        {
                            "level": 17,
                            "scale": 4513.988705,
                            "resolution": 1.19432856685505,
                        },
                        {
                            "level": 18,
                            "scale": 2256.994353,
                            "resolution": 0.597164283559817,
                        },
                        {
                            "level": 19,
                            "scale": 1128.497176,
                            "resolution": 0.298582141647617,
                        },
                        {
                            "level": 20,
                            "scale": 564.248588,
                            "resolution": 0.14929107082380833,
                        },
                        {
                            "level": 21,
                            "scale": 282.124294,
                            "resolution": 0.07464553541190416,
                        },
                        {
                            "level": 22,
                            "scale": 141.062147,
                            "resolution": 0.03732276770595208,
                        },
                    ],
                }
            pp = {
                "minScale": min_scale,
                "maxScale": max_scale,
                "name": title,
                "tilingSchema": {
                    "tileCacheInfo": cache_info,
                    "tileImageInfo": {
                        "format": "PNG32",
                        "compressionQuality": 0,
                        "antialiasing": True,
                    },
                    "cacheStorageInfo": {
                        "storageFormat": "esriMapCacheStorageModeExploded",
                        "packetSize": 128,
                    },
                },
                "cacheOnDemand": True,
                "cacheOnDemandMinScale": 144448,
                "capabilities": "Map,ChangeTracking",
            }
            params = {
                "f": "json",
                "outputType": "tiles",
                "buildInitialCache": build_cache,
                "itemid": self.itemid,
                "filetype": "featureService",
                "publishParameters": json.dumps(pp),
            }
            url = "%s/content/users/%s/publish" % (
                self._portal.resturl,
                self._user_id,
            )
            res = self._gis._con.post(url, params)
            serviceitem_id = self._check_publish_status(res["services"], folder=None)
            if self._gis._portal.is_arcgisonline:
                from ..mapping._types import MapImageLayer

                ms_url = self._gis.content.get(serviceitem_id).url
                ms = MapImageLayer(url=ms_url, gis=self._gis)
                extent = ",".join(
                    [
                        str(ms.properties["fullExtent"]["xmin"]),
                        str(ms.properties["fullExtent"]["ymin"]),
                        str(ms.properties["fullExtent"]["xmax"]),
                        str(ms.properties["fullExtent"]["ymax"]),
                    ]
                )
                lods = []
                for lod in cache_info["lods"]:
                    if lod["scale"] <= min_scale and lod["scale"] >= max_scale:
                        lods.append(str(lod["level"]))
                ms.manager.update_tiles(levels=",".join(lods), extent=extent)
            return self._gis.content.get(serviceitem_id)
        else:
            raise ValueError("Input must of type FeatureService")
        return

    # ----------------------------------------------------------------------
    def protect(self, enable: bool = True):
        """
        The ``protect`` method enables or disables delete protection on this item, essentially allowing the item to be
        deleted or protecting it from deletion.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            Optional boolean. Default is True which enables delete
                          protection, False to disable delete protection.
        ================  ===============================================================

        :return:
            A json object in the following format:
            {"success": true | false}

        """

        try:
            folder = self.ownerFolder
        except:
            folder = None
        res = self._portal.protect_item(self.itemid, self._user_id, folder, enable)
        self._hydrated = False
        self._hydrate()
        return res

    # ----------------------------------------------------------------------
    def _check_publish_status(self, ret, folder):
        """Internal method to check the status of a publishing job.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        ret                 Required dictionary. Represents the result of a publish REST call.
                            This dict should contain the `serviceItemId` and `jobId` of the publishing job.
        ---------------     --------------------------------------------------------------------
        folder              Required string. Obtained from self.ownerFolder
        ===============     ====================================================================


        :return:
           The status.
        """

        import time

        try:
            serviceitem_id = ret[0]["serviceItemId"]
        except KeyError as ke:
            raise RuntimeError(ret[0]["error"]["message"])

        if "jobId" in ret[0]:
            job_id = ret[0]["jobId"]
            path = "content/users/" + self.owner
            if folder is not None:
                path = path + "/" + folder + "/"

            path = path + "/items/" + serviceitem_id + "/status"
            params = {"f": "json", "jobid": job_id}
            job_response = self._portal.con.post(path, params)

            # Query and report the Analysis job status.
            #
            num_messages = 0
            # print(str(job_response))
            if "status" in job_response:
                while not job_response.get("status") == "completed":
                    time.sleep(5)

                    job_response = self._portal.con.post(path, params)

                    # print(str(job_response))
                    if job_response.get("status") in (
                        "esriJobFailed",
                        "failed",
                    ):
                        raise Exception("Job failed.")
                    elif job_response.get("status") == "esriJobCancelled":
                        raise Exception("Job cancelled.")
                    elif job_response.get("status") == "esriJobTimedOut":
                        raise Exception("Job timed out.")
            elif (
                not "jobId" in ret[0]
                and "serviceItemId" in ret[0]
                and ret[0]["type"] == "Map Service"
            ):
                return ret[0]["serviceItemId"]
            else:
                raise Exception("No job results.")
        else:
            raise Exception("No job id")

        return serviceitem_id

    # ----------------------------------------------------------------------
    @property
    def comments(self):
        """
        The ``comments`` property gets a list of comments for a given item.
        """
        from .._impl.comments import Comment

        cs = []
        start = 1
        num = 100
        nextStart = 0
        url = "%s/sharing/rest/content/items/%s/comments" % (
            self._portal.url,
            self.id,
        )
        while nextStart != -1:
            params = {"f": "json", "start": start, "num": num}
            res = self._portal.con.post(url, params)
            for c in res["comments"]:
                cs.append(
                    Comment(
                        url="%s/%s" % (url, c["id"]),
                        item=self,
                        initialize=True,
                    )
                )
            start += num
            nextStart = res["nextStart"]
        return cs

    # ----------------------------------------------------------------------
    def add_comment(self, comment: str):
        """
        The ``add_comment`` method adds a comment to an item.

        .. note::
            The ``add_comment`` method is only available only to authenticated users who have access to the item.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        comment             Required string. Text to be added as a comment to a specific item.
        ===============     ====================================================================

        :return:
           Comment ID if successful, None if failure occurs.

        .. code-block:: python

            # Usage Example

            >>> item.add_comment("Detailed Comment on the Item")
        """
        params = {"f": "json", "comment": comment}
        url = "%s/sharing/rest/content/items/%s/addComment" % (
            self._portal.url,
            self.id,
        )
        try:
            res = self._portal.con.post(url, params)
            if "commentId" in res:
                return res["commentId"]
            return None
        except Exception as e:
            if e.args[0].find("Too many failures") > -1:
                raise RuntimeError(
                    "The number of comments allowed has been exceeded, please wait to post more comments"
                )
            else:
                raise e

    # ----------------------------------------------------------------------
    @property
    def rating(self):
        """
        Get/Set the rating given by the current user to the item.
        Set adds a rating to an item to which you have access - Only one rating
        can be given to an item per user. If this call is made on a
        currently rated item, the new rating will overwrite the existing
        rating. A user cannot rate their own item. Available only to
        authenticated users.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required float. The rating to be applied for the item. The value
                            must be a floating point number between 1.0 and 5.0.
        ===============     ====================================================================
        """
        url = "%s/sharing/rest/content/items/%s/rating" % (
            self._portal.url,
            self.id,
        )
        params = {"f": "json"}
        res = self._portal.con.get(url, params)
        if "rating" in res:
            return res["rating"]
        return None

    # ----------------------------------------------------------------------
    @rating.setter
    def rating(self, value: float):
        """
        See main ``rating`` property docstring
        """
        url = "%s/sharing/rest/content/items/%s/addRating" % (
            self._portal.url,
            self.id,
        )
        params = {"f": "json", "rating": float(value)}
        self._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def delete_rating(self):
        """
        The ``delete_rating`` method removes the rating the calling user added for the specified item.
        """
        url = "%s/sharing/rest/content/items/%s/deleteRating" % (
            self._portal.url,
            self.id,
        )
        params = {"f": "json"}
        res = self._portal.con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    @property
    def proxies(self):
        """
        The ``proxies`` property gets the ArcGIS Online hosted proxy services, set on a registered app,
        item with the Registered App type keyword. Additionally, this resource is only
        available to the item owner and the organization administrator.
        """
        url = "%s/sharing/rest/content/users/%s/items/%s/proxies" % (
            self._portal.url,
            self._user_id,
            self.id,
        )
        params = {"f": "json"}
        ps = []
        try:
            res = self._portal.con.get(url, params)
            if "appProxies" in res:
                for p in res["appProxies"]:
                    ps.append(p)
        except:
            return []
        return ps

    # ----------------------------------------------------------------------
    def _create_proxy(
        self,
        url: Optional[str] = None,
        hit_interval: Optional[int] = None,
        interval_length: int = 60,
        proxy_params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        A service proxy creates a new endpoint for a service that is
        specific to your application. Only allowed domains that you
        specify will be able to access the service.

        ===================    ===============================================================
        **Parameter**           **Description**
        -------------------    ---------------------------------------------------------------
        url                    Optional string. Represents the hosted service URLs to proxy.
        -------------------    ---------------------------------------------------------------
        hit_interval           Optional Integer. Number of times a service can be used in the
                               given interval_length.
        -------------------    ---------------------------------------------------------------
        interval_length        Optional Integer. The time gap for the total hit_interval that
                               a service can be used.  The number is in seconds.
        -------------------    ---------------------------------------------------------------
        proxy_params           Optional dict. Dictionary that provides referrer checks when
                               accessing the premium content and optionally rate limiting if
                               it is not set for each service in proxies.
                                Example:

                                {
                                   "referrers": ["http://foo.example.com", "http://bar.example.com"],
                                   "hitsPerInterval": 1000,
                                   "intervalSeconds": 60
                                }
        ===================    ===============================================================


        :return: Item

        """
        url = "%s/sharing/rest/content/users/%s/items/%s/createProxies" % (
            self._portal.url,
            self._user_id,
            self.id,
        )
        params = {"f": "json", "proxies": [], "serviceProxyParams": {}}
        if url and hit_interval and interval_length:
            params["proxies"].append(
                {
                    "sourceUrl": url,
                    "hitPerInterval": hit_interval,
                    "intervalSeconds": interval_length,
                }
            )
        if proxy_params is not None:
            params["serviceProxyParams"] = proxy_params
        res = self._portal.con.post(url, params)
        return Item(gis=self._gis, itemid=res["id"])

    # ----------------------------------------------------------------------
    def _delete_proxy(self, proxy_id: str) -> dict:
        """
        The delete proxies removes a hosted proxies set on an item. The
        operation can only be made by the item owner or the organization
        administrator.

        ===================    ===============================================================
        **Parameter**           **Description**
        -------------------    ---------------------------------------------------------------
        proxy_id               Required string. This is a comma seperated list of proxy ids.
        ===================    ===============================================================


        :return: dict

        """
        params = {"f": "json", "proxies": proxy_id}
        url = "%s/sharing/rest/content/users/%s/items/%s/deleteProxies" % (
            self._portal.url,
            self._user_id,
            self.id,
        )
        return self._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def copy_item(
        self,
        *,
        title: Optional[str] = None,
        tags: Optional[Union[list[str], str]] = None,
        folder: Optional[str] = None,
        include_resources: bool = False,
        include_private: bool = False,
    ):
        """
        The ``copy_item`` operation creates a new :class:`~arcgis.gis.Item` that is a copy of the original
        :class:`~arcgis.gis.Item` on the server side. It is
        quite similar to the ``copy`` method, but only creates a new :class:`~arcgis.gis.Item`.

        The `copy_item` method is allowed for the following:

           1. Original item being copied is owned by the user invoking the copy operation.
           2. The :class:`~arcgis.gis.User` object is an administrator.
           3. The :class:`~arcgis.gis.User` object has itemControl update capability.

        Additionally, there are several caveats to the ``copy_item`` method. First, the new item created by the
        ``copy_item`` operation will have a system generated itemID. Additionally, hosted services are copied as
        reference only. Reserved keywords, ratings, views, comments and listing properties are reset for the new item.
        Sharing access of the original item is not preserved and sharing access of new item is set to private.
        Lastly, relationships and dependencies of the original item are not maintained in the new item.

        .. note::
            This method is only available on ArcGIS Online or ArcGIS Enterprise 10.9 or higher

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        title                      Optional string. The title of the destination item. If not specified, title of the original item is used.
        -----------------------    -------------------------------------------------------------
        tags                       Optional String. New set of tags (comma separated) of the destination item.
        -----------------------    -------------------------------------------------------------
        folder                     Optional String. Folder Id of the destination item. If the folder Id is not specified, then the item remains in the same folder.

                                   If the administrator invokes a copy of an item belonging to another user, and does not specify the folder Id, the item gets created in the root folder of the administrator.
        -----------------------    -------------------------------------------------------------
        include_resources    Optional boolean. If true, the file resources of the original
                                   item will be copied over to the new item. Private file resources
                                   will not be copied over. If false, the file resources of the
                                   original item will not be copied over to the new item. The
                                   default is false.

        -----------------------    -------------------------------------------------------------
        include_private            If true, and if `include_resources` is set to true as well, then
                                   the private resources of the original item will be copied over to
                                   the new item. If false, the private file resources of the original
                                   item will not be copied over to the new item. The default is false.
        =======================    =============================================================



        :return: An :class:`~arcgis.gis.Item` object
        """
        if tags and type(tags) is list:
            tags = ",".join(tags)

        url = "%s/sharing/rest/content/users/%s/items/%s/copy" % (
            self._portal.url,
            self._user_id,
            self.id,
        )
        params = {
            "f": "json",
            "title": title,
            "tags": tags,
            "includeResources": include_resources,
            "copyPrivateResources": include_private,
        }
        res = self._portal.con.post(url, params)
        if "itemId" in res:
            return self._gis.content.get(res["itemId"])
        elif "id" in res:
            return self._gis.content.get(res["id"])
        else:
            return res

    # ----------------------------------------------------------------------
    def copy(
        self,
        title: Optional[str] = None,
        tags: Optional[Union[list[str], str]] = None,
        snippet: Optional[str] = None,
        description: Optional[str] = None,
        layers: Optional[list[int]] = None,
    ):
        """
        The ``copy`` method allows for the creation of an item that is derived from the current Item.

        For layers, ``copy`` will create a new item that uses the URL as a reference.
        For non-layer based items, these will be copied and the exact same data will be
        provided.

        If title, tags, snippet of description is not provided the values from `item` will be used.
        The ``copy`` method can be used in a variety of situations, such as:
            1. Vector tile service sprite customization
            2. Limiting feature service exposure
            3. Sharing content by reference with groups
            4. Creating backup items.


        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        title                      Optional string. The name of the new item.
        -----------------------    -------------------------------------------------------------
        tags                       Optional list of string. Descriptive words that help in the
                                   searching and locating of the published information.
        -----------------------    -------------------------------------------------------------
        snippet                    Optional string. A brief summary of the information being
                                   published.
        -----------------------    -------------------------------------------------------------
        description                Optional string. A long description of the Item being
                                   published.
        -----------------------    -------------------------------------------------------------
        layers                     Optional list of integers.  If you have a layer with multiple
                                   and you only want specific layers, an index can be provided
                                   those layers.  If nothing is provided, all layers will be
                                   visible.

                                   .. code-block:: python
                                        # Example Usage #1:
                                        >>> item.copy(title="Atlantic_Hurricanes",
                                        >>>           layers=[0,3])
                                       # Example Usage #2:
                                        >>> item.copy(title="Weather_Data",
                                        >>>          layers = [9])
        =======================    =============================================================

        :return: An :class:`~arcgis.gis.Item` object

         .. code-block:: python

            **Usage Example**

            >>> item.copy()
            <Item title:"gisslideshow - Copy 94452b" type:Microsoft Powerpoint owner:geoguy>

            >>> item.copy(title="GIS_Tutorial")
            <Item title:"GIS_Tutorial" type:Microsoft Powerpoint owner:geoguy>

            >>> item.copy()
            <Item title:"NZTiles - Copy 021a06" type:Vector Tile Layer owner:geoguy>


        """
        TEXT_BASED_ITEM_TYPES = [
            "Web Map",
            "Web Scene",
            "360 VR Experience",
            "Operation View",
            "Workforce Project",
            "Insights Model",
            "Insights Page",
            "Dashboard",
            "Feature Collection",
            "Insights Workbook",
            "Feature Collection Template",
            "Hub Initiative",
            "Hub Site Application",
            "Hub Page",
            "Web Mapping Application",
            "Mobile Application",
            "Symbol Set",
            "Color Set",
            "Content Category Set",
            "Windows Viewer Configuration",
        ]
        FILE_BASED_ITEM_TYPES = [
            "Notebook",
            "CityEngine Web Scene",
            "Pro Map",
            "Map Area",
            "KML Collection",
            "Code Attachment",
            "Operations Dashboard Add In",
            "Native Application",
            "Native Application Template",
            "KML",
            "Native Application Installer",
            "Form",
            "AppBuilder Widget Package",
            "File Geodatabase",
            "CSV",
            "Image",
            "Locator Package",
            "Map Document",
            "Shapefile",
            "Microsoft Word",
            "PDF",
            "CAD Drawing",
            "Service Definition",
            "Image",
            "Visio Document",
            "iWork Keynote",
            "iWork Pages",
            "iWork Numbers",
            "Report Template",
            "Statistical Data Collection",
            "SQLite Geodatabase",
            "Mobile Basemap Package",
            "Project Package",
            "Task File",
            "ArcPad Package",
            "Explorer Map",
            "Globe Document",
            "Scene Document",
            "Published Map",
            "Map Template",
            "Windows Mobile Package",
            "Layout",
            "Project Template",
            "Layer",
            "Explorer Package",
            "Image Collection",
            "Desktop Style",
            "Geoprocessing Sample",
            "Locator Package",
            "Rule Package",
            "Raster function template",
            "ArcGIS Pro Configuration",
            "Workflow Manager Package",
            "Desktop Application",
            "Desktop Application Template",
            "Code Sample",
            "Desktop Add In",
            "Explorer Add In",
            "ArcGIS Pro Add In",
            "Microsoft Powerpoint",
            "Microsoft Excel",
            "Layer Package",
            "Mobile Map Package",
            "Geoprocessing Package",
            "Scene Package",
            "Tile Package",
            "Vector Tile Package",
        ]
        SERVICE_BASED_ITEM_TYPES = [
            "Vector Tile Service",
            "Scene Service",
            "WMS",
            "WFS",
            "WMTS",
            "Geodata Service",
            "Globe Service",
            "Scene Service",
            "Relational Database Connection",
            "AppBuilder Extension",
            "Document Link",
            "Geometry Service",
            "Geocoding Service",
            "Network Analysis Service",
            "Geoprocessing Service",
            "Workflow Manager Service",
            "Image Service",
            "Map Service",
            "Feature Service",
        ]
        item = self
        from datetime import timezone
        from uuid import uuid4

        now = datetime.now(timezone.utc)
        if title is None:
            title = item.title + " - Copy %s" % uuid4().hex[:6]
        if tags is None:
            tags = item.tags
        if snippet is None:
            snippet = item.snippet
        if description is None:
            description = item.description

        if (
            item.type in SERVICE_BASED_ITEM_TYPES
            or item.type == "KML"
            and item.url is not None
        ):
            params = {
                "f": "json",
                "item": item.title.replace(" ", "_")
                + "-_copy_%s" % int(now.timestamp() * 1000),
                "type": item.type,
                "url": item.url,
                "typeKeywords": ",".join(item.typeKeywords),
            }

            params["title"] = title
            params["tags"] = ",".join(tags)
            params["snippet"] = snippet
            params["description"] = description
            if not layers is None:
                text = {"layers": []}
                lyrs = item.layers
                for idx, lyr in enumerate(lyrs):
                    if idx in layers:
                        text["layers"].append(
                            {
                                "layerDefinition": {"defaultVisibility": True},
                                "id": idx,
                            }
                        )
                params["text"] = text
            url = "%s/content/users/%s/addItem" % (
                self._gis._portal.resturl,
                self._user_id,
            )
            res = self._gis._con.post(url, params)
            if "id" in res:
                itemid = res["id"]
            else:
                return None

            if itemid is not None:
                return Item(self._gis, itemid)
            else:
                return None
        elif item.type.lower() == "notebook":
            with tempfile.TemporaryDirectory() as d:
                import shutil

                fp = item.download(save_path=d)
                if fp.find("..") > -1:
                    shutil.copy(fp, fp.replace("..", "."))
                    fp = fp.replace("..", ".")
                sfp = os.path.split(fp)
                fname, ext = os.path.splitext(sfp[1])
                ext = ext.replace(".", "")
                nfp = os.path.join(sfp[0], "%s_%s.%s" % (fname, uuid4().hex[:5], ext))
                os.rename(fp, nfp)
                ip = {
                    "type": item.type,
                    "tags": ",".join(item.tags),
                    "snippet": snippet,
                    "description": description,
                    "typeKeywords": ",".join(item.typeKeywords),
                    "title": title,
                }

                item = self._gis.content.add(item_properties=ip, data=nfp)
                return item
        elif item.type in FILE_BASED_ITEM_TYPES:
            fp = self.get_data()
            sfp = os.path.split(fp)
            fname, ext = os.path.splitext(sfp[1])
            nfp = os.path.join(sfp[0], "%s_%s.%s" % (fname, uuid4().hex[:5], ext))
            os.rename(fp, nfp)
            ip = {
                "type": item.type,
                "tags": ",".join(item.tags),
                "snippet": snippet,
                "description": description,
                "typeKeywords": ",".join(item.typeKeywords),
                "title": title,
            }
            item = self._gis.content.add(item_properties=ip, data=nfp)
            os.remove(nfp)
            return item
        elif item.type in TEXT_BASED_ITEM_TYPES:
            data = self.get_data()
            ip = {
                "type": item.type,
                "tags": ",".join(item.tags),
                "typeKeywords": ",".join(item.typeKeywords),
                "snippet": snippet,
                "description": description,
                "text": data,
                "title": title,
            }
            if item.type == "Notebook":
                ip["properties"] = item.properties
            new_item = self._gis.content.add(item_properties=ip)
            if item.url and item.url.find(item.id) > -1:
                new_item.update({"url": item.url.replace(item.id, new_item.id)})
            return new_item

        else:
            raise ValueError("Item of type: %s is not supported by copy" % (item.type))
        return

    # ----------------------------------------------------------------------
    @property
    def dependencies(self):
        """The ``dependencies`` property returns a class to manage an item's
        dependencies.

        :return: :class:`~arcgis.gis.ItemDependency` object.
        """
        if self._depend is None:
            self._depend = ItemDependency(self)
        return self._depend

    # ----------------------------------------------------------------------
    def register(
        self,
        app_type: str,
        redirect_uris: Optional[list[str]] = None,
        http_referers: Optional[list[str]] = None,
        privileges: Optional[list[str]] = None,
    ):
        """

        The ``register`` method registers an app item with the enterprise, resulting in an APPID and APPSECRET
        (also known as client_id and client_secret in OAuth speak, respectively) being
        generated for that app. Upon successful registration, a Registered
        App type keyword gets appended to the app item.

        .. note::
            The ``register`` method is available to the item owner.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        app_type            Required string. The type of app that was registered indicating
                            whether it's a browser app, native app, server app, or a multiple
                            interface app.
                            Values: browser, native, server, or multiple
        ---------------     --------------------------------------------------------------------
        redirect_uris       Optional list.  The URIs where the access_token or authorization
                            code will be delivered upon successful authorization. The
                            redirect_uri specified during authorization must match one of the
                            registered URIs, otherwise authorization will be rejected.

                            A special value of urn:ietf:wg:oauth:2.0:oob can also be specified
                            for authorization grants. This will result in the authorization
                            code being delivered to a portal URL (/oauth2/approval). This
                            value is typically used by apps that don't have a web server or a
                            custom URI scheme where the code can be delivered.

                            The value is a JSON string array.
        ---------------     --------------------------------------------------------------------
        http_referers       Optional List. A list of the http referrers for which usage of the
                            :class:`~arcgis.gis._impl.APIKey` will be restricted to.

                            .. note::
                                Http Referrers can be configured for non apiKey type apps as
                                well. The list configured here will be used to validate the app
                                tokens sent in while accessing the sharing API. The referrer checks
                                will not be applied to user tokens.
        ---------------     --------------------------------------------------------------------
        privileges          Optional List. A list of the privileges that will be available for
                            this :class:`~arcgis.gis._impl.APIKey`.

                            .. note::
                                Privileges can be configured for non  `API Key` type apps as
                                well. The list configured here will be used to grant access to items
                                when item endpoint is accessed with app tokens. The checks will not
                                be applied to user tokens and they can continue accessing items
                                based on the current item sharing model. With app tokens, all items
                                of app owner can be accessed if the privileges list is not
                                configured.
        ===============     ====================================================================

        :return: A dictionary indicating 'success' or 'error'

        .. code-block:: python

            # Usage Example

            >>> item.register(app_type = "browser",
            >>>             redirect_uris = [ "https://app.example.com", "urn:ietf:wg:oauth:2.0:oob" ],
            >>>             http_referers = [ "https://foo.com", "https://bar.com" ],
            >>>            privileges = ["portal:apikey:basemaps", "portal:app:access:item:itemId",
            >>>                         "premium:user:geocode", "premium:user:networkanalysis"]
                              )

        """
        if redirect_uris is None:
            redirect_uris = []
        if str(app_type).lower() not in [
            "browser",
            "native",
            "server",
            "multiple",
            "apikey",
        ]:
            raise ValueError(
                (
                    "Invalid app_type of : %s. Allowed values"
                    ": browser, native, server or multiple." % app_type
                )
            )
        params = {
            "f": "json",
            "itemId": self.id,
            "appType": app_type,
            "redirect_uris": redirect_uris,
        }
        if http_referers:
            params["httpReferrers"] = http_referers
        if privileges:
            params["privileges"] = privileges
        url = "%soauth2/registerApp" % self._portal.resturl
        res = self._portal.con.post(url, params)
        self._hydrated = False
        self._hydrate()
        return res

    # ----------------------------------------------------------------------
    def unregister(self):
        """

        The ``unregister`` property removes the application registration from an app
        Item, along with the Registered App type keyword.

        .. note::
            The ``unregister`` method is available to the item owner and organization administrators.

        :return:
            A boolean indicating success (True), or failure (False)


        """
        appinfo = self.app_info
        if "Registered App" not in self.typeKeywords:
            return False
        if appinfo == {} or len(appinfo) == 0:
            return False
        params = {"f": "json"}
        url = "%soauth2/apps/%s/unregister" % (
            self._portal.resturl,
            appinfo["client_id"],
        )
        res = self._portal.con.post(url, params)
        if res["success"]:
            self._hydrated = False
            self._hydrate()
            return True
        return res["success"]

    # ----------------------------------------------------------------------
    def package_info(self, folder: Optional[str] = None) -> str:
        """
        Items will have a package info file available only if that item is
        an ArcGIS package (for example, a layer package or map package). It
        contains information that is used by clients (ArcGIS Pro, ArcGIS
        Explorer, and so on) to work appropriately with downloaded
        packages. Navigating to the URL will result in a package info file
        (.pkinfo) being downloaded.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Optional string. The save location of the pkinfo file.
        ===============     ====================================================================

        :returns: str
        """

        url = f"{self._gis._portal.resturl}content/items/{self.itemid}/item.pkinfo"
        res = self._portal.con.get(url, {}, try_json=False, out_folder=folder)
        return res

    # ----------------------------------------------------------------------
    @property
    def item_card(self) -> str:
        """
        Returns an XML representation of the Item

        :returns: A string path to the downloaded XML file.
        """
        url = (
            f"{self._gis._portal.resturl}content/items/{self.itemid}/info/iteminfo.xml"
        )
        res = self._portal.con.get(url, {"f": "json"})
        return res

    # ----------------------------------------------------------------------
    @property
    def app_info(self):
        """
        The ``app_info`` property is a resource for accessing application information. If the parent item is registered
        using the register app operation, ``app_info`` returns information pertaining to the registered app.
        Every registered app gets an App ID and App Secret, which are titled client_id and client_secret, respectively,
        in the terminology of OAuth.

        :return: A `Dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_

        """
        if "Registered App" not in self.typeKeywords:
            return {}
        url = "{base}content/users/{user}/items/{itemid}/registeredAppInfo".format(
            base=self._portal.resturl, user=self._user_id, itemid=self.id
        )
        params = {"f": "json"}
        try:
            return self._portal.con.get(url, params)
        except:
            return {}

    # ----------------------------------------------------------------------
    @functools.lru_cache(maxsize=255)
    def _validate_url(
        self, url: str | None = None, return_type: str | None = None
    ) -> str:
        """checks if the URL endpoint is reachable via public and/or private url

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Optional String. URL to check. If `None` it is obtained from the Item['url'] key
        ---------------     --------------------------------------------------------------------
        return_type         Option String. The options are "BOTH", "PUBLIC_ONLY", "PRIVATE_ONLY".
                            When "BOTH" is selected, which is defualt first the public URL is
                            validated if that fails, then the private URL is given. For
                            "PUBLIC_ONLY", the public URL is given back. For "PRIVATE_ONLY" only
                            the private URL is given. If no private URL exists, then the public is
                            returned.

        ===============     ====================================================================

        :returns: string
        """
        if return_type is None:
            return_type = "BOTH"
        elif str(return_type).upper() in [
            "BOTH",
            "PRIVATE_ONLY",
            "PUBLIC_ONLY",
        ]:
            return_type = str(return_type).upper()
        else:
            raise ValueError("Invalid `return_type`.")
        if url is None:
            url = self["url"]
        res = self._gis._private_service_url(url)
        private_url = res.get("privateServiceUrl", None)
        public_url = res.get("serviceUrl", None)
        if return_type == "BOTH":
            if private_url is None and public_url:
                return public_url
            elif private_url is None and public_url is None:
                return url
            elif public_url == private_url:
                return public_url
            elif public_url != private_url:
                for purl in [public_url, private_url]:
                    try:
                        if purl:
                            self._gis._con.get(purl)
                            return purl
                    except:
                        ...
        elif return_type == "PUBLIC_ONLY":
            return public_url
        elif return_type == "PRIVATE_ONLY":
            return private_url or public_url
        return url


########################################################################
class ViewManager:
    """
    A helper class to work with hosted feature layer views created from
    :class:`items <arcgis.gis.Item>` whose `type` property value is ``feature
    service.``

    This class is not meant to be created directly, but instead returned
    from the :attr:`~arcgis.gis.Item.view_manager` property on an
    :class:`~arcgis.gis.Item`.
    """

    _item = None
    _gis = None

    def __init__(self, item: Item):
        self._item = item
        self._gis = item._gis

    # ----------------------------------------------------------------------
    def list(self) -> list[Item]:
        """
        Returns all views for a given item

        :returns:
            List of feature layer view :class:`items <arcgis.gis.Item>`
        """
        return [
            i
            for i in self._item.related_items("Service2Data", "reverse")
            if "View Service" in i.typeKeywords
        ]

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        spatial_reference: dict[str, Any] | None = None,
        extent: dict[str, int | float] | None = None,
        allow_schema_changes: bool = True,
        updateable: bool = True,
        capabilities: str = "Query",
        view_layers: list[int] | None = None,
        view_tables: list[int] | None = None,
        *,
        description: str | None = None,
        tags: str | None = None,
        snippet: str | None = None,
        overwrite: bool | None = None,
        set_item_id: str | None = None,
        preserve_layer_ids: bool = False,
    ) -> Item:
        """
        Creates a view of an existing feature service Item. You can create a view if you need a different view of the data
        represented by a hosted feature layer. For example, you want to apply different editor settings,
        styles or filters or define which features or fields are available, or share the data to different groups than
        the hosted feature layer.

        When you create a feature layer view, a new hosted feature layer :class:`~arcgis.gis.Item` is added to your content. This new layer is a
        view of the data in the :class:`hosted feature layer <arcgis.features.FeatureLayerCollection>`, which means updates made to the data appear in the hosted feature layer and all of its hosted feature layer views. However, since the view is a separate layer, you can change
        properties and settings on this item separately from the hosted feature layer from which it is created.

        For example, you can allow members of your organization to edit the hosted feature layer but share a read-only
        feature layer view with the public.

        See `Create hosted feature layer views <https://doc.arcgis.com/en/arcgis-online/manage-data/create-hosted-views.htm>`_
        to learn more details.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        name                     Required string. Name of the new view item
        --------------------     --------------------------------------------------------------------
        spatial_reference        Optional dict. Specify the spatial reference of the view
        --------------------     --------------------------------------------------------------------
        extent                   Optional dict. Specify the extent of the view
        --------------------     --------------------------------------------------------------------
        allow_schema_changes     Optional bool. Default is True. Determines if a view can alter a
                                 service's schema.
        --------------------     --------------------------------------------------------------------
        updateable               Optional bool. Default is True. Determines if view can update values
        --------------------     --------------------------------------------------------------------
        capabilities             Optional string. Specify capabilities as a comma separated string.
                                 For example "Query, Update, Delete". Default is 'Query'.
        --------------------     --------------------------------------------------------------------
        view_layers              Optional list. Specify list of layers present in the
                                 :class:`~arcgis.features.FeatureLayerCollection` you want in the view.
        --------------------     --------------------------------------------------------------------
        view_tables              Optional list. Specify list of tables present in the
                                 :class:`~arcgis.features.FeatureLayerCollection` you want in the view
        --------------------     --------------------------------------------------------------------
        description              Optional String. A user-friendly description for the published dataset.
        --------------------     --------------------------------------------------------------------
        tags                     Optional String. The comma separated string of descriptive words.
        --------------------     --------------------------------------------------------------------
        snippet                  Optional String. A short description of the view item.
        --------------------     --------------------------------------------------------------------
        overwrite                Optional Boolean.  If true, the view is overwritten, False is the default.
        --------------------     --------------------------------------------------------------------
        set_item_id              Optional String. If set, the ItemId is defined by the user, not the system.
        --------------------     --------------------------------------------------------------------
        preserve_layer_ids       Optional Boolean. Preserves the layer's `id` on it's definition when `True`.
                                 The default is `False`.
        ====================     ====================================================================

        .. code-block:: python

            # USAGE EXAMPLE: Create a veiw from a hosted feature layer

            >>> crime_fl_item = gis.content.search("2012 crime")[0]
            >>> view = crime_fl_item.view_manager.create(name=uuid.uuid4().hex[:9], # create random name
                                                         updateable=True,
                                                         allow_schema_changes=False,
                                                         capabilities="Query,Update,Delete")

        :return:
            The :class:`~arcgis.gis.Item` for the view.
        """
        flc = arcgis.features.FeatureLayerCollection.fromitem(self._item)
        mgr = flc.manager
        return mgr.create_view(
            name=name,
            spatial_reference=spatial_reference,
            extent=extent,
            allow_schema_changes=allow_schema_changes,
            updateable=updateable,
            capabilities=capabilities,
            view_layers=view_layers,
            view_tables=view_tables,
            description=description,
            tags=tags,
            snippet=snippet,
            overwrite=overwrite,
            set_item_id=set_item_id,
            preserve_layer_ids=preserve_layer_ids,
        )

    # ----------------------------------------------------------------------
    def get_definitions(self, item: Item) -> list[ViewLayerDefParameter]:
        """Gets the View Definition Parameters for a Given Item

        =============     =====================================================
        **Argument**      **Description**
        -------------     -----------------------------------------------------
        item              The :class:`~arcgis.gis.Item` to return the
                          view layer definitions for.
        =============     =====================================================


        :return:
            List of :class:`~arcgis.gis._impl._dataclasses.ViewLayerDefParameter`
            objects or None.
        """
        if "View Service" in item.typeKeywords:
            from arcgis.gis._impl._dataclasses import ViewLayerDefParameter

            services = item.layers + item.tables
            return [ViewLayerDefParameter.fromlayer(lyr) for lyr in services]
        return []

    # ----------------------------------------------------------------------
    def update(self, layer_def: list[ViewLayerDefParameter] | None = None) -> bool:
        """
        Updates a set of layers with new queries, geometries, and column visibilities.

        =============     =====================================================
        **Argument**      **Description**
        -------------     -----------------------------------------------------
        layer_def         List of
                          :class:`~arcgis.gis._impl._dataclasses.ViewLayerDefParameter`
                          objects for modifying the layers.
        =============     =====================================================


        :returns: Boolean
        """
        results = []
        assert isinstance(layer_def, (list, tuple))
        for lyrdef in layer_def:
            assert isinstance(lyrdef, ViewLayerDefParameter)
            layer = lyrdef.layer
            assert isinstance(layer, arcgis.features.FeatureLayer)
            if "isView" in lyrdef.layer.properties and lyrdef.layer.properties.isView:
                results.append(
                    {
                        layer._url: layer.container.manager.update_definition(
                            lyrdef.as_json()
                        )
                    }
                )
            else:
                raise ValueError("The layer is not a view.")
            del lyrdef
        return results


########################################################################
class ItemDependency(object):
    """
    Manage, monitor, and control Item dependencies.

    Dependencies allow users to better understand the relationships between
    their organizational items. This class provides the users with the following:

    - Warnings during item deletion if the deletion is going to break item/layer references in a web map or web application.
    - Ability to explore the dependents and dependencies of a specific item.
    - Administrator ability to efficiently update the URLs of their hosted/federated services in a single edit operation.

    When an item is updated, its dependencies are updated as well and always be kept in sync.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Required Item. Item object to examine dependencies on.
    ===============     ====================================================================

    .. note::
        Instances of this class are not created directly. Objects of this type
        are returned by the :attr:`~arcgis.gis.Item.dependencies` property on
        :class:`~arcgis.gis.Item` objects.

    """

    _url = None
    _gis = None
    _con = None
    _item = None
    _portal = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, item):
        """Constructor"""
        self._item = item
        self._gis = item._gis
        self._con = self._gis._con
        self._portal = self._gis._portal
        self._url = "%scontent/items/%s/dependencies" % (
            self._portal.resturl,
            item.itemid,
        )

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<Dependencies for %s>" % self._item.itemid

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<Dependencies for %s>" % self._item.itemid

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(dict(self.properties)["items"])

    # ----------------------------------------------------------------------
    def _init(self):
        params = {"f": "json", "num": 100, "start": 0}
        res = self._con.get(self._url, params)
        items = res["list"]
        start = 0
        num = 100
        while res["nextStart"] > -1:
            start += num
            params = {"f": "json", "num": 100, "start": res["nextStart"]}
            res = self._con.get(self._url, params)
            if "list" in res:
                items += res["list"]
        self._properties = _mixins.PropertyMap({"items": items})

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the dependencies properties"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def add(self, depend_type: str, depend_value: str):
        """
        Assigns a dependency to the current item

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        depend_type         Required String. This is the type of dependency that is registered
                            for the item. Allowed values:

                             - ``table``
                             - ``url``
                             - ``itemid``
                             - ``serverId``
        ---------------     --------------------------------------------------------------------
        depend_value        Required string. This is the associated value for the type above.
        ===============     ====================================================================

        :return: Boolean

        """
        dtlu = {"table": "table", "url": "url", "itemid": "id"}
        params = {"f": "json", "type": dtlu[depend_type], "id": depend_value}
        url = "%s/addDependency" % self._url
        res = self._con.post(url, params)
        if "error" in res:
            return res
        self._properties = None
        return True

    # ----------------------------------------------------------------------
    def remove(self, depend_type: str, depend_value: str):
        """
        Deletes a dependency to the current item

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        depend_type         Required String. This is the type of dependency that is registered
                            for the item. The allowed values are: table, url, or itemid.
        ---------------     --------------------------------------------------------------------
        depend_value        Required string. This is the associated value for the type above.
        ===============     ====================================================================

        :return: Boolean

        """
        dtlu = {"table": "table", "url": "url", "itemid": "id", "id": "id"}
        params = {"f": "json", "type": dtlu[depend_type], "id": depend_value}
        url = "%s/removeDependency" % self._url
        res = self._con.post(url, params)
        if "error" in res:
            return res
        self._properties = None
        return True

    # ----------------------------------------------------------------------
    def remove_all(self):
        """
        Revokes all dependencies for the current item

        :return: Boolean

        """
        if self._gis.version <= [8, 2]:
            for i in dict(self.properties)["items"]:
                if "url" in i:
                    self.remove(i["dependencyType"], i["id"])
                elif "id" in i:
                    self.remove(i["dependencyType"], i["id"])
                elif "table" in i:
                    self.remove(i["dependencyType"], i["id"])
            self._properties = None
        else:
            url = "%s/removeAllDependencies" % self._url
            params = {"f": "json"}
            res = self._con.post(url, params)
            if "error" in res:
                return res
            self._properties = None
        return True

    # ----------------------------------------------------------------------
    @property
    def to_dependencies(self):
        """
        Returns a list of items that are dependent on the current Item
        """
        url = "%s/listDependentsTo" % self._url
        params = {"f": "json", "num": 100, "start": 0}
        res = self._con.get(url, params)

        items = res["list"]
        num = 100
        while res["nextStart"] > -1:
            params = {"f": "json", "num": 100, "start": res["nextStart"]}
            res = self._con.get(url, params)
            if "list" in res:
                items += res["list"]
        return items


def rot13(s, b64: bool = False, of: bool = False):
    if s is None:
        return None
    result = ""

    # If b64 is True, then first convert back to a string
    if b64:
        try:
            s = base64.b64decode(s).decode()
        except:
            raise RuntimeError(
                "Reading value from profile is not correctly formatted. "
                + "Update by creating a new connection using the profile option."
            )

    # Loop over characters.
    for v in s:
        # Convert to number with ord.
        c = ord(v)

        # Shift number back or forward.
        if c >= ord("a") and c <= ord("z"):
            if c > ord("m"):
                c -= 13
            else:
                c += 13
        elif c >= ord("A") and c <= ord("Z"):
            if c > ord("M"):
                c -= 13
            else:
                c += 13

        # Append to result.
        result += chr(c)

    # Return transformation.
    if of:
        return result
    if not b64:
        # if not base64 to start, need to convert to base64 for saving to file
        return (base64.b64encode(result.encode())).decode()
    else:
        return result


class _GISResource(object):
    """a GIS service"""

    def __init__(self, url, gis=None):
        from ._impl._con import Connection

        self._hydrated = False
        if str(url).lower().endswith("/"):
            url = url[:-1]
        self.url = url
        self._url = url

        if gis is None:
            gis = GIS(set_active=False)
            self._gis = gis
            self._con = gis._con
        else:
            self._gis = gis
            if isinstance(gis, Connection):
                self._con = gis
            else:
                self._con = gis._con

    @classmethod
    def fromitem(cls, item: Item):
        """
        The ``fromitem`` method is used to create a :class:`~arcgis.features.FeatureLayerCollection` from a
        :class:`~arcgis.gis.Item` class.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        item                       A required :class:`~arcgis.gis.Item` object. The item needed to convert to
                                   a :class:`~arcgis.features.FeatureLayerCollection` object.
        ======================     ====================================================================

        :return:
            A :class:`~arcgis.features.FeatureLayerCollection` object.

        """
        if not item.type.lower().endswith("service"):
            raise TypeError("item must be a type of service, not " + item.type)
        return cls(item.url, item._gis)

    def _refresh(self):
        params = {"f": "json"}
        is_raster = False
        if (
            type(self).__name__ == "ImageryLayer"
            or type(self).__name__ == "_ImageServerRaster"
        ):
            is_raster = True
            if self._fn is not None:
                params["renderingRule"] = self._fn
            if hasattr(self, "_uri"):
                if isinstance(self._uri, bytes):
                    if "renderingRule" in params.keys():
                        del params["renderingRule"]
                params["Raster"] = self._uri

        if type(self).__name__ == "VectorTileLayer":  # VectorTileLayer is GET only
            dictdata = self._con.get(self.url, params)  # , token=self._lazy_token)
        else:
            try:
                if is_raster:
                    dictdata = self._con.post(
                        self.url,
                        params,
                        timeout=None,  # token=self._lazy_token,
                    )
                else:
                    dictdata = self._con.post(
                        self.url, params
                    )  # , token=self._lazy_token)
            except Exception as e:
                if hasattr(e, "msg") and e.msg == "Method Not Allowed":
                    dictdata = self._con.get(
                        self.url, params
                    )  # , token=self._lazy_token)
                elif str(e).lower().find("token required") > -1:
                    dictdata = self._con.get(self.url, params)
                else:
                    raise e

        self._lazy_properties = _mixins.PropertyMap(dictdata)

    @property
    def properties(self):
        """
        The ``properties`` property retrieves and set properties of this object.
        """
        if self._hydrated:
            return self._lazy_properties
        else:
            self._hydrate()
            return self._lazy_properties

    @properties.setter
    def properties(self, value):
        self._lazy_properties = value

    def _hydrate(self):
        """Fetches properties and deduces token while doing so"""
        self._lazy_token = None
        err = None

        with _common_utils._DisableLogger():
            try:
                self._refresh()

            except HTTPError as httperror:  # service maybe down
                _log.error(httperror)
                err = httperror
            except RuntimeError as e:
                try:
                    # try as a public server
                    self._lazy_token = None
                    self._refresh()

                except HTTPError as httperror:
                    _log.error(httperror)
                    err = httperror
                except RuntimeError as e:
                    if "Token Required" in e.args[0]:
                        # try token in the provided gis
                        self._lazy_token = self._con.token
                        self._refresh()
            except:
                try:
                    # try as a public server
                    self._lazy_token = None
                    self._refresh()

                except HTTPError as httperror:
                    _log.error(httperror)
                    err = httperror
                except RuntimeError as e:
                    if "Token Required" in e.args[0]:
                        # try token in the provided gis
                        self._lazy_token = self._con.token
                        self._refresh()

        if err is not None:
            raise RuntimeError(
                "HTTPError: this service url encountered an HTTP Error: " + self.url
            )

        self._hydrated = True

    @property
    def _token(self):
        if self._hydrated:
            return self._lazy_token
        else:
            self._hydrate()
            return self._lazy_token

    def __str__(self):
        return '<%s url:"%s">' % (type(self).__name__, self.url)

    def __repr__(self):
        return '<%s url:"%s">' % (type(self).__name__, self.url)

    def _invoke(self, method, **kwargs):
        """Invokes the specified method on this service passing in parameters from the kwargs name-value pairs"""
        url = self._url + "/" + method
        params = {"f": "json"}
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                params[k] = v
                del k, v
        return self._con.post(path=url, postdata=params, token=self._token)


class Layer(_GISResource):
    """
    The ``Layer`` class is a primary concept for working with data in a GIS.

    Users create, import, export, analyze, edit, and visualize layers.

    Layers can be added to and visualized using maps. They act as inputs to and outputs from analysis tools.

    Layers are created by publishing data to a GIS, and are exposed as a broader resource (Item) in the
    GIS. ``Layer`` objects can be obtained through the layers attribute on layer :class:`~arcgis.gis.Item` objects in
    the GIS.
    """

    def __init__(self, url, gis=None):
        super(Layer, self).__init__(url, gis)
        self.filter = None
        self._time_filter = None
        """optional attribute query string to select features to process by geoanalytics or spatial analysis tools"""

    @classmethod
    def fromitem(cls, item: str, index: int = 0):
        """
        The ``fromitem`` method returns the layer at the specified index from a layer :class:`~arcgis.gis.Item` object.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. An item containing layers.
        ------------------     --------------------------------------------------------------------
        index                  Optional int. The index of the layer amongst the item's layers
        ==================     ====================================================================

        :return:
           The layer at the specified index.

        .. code-block:: python

            # Usage Example

            >>> layer.fromitem(item="9311d21a9a2047d19c0faaebd6f2cca6", index=3)
        """
        return item.layers[index]

    @property
    def _lyr_dict(self):
        url = self.url

        lyr_dict = {"type": type(self).__name__, "url": url}
        if self._token is not None:
            lyr_dict["serviceToken"] = self._token

        if self.filter is not None:
            lyr_dict["filter"] = self.filter
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    @property
    def _lyr_json(self):
        url = self.url
        if self._token is not None:  # causing geoanalytics Invalid URL error
            url += "?token=" + self._token

        lyr_dict = {"type": type(self).__name__, "url": url}

        if self.filter is not None:
            lyr_dict["options"] = json.dumps({"definition_expression": self.filter})
        if self._time_filter is not None:
            lyr_dict["time"] = self._time_filter
        return lyr_dict

    @property
    def _lyr_domains(self):
        """
        returns the domain information for any fields in the layer with domains
        """
        domains = []
        for field in [
            field for field in self.properties.fields if field["domain"] != None
        ]:
            field_domain = dict(field.domain)
            field_domain["fieldName"] = field.name
            domains.append({field.name: field_domain})
        return domains


from arcgis.gis._impl._profile import ProfileManager

login_profiles = ProfileManager()
