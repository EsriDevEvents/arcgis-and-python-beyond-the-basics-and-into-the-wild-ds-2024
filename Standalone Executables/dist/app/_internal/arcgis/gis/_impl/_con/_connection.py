"""
Connection Object that uses Python Requests
"""
from arcgis.auth.tools import LazyLoader
from typing import Union
from arcgis.auth.tools._util import check_module_exists

try:
    arcpy = LazyLoader("arcpy", strict=True)

    HASARCPY = True
except ImportError:
    HASARCPY = False
except:
    HASARCPY = False

try:
    requests_gssapi = LazyLoader("requests_gssapi", strict=True)
    HAS_GSSAPI = True
except:
    HAS_GSSAPI = False

import sys

if sys.platform == "win32" and check_module_exists("certifi_win32"):
    try:
        import certifi_win32

        certifi_win32.wincerts.where()

        if certifi_win32.wincerts.verify_combined_pem() == False:
            certifi_win32.generate_pem()

    except ImportError:
        pass
elif check_module_exists("truststore"):  # pragma: no cover
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError as ie:
        pass
    except Exception as e:
        pass

import os
import copy
import json
import uuid
import datetime
import mimetypes
import logging
import warnings
import tempfile
from functools import lru_cache
from urllib.request import urlparse
import requests
from urllib3 import exceptions as _exceptions

# from requests import Session
from requests_toolbelt.downloadutils import stream
from json import JSONDecodeError
from ._helpers import _filename_from_headers, _filename_from_url
from ._authguess import GuessAuth
from arcgis._impl.common._utils import _date_handler
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._isd import InsensitiveDict
from arcgis.auth import EsriSession
from arcgis.auth import (
    EsriBuiltInAuth,
    EsriAPIKeyAuth,
    EsriGenTokenAuth,
    ArcGISProAuth,
    EsriOAuth2Auth,
    EsriUserTokenAuth,
)
from arcgis.auth._auth._notebook import EsriNotebookAuth

try:
    from arcgis.auth import EsriWindowsAuth

    HAS_SSPI = True
except ImportError:
    HAS_SSPI = False

try:
    from arcgis.auth import EsriKerberosAuth

    HAS_KERBEROS = True
except ImportError:
    HAS_KERBEROS = False

from arcgis.auth import EsriBasicAuth

__version__ = "2.2.0.1"

_DEFAULT_TOKEN = uuid.uuid4()
_log = logging.getLogger(__name__)


class Connection(object):
    """
    Universal Connection Object
    """

    _timeout = None
    _refresh_token = None
    _token = None
    _token_url = None
    _create_time = None
    _session = None
    _header = None
    _username = None
    _password = None
    _proxy_url = None
    _proxy_port = None
    _proxy_password = None
    _proxy_username = None
    _verify_cert = None
    _baseurl = None
    baseurl = None
    _referer = None
    _expiration = None
    _auth = None
    _product = None
    _custom_auth = None
    _custom_adapter = None
    legacy = None
    _server_log = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        baseurl: str = None,
        username: str = None,
        password: str = None,
        **kwargs,
    ):
        """initializer

        Optional Kwargs

        client_id=None
        client_secret=None
        expiration=60
        cert_file
        key_file
        proxy_url
        proxy_port
        proxy : Dict - Dict
        verify_cert
        referer
        all_ssl = True
        portal_connection = None
        product = "UNKNOWN", "PORTAL", "SERVER", "AGOL", "GEOEVENT"  # SHOULD BE SET BY GIS object, or Server Object, etc...
        token= None If token, _AUTH = BUILTIN
        token_url
        AUTH keys = HOME, BUILTIN, PRO, ANON, PKI, HANDLER, UNKNOWN (Internal)
        custom_auth = Requests authencation handler
        trust_env = T/F if to ignore netrc files
        legacy boolean. If True the token will be appended to the URL for GET and in the FORM POST.
        timeout:int=600
        use_gen_token = boolean - Uses the GenTokenAuth over EsriBuiltInAuth
        security_kwargs = dict - a set of optional arguments for Kerberos Auth
        """
        from arcgis.gis import GIS

        self._ags_file = kwargs.pop("ags_file", None)
        self._security_kwargs = kwargs.pop("security_kwargs", {})
        self._use_gen_token = kwargs.pop("use_gen_token", False)
        self._is_hosted_nb_home = kwargs.pop("is_hosted_nb_home", False)
        self._proxy = kwargs.pop("proxy", None)
        self._timeout = kwargs.pop("timeout", 600)
        self._all_ssl = kwargs.pop("all_ssl", True)
        self.trust_env = kwargs.pop("trust_env", None)
        self._custom_adapter = kwargs.pop("custom_adapter", None)
        self.legacy = kwargs.pop("legacy", False)
        if baseurl:
            while baseurl.endswith("/"):
                baseurl = baseurl[:-1]
        self._custom_auth = kwargs.pop("custom_auth", None)
        self._product = kwargs.pop("product", "UNKNOWN")
        if baseurl is None:
            self._product = "AGOL"
            baseurl = "https://www.arcgis.com"

        self._baseurl = baseurl
        self._username = username
        self._password = password

        self._expiration = kwargs.pop("expiration", 60) or 60
        if self._expiration < 60:
            _log.warning("Expiration is less than 60 seconds, changing value to 60")
            self._expiration = 60
        self._portal_connection = kwargs.pop(
            "portal_connection", None
        )  # For Federated Objects (Portal Connection)
        if self._ags_file and os.path.isfile(self._ags_file):
            res = arcpy.gp.getStandaloneServerToken(self._ags_file)
            self._referer = res.pop("referer", "http")
            self._baseurl = res.pop("serverUrl", None)
            baseurl = self._baseurl
        elif isinstance(self._portal_connection, GIS):
            self._portal_connection = self._portal_connection._con

        if (
            (self._referer or self._referer is None)
            and self._portal_connection
            and str(self._portal_connection._auth).lower() == "home"
        ):
            self._referer = None
        elif baseurl.lower() == "pro":
            self._auth = "PRO"
            try:
                self._referer = arcpy.GetSigninToken().pop("referer", "http")
            except:
                self._referer = kwargs.pop("referer", "http")
        else:
            self._referer = kwargs.pop("referer", "http")

        self._verify_cert = kwargs.pop("verify_cert", False)  # True)
        if self._verify_cert == False:
            warnings.simplefilter("ignore", _exceptions.InsecureRequestWarning)

        self._cert_file = kwargs.pop("cert_file", None)
        self._key_file = kwargs.pop("key_file", None)
        if "proxy_host" in kwargs:
            self._proxy_url = kwargs.pop("proxy_host", None)
        else:
            self._proxy_url = kwargs.pop("proxy_url", None)
        self._proxy_port = kwargs.pop("proxy_port", None)
        self._proxy_username = kwargs.pop("proxy_username", None)
        self._proxy_password = kwargs.pop("proxy_password", None)
        self._header = {"User-Agent": "Geosaurus/%s" % __version__}
        self._client_id = kwargs.pop("client_id", None)
        self._client_secret = kwargs.pop("client_secret", None)
        self._token_url = kwargs.pop("token_url", None)
        if str(baseurl).lower() == "pro":
            self._auth = "PRO"
            auth_check = [""]
        elif str(baseurl).lower() != "pro":
            auth_check = [""]
        if self._is_hosted_nb_home:
            auth_check = [""]
        elif self._key_file is None and self._cert_file is None:
            auth_check = self._auth_check(baseurl, proxies=self._assemble_proxy())
        else:
            auth_check = [""]
        if self._is_hosted_nb_home:
            self._auth = "HOME"  # NB AUTH
            self._token = kwargs.pop("token", None)
            self._expiration = 10080
            self._referer = ""
        elif self._ags_file:
            self._auth = "AGS_AUTH"  # AGS AUTH
        elif "token" in kwargs and kwargs["token"]:
            self._auth = "USER_TOKEN"
            self._token = kwargs.pop("token", None)
        elif "api_key" in kwargs and kwargs["api_key"]:
            self._auth = "API_KEY"
            self._api_key = kwargs.pop("api_key", None)
        elif (
            self._key_file is None
            and self._key_file is None
            and username is None
            and password is None
            and self._portal_connection is None
            and self._client_id is None
            and str(baseurl).lower() != "pro"
            and any(
                [
                    a in auth_check
                    for a in [
                        "Negotiate",
                        "NTLM",
                        "Negotiate, NTLM",
                        "Basic",
                        "basic",
                    ]
                ]
            )
            == False
        ):
            self._auth = "ANON"
        elif self._client_id:
            self._auth = "OAUTH"
        elif (not username is None and not password is None) and (
            "Negotiate" in auth_check or "Negotiate, NTLM" in auth_check
        ):
            self._auth = "KERBEROS"
        elif (username is None and password is None) and (
            "Negotiate" in auth_check or "Negotiate, NTLM" in auth_check
        ):
            self._auth = "KERBEROS"
        elif (not username is None and not password is None) and len(
            username.split("\\")
        ) > 1:
            self._auth = "IWA"
        elif (
            not username is None
            and not password is None
            and any([ac.lower().find("basic") > -1 for ac in auth_check])
        ):
            self._auth = "BASIC_REALM"
        elif (
            not username is None
            and not password is None
            and any([ac.lower().find("ntlm") > -1 for ac in auth_check])
        ):
            self._auth = "NTLM"
        elif (not username is None and not password is None) or (
            self._portal_connection and self._portal_connection._auth == "BUILTIN"
        ):
            self._auth = "BUILTIN"
        elif (not username is None and not password is None) or (
            self._portal_connection and self._portal_connection._auth == "BASIC_REALM"
        ):
            self._auth = "BASIC_REALM"
        elif (not username is None and not password is None) or (
            self._portal_connection and self._portal_connection._auth == "NTLM"
        ):
            self._auth = "NTLM"
        elif (
            (username and password)
            and self._client_id is None
            and str(baseurl).lower() != "pro"
        ):
            self._auth = "BUILTIN"
        elif self._portal_connection:
            self._auth = "BUILTIN"
        elif baseurl.lower() == "pro":
            self._auth = "PRO"
            portal_url = arcpy.GetActivePortalURL()
            if portal_url.lower().find("/sharing/rest") == -1:
                if arcpy.GetActivePortalURL().endswith("/"):
                    self._baseurl = arcpy.GetActivePortalURL() + "sharing/rest"
                else:
                    self._baseurl = arcpy.GetActivePortalURL() + "/sharing/rest"
            else:
                self._baseurl = arcpy.GetActivePortalURL()
        elif self._cert_file or (self._cert_file and self._key_file):
            self._auth = "PKI"

        if self._portal_connection and self._portal_connection._auth in [
            "BASIC_REALM",
            "IWA",
            "NTLM",
            "KERBEROS",
            "PKI",
        ]:
            self._session = self._portal_connection._session
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._create_session()

        #  Product Info
        if self._client_id:
            self._product = "PORTAL"
        elif self._is_hosted_nb_home:
            self._product = "NOTEBOOK_SERVER"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._product = self._check_product()
        self._baseurl = self._validate_url(self._baseurl)
        self.baseurl = self._baseurl

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=10)
    def _parsed(self, url: str):
        return urlparse(url)

    # ----------------------------------------------------------------------
    def _auth_check(self, url: str, proxies: dict = None) -> list:
        import requests

        if str(url).lower() == "pro":
            return [""]
        if self._cert_file and self._key_file:
            cert = (self._cert_file, self._key_file)

        elif self._cert_file and self._password:
            from arcgis.gis._impl._con._cert import pfx_to_pem

            self._key_file, self._cert_file = pfx_to_pem(
                pfx_path=self._cert_file, pfx_password=self._password
            )
            cert = (self._cert_file, self._key_file)
        elif self._cert_file:
            cert = self._cert_file
        else:
            cert = None
        with requests.Session() as s:
            s.cert = cert
            s.verify = self._verify_cert
            s.trust_env = True
            if self._custom_adapter:
                for k, v in self._custom_adapter.items():
                    s.mount(k, v)
            if self._custom_auth:
                s.auth = self._custom_auth
            parsed = self._parsed(url)
            root = (
                rf"{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split(r'/')[0]}"
            )
            params = {"f": "json"}
            results = []
            for pt in [
                "/info",
                "/rest/info",
                "/sharing/rest/info",
                "/rest/services",
            ]:
                try:
                    www_auth = s.get(
                        root + pt,
                        params=params,
                        verify=self._verify_cert,
                        proxies=proxies,
                    ).headers.get("www-authenticate", "")
                    results.append(www_auth)
                except:
                    results.append("")
        return list(set(results))

    # ----------------------------------------------------------------------
    def _validate_url(self, url: str) -> str:
        """ensures the base url has the /sharing/rest"""
        if url.lower().find("arcgis.com") > -1:
            self._product = "AGOL"
        if self._product in ["AGO", "AGOL", "PORTAL"]:
            if not url[-1] == "/":
                url += "/"
            if url.lower().find("www.arcgis.com") > -1:
                urlscheme = urlparse(url).scheme
                return "{scheme}://www.arcgis.com/sharing/rest/".format(
                    scheme=urlscheme
                )
            elif url.lower().endswith("sharing/"):
                return url + "rest/"
            elif url.lower().endswith("sharing/rest/"):
                return url
            else:
                return url + "sharing/rest/"
        elif self._product in ["SERVER", "FEDERATED_SERVER", "FEDERATED"]:
            parsed = urlparse(url)
            path = parsed.path
            if str(path).startswith("/"):
                path = path[1:]
            url = "%s://%s/%s/rest/services/" % (
                parsed.scheme,
                parsed.netloc,
                path.split("/")[0],
            )
        return url

    # ----------------------------------------------------------------------
    def _assemble_proxy(self) -> dict:
        if self._proxy and isinstance(self._proxy, dict):
            return self._proxy
        elif self._proxy_port and self._proxy_url:
            url = "%s:%s" % (self._proxy_url, self._proxy_port)
            if self._proxy_password and self._proxy_username:
                proxies = {
                    "http": "http://%s:%s@%s"
                    % (self._proxy_username, self._proxy_password, url),
                    "https": "https://%s:%s@%s"
                    % (self._proxy_username, self._proxy_password, url),
                }
            else:
                proxies = {
                    "http": "http://%s" % url,
                    "https": "https://%s" % url,
                }
            return proxies
        return

    # ----------------------------------------------------------------------
    def _create_session(self) -> requests.Session:
        if self._proxy and isinstance(self._proxy, dict):
            proxies = self._proxy
            self._proxy = proxies
        elif self._proxy_port and self._proxy_url:
            url = "%s:%s" % (self._proxy_url, self._proxy_port)
            if self._proxy_password and self._proxy_username:
                proxies = {
                    "http": "http://%s:%s@%s"
                    % (self._proxy_username, self._proxy_password, url),
                    "https": "https://%s:%s@%s"
                    % (self._proxy_username, self._proxy_password, url),
                }
            else:
                proxies = {
                    "http": "http://%s" % url,
                    "https": "https://%s" % url,
                }
            self._proxy = proxies
        else:
            proxies = None

        if self._cert_file and self._key_file:
            cert = (self._cert_file, self._key_file)

        elif self._cert_file and self._password:
            from arcgis.gis._impl._con._cert import pfx_to_pem

            self._key_file, self._cert_file = pfx_to_pem(
                pfx_path=self._cert_file, pfx_password=self._password
            )
            cert = (self._cert_file, self._key_file)
        elif self._cert_file:
            cert = self._cert_file
        else:
            cert = None

        self._session = EsriSession(
            cert=cert,
            verify_cert=self._verify_cert,
            proxies=proxies,
            retries=5,
        )
        self._session.verify = self._verify_cert
        self._session.stream = True
        self._session.trust_env = self.trust_env
        self._session.headers.update(self._header)
        self._session.proxies = proxies

        from urllib3.util import Retry
        from urllib3 import __version__ as __urllib3_version__

        if self._custom_adapter:
            for k, v in self._custom_adapter.items():
                self._session.mount(k, v)

        if self._referer is None and (
            self._portal_connection
            and str(self._portal_connection._auth).lower() == "home"
        ):
            self._referer = "http"
            self._session.headers.pop("Referer", None)
            self._session.headers["Referer"] = json.dumps("")
        elif (
            self._portal_connection
            and str(self._portal_connection._auth).lower() == "home"
        ):
            self._referer = "http"
            self._session.headers.pop("Referer", None)
            self._session.headers["Referer"] = json.dumps("")
        elif self._referer is None:
            self._referer = "http"
            self._session.headers.update({"Referer": self._referer})
        else:
            self._session.headers.update({"Referer": self._referer})
        if self._custom_auth:
            self._session.auth = self._custom_auth
            self._auth = "CUSTOM"
        elif self._auth.lower() == "home":
            from arcgis.auth._auth._notebook import EsriNotebookAuth

            self._session.verify = False
            self._session.auth = EsriNotebookAuth(
                token=self._token,
                referer=self._referer,
                auth=GuessAuth(username=None, password=None),
            )
        elif self._auth.lower() == "ags_auth":
            from arcgis.auth._auth import ArcGISServerAuth

            self._session.auth = ArcGISServerAuth(
                legacy=self.legacy,
                ags_file=self._ags_file,
            )
        elif self._auth.lower() == "oauth":
            self._session.auth = EsriOAuth2Auth(
                base_url=self._baseurl,
                client_id=self._client_id,
                client_secret=self._client_secret,
                username=self._username,
                password=self._password,
                referer=self._referer,
                expiration=self._expiration,
                verify_cert=self._verify_cert,
                proxies=proxies,
            )
        elif self._auth.lower() == "builtin":
            if self._check_product() == "SERVER":
                pauth = None
                if self._portal_connection:
                    pauth = self._portal_connection._con._auth
                if self._token_url is None:
                    self._check_product()
                self._session.auth = EsriGenTokenAuth(
                    token_url=self._token_url,
                    referer=self._referer,
                    username=self._username,
                    password=self._password,
                    portal_auth=pauth,
                    time_out=self._timeout,
                    verify_cert=self._verify_cert,
                    legacy=self.legacy,
                    proxies=proxies,
                )
            else:
                if self._use_gen_token:
                    if self._token_url is None:
                        self._check_product()
                    self._session.auth = EsriGenTokenAuth(
                        token_url=self._token_url,
                        referer=self._referer,
                        username=self._username,
                        password=self._password,
                        portal_auth=None,
                        time_out=1440,
                        verify_cert=self._verify_cert,
                        legacy=self.legacy,
                        proxies=proxies,
                    )
                else:
                    self._session.auth = EsriBuiltInAuth(
                        url=self._baseurl,
                        username=self._username,
                        password=self._password,
                        expiration=self._timeout,
                        legacy=False,
                        verify_cert=self._verify_cert,
                        referer=self._referer,
                        proxies=proxies,
                    )
        elif self._auth.lower() == "user_token":
            self._session.auth = EsriUserTokenAuth(
                token=self._token,
                referer=self._referer,
                verify_cert=self._verify_cert,
            )
        elif self._auth.lower() == "api_key":
            from arcgis.auth._auth._apikey import EsriAPIKeyAuth

            self._session.auth = EsriAPIKeyAuth(
                api_key=self._api_key,
                referer=self._referer,
                verify_cert=self._verify_cert,
            )
            self._token = self._api_key
        elif self._auth.lower() == "basic_realm":
            self._session.auth = EsriBasicAuth(
                username=self._username,
                password=self._password,
                referer=self._referer,
                verify_cert=self._verify_cert,
            )
        elif self._auth.lower() in ["kerberos"] and HAS_KERBEROS:
            if self._security_kwargs:
                self._session.auth = EsriKerberosAuth(
                    proxies=self._proxy,
                    username=self._username,
                    password=self._password,
                    verify_cert=self._verify_cert,
                    legacy=False,
                    **self._security_kwargs,
                )
            elif HAS_GSSAPI:
                self._session.auth = EsriWindowsAuth(
                    username=self._username,
                    password=self._password,
                    verify_cert=self._verify_cert,
                    legacy=False,
                    proxies=self._proxy,
                )
            else:
                self._session.auth = EsriKerberosAuth(
                    proxies=self._proxy,
                    username=self._username,
                    password=self._password,
                    verify_cert=self._verify_cert,
                    legacy=False,
                    **self._security_kwargs,
                )
        elif self._username and self._password and self._auth.lower() != "iwa":
            self._session.auth = GuessAuth(
                username=self._username, password=self._password
            )
        elif self._auth.lower() in ["iwa", "ntlm"] and HAS_SSPI:
            self._session.auth = EsriWindowsAuth(
                username=self._username,
                password=self._password,
                verify_cert=self._verify_cert,
                legacy=False,
                proxies=self._proxy,
            )
        elif self._auth.lower() == "pro":
            self._session.auth = (
                GuessAuth(None, None, legacy=False) + ArcGISProAuth()
            )  # GuessAuth(None, None, legacy=False)
        elif not self._cert_file and not self._key_file:
            # else:

            if HAS_SSPI:
                try:
                    self._session.auth = EsriWindowsAuth(
                        verify_cert=self._verify_cert,
                        legacy=False,
                        proxies=self._proxy,
                    )
                except:
                    ...
            elif HAS_KERBEROS:
                try:
                    self._session.auth = EsriKerberosAuth(
                        verify_cert=self._verify_cert,
                        legacy=False,
                        proxies=self._proxy,
                    )
                except:
                    ...

    # ----------------------------------------------------------------------
    def get(
        self, path: str, params: dict = None, **kwargs
    ) -> Union[dict, requests.Response]:
        """

        sends a GET request.

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        path                          required string. URI resource to access.
        ===========================   =====================================================

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        params                        optional dictionary.  This is the data payload to the
                                      web resource
        ---------------------------   -----------------------------------------------------
        out_folder                    optional string. Save path on disk.  The default is
                                      the user's temp directory.
        ---------------------------   -----------------------------------------------------
        file_name                     optional string. Name of the file to save. If not
                                      provided, name will try to be parsed from the response.
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If True, the marker f='json' will
                                      be appended to the parameters.
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean. Default True, if true, the call will
                                      add a token to any token based security.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Boolean.  When False, the JSON values will not be encoded.
        ---------------------------   -----------------------------------------------------
        ignore_error_key              optional Boolean. The default is False. If true, JSON will be returned and no exception is raised when 'error' is present in the response
        ---------------------------   -----------------------------------------------------
        return_raw_response           optional Boolean. Returns the requests' Response object.
        ---------------------------   -----------------------------------------------------
        allow_redirects               Optional Boolean. Does not allow the call to follow the 3xx status codes. The default is True
        ---------------------------   -----------------------------------------------------
        drop_auth                     Optional Boolean. Drop this when auth handlers cause issues. The default is False
        ===========================   =====================================================
        """
        drop_auth = kwargs.pop("drop_auth", False)
        ignore_error_key = kwargs.pop("ignore_error_key", False)
        allow_redirects = kwargs.pop("allow_redirects", True)
        return_raw_response = kwargs.pop("return_raw_response", False)
        json_encode = kwargs.pop("json_encode", True)
        add_headers = kwargs.pop("add_headers", {})
        if self._baseurl.endswith("/") == False:
            self._baseurl += "/"
        url = path
        if url.find("://") == -1:
            if url.startswith("/") == False and self._baseurl.endswith("/") == False:
                url = "/" + url
            url = self._baseurl + url
        if kwargs.pop("ssl", False) or self._all_ssl:
            url = url.replace("http://", "https://")
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()

        try_json = kwargs.pop("try_json", True)
        if try_json and isinstance(params, dict):
            params["f"] = "json"
        if params == {}:
            params = None

        if "out_folder" in kwargs:
            out_path = kwargs.pop("out_folder", tempfile.gettempdir())
        else:
            out_path = kwargs.pop("out_path", tempfile.gettempdir())
        file_name = kwargs.pop("file_name", None)
        if isinstance(params, dict):
            if params and json_encode:
                for k, v in copy.copy(params).items():
                    if isinstance(v, (tuple, dict, list, bool)):
                        params[k] = json.dumps(v, default=_date_handler)
                    elif isinstance(v, PropertyMap):
                        params[k] = json.dumps(dict(v), default=_date_handler)
                    elif isinstance(v, InsensitiveDict):
                        params[k] = v.json
        if add_headers:
            original_headers = copy.deepcopy(self._session.headers)
            self._session.headers.update(add_headers)
        try:
            if self._cert_file:
                cert = (self._cert_file, self._key_file)
            else:
                cert = None
            if self._session.auth and drop_auth:
                auth = self._session.auth
                self._session.auth = None
                resp = self._session.get(
                    url=url,
                    params=params,
                    cert=cert,
                    verify=self._verify_cert,
                    allow_redirects=allow_redirects,
                )
                self._session.auth = auth
            else:
                resp = self._session.get(
                    url=url,
                    params=params,
                    cert=cert,
                    verify=self._verify_cert,
                    allow_redirects=allow_redirects,
                )
        except requests.exceptions.SSLError as err:
            raise requests.exceptions.SSLError(
                "Please set verify_cert=False due to encountered SSL error: %s" % err
            )
        except requests.exceptions.InvalidURL as errIU:
            raise requests.exceptions.SSLError("Invalid URL provided: %s" % errIU)
        except requests.exceptions.ConnectionError as errCE:
            raise requests.exceptions.ConnectionError(
                "A connection error has occurred: %s" % errCE
            )
        except requests.exceptions.InvalidHeader as errIH:
            raise requests.exceptions.InvalidHeader(
                "A invalid header was provided: %s" % errIH
            )
        except requests.exceptions.HTTPError as errh:
            raise requests.exceptions.HTTPError("Http Error: %s" % errh)
        except requests.exceptions.MissingSchema as errMS:
            raise requests.exceptions.MissingSchema(
                "URL scheme must be provided: %s" % errMS
            )
        except requests.exceptions.RequestException as errRE:
            raise requests.exceptions.RequestException(
                "A general exception was raised: %s" % errRE
            )
        except Exception as e:
            raise Exception("A general error occurred: %s" % e)
        except:
            import traceback

            raise Exception("An unknown error occurred: %s" % traceback.format_exc())
        if add_headers:
            self._session.headers.clear()
            self._session.headers.update(original_headers)
        if return_raw_response:
            return resp
        return self._handle_response(
            resp,
            file_name,
            out_path,
            try_json,
            force_bytes=kwargs.pop("force_bytes", False),
            ignore_error_key=ignore_error_key,
        )

    # ----------------------------------------------------------------------
    def _handle_response(
        self,
        resp: requests.Response,
        file_name: str,
        out_path: str,
        try_json: bool,
        force_bytes: bool = False,
        ignore_error_key: bool = False,
    ) -> dict:
        """
        handles the request responses

        ===========================   =====================================================
        **optional keywords**         **description**
        ---------------------------   -----------------------------------------------------
        resp                          required resquests.Rsponse object.
        ---------------------------   -----------------------------------------------------
        file_name                     required string.  Name of the output file if needed.
        ---------------------------   -----------------------------------------------------
        out_path                      required string. Name of the save folder.
        ---------------------------   -----------------------------------------------------
        try_json                      required boolean. Determines if the response should
                                      be returned as a dictionary or the native format
                                      send back by a server
        ---------------------------   -----------------------------------------------------
        force_bytes                   Optional Boolean. If True, the results returns as bytes
                                      instead of a file path.
        ===========================   =====================================================

        returns: string, dictionary, or bytes depending on the response.

        """
        if "Set-Cookie" in resp.headers and (
            str(self._auth).lower() in ["anon"]
            or (
                str(self._username).find("\\") > -1
                and str(self._auth).lower() in ["builtin"]
            )
        ):
            self._auth = "IWA"

        data = None
        url = resp.url
        if out_path and os.path.isdir(out_path) == False:
            os.makedirs(out_path)
        if out_path is None:
            out_path = tempfile.gettempdir()
        if (
            file_name is None
            and "Content-Type" in resp.headers
            and (
                resp.headers["Content-Type"].lower().find("json") == -1
                and resp.headers["Content-Type"].lower().find("text") == -1
            )
        ):
            file_name = (
                _filename_from_url(url) or _filename_from_headers(resp.headers) or None
            )
        elif file_name is None and "Content-Disposition" in resp.headers:
            file_name = (
                _filename_from_url(url) or _filename_from_headers(resp.headers) or None
            )
        if force_bytes:
            try:
                return bytes(resp.content)
            except:
                return resp.content
        if file_name is not None:
            file_name = os.path.join(out_path, file_name)
            if os.path.isfile(file_name):
                os.remove(file_name)
            stream_size = 512 * 2
            if "Content-Length" in resp.headers:
                max_length = int(resp.headers["Content-Length"])
                if max_length > stream_size * 2 and max_length < 1024 * 1024:
                    stream_size = 1024 * 2
                elif max_length > 5 * (1024 * 1024) and max_length < 10 * (1024 * 1024):
                    stream_size = 5 * (1024 * 1024)  # 5 mb
                elif max_length >= 10 * (1024 * 1024):
                    stream_size = 10 * (1024 * 1024)  # 10 mb
                elif max_length > (1024 * 1024):
                    stream_size = 1024 * 1024  # 1 mb
                else:
                    stream_size = 512 * 2

            fp = stream.stream_response_to_file(
                response=resp, path=file_name, chunksize=stream_size
            )
            return fp

        if try_json:
            if (
                "Content-Length" in resp.headers
                and int(resp.headers.get("Content-Length")) == 0
            ):
                data = {}
            elif (
                "Transfer-Encoding" in resp.headers
                and resp.headers["Transfer-Encoding"].lower() == "chunked"
            ):
                data = None
                for it in resp.iter_lines(
                    chunk_size=None, decode_unicode=True, delimiter=None
                ):
                    if data is None:
                        data = it
                    else:
                        data += it
                data = json.loads(data)
                if "error" in data and ignore_error_key == False:
                    raise Exception(data["error"])
            else:
                try:
                    data = resp.json()
                except JSONDecodeError:
                    if resp.text:
                        raise Exception(resp.text)
                    else:
                        raise
            # if 'error' in data:
            # raise Exception(data['error'])
            # return data
            # else:
            # return resp.text
            if "error" in data and ignore_error_key == False:
                if "messages" in data:
                    return data
                errorcode = data["error"]["code"] if "code" in data["error"] else 0
                self._handle_json_error(data["error"], errorcode)
            return data
        else:
            return resp.text

    # ----------------------------------------------------------------------
    def _handle_json_error(self, error, errorcode):
        errormessage = error.get("message")
        # handles case where message exists in the dictionary but is None
        if errormessage is None:
            errormessage = "Unknown Error"
        # _log.error(errormessage)
        if "details" in error and error["details"] is not None:
            if isinstance(error["details"], str):
                errormessage = f"{errormessage} \n {error['details']}"
                # _log.error(error['details'])
            else:
                for errordetail in error["details"]:
                    if isinstance(errordetail, str):
                        errormessage = errormessage + "\n" + errordetail
                        # _log.error(errordetail)

        errormessage = errormessage + "\n(Error Code: " + str(errorcode) + ")"
        raise Exception(errormessage)

    def post_multipart(
        self, path: str, params: dict = None, files: list = None, **kwargs
    ) -> Union[dict, requests.Response]:
        """
        sends a MultiPart Form POST request.

        ===========================   =====================================================
        **Parameter**                **Description**
        ---------------------------   -----------------------------------------------------
        path                          optional string.  URL or part of the url resource
                                      to call.
        ---------------------------   -----------------------------------------------------
        params                        optional dict.  Contains data to pass to the web
                                      resource.
        ---------------------------   -----------------------------------------------------
        files                         optional list of files

                                      Files can be provided two ways:

                                      The most basic way is:
                                      Way1: {key : r"c:\temp\myfile.foo}
                                      This is just the file path and the key.


                                      The preferred way:

                                      Way 2: {key : (file_name, open(c:\temp\myfile.foo, 'rb'), image\jpeg)}

                                      Way 2 requires providing the filename, IO object as 'rb', and the mimetype.
        ===========================   =====================================================


        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call. If
                                      is_geoevent is True, then the token will be appended
                                      to the header.
        ---------------------------   -----------------------------------------------------
        is_geoevent                   optional boolean. True means the auth token will be placed in the header
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        out_folder                    optional string.  This is the save folder for the data.
        ---------------------------   -----------------------------------------------------
        file_name                     optional string. The save name of the file. This will override the file name if provided in the response.
        ---------------------------   -----------------------------------------------------
        force_bytes                   optional boolean.  Deprecated.
        ---------------------------   -----------------------------------------------------
        add_headers                   optional dict.  If provided, additional headers will be given for a single call.
        ---------------------------   -----------------------------------------------------
        post_json                     optional bool. If True, the data is pushed in the request's json parameter.  This is an edge case for Workflow Manager. The default is `False`.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Bool. If False, the key/value parameters will not be JSON encoded.
        ---------------------------   -----------------------------------------------------
        timeout                       optional Integer. Timeout in seconds
        ---------------------------   -----------------------------------------------------
        return_raw_response           Optional Boolean. If True, returns the requests.Response object.
        ---------------------------   -----------------------------------------------------
        allow_redirects               Optional Boolean. Does not allow the call to follow the 3xx status codes. The default is True
        ---------------------------   -----------------------------------------------------
        drop_auth                     Optional Boolean. Drop this when auth handlers cause issues. The default is False
        ===========================   =====================================================

        :returns: data returned from the URL call.
        """
        import io

        drop_auth = kwargs.pop("drop_auth", False)
        allow_redirects = kwargs.pop("allow_redirects", True)
        timeout = kwargs.pop("timeout", self._timeout)
        return_raw_response = kwargs.pop("return_raw_response", False)
        json_encode = kwargs.pop("json_encode", True)
        if self._baseurl.endswith("/") == False:
            self._baseurl += "/"
        url = path

        post_json = kwargs.pop("post_json", False)

        # if self._auth == "IWA":
        #    self._session = None
        if "postdata" in kwargs:  # handles legacy issues
            params = kwargs.pop("postdata")
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)

        if url.find("://") == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False) or self._all_ssl:
            url = url.replace("http://", "https://")

        if try_json:
            params["f"] = "json"
        fields = {}
        if files:
            if isinstance(files, dict):
                for k, v in files.items():
                    if isinstance(v, (list, tuple)):
                        fields[k] = v
                    else:
                        fields[k] = (
                            os.path.basename(v),
                            open(v, "rb"),
                            mimetypes.guess_type(v)[0],
                        )
            elif isinstance(files, (list, tuple)):
                for key, filePath, fileName in files:
                    if (
                        isinstance(fileName, str)
                        and isinstance(filePath, (io.StringIO, io.BytesIO)) == False
                    ):
                        fields[key] = (
                            fileName,
                            open(filePath, "rb"),
                            mimetypes.guess_type(filePath)[0],
                        )
                    elif isinstance(fileName, str) and isinstance(
                        filePath, (io.StringIO, io.BytesIO)
                    ):
                        fields[key] = (
                            fileName,
                            filePath,
                            None,
                        )
                    else:
                        fields[key] = v
            files = fields

        out_path = (
            kwargs.pop("out_path", None)
            or kwargs.pop("out_folder", None)
            or tempfile.gettempdir()
        )
        file_name = kwargs.pop("file_name", None)
        try:
            if self._cert_file:
                cert = (self._cert_file, self._key_file)
            else:
                cert = None
            if json_encode:
                for k, v in params.items():
                    if isinstance(v, (dict, list, tuple, bool)):
                        params[k] = json.dumps(v, default=_date_handler)
                    elif isinstance(v, PropertyMap):
                        params[k] = json.dumps(dict(v), default=_date_handler)
                    elif isinstance(v, InsensitiveDict):
                        params[k] = v.json
            # When data and files are present, they need to be combined
            # https://stackoverflow.com/a/12385661

            params.update(fields)
            if self._session.auth and drop_auth:
                auth = self._session.auth
                self._session.auth
            else:
                auth = None
            if post_json:  # edge case workflow
                if timeout:
                    resp = self._session.post(
                        url=url,
                        json=params,
                        cert=cert,
                        files=files,
                        allow_redirects=allow_redirects,
                        verify=self._verify_cert,
                        timeout=timeout,
                    )
                else:
                    resp = self._session.post(
                        url=url,
                        json=params,
                        cert=cert,
                        allow_redirects=allow_redirects,
                        verify=self._verify_cert,
                        files=files,
                    )
            else:
                if files is None:
                    files = {}
                multipart_form_data = {
                    k: (None, v) for k, v in params.items() if not k in files
                }
                multipart_form_data.update(fields)

                if timeout:
                    resp = self._session.post(
                        url=url,
                        files=multipart_form_data,
                        cert=cert,
                        allow_redirects=allow_redirects,
                        timeout=timeout,
                        verify=self._verify_cert,
                    )
                else:
                    resp = self._session.post(
                        url=url,
                        cert=cert,
                        files=multipart_form_data,
                        allow_redirects=allow_redirects,
                        verify=self._verify_cert,
                    )
            if auth and drop_auth:
                self._session.auth = auth
        except requests.exceptions.SSLError as err:
            raise requests.exceptions.SSLError(
                "Please set verify_cert=False due to encountered SSL error: %s" % err
            )
        except requests.exceptions.InvalidURL as errIU:
            raise requests.exceptions.SSLError("Invalid URL provided: %s" % errIU)
        except requests.exceptions.ConnectionError as errCE:
            raise requests.exceptions.ConnectionError(
                "A connection error has occurred: %s" % errCE
            )

        except requests.exceptions.InvalidHeader as errIH:
            raise requests.exceptions.InvalidHeader(
                "A invalid header was provided: %s" % errIH
            )
        except requests.exceptions.HTTPError as errh:
            raise requests.exceptions.HTTPError("Http Error: %s" % errh)
        except requests.exceptions.MissingSchema as errMS:
            raise requests.exceptions.MissingSchema(
                "URL scheme must be provided: %s" % errMS
            )
        except requests.exceptions.RequestException as errRE:
            raise requests.exceptions.RequestException(
                "A general exception was raised: %s" % errRE
            )
        except Exception as e:
            raise Exception("A general error occurred: %s" % e)
        except:
            import traceback

            raise Exception("An unknown error occurred: %s" % traceback.format_exc())
        if return_raw_response:
            return resp
        return self._handle_response(
            resp=resp,
            out_path=out_path,
            file_name=file_name,
            try_json=try_json,
            force_bytes=kwargs.pop("force_bytes", False),
        )

    # ----------------------------------------------------------------------
    def post(
        self, path: str, params: dict = None, files: list = None, **kwargs
    ) -> Union[dict, requests.Response]:
        """
        sends a POST request.

        ===========================   =====================================================
        **Parameter**                **Description**
        ---------------------------   -----------------------------------------------------
        path                          optional string.  URL or part of the url resource
                                      to call.
        ---------------------------   -----------------------------------------------------
        params                        optional dict.  Contains data to pass to the web
                                      resource.
        ---------------------------   -----------------------------------------------------
        files                         optional list of files

                                      Files can be provided two ways:

                                      The most basic way is:
                                      Way1: {key : r"c:\temp\myfile.foo}
                                      This is just the file path and the key.


                                      The preferred way:

                                      Way 2: {key : (file_name, open(c:\temp\myfile.foo, 'rb'), image\jpeg)}

                                      Way 2 requires providing the filename, IO object as 'rb', and the mimetype.
        ===========================   =====================================================


        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call. If
                                      is_geoevent is True, then the token will be appended
                                      to the header.
        ---------------------------   -----------------------------------------------------
        is_geoevent                   optional boolean. True means the auth token will be placed in the header
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        out_folder                    optional string.  This is the save folder for the data.
        ---------------------------   -----------------------------------------------------
        file_name                     optional string. The save name of the file. This will override the file name if provided in the response.
        ---------------------------   -----------------------------------------------------
        force_bytes                   optional boolean.  Deprecated.
        ---------------------------   -----------------------------------------------------
        add_headers                   optional dict.  If provided, additional headers will be given for a single call.
        ---------------------------   -----------------------------------------------------
        post_json                     optional bool. If True, the data is pushed in the request's json parameter.  This is an edge case for Workflow Manager. The default is `False`.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Bool. If False, the key/value parameters will not be JSON encoded.
        ---------------------------   -----------------------------------------------------
        timeout                       optional Integer. The number of seconds to timeout a service without a response.  The default is 600 seconds.
        ---------------------------   -----------------------------------------------------
        return_raw_response           Optional boolean. Returns the requests.Response object.
        ---------------------------   -----------------------------------------------------
        allow_redirects               Optional Boolean. Does not allow the call to follow the 3xx status codes. The default is True
        ---------------------------   -----------------------------------------------------
        drop_auth                     Optional Boolean. Drop this when auth handlers cause issues. The default is False
        ===========================   =====================================================

        :returns: data returned from the URL call.

        """
        drop_auth = kwargs.pop("drop_auth", False)
        allow_redirects = kwargs.pop("allow_redirects", True)
        timeout = kwargs.pop("timeout", self._timeout)
        return_raw_response = kwargs.pop("return_raw_response", False)
        json_encode = kwargs.pop("json_encode", True)
        add_headers = kwargs.pop("add_headers", {})
        buffer_reader = None

        if self._baseurl.endswith("/") == False:
            self._baseurl += "/"
        url = path

        post_json = kwargs.pop("post_json", False)

        if "postdata" in kwargs:  # handles legacy issues
            params = kwargs.pop("postdata")
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)

        if url.find("://") == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False) or self._all_ssl:
            url = url.replace("http://", "https://")

        if try_json:
            params["f"] = "json"
        if files:
            fields = {}
            if isinstance(files, dict):
                for k, v in files.items():
                    if isinstance(v, (list, tuple)):
                        fields[k] = v
                    else:
                        buffer_reader = open(v, "rb")
                        fields[k] = (
                            os.path.basename(v),
                            buffer_reader,
                            mimetypes.guess_type(v)[0],
                        )
            elif isinstance(files, (list, tuple)):
                for key, filePath, fileName in files:
                    import io

                    if (
                        isinstance(fileName, str)
                        and isinstance(filePath, (io.StringIO, io.BytesIO)) == False
                    ):
                        buffer_reader = open(filePath, "rb")
                        fields[key] = (
                            fileName,
                            buffer_reader,
                            mimetypes.guess_type(filePath)[0],
                        )
                    elif isinstance(fileName, str) and isinstance(
                        filePath, (io.StringIO, io.BytesIO)
                    ):
                        fields[key] = (
                            fileName,
                            filePath,
                            None,
                        )
                    else:
                        fields[key] = v
            files = fields

        out_path = (
            kwargs.pop("out_path", None)
            or kwargs.pop("out_folder", None)
            or tempfile.gettempdir()
        )
        file_name = kwargs.pop("file_name", None)
        if add_headers:
            original_headers = copy.deepcopy(self._session.headers)
            self._session.headers.update(add_headers)
        try:
            if self._cert_file:
                cert = (self._cert_file, self._key_file)
            else:
                cert = None
            if json_encode:
                for k, v in params.items():
                    if isinstance(v, (dict, list, tuple, bool)):
                        params[k] = json.dumps(v, default=_date_handler)
                    elif isinstance(v, PropertyMap):
                        params[k] = json.dumps(dict(v), default=_date_handler)
                    elif isinstance(v, InsensitiveDict):
                        params[k] = v.json
            if self._session.auth and drop_auth:
                auth = self._session.auth
                self._session.auth
            else:
                auth = None
            if post_json:  # edge case workflow
                if timeout:
                    resp = self._session.post(
                        url=url,
                        json=params,
                        cert=cert,
                        files=files,
                        timeout=timeout,
                        verify=self._verify_cert,
                        allow_redirects=allow_redirects,
                    )
                else:
                    resp = self._session.post(
                        url=url,
                        json=params,
                        cert=cert,
                        files=files,
                        verify=self._verify_cert,
                        allow_redirects=allow_redirects,
                    )

            else:
                if timeout:
                    resp = self._session.post(
                        url=url,
                        data=params,
                        cert=cert,
                        files=files,
                        timeout=timeout,
                        verify=self._verify_cert,
                        allow_redirects=allow_redirects,
                    )
                else:
                    resp = self._session.post(
                        url=url,
                        data=params,
                        cert=cert,
                        files=files,
                        verify=self._verify_cert,
                        allow_redirects=allow_redirects,
                    )
            if auth:
                self._session.auth = auth
            if add_headers:
                self._session.headers.clear()
                self._session.headers.update(original_headers)
        except requests.exceptions.SSLError as err:
            raise requests.exceptions.SSLError(
                "Please set verify_cert=False due to encountered SSL error: %s" % err
            )
        except requests.exceptions.InvalidURL as errIU:
            raise requests.exceptions.SSLError("Invalid URL provided: %s" % errIU)
        except requests.exceptions.ConnectionError as errCE:
            raise requests.exceptions.ConnectionError(
                "A connection error has occurred: %s" % errCE
            )

        except requests.exceptions.InvalidHeader as errIH:
            raise requests.exceptions.InvalidHeader(
                "A invalid header was provided: %s" % errIH
            )
        except requests.exceptions.HTTPError as errh:
            raise requests.exceptions.HTTPError("Http Error: %s" % errh)
        except requests.exceptions.MissingSchema as errMS:
            raise requests.exceptions.MissingSchema(
                "URL scheme must be provided: %s" % errMS
            )
        except requests.exceptions.RequestException as errRE:
            raise requests.exceptions.RequestException(
                "A general exception was raised: %s" % errRE
            )
        except Exception as e:
            raise Exception("A general error occurred: %s" % e)
        except:
            import traceback

            raise Exception("An unknown error occurred: %s" % traceback.format_exc())
        if buffer_reader:
            buffer_reader.close()
        if return_raw_response:
            return resp
        return self._handle_response(
            resp=resp,
            out_path=out_path,
            file_name=file_name,
            try_json=try_json,
            force_bytes=kwargs.pop("force_bytes", False),
        )

    # ----------------------------------------------------------------------
    def put_raw(self, url: str, data: dict, **kwargs) -> requests.Response:
        """
        performs a raw PUT operation

        url: str
        data: bytes or open() object
        kwargs - optional requests.put parameters.  headers is not supported, use additional_headers
        """
        verify = kwargs.pop("verify", True)
        original_headers = copy.deepcopy(self._session.headers)
        self._session.headers.update(kwargs.pop("additional_headers", {}))
        token_header = "X-Esri-Authorization"
        if self.token and not "X-Esri-Authorization" in original_headers:
            token = self.token
            self._session.headers.update({token_header: "Bearer %s" % token})

        resp = self._session.put(
            url=url,
            data=data,
            verify=verify,
            headers=self._session.headers,
            **kwargs,
        )
        self._session.headers.clear()
        self._session.headers.update(original_headers)

        return resp

    # ----------------------------------------------------------------------
    def put(
        self, url: str, params: dict = None, files: list = None, **kwargs
    ) -> Union[dict, requests.Response]:
        """
        sends a PUT request

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        url                           Optional String. The web endpoint.
        ---------------------------   -----------------------------------------------------
        params                        Optional dict. The Key/value pairs to send along with the request.
        ---------------------------   -----------------------------------------------------
        files                         Optional list. Allows users to provide a file or files
                                      to the operation.  The format should be:
                                      files = [[key, filePath, fileName], ...,[keyN, filePathN, fileNameN]]
        ===========================   =====================================================

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call. (deprecated)
        ---------------------------   -----------------------------------------------------
        token_as_header               Optional boolean.  If True, the token will go into
                                      the header as `Authorization` header.  This can be
                                      overwritten using the `token_header` parameter. (deprecated)
        ---------------------------   -----------------------------------------------------
        token_header                  Optional String. If provided and token_as_header is
                                      True, authentication token will be placed in this
                                      instead on URL string. (deprecated)
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        post_json                     optional bool. If True, the data is pushed in the request's json parameter.  This is an edge case for Workflow Manager. The default is `False`.
        ---------------------------   -----------------------------------------------------
        json_encode                   optional Bool. If False, the key/value parameters will not be JSON encoded.
        ---------------------------   -----------------------------------------------------
        return_raw_response           Optional boolean. When true, it returns the requests.Response object
        ---------------------------   -----------------------------------------------------
        allow_redirects               Optional Boolean. Does not allow the call to follow the 3xx status codes. The default is True
        ---------------------------   -----------------------------------------------------
        drop_auth                     Optional Boolean. Drop this when auth handlers cause issues. The default is False
        ===========================   =====================================================

        :returns: dict or string depending on the response

        """
        drop_auth = kwargs.pop("drop_auth", False)
        allow_redirects = kwargs.pop("allow_redirects", True)
        return_raw_response = kwargs.pop("return_raw_response", False)
        post_json = kwargs.pop("post_json", False)
        json_encode = kwargs.pop("json_encode", True)
        out_path = kwargs.pop("out_path", None)
        file_name = kwargs.pop("file_name", None)
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)
        if url.find("://") == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False):
            url = url.replace("http://", "https://")

        if try_json:
            params["f"] = "json"

        if files:
            fields = {}
            if isinstance(files, dict):
                for k, v in files.items():
                    if isinstance(v, (list, tuple)):
                        fields[k] = v
                    else:
                        fields[k] = (
                            os.path.basename(v),
                            open(v, "rb"),
                            mimetypes.guess_type(v)[0],
                        )
            elif isinstance(files, (list, tuple)):
                for key, filePath, fileName in files:
                    if isinstance(fileName, str):
                        fields[key] = (
                            fileName,
                            open(filePath, "rb"),
                            mimetypes.guess_type(filePath)[0],
                        )
                    else:
                        fields[key] = v

            params.update(fields)
            del files, fields
        if self._cert_file:
            cert = (self._cert_file, self._key_file)
        else:
            cert = None
        if json_encode:
            for k, v in params.items():
                if isinstance(v, (dict, list, tuple, bool)):
                    params[k] = json.dumps(v, default=_date_handler)
                elif isinstance(v, PropertyMap):
                    params[k] = json.dumps(dict(v), default=_date_handler)
                elif isinstance(v, InsensitiveDict):
                    params[k] = v.json
        if self._session.auth and drop_auth:
            auth = self._session.auth
            self._session.auth
        else:
            auth = None
        if post_json:  # edge case workflow
            resp = self._session.put(
                url=url,
                json=params,
                cert=cert,
                files=files,
                allow_redirects=allow_redirects,
            )
        else:
            resp = self._session.put(
                url=url,
                data=params,
                cert=cert,
                files=files,
                allow_redirects=allow_redirects,
            )
        if auth and drop_auth:
            self._session.auth = auth
        if return_raw_response:
            return resp
        return self._handle_response(
            resp=resp,
            out_path=out_path,
            file_name=file_name,
            try_json=try_json,
            force_bytes=kwargs.pop("force_bytes", False),
        )

    # ----------------------------------------------------------------------
    def delete(self, url, params=None, **kwargs):
        """
        sends a PUT request

        ===========================   =====================================================
        **Parameter**                **Description**
        ---------------------------   -----------------------------------------------------
        url                           Optional String. The web endpoint.
        ---------------------------   -----------------------------------------------------
        params                        Optional dict. The Key/value pairs to send along with the request.
        ===========================   =====================================================

        ===========================   =====================================================
        **Optional Parameters**       **Description**
        ---------------------------   -----------------------------------------------------
        add_token                     optional boolean.  True means try to add the boolean,
                                      else do not add a ?token=<foo> to the call.
        ---------------------------   -----------------------------------------------------
        token_as_header               Optional boolean.  If True, the token will go into
                                      the header as `Authorization` header.  This can be
                                      overwritten using the `token_header` parameter
        ---------------------------   -----------------------------------------------------
        token_header                  Optional String. If provided and token_as_header is
                                      True, authentication token will be placed in this
                                      instead on URL string.
        ---------------------------   -----------------------------------------------------
        try_json                      optional boolean.  If true, the call adds the ?f=json.
        ---------------------------   -----------------------------------------------------
        ssl                           optional boolean. If true all calls are forced to be
                                      https.
        ---------------------------   -----------------------------------------------------
        return_raw_response           Optional Boolean.  Returns the raw requests.Response object
        ---------------------------   -----------------------------------------------------
        allow_redirects               Optional Boolean. Does not allow the call to follow the 3xx status codes. The default is True
        ---------------------------   -----------------------------------------------------
        drop_auth                     Optional Boolean. Drop this when auth handlers cause issues. The default is False
        ===========================   =====================================================

        :returns: dict or string depending on the response
        """
        allow_redirects = kwargs.pop("allow_redirects", True)
        drop_auth = kwargs.pop("drop_auth", False)
        out_path = kwargs.pop("out_path", None)
        return_raw_response = kwargs.pop("return_raw_response", False)
        file_name = kwargs.pop("file_name", None)
        if params is None:
            params = {}
        if self._session is None:
            self._create_session()
        try_json = kwargs.pop("try_json", True)
        if url.find("://") == -1:
            url = self._baseurl + url
        if kwargs.pop("ssl", False):
            url = url.replace("http://", "https://")

        if try_json:
            params["f"] = "json"
        if drop_auth and self._session.auth:
            auth = self._session.auth
            self._session.auth = None
            resp = self._session.delete(
                url=url, data=params, allow_redirects=allow_redirects
            )
            self._session.auth = auth
        else:
            resp = self._session.delete(
                url=url, data=params, allow_redirects=allow_redirects
            )
        if return_raw_response:
            return resp
        return self._handle_response(
            resp=resp,
            out_path=out_path,
            file_name=file_name,
            try_json=try_json,
            force_bytes=kwargs.pop("force_bytes", False),
        )

    # ----------------------------------------------------------------------
    def streaming_method(
        self,
        url: str,
        callback: object,
        data: dict = None,
        json_data: dict = None,
        verb: str = "GET",
        **kwargs,
    ) -> Union[dict, requests.Response]:
        """
        Performs streaming web requests.

        =======================     ===========================================================
        **Parameter**              **Description**
        -----------------------     -----------------------------------------------------------
        url                         Required String. The web resource location.
        -----------------------     -----------------------------------------------------------
        callback                    Required Method.  The callback function to handle the response from the streaming request.

                                    **Example**
                                    ```
                                    def hook(r, *args, **kwargs):
                                        print('called a hook')
                                        return r
                                    ```
                                    See: https://requests.kennethreitz.org/en/master/user/advanced/#event-hooks


        -----------------------     -----------------------------------------------------------
        data                        Optional Dict. The parameters to pass to the method.
        -----------------------     -----------------------------------------------------------
        json_data                   Optional Dict. The parameters to pass to the method. This applies to POST only
        -----------------------     -----------------------------------------------------------
        verb                        Optional String.  The default is GET.  The allowed values are POST, GET, or PUT.
        -----------------------     -----------------------------------------------------------
        kwargs                      Optional Dict.  See https://requests.readthedocs.io/en/master/user/advanced/#request-and-response-objects
        =======================     ===========================================================

        """
        verbs = ["post", "put", "get"]
        if verb.lower() in verbs:
            hooks = {"response": callback}
            fn = getattr(self._session, verb.lower())
            if verb.lower() == "post":
                return fn(
                    url=url,
                    data=data,
                    json_data=json,
                    hooks=hooks,
                    stream=True,
                    **kwargs,
                )
            else:
                return fn(url=url, data=data, hooks=hooks, stream=True, **kwargs)
        else:
            allowed_verb = ",".join(verbs)
            raise ValueError(f"Invalid web method only {allowed_verb} as allowed")
        self._session.post(url=url, data=data, json_data=json, stream=True)

    # ----------------------------------------------------------------------
    def login(
        self,
        username: str,
        password: str,
        expiration: Union[int, float] = None,
    ):
        """allows a user to login to a site with different credentials"""
        if expiration is None:
            expiration = 1440
        try:
            if self._username != username and self._password != password:
                c = Connection(
                    baseurl=self._baseurl,
                    username=username,
                    password=password,
                )
                self = c
        except:
            raise Exception("Could not create a new login.")

    # ----------------------------------------------------------------------
    def relogin(self, expiration: Union[float, int] = None) -> str:
        """Re-authenticates with the portal using the same username/password."""
        if expiration is None:
            expiration = self._expiration
        self.logout()
        return self.token

    # ----------------------------------------------------------------------
    def logout(self):
        """Logs out of the portal."""
        self._token = None
        self._create_time = None

    # ----------------------------------------------------------------------
    @property
    def is_logged_in(self) -> bool:
        """Returns true if logged into the portal."""
        return (self._auth in ["ANON", "UNKNOWN"]) == False

    # ----------------------------------------------------------------------
    @property
    def product(self) -> str:
        """Returns true if logged into the portal."""
        return self._product

    # ----------------------------------------------------------------------
    @property
    def token(self) -> str:
        """Gets a Token"""
        if isinstance(self._session.auth, EsriBuiltInAuth):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriGenTokenAuth):
            return self._session.auth.token()
        elif isinstance(self._session.auth, ArcGISProAuth):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriOAuth2Auth):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriUserTokenAuth):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriNotebookAuth):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriAPIKeyAuth):
            return self._session.auth.token
        return None

    # ----------------------------------------------------------------------
    @token.setter
    def token(self, value):
        """gets/sets the token"""
        if self._token != value:
            self._token = value

    # ----------------------------------------------------------------------
    def _check_product(self) -> str:
        """
        determines if the product is portal, arcgis online or arcgis server
        """
        from urllib.error import HTTPError

        baseurl = self._baseurl
        if self._product == "SERVER":
            if self._token_url is None:
                parsed = urlparse(self._baseurl)
                path = parsed.path[1:].split("/")[0]
                self._token_url = "https://%s/%s/admin/generateToken" % (
                    parsed.netloc,
                    path,
                )
            self._product = "SERVER"
            return "SERVER"
        if baseurl is None:
            return "UNKNOWN"
        if baseurl.lower().find("arcgis.com") > -1:
            parsed = urlparse(self._baseurl)
            self._product = "AGOL"
            self._token_url = "https://%s/sharing/rest/generateToken" % parsed.netloc
            return "AGOL"
        elif baseurl.lower().find("/sharing/rest") > -1:
            if baseurl.endswith("/"):
                try:
                    res = self.get(
                        baseurl + "info",
                        params={"f": "json"},
                        add_token=False,
                    )
                except:
                    res = self.get(baseurl + "info", params={"f": "json"})
            else:
                try:
                    res = self.get(
                        baseurl + "/info",
                        params={"f": "json"},
                        add_token=False,
                    )
                except:
                    res = self.get(baseurl + "/info", params={"f": "json"})
            if (
                self._token_url is None
                and res is not None
                and isinstance(res, dict)
                and "authInfo" in res
                and "tokenServicesUrl" in res["authInfo"]
                and res["authInfo"]["isTokenBasedSecurity"]
            ):
                parsed_from_system = urlparse(res["authInfo"]["tokenServicesUrl"])
                parsed = urlparse(baseurl)
                if (
                    parsed.netloc.lower() != parsed_from_system.netloc.lower()
                    and parsed.netloc.find(":7443") > -1
                ):  # WA not being used for token url
                    self._token_url = os.path.join(
                        parsed_from_system.scheme + "://",
                        parsed.netloc
                        + "/arcgis/"
                        + "/".join(parsed_from_system.path[1:].split("/")[1:]),
                    )
                    url_test = self._session.post(
                        self._token_url, {"f": "json"}, allow_redirects=False
                    )
                    if url_test.status_code == 301:
                        self._token_url = url_test.headers["location"]
                        if self._baseurl != os.path.dirname(
                            url_test.headers["location"]
                        ):
                            self._baseurl = os.path.dirname(
                                url_test.headers["location"]
                            )
                else:
                    self._token_url = res["authInfo"]["tokenServicesUrl"]
            elif (
                self._token_url is None
                and res is not None
                and isinstance(res, dict)
                and "authInfo" in res
                and "tokenServicesUrl" in res["authInfo"]
                and res["authInfo"]["isTokenBasedSecurity"] == False
            ):
                self._token_url = None
                self._auth = "OTHER"
            self._product = "PORTAL"
            return "PORTAL"
        else:
            # Brute Force Method
            parsed = urlparse(baseurl)
            root = (
                rf"{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split(r'/')[0]}"
            )
            parts = [
                "/info",
                "/rest/services",
                "/rest/info",
                "/sharing/rest/info",
            ]
            params = {"f": "json"}
            for pt in parts:
                try:
                    res = self.get(
                        root + pt,
                        params=params,
                        add_token=False,
                        allow_redirects=False,
                    )
                    if "folders" in res:
                        self._product = "SERVER"
                        return self._check_product()
                    elif (
                        self._token_url is None
                        and res is not None
                        and isinstance(res, dict)
                        and "authInfo" in res
                        and "tokenServicesUrl" in res["authInfo"]
                        and res["authInfo"]["isTokenBasedSecurity"]
                    ):
                        self._token_url = res["authInfo"]["tokenServicesUrl"]
                    elif (
                        self._token_url is None
                        and res is not None
                        and isinstance(res, dict)
                        and "authInfo" in res
                        and "tokenServicesUrl" in res["authInfo"]
                        and res["authInfo"]["isTokenBasedSecurity"] == False
                    ):
                        self._token_url = None
                        self._auth = "OTHER"
                except HTTPError as e:
                    _log.warning(str(e))
                    res = ""
                except json.decoder.JSONDecodeError:
                    res = ""
                except Exception as e:
                    res = ""
                if (
                    isinstance(res, dict)
                    and "currentVersion" in res
                    and self._token_url
                ):
                    t_parsed = urlparse(self._token_url[1:]).path
                    b_parsed = urlparse(self._baseurl[1:]).path
                    if t_parsed.startswith("/"):
                        t_parsed = t_parsed[1:].split("/")[0]
                    else:
                        t_parsed = t_parsed.split("/")[0]
                    if b_parsed.startswith("/"):
                        b_parsed = b_parsed[1:].split("/")[0]
                    else:
                        b_parsed = b_parsed.split("/")[0]
                    if t_parsed.lower() != b_parsed.lower():
                        self._token_url = None
                        if self._portal_connection:
                            self._product = "FEDERATED_SERVER"
                            return "FEDERATED_SERVER"
                        else:
                            from arcgis.gis import GIS

                            self._portal_connection = GIS(
                                url=res["authInfo"]["tokenServicesUrl"].split(
                                    "/sharing/"
                                )[0],
                                username=self._username,
                                password=self._password,
                                verify_cert=self._verify_cert,
                            )._con
                            self._product = "FEDERATED_SERVER"
                            return "FEDERATED_SERVER"
                    self._product = "SERVER"
                    return "SERVER"
                elif (
                    isinstance(res, dict)
                    and "currentVersion" in res
                    and self._token_url is None
                ):
                    self._product = "SERVER"
                    return "SERVER"
                del pt
                del res
        self._product = "PORTAL"
        return "PORTAL"

    def _create_token(self, url: str) -> str:
        """
        When a service needs a token that is not in the token, this method obtains it.
        Only the following authentications require a URL: EsriPKIAuth, EsriBasicAuth, EsriKerberosAuth, EsriWindowsAuth, or EsriGenTokenAuth
        :returns: str (token is possible)
        """
        from arcgis.auth import (
            EsriNotebookAuth,
            EsriAPIKeyAuth,
            EsriPKIAuth,
            EsriBasicAuth,
            EsriKerberosAuth,
            EsriWindowsAuth,
            BaseEsriAuth,
        )
        from arcgis.auth.tools import parse_url

        if isinstance(
            self._session.auth,
            (
                EsriUserTokenAuth,
                EsriOAuth2Auth,
                EsriNotebookAuth,
                EsriAPIKeyAuth,
                ArcGISProAuth,
                EsriBuiltInAuth,
            ),
        ):
            return self._session.auth.token
        elif isinstance(self._session.auth, EsriGenTokenAuth):
            return self._session.auth.token(url)
        elif isinstance(
            self._session.auth,
            (EsriPKIAuth, EsriBasicAuth, EsriKerberosAuth, EsriWindowsAuth),
        ):
            if self._server_log is None:
                self._server_log = {}
            parsed = parse_url(url)
            expiration = 16000
            if parsed.port:
                if parsed.port in parsed.netloc:
                    server_url = f'{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split("/")[0]}'
                else:
                    server_url = f'{parsed.scheme}://{parsed.netloc}:{parsed.port}/{parsed.path[1:].split("/")[0]}'
            else:
                server_url = (
                    f'{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split("/")[0]}'
                )
            postdata = {
                "request": "getToken",
                "serverURL": server_url,
                "referer": self._referer or "http",
                "f": "json",
            }

            if expiration:
                postdata["expiration"] = expiration
            if parsed.netloc in self._server_log:
                token_url = self._server_log[parsed.netloc]
            else:
                info = self._session.get(
                    server_url + "/rest/info?f=json",
                    auth=self._session.auth,
                    verify=self._verify_cert,
                ).json()
                token_url = info["authInfo"]["tokenServicesUrl"]
                self._server_log[parsed.netloc] = token_url

            token = self._session.post(
                token_url,
                data=postdata,
                auth=self._session.auth,
                verify=self._verify_cert,
            )
            return token.json().get("token", None)
        elif isinstance(self._session.auth, BaseEsriAuth):
            return self._session.auth.token
        return None
