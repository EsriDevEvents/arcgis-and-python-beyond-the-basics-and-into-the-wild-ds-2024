from __future__ import annotations
from cachetools import cached, TTLCache
import lxml.html
from urllib.parse import urlunparse, quote, parse_qsl, parse_qs, urlparse
from functools import lru_cache
from typing import Any
from getpass import getpass
from requests.auth import AuthBase
from requests_oauthlib import OAuth2Session
from requests.cookies import extract_cookies_to_jar

from ._schain import SupportMultiAuth
from ..tools._lazy import LazyLoader
from ..tools import parse_url, assemble_url
from .._error import ArcGISLoginError

warnings = LazyLoader("warnings")
re = LazyLoader("re")
json = LazyLoader("json")
threading = LazyLoader("threading")
_dt = LazyLoader("datetime")
requests = LazyLoader("requests")
requests_oauthlib = LazyLoader("requests_oauthlib")
warnings = LazyLoader("warnings")

_MSG = """

You need to a security question by integer:

1. What city were you born in?
2. What was your high school mascot?
3. What is your mother's maiden name?
4. What was the make of your first car?
5. What high school did you go to?
6. What is the last name of your best friend?
7. What is the middle name of your youngest sibling?
8. What is the name of the street on which you grew up?
9. What is the name of your favorite fictional character?
10. What is the name of your favorite pet?
11. What is the name of your favorite restaurant?
12. What is the title of your favorite book?
13. What is your dream job?
14. Where did you go on your first date?
"""


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _token_url_validator(
    url: str,
    session: "EsriSession",
    verify: bool = False,
    proxies: frozenset = None,
) -> str:
    """validates the token url from the give URL"""
    parts = ["/info", "/rest/info", "/sharing/rest/info"]
    params = {"f": "json"}
    if proxies:
        proxies = dict(proxies)
    parsed_url = _parse_arcgis_url(url=url)
    token_url = None  # parsed_url + "/sharing/rest/generateToken"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pt in parts:
            try:
                resp = session.get(
                    f"{parsed_url}{pt}?f=json",
                    proxies=proxies,
                    verify=verify,
                )  # need to include proxies, verify parameter
                token_url = resp.json()["authInfo"]["tokenServicesUrl"]
                if token_url:
                    break
                del pt
            except Exception as e:
                pass
    return token_url


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _parse_arcgis_url(url: str) -> str:
    """
    Returns a valid ArcGIS Online or ArcGIS Enterprise base URL

    :returns: str
    """
    if url is None or url.lower().find("www.arcgis.com") > -1:
        return "https://www.arcgis.com"
    parsed = parse_url(url)
    if len(parsed.path) <= 1:
        return f"{parsed.scheme}://{parsed.netloc}"
    else:
        idx = parsed.path.lower().find("sharing/rest")
        if idx != -1:
            wa = parsed.path[1:idx]
            wa = wa.replace("/", "")
        else:
            wa = parsed.path[1:].split("/")
            wa = wa[0].replace("/", "")
        if len(wa) == 0:
            return f"{parsed.scheme}://{parsed.netloc}"
        return f"{parsed.scheme}://{parsed.netloc}/{wa}"


###########################################################################
class ArcGISServerAuth(AuthBase, SupportMultiAuth):
    """
    Performs the ArcGIS Server (ags) Authentication for a given request.
    """

    _arcpy = None
    _referer = None
    _invalid_token_urls = None
    _401_counters = None
    _ags = None

    # ----------------------------------------------------------------------
    def __init__(self, ags_file: str, legacy: bool = False):
        try:
            self._arcpy = LazyLoader("arcpy", strict=True)
            self.legacy = legacy
            self._invalid_token_urls = set()
            self._401_counters = dict()
            self._ags = ags_file
        except:
            raise

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    @property
    def token(self):
        """obtains the login token"""
        return self._ags_token()

    @lru_cache(maxsize=255)
    def _read_ags_file(self, ags_file: str) -> dict[str, Any]:
        """reads the ags file into cache"""
        return self._arcpy.gp.getStandaloneServerToken(self._ags) or {}

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def _url(self, ags_file) -> str:
        if self._arcpy:
            resp = self._read_ags_file(ags_file)
            return _parse_arcgis_url(resp.get("serverUrl")) + "/rest/services"
        else:
            raise Exception("ArcPy not found, please install arcpy")

    # ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """gets the token for various products"""
        if self._arcpy:
            return self._url(ags_file=self._ags)

        else:
            raise Exception("ArcPy not found, please install arcpy")

    # ----------------------------------------------------------------------
    def _ags_token(self):
        """gets the token for various products"""
        if self._arcpy:
            resp = self._read_ags_file(self._ags)
            if resp:
                if "referer" in resp:
                    self._referer = resp["referer"]
                if "token" in resp:
                    return resp["token"]
                else:
                    raise Exception("Could not generate token.")
            else:
                raise Exception(
                    (
                        "Could not login using Pro authencation."
                        "Please verify in Pro that you are logged in."
                    )
                )
        else:
            raise Exception("ArcPy not found, please install arcpy")

    # ----------------------------------------------------------------------
    def handle_40x(self, r, **kwargs):
        """Handles Case where token is invalid"""
        parsed = parse_url(r.url)
        server_url = assemble_url(parsed)
        if (r.status_code < 500 and r.status_code > 399) and str(r.text).lower().find(
            "invalid token"
        ) > -1:
            # Recreate the request without the token
            #
            parsed = parse_url(r.url)

            self._invalid_token_urls.add(server_url)
            r.content
            r.raw.release_conn()
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer or "http"
            _r.history.append(r)
            return _r
        elif str(r.text).lower().find("invalid token") > -1:
            # Recreate the request without the token
            #
            parsed = parse_url(r.url)
            self._invalid_token_urls.add(server_url)
            r.content
            r.raw.release_conn()
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer or "http"
            _r.history.append(r)
            return _r
        return r

    # ----------------------------------------------------------------------
    def __call__(self, r):
        """Handles the Token Authorization Logic"""
        if self._invalid_token_urls is None:
            self._invalid_token_urls = set()
        parsed = parse_url(r.url)
        server_url = assemble_url(parsed)
        if not server_url in self._invalid_token_urls:
            r.register_hook("response", self.handle_40x)
            if self.legacy == False and self.token:
                r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
                r.headers["referer"] = self._referer or ""
            elif self.legacy == False and not self.token:
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "GET" and self.token:
                r.prepare_url(url=r.url, params={"token": self.token})
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "GET" and not self.token:
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "POST" and self.token:
                data = parse_qs(r.body)
                data["token"] = self.token
                r.prepare_body(data, None, None)
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "POST" and not self.token:
                data = parse_qs(r.body)
                r.prepare_body(data, None, None)
                r.headers["referer"] = self._referer or ""
            else:
                raise Exception(
                    "Only POST and GET are supported with legacy methods of authentication."
                )
            try:
                self.pos = r.body.tell()
            except AttributeError:
                self.pos = None

            return r
        else:
            r.headers.pop("X-Esri-Authorization", None)
        return r


###########################################################################
class ArcGISProAuth(AuthBase, SupportMultiAuth):
    """
    Performs the ArcGIS Pro Authentication for a given request.
    """

    _arcpy = None
    _referer = None
    _invalid_token_urls = None
    _401_counters = None

    def __init__(self, legacy: bool = False):
        try:
            self._arcpy = LazyLoader("arcpy", strict=True)
            self.legacy = legacy
            self._invalid_token_urls = set()
            self._401_counters = dict()
        except:
            raise

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    @property
    def token(self):
        """obtains the login token"""
        return self._pro_token()

    # ----------------------------------------------------------------------
    def _pro_token(self):
        """gets the token for various products"""
        if self._arcpy:
            resp = self._arcpy.GetSigninToken()
            if resp:
                if "referer" in resp:
                    self._referer = resp["referer"]
                if "token" in resp:
                    return resp["token"]
                else:
                    raise Exception("Could not generate token.")
            else:
                raise Exception(
                    (
                        "Could not login using Pro authencation."
                        "Please verify in Pro that you are logged in."
                    )
                )
        else:
            raise Exception("ArcPy not found, please install arcpy")

    # ----------------------------------------------------------------------
    def handle_40x(self, r, **kwargs):
        """Handles Case where token is invalid"""
        parsed = parse_url(r.url)
        server_url = assemble_url(parsed)
        if (r.status_code < 500 and r.status_code > 399) and str(r.text).lower().find(
            "invalid token"
        ) > -1:
            # Recreate the request without the token
            #
            parsed = parse_url(r.url)

            self._invalid_token_urls.add(server_url)
            r.content
            r.raw.release_conn()
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer or "http"
            _r.history.append(r)
            return _r
        elif str(r.text).lower().find("invalid token") > -1:
            # Recreate the request without the token
            #
            parsed = parse_url(r.url)
            self._invalid_token_urls.add(server_url)
            r.content
            r.raw.release_conn()
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer or "http"
            _r.history.append(r)
            return _r
        return r

    # ----------------------------------------------------------------------
    def __call__(self, r):
        """Handles the Token Authorization Logic"""
        if self._invalid_token_urls is None:
            self._invalid_token_urls = set()
        parsed = parse_url(r.url)
        server_url = assemble_url(parsed)
        if not server_url in self._invalid_token_urls:
            r.register_hook("response", self.handle_40x)
            if self.legacy == False:
                r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "GET":
                r.prepare_url(url=r.url, params={"token": self.token})
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "POST":
                data = parse_qs(r.body)
                data["token"] = self.token
                r.prepare_body(data, None, None)
                r.headers["referer"] = self._referer or ""
            else:
                raise Exception(
                    "Only POST and GET are supported with legacy methods of authentication."
                )
            try:
                self.pos = r.body.tell()
            except AttributeError:
                self.pos = None

            return r
        else:
            r.headers.pop("X-Esri-Authorization", None)
        return r


###########################################################################
class EsriBuiltInAuth(AuthBase, SupportMultiAuth):
    """
    Performs the BUILT-IN Login Authorization for ArcGIS Online and Enterprise
    """

    _params = None
    _auth_token = None
    _authorization_url = None
    _state = None
    _auto_refresh_extra_params = None
    _clientid = None
    _username = None
    _password = None
    _re_expiration = None
    _create_time = None
    _expiration_time = None
    _oauth_info = None
    _referer = None
    _no_go_token = None
    _response_type = None
    _client = None
    _expiration = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        expiration: int = None,
        legacy: bool = False,
        verify_cert: bool = True,
        referer: str = None,
        **kwargs,
    ):
        """init"""
        from requests.packages import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._expiration = expiration or 20160
        self._response_type = kwargs.pop("response_type", "token")
        if self._response_type == "token":
            self._clientid = "pythonapi"  # "arcgisonline"
        else:
            self._clientid = kwargs.get("clientid", "pythonapi")  # "arcgispro"
        if self._response_type == "token":
            from oauthlib.oauth2 import MobileApplicationClient

            self._client = MobileApplicationClient(client_id=self._clientid)

        if referer is None:
            self._referer = ""
        url = _parse_arcgis_url(url=url)
        self.legacy = legacy
        self._verify_cert = verify_cert
        self.proxies = kwargs.pop("proxies", {})
        self._base_url = url
        self._auth_url = f"{url}/sharing/rest/oauth2/authorize"
        self._token_url = f"{url}/sharing/rest/oauth2/token"
        self._signin_url = f"{url}/sharing/oauth2/signin"
        self._reset_password_url = f"{url}/sharing/oauth2/resetPassword"
        self._update_profile_url = f"{url}/sharing/oauth2/updateUserProfile"
        self._mfa_url = f"{url}/sharing/oauth2/mfa"
        self._mfa_code = kwargs.pop("mfa_code", None)
        self._no_go_token = set()
        self._username = username
        self._password = password
        self._referer = referer or ""

        self._re_expressions = {
            "step-1a": re.compile("var oAuthInfo = ({.*?});", re.DOTALL),
            "step-1b": re.compile("var oAuthInfo = ({.*?})", re.DOTALL),
            "step-1c": re.compile("var\s+(\w+)\s*=\s*({.*?})", re.DOTALL),
            "step-2": re.compile(r"<title>SUCCESS code=(.*?)</title>", re.DOTALL),
            "password_reset": re.compile(r"{.*\:.*}"),
        }
        self._params = {"expiration": expiration or 1440}

        self._auto_refresh_extra_params = {"client_id": self._clientid}

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def suspend(self) -> bool:
        """
        Invalidates the login and checks any licenses back in

        :return: Bool.  If True, the licenses is checked back in successfully,
                        False, the process failed.
        """

        if self._auth_token and "refresh_token" in self._auth_token:
            params = {
                "grant_type": "suspend_session",
                "client_id": self._clientid,  # "arcgispro",
                "refresh_token": self._auth_token["refresh_token"],
                "f": "json",
            }
        else:
            return True
        try:
            return (
                requests.post(
                    self._token_url,
                    data=params,
                    verify=self._verify_cert,
                    proxies=self.proxies,
                )
                .json()
                .get("success", False)
            )
        except:
            return False

    def _init_response_type_token(self):
        """"""
        import requests_oauthlib

        redirect_uri = f"https://{parse_url(self._auth_url).netloc}"  # "urn:ietf:wg:oauth:2.0:oob" does not work for MobileApplicationClient

        session = requests_oauthlib.OAuth2Session(
            self._clientid,
            client=self._client,
            redirect_uri=redirect_uri,
        )
        auth_url, state = session.authorization_url(
            self._auth_url,
            expiration=self._expiration,
            **{
                "allow_verification": "false",
                "style": "dark",
                "locale": "en-US",
            },
        )
        auth_response = requests.get(
            url=auth_url,
            proxies=self.proxies,
            verify=self._verify_cert,
        ).text
        if "oauth_state" in auth_response:
            oauth_state = auth_response.split('"oauth_state":"')[1].split('"')[0]
        else:
            raise Exception("Unable to generate oauth token")
        params = {
            "expiration": self._expiration,
            "oauth_state": oauth_state,
            "username": self._username,
            "password": self._password,
        }

        response = requests.post(
            self._signin_url,
            data=params,
            allow_redirects=False,
            proxies=self.proxies,
            verify=self._verify_cert,
        )
        #
        # After authenticating, ArcGIS Online/Enterprise can prompt for a
        # terms and conditions acceptance.
        #
        if response.text.find("OAUTH_0015") > -1:
            raise ArcGISLoginError()
        callback_url = response.headers["location"]
        if callback_url.find("acceptTermsAndConditions") > -1:
            parsed = parse_url(response.headers["location"])
            oauth_state = parse_qs(parsed.query)["oauth_state"][0]

            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            params = {
                "oauth_state": oauth_state,
                "acceptTermsAndConditions": True,
            }
            response = requests.post(
                url,
                params,
                verify=self._verify_cert,
                allow_redirects=False,
                proxies=self.proxies,
            )
            callback_url = response.headers["location"]
        self._expiration_time = _dt.datetime.now() + _dt.timedelta(seconds=1440)

        # Now we extract the token from the URL to make use of it.
        self._auth_token = session.token_from_fragment(callback_url)
        if "expires_at" in self._auth_token:
            self._expiration_time = _dt.datetime.fromtimestamp(
                self._auth_token["expires_at"]
            )
        elif "expires" in self._auth_token:
            self._expiration_time = _dt.datetime.now() + _dt.timedelta(
                seconds=self._auth_token["expiration"]
            )
        else:
            self._expiration_time = _dt.datetime.now() + _dt.timedelta(seconds=300)

    # ----------------------------------------------------------------------
    def _init_token_auth_handshake(self):
        """perform initial handshake"""
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        self._oauth = OAuth2Session(self._clientid, redirect_uri=redirect_uri)
        authorization_url, state = self._oauth.authorization_url(
            self._auth_url,
            expiration=20160,
            **{
                "allow_verification": "false",
                "style": "dark",
                "locale": "en-US",
            },
        )
        self._authorization_url = authorization_url
        self._state = state
        content = requests.get(
            self._authorization_url,
            verify=self._verify_cert,
            proxies=self.proxies,
        ).text
        if content.find("Error: Invalid client_id") > -1:
            self._clientid = "arcgisonline"
            from oauthlib.oauth2 import MobileApplicationClient

            self._client = MobileApplicationClient(client_id=self._clientid)
            self._init_response_type_token()
            return
        oauth_info = None
        pattern = self._re_expressions["step-1a"]
        if len(pattern.findall(content)) == 0:
            pattern = self._re_expressions["step-1b"]
        soup = lxml.html.fromstring(content)
        for script in soup.xpath("//script/text()"):
            script_code = str(script).strip()
            matches = pattern.search(script_code)
            if not matches is None:
                js_object = matches.groups()[0]
                try:
                    oauth_info = json.loads(js_object)
                except:
                    oauth_info = json.loads(js_object + "}")
                break
        if oauth_info:
            oauth_state = oauth_info["oauth_state"]
        else:
            raise ArcGISLoginError(
                "Could not login. Please ensure you have valid credentials and set your security login question."
            )

        signin_params = {
            "oauth_state": oauth_state,
            "authorize": True,
            "username": self._username,
            "password": self._password,
        }
        signin_resp = requests.post(
            self._signin_url,
            signin_params,
            verify=self._verify_cert,
            allow_redirects=True,
            proxies=self.proxies,
        )
        matches = pattern.findall(signin_resp.text)
        if len(matches) > 0:
            try:
                sign_json = json.loads(matches[0].strip())
            except:
                sign_json = json.loads(matches[0].strip() + "}")
        else:
            sign_json = {}
        if "messages" in sign_json and (
            any(
                [
                    msg.lower().find("invalid username or password") > -1
                    for msg in sign_json["messages"]
                ]
            )
            or any(
                [
                    msg.find("Too many invalid attempts. Please try again later.") > -1
                    for msg in sign_json["messages"]
                ]
            )
        ):
            raise ValueError(",".join(sign_json["messages"]))
        elif signin_resp.url.lower().find("accepttermsandconditions") > -1:
            parsed = parse_url(signin_resp.url)
            oauth_state = parse_qs(signin_resp.url.split("?")[-1])["oauth_state"][0]
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            params = {
                "oauth_state": oauth_state,
                "acceptTermsAndConditions": True,
            }
            signin_resp = requests.post(
                url,
                params,
                verify=self._verify_cert,
                allow_redirects=True,
                proxies=self.proxies,
            )
            resp_text = signin_resp.text
            exp = r"<title>SUCCESS code=(.*?)</title>"
            pattern = self._re_expressions["step-2"]
            code = pattern.findall(resp_text)[0]
        elif signin_resp.url.lower().find("updateuserprofile") > -1:
            # raise Exception(
            # (
            # "This is your first time logging in and you are required"
            # " to setup a new password manually before logging in."
            # )
            # )

            print(
                "This is your first time logging in and you are required to setup a new password."
            )
            new_password = getpass(prompt="Please Enter a New Password: ")
            assert self._password != new_password

            resp_text = signin_resp.text
            pattern = self._re_expressions["password_reset"]
            oauth_state = json.loads(pattern.findall(resp_text)[0].replace(" ", ""))[
                "oauth_state"
            ]
            params = {
                "password": self._password,
                "newPassword": new_password,
                "newPassword2": new_password,
                "oauth_state": oauth_state,
                "f": "json",
            }
            resp = requests.post(
                self._reset_password_url,
                params,
                verify=self._verify_cert,
                allow_redirects=True,
                proxies=self.proxies,
            )
            self._password = new_password
            oauth_state = json.loads(pattern.findall(resp_text)[0].replace(" ", ""))[
                "oauth_state"
            ]
            print(_MSG)
            question = int(getpass(prompt="Select a question by integer: "))
            answer = getpass("Answer to the question: ")
            params = {
                "f": "json",
                "securityQuestionIdx": question,
                "securityAnswer": answer,
                "oauth_state": oauth_state,
            }
            resp = requests.post(
                self._update_profile_url,
                params,
                verify=self._verify_cert,
                allow_redirects=True,
                proxies=self.proxies,
            )
            self._init_token_auth_handshake()
        elif signin_resp.url.lower().find("/mfa") > -1:
            from ..tools._util import mfa_otp

            mfa_code = self._mfa_code
            if self._mfa_code is None:
                verify_code = input("Please input your 2FA code: ")
            else:
                verify_code = mfa_otp(mfa_code)

            oauth_info = None
            pattern = self._re_expressions["step-1a"]
            if len(pattern.findall(content)) == 0:
                pattern = self._re_expressions["step-1b"]
            soup = lxml.html.fromstring(signin_resp.text)
            for script in soup.xpath("//script/text()"):
                script_code = str(script).strip()
                matches = pattern.search(script_code)
                if not matches is None:
                    js_object = matches.groups()[0]
                    try:
                        oauth_info = json.loads(js_object)
                    except:
                        oauth_info = json.loads(js_object + "}")
                    break
            mfa_params: dict[str, str] = {
                "oauth_state": oauth_info["oauth_state"],
                "authResponse": "",
                "totp": "",
                "mfa_code": verify_code,
                "recovery_code": "",
            }

            resp = requests.post(
                self._mfa_url,
                data=mfa_params,
                verify=self._verify_cert,
                allow_redirects=False,
                proxies=self.proxies,
            )
            resp_text = requests.get(resp.headers["location"]).text
            exp = r"<title>SUCCESS code=(.*?)</title>"
            pattern = self._re_expressions["step-2"]
            code = pattern.findall(resp_text)[0]
            self._auth_token = self._oauth.fetch_token(
                token_url=self._token_url,
                code=code,
                verify=self._verify_cert,
                include_client_id=True,
                proxies=self.proxies,
                **{"expiration": 20160},
            )
        else:
            resp_text = signin_resp.text
            exp = r"<title>SUCCESS code=(.*?)</title>"
            pattern = self._re_expressions["step-2"]
            code = pattern.findall(resp_text)[0]
        self._auth_token = self._oauth.fetch_token(
            token_url=self._token_url,
            code=code,
            verify=self._verify_cert,
            include_client_id=True,
            proxies=self.proxies,
            **{"expiration": 20160},
        )
        if "expires_at" in self._auth_token:
            self._expiration_time = _dt.datetime.fromtimestamp(
                self._auth_token["expires_at"]
            )
        elif "expires" in self._auth_token:
            self._expiration_time = _dt.datetime.now() + _dt.timedelta(
                seconds=self._auth_token["expiration"]
            )
        else:
            self._expiration_time = _dt.datetime.now() + _dt.timedelta(seconds=300)

    # ----------------------------------------------------------------------
    @property
    def token(self):
        """obtains the login token"""
        try:
            if self._auth_token:
                if (
                    _dt.datetime.now() + _dt.timedelta(minutes=5)
                ) >= self._expiration_time:
                    self._refresh()
                return self._auth_token["access_token"]
            else:
                if self._response_type == "token":
                    self._init_response_type_token()
                else:
                    self._init_token_auth_handshake()
                return self.token
        except:
            self._auth_token = None
            self._init_token_auth_handshake()
            if self._auth_token:
                return self.token
            else:
                raise
        return None

    # ----------------------------------------------------------------------
    def __call__(self, r):
        if self.legacy == False:
            r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
        elif self.legacy and r.method == "GET":
            r.prepare_url(url=r.url, params={"token": self.token})
        elif self.legacy and r.method == "POST":
            data = parse_qs(r.body)
            data["token"] = self.token
            r.prepare_body(data, None, None)
        else:
            raise Exception(
                "Only POST and GET are supported with legacy methods of authentication."
            )
        r.register_hook("response", self.handle_401)
        try:
            self.pos = r.body.tell()
        except AttributeError:
            # In the case of HTTPKerberosAuth being reused and the body
            # of the previous request was a file-like object, pos has
            # the file position of the previous body. Ensure it's set to
            # None.
            self.pos = None
        # return request

        return r

    # ----------------------------------------------------------------------
    def handle_401(self, r, **kwargs):
        """
        handles the issues in the response where token might be rejected
        """
        parsed = parse_url(r.url)
        if (
            r.text.lower().find("invalid token") > -1
            or r.text.find("Token is valid but access is denied.") > -1
            or (parsed.scheme, parsed.netloc, parsed.path) in self._no_go_token
        ):
            # parsed = parse.urlparse(r.url)
            self._no_go_token.add((parsed.scheme, parsed.netloc, parsed.path))
            # Recreate the request without the token
            #
            r.content
            r.raw.release_conn()
            r.request.headers["referer"] = self._referer  # or "http"
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer  # or "http"
            _r.headers.pop("X-Esri-Authorization", None)
            _r.history.append(r)
            return _r
        elif r.text.lower().find("token required") > -1:
            r.content
            r.raw.release_conn()
            r.request.headers["referer"] = self._referer  # or "http"
            r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self._referer  # or "http"
            _r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
            _r.history.append(r)
            return _r
        return r

    # ----------------------------------------------------------------------
    def _refresh(self):
        """renews the token"""
        if self._response_type == "token":
            self._expiration_time = _dt.datetime.now() + _dt.timedelta(
                seconds=self._expiration - 1
            )
            self._init_response_type_token()
        else:
            self._auth_token = self._oauth.refresh_token(
                token_url=self._token_url,
                verify=self._verify_cert,
                client_id=self._oauth.client_id,
                expiration=20160,
            )
            if "expires_at" in self._auth_token:
                self._expiration_time = _dt.datetime.fromtimestamp(
                    self._auth_token["expires_at"]
                )
            elif "expires" in self._auth_token:
                self._expiration_time = _dt.datetime.now() + _dt.timedelta(
                    seconds=self._auth_token["expiration"]
                )
            else:
                self._expiration_time = _dt.datetime.now() + _dt.timedelta(seconds=300)


###########################################################################
class EsriGenTokenAuth(AuthBase, SupportMultiAuth):
    """
    This form of Authentication leverages the `generateToken` endpoint from
    the ArcGIS Product.  This is supported for ArcGIS Online, ArcGIS Enterprise
    and ArcGIS Server.

    This form of authentication is considered legacy and should only be used
    with unfederated server products.
    """

    _anon_urls = None
    _session = None
    _token = None
    _expires_on = None
    _portal_auth = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        token_url: str,
        referer: str,
        username: str = None,
        password: str = None,
        portal_auth: "EsriGenTokenAuth" = None,
        time_out: int = 1440,
        verify_cert: bool = True,
        legacy: bool = False,
        **kwargs,
    ) -> None:
        """init"""
        self.proxies = kwargs.pop("proxies", None)
        if username is None and portal_auth is None:
            raise Exception(
                "A portal_auth or username/password is required for GenerateToken"
            )
        self._anon_urls = set()
        self._legacy_auth = legacy
        self._portal_auth = portal_auth
        has_session = "session" in kwargs
        self._session = kwargs.pop("session", None) or requests.sessions.Session()
        self.verify_cert = verify_cert
        if has_session == False:
            self._session.verify = verify_cert
            self._session.allow_redirects = True
        if self.proxies:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                token_url = _token_url_validator(
                    _parse_arcgis_url(token_url),
                    session=self._session,
                    verify=False,
                    proxies=frozenset(self.proxies.items()),
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                token_url = _token_url_validator(
                    _parse_arcgis_url(token_url),
                    session=self._session,
                    verify=False,
                )
        self._thread_local = threading.local()

        self._expires_on = None
        self._token_url = token_url

        self.username = username
        self.password = password
        if referer is None and self.referer is None:
            self.referer = "http"
        else:
            self.referer = referer
        if not time_out or time_out < 5:
            self._time_out = 1440
        else:
            self._time_out = time_out

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    @property
    def referer(self) -> str:
        if "referer" in self._session.headers:
            return self._session.headers["referer"]
        else:
            self.referer = "http"
            return self.referer

    # ----------------------------------------------------------------------
    @referer.setter
    def referer(self, value: str) -> None:
        """Gets/Sets the referer"""
        if (
            "referer" in self._session.headers
            and self._session.headers["referer"] != value
        ):
            self._session.headers["referer"] = value
        else:
            self._session.headers["referer"] = value

    # ----------------------------------------------------------------------
    @property
    def time_out(self) -> int:
        """returns the time out time in minutes"""
        return self._time_out

    # ----------------------------------------------------------------------
    @time_out.setter
    def time_out(self, value: int):
        if value is None:
            value = 60
        if self._time_out != value:
            self._time_out = value

    # ----------------------------------------------------------------------
    @property
    def expiration(self) -> _dt.datetime:
        """Gets the time the token will expire on"""
        if self._expires_on:
            return self._expires_on
        else:
            self.token
            return self.expiration

    # ----------------------------------------------------------------------
    @cached(cache=TTLCache(maxsize=255, ttl=60))
    def token(self, server_url=None) -> str:
        if self._token:
            if (_dt.datetime.now() - _dt.timedelta(minutes=5)) >= self.expiration:
                self._token = None
            return self._token
        elif self.username and self.password:
            return self._init_token_auth_handshake()
        elif self._portal_auth:
            return self._init_token_auth_handshake(server_url)

        return None

    # ----------------------------------------------------------------------
    def __call__(self, r):
        """
        Handles the token authentication for the call
        """
        parsed = parse_url(r.url)
        server_url = f"{parsed.scheme}://{parsed.netloc}"
        self.init_per_thread_state()
        if self._legacy_auth and r.method == "GET":
            r.prepare_url(url=r.url, params={"token": self.token(server_url)})
        elif self._legacy_auth and r.method == "POST":
            data = parse_qs(r.body)
            data["token"] = self.token(server_url)
            r.prepare_body(data, None, None)
        else:
            r.headers["X-Esri-Authorization"] = f"Bearer {self.token(server_url)}"
        r.headers["referer"] = self.referer
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        try:
            self.pos = r.body.tell()
        except AttributeError:
            # In the case of HTTPKerberosAuth being reused and the body
            # of the previous request was a file-like object, pos has
            # the file position of the previous body. Ensure it's set to
            # None.
            self.pos = None
        # return request
        self._thread_local.num_401_calls = 1
        return r

    # ----------------------------------------------------------------------
    def init_per_thread_state(self):
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    # ----------------------------------------------------------------------

    def handle_401(self, r, **kwargs):
        # if r.status_code in [401, 402, 403]:
        # raise Exception(f"Error: {r.status_code}, {r.text}")
        if not 400 <= r.status_code < 500 and (
            r.status_code == 200 and r.text.find("Invalid Token") == -1
        ):
            self._thread_local.num_401_calls = 1
            return r
        elif r.status_code == 200 and r.text.find("Invalid Token") > -1:
            if self._thread_local.num_401_calls < 2:
                self._thread_local.num_401_calls += 1
                # Consume content and release the original connection
                # to allow our new request to reuse the same one.
                r.content
                r.close()
                prep = r.request.copy()
                if self._legacy_auth and prep.method == "GET":
                    parsed = parse_url(prep.url)
                    url = urlunparse(
                        (
                            parsed.scheme,
                            parsed.netloc,
                            parsed.path,
                            "",
                            "",
                            "",
                        )
                    )
                    kv = dict(parse_qsl(parsed.query))
                    kv.pop("token", None)
                    prep.prepare_url(url=url, params=kv)
                elif self._legacy_auth and prep.method == "POST":
                    data = parse_qs(prep.body)
                    data.pop("token", None)
                    prep.prepare_body(data, None, None)

                extract_cookies_to_jar(prep._cookies, r.request, r.raw)
                prep.prepare_cookies(prep._cookies)
                prep.headers.pop("X-Esri-Authorization", None)
                _r = r.connection.send(prep, **kwargs)
                _r.history.append(r)
                _r.request = prep
                return _r
        return r

    # ----------------------------------------------------------------------

    def handle_redirect(self, r, **kwargs):
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    # ----------------------------------------------------------------------
    @cached(cache=TTLCache(maxsize=255, ttl=60))
    def _init_token_auth_handshake(self, server_url=None):
        """gets the token"""
        if self.username and self.password:  # Basic Generate Token Logic
            self.time_out = 60
            postdata = {
                "username": self.username,
                "password": self.password,
                "referer": self.referer,
                "client": "referer",
                "expiration": 60,  # self.time_out,
                "f": "json",
            }
            resp = self._session.post(
                url=self._token_url,
                data=postdata,
                verify=self.verify_cert,
                proxies=self.proxies,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    raise Exception(data)
                if "expires" in data:
                    self._expires_on = _dt.datetime.fromtimestamp(
                        int(data["expires"]) / 1000
                    )
                elif "expiration" in data:
                    self._expires_on = _dt.datetime.fromtimestamp(
                        int(data["expiration"])
                    )
                else:
                    self._expires_on = _dt.datetime.now() + _dt.timedelta(
                        minutes=self._time_out - 3
                    )
                self._token = data["token"]
                return self._token
            else:
                raise Exception("Could not generate the token")
        elif self._portal_auth:  # Federated Token Logic
            postdata = {
                "serverURL": server_url,
                "token": self._portal_auth.token(),
                "expiration": str(self.time_out),
                "f": "json",
                "request": "getToken",
                "referer": self.referer,
            }
            resp = self._session.post(
                url=self._token_url,
                data=postdata,
                verify=self.verify_cert,
                proxies=self.proxies,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    raise Exception(data)
                if "expires" in data:
                    self._expires_on = _dt.datetime.fromtimestamp(
                        int(data["expires"]) / 1000
                    )
                elif "expiration" in data:
                    self._expires_on = _dt.datetime.fromtimestamp(
                        int(data["expiration"])
                    )
                else:
                    self._expires_on = _dt.datetime.now() + _dt.timedelta(
                        minutes=self._time_out - 3
                    )
                self._token = data["token"]
                return self._token
        else:
            raise Exception("Invalid Credentials")
