import uuid
import base64
import socket
import hashlib
from requests import PreparedRequest, Response, Session
from arcgis.auth._auth._base import BaseEsriAuth
import datetime as _dt
import secrets
import hashlib
import base64
import requests
from arcgis.auth.tools import parse_url
from urllib.parse import parse_qs


###########################################################################
class EsriPKCEAuth(BaseEsriAuth):
    """Implements OAuth 2.0 PKCE Workflow"""

    _url = None
    _token = None
    _verify = None
    _proxies = None
    _referer = None
    _session = None
    _hostname = None
    _password = None
    _username = None
    _client_id = None
    _expiration = None
    _expires_in = None
    _tokens = None

    # ---------------------------------------------------------------------
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        *,
        legacy: bool = False,
        **kwargs,
    ):
        """initializer"""
        self._no_go_token = set()
        self.legacy = legacy
        self._hostname = kwargs.pop("hostname", socket.getfqdn().lower())

        self._redirect_url = kwargs.pop("redirect_url", url)
        self._client_id = kwargs.pop("client_id", "arcgisonline")
        self._referer = kwargs.pop("referer", "http")
        self._expiration = kwargs.pop("expiration", 1440)
        self._username = username
        self._url = url
        self._password = password
        self._verify = kwargs.pop("verify_cert", True)
        self._proxies = kwargs.pop("proxies", None)
        self._code_verifier = (uuid.uuid4().hex + uuid.uuid4().hex)[:44]
        self._code_challenge = None
        self._session = Session()
        self._session.proxies = self._proxies
        self._session.verify = self._verify

    # ---------------------------------------------------------------------
    @property
    def _create_challenge(self) -> str:
        """creates the challenge string for the PKCE login"""
        if self._code_challenge is None:
            self._code_challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(self._code_verifier.encode("ascii")).digest()
                )
                .decode()
                .strip("=")
            )
        return self._code_challenge

    # ---------------------------------------------------------------------
    def _signin(self, username: str, password: str) -> dict:
        """Signs into the enterprise or online site"""

        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_url,
            "response_type": "code",
            "code_challenge": self._create_challenge,
            "code_challenge_method": "S256",
            "f": "json",
            "expiration": self._expiration,
        }
        url = f"{self._url}/sharing/rest/oauth2/authorize"
        response = self._session.get(
            url, params=params, verify=self._verify, proxies=self._proxies
        ).text
        if "oauth_state" in response:
            oauth_state = response.split('"oauth_state":"')[1].split('"')[0]
        else:
            raise Exception("Unable to generate oauth token")
        params = {
            "oauth_state": oauth_state,
            "username": username,
            "password": password,
        }
        url = f"{self._url}/sharing/rest/oauth2/signin"
        response = self._session.post(
            url,
            data=params,
            verify=self._verify,
            proxies=self._proxies,
            allow_redirects=False,
        )
        if response.headers["Location"].lower().find("accepttermsandconditions") > -1:
            url: str = response.headers["Location"]
            params = {"acceptTermsAndConditions": True}
            response: requests.Response = self._session.post(
                url,
                data=params,
                verify=self._verify,
                proxies=self._proxies,
                allow_redirects=False,
            )

        if response.status_code == 302:
            if "Location" in response.headers:
                oauth_code = (
                    response.headers["Location"].split("code=")[1].split("&")[0]
                )

            else:
                raise Exception("Unable to generate oauth authorization code")
        else:
            raise Exception("Unable to generate oauth token\n{}".format(response))

        params = {
            "client_id": self._client_id,
            "grant_type": "authorization_code",
            "redirect_uri": self._url,
            "code_verifier": self._code_verifier,
            "code": oauth_code,
        }
        url = f"{self._url}/sharing/rest/oauth2/token"

        tokens = self._session.post(
            url, data=params, verify=self._verify, proxies=self._proxies
        ).json()
        self._expires_in = _dt.datetime.now() + _dt.timedelta(
            seconds=tokens.get("expires_in", 1800) - 300
        )
        self._refresh_token_value = tokens.get("refresh_token")
        self._refresh_expires_in = _dt.datetime.now() + _dt.timedelta(
            seconds=(tokens.get("refresh_token_expires_in", 86300) / 60) - 300
        )
        if "error" in tokens.keys():
            if "message" in tokens["error"]:
                raise Exception(
                    "Error generating access token\n{}".format(
                        tokens["error"]["message"]
                    )
                )
            else:
                raise Exception("Error generating access token")

        self._tokens = tokens

    # ---------------------------------------------------------------------
    def _refresh_token(self) -> None:
        """
        refreshes the `token` by using the refresh token
        This method updates the self._tokens value with the new access_token
        :returns: None
        """
        if _dt.datetime.now() >= self._refresh_expires_in:
            self._signin(username=self._username, password=self._password)
        params = {
            "client_id": self._client_id,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token_value,
        }
        url = f"{self._url}/sharing/rest/oauth2/token"
        response = self._session.post(
            url, data=params, verify=self._verify, proxies=self._proxies
        )
        tokens = response.json()
        self._expires_in = _dt.datetime.now() + _dt.timedelta(
            seconds=tokens.get("expires_in", 1800) - 300
        )
        self._tokens.update(tokens)

    # ---------------------------------------------------------------------
    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        """
        Called Before PrepareRequest is Completed.

        The logic to attached preemptive authentication should go here.
        If you want to wait for response, register a response_hook within the call.

        :returns: PrepareRequest

        ```
        def __call__(self, r:PreparedRequest) -> PreparedRequest:
            r.register_hook(event="response", hook=self.response_hook)
            return r
        ```

        """
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
        r.register_hook("response", self.response_hook)
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

    # ---------------------------------------------------------------------
    def response_hook(self, r: Response, **kwargs) -> Response:
        """
        response hook logic

        :return: Response
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

    # ---------------------------------------------------------------------
    @property
    def token(self) -> str:
        """
        returns a ArcGIS Token as a string

        :return: string
        """
        if self._tokens is None:
            self._signin(username=self._username, password=self._password)
            return self.token
        elif _dt.datetime.now() >= self._expires_in:
            self._refresh_token()
            return self.token
        elif _dt.datetime.now() >= self._refresh_expires_in:
            self._signin(username=self._username, password=self._password)
            return self.token
        else:
            return self._tokens.get("access_token")
