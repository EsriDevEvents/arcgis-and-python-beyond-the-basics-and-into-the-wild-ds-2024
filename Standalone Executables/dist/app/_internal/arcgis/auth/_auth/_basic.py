from requests.auth import (
    _basic_auth_str,
    HTTPBasicAuth,
    HTTPDigestAuth,
    HTTPProxyAuth,
)
from ._schain import SupportMultiAuth
from ..tools._lazy import LazyLoader
from ..tools import parse_url, assemble_url

requests = LazyLoader("requests")


###########################################################################
class ProxyAuth(HTTPProxyAuth, SupportMultiAuth):  # pragma: no cover
    """Extends ProxyAuth for Chaining"""

    def __init__(self, username: str, password: str):  # pragma: no cover
        HTTPProxyAuth.__init__(username, password)


###########################################################################
class DigestAuth(HTTPDigestAuth, SupportMultiAuth):  # pragma: no cover
    """Implements HTTPDigestAuth for chaining"""

    def __init__(self, username: str, password: str):
        HTTPDigestAuth.__init__(self, username, password)

    @property
    def token(self) -> str:
        """
        returns the authentication token
        """
        return None


###########################################################################
class EsriBasicAuth(HTTPBasicAuth, SupportMultiAuth):
    """Describes a basic requests authentication."""

    _server_log = None
    auth = None

    def __init__(
        self,
        username: str,
        password: str,
        referer: str = "http",
        verify_cert: bool = True,
        **kwargs,
    ):
        self.username = username
        self.password = password
        self._server_log = dict()
        self._tokens = dict()
        self.referer = referer or ""
        self.verify_cert = verify_cert
        self._session = requests.Session()
        self._session.verify = verify_cert
        self._session.headers.update({"referer": referer})
        self._session.auth = (self.username, self.password)
        self._proxies = kwargs.pop("proxies", None)

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

        # ----------------------------------------------------------------------

    @property
    def token(self) -> str:
        """
        Gets the token.  This is always `None` for `EsriBasicAuth`

        :returns: String
        """
        return None

    def generate_portal_server_token(self, r, **kwargs):
        """generates a server token using Portal token"""
        parsed = parse_url(r.url)
        server_url: str = assemble_url(parsed)

        if (
            r.text.lower().find("invalid token") > -1
            or r.text.lower().find("token required") > -1
            or r.text.lower().find("unauthorized") > -1
        ) or server_url in self._server_log:
            expiration = 16000

            postdata = {
                "request": "getToken",
                "serverURL": server_url,
                "referer": self.referer or "http",
                "f": "json",
            }
            if expiration:
                postdata["expiration"] = expiration
            if server_url in self._server_log:
                token_url = self._server_log[server_url]
            else:
                info = self._session.get(
                    server_url + "/rest/info?f=json",
                    auth=self._session.auth,
                    verify=self.verify_cert,
                    proxies=self._proxies,
                    timeout=5,
                ).json()
                token_url = info["authInfo"]["tokenServicesUrl"]
                self._server_log[server_url] = token_url
            if server_url in self._tokens:
                token_str = self._tokens[server_url]
            else:
                token = self._session.post(
                    token_url,
                    data=postdata,
                    auth=self._session.auth,
                    verify=self.verify_cert,
                    proxies=self._proxies,
                )
                token_str = token.json().get("token", None)
                if token_str is None:
                    return r
                self._tokens[server_url] = token_str
            # Recreate the request with the token
            #
            r.content
            r.raw.release_conn()
            r.request.headers["referer"] = self.referer or "http"
            r.request.headers["X-Esri-Authorization"] = f"Bearer {token_str}"
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self.referer or "http"
            _r.headers["X-Esri-Authorization"] = f"Bearer {token_str}"
            _r.history.append(r)
            return _r
        return r

    def __call__(self, r):
        if self.auth:
            self.auth.__call__(r)
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        r.register_hook("response", self.generate_portal_server_token)
        return r
