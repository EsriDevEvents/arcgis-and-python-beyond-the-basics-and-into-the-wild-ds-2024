from requests.auth import AuthBase
from urllib import parse
from arcgis.auth.tools import parse_url

from ._schain import SupportMultiAuth


class EsriAPIKeyAuth(AuthBase, SupportMultiAuth):
    """authentication for API Keys"""

    api_key = None
    auth = None
    _no_go_token = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        api_key: str,
        referer: str = None,
        verify_cert: bool = True,
        auth: AuthBase = None,
    ):
        self._no_go_token = set()
        self.api_key = api_key
        self.verify_cert = verify_cert
        self.auth = auth
        if referer is None:
            self.referer = ""
        else:
            self.referer = referer

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    @property
    def token(self) -> str:
        """
        Gets/Sets the API token

        :returns: String
        """
        return self.api_key

    # ----------------------------------------------------------------------
    @token.setter
    def token(self, api_key: str):
        """Gets/Sets the API token"""
        if self.api_key != api_key:
            self.api_key = api_key

    # ----------------------------------------------------------------------
    def add_api_token(self, r, **kwargs):
        """generates a server token using Portal token"""
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
            r.request.headers["referer"] = self.referer  # or "http"
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self.referer  # or "http"
            _r.headers.pop("X-Esri-Authorization", None)
            _r.history.append(r)
            return _r
        elif r.text.lower().find("token required") > -1:
            r.content
            r.raw.release_conn()
            r.request.headers["referer"] = self.referer  # or "http"
            r.headers["X-Esri-Authorization"] = f"Bearer {self.api_key}"
            # r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self.referer  # or "http"
            # _r.headers.pop("X-Esri-Authorization", None)
            _r.headers["X-Esri-Authorization"] = f"Bearer {self.api_key}"
            _r.history.append(r)
            return _r
        return r

    # ----------------------------------------------------------------------
    def __call__(self, r):
        parsed = parse_url(r.url)
        if self.auth:
            self.auth.__call__(r)
        if (
            # not "X-Esri-Authorization" in r.headers
            # and
            not (parsed.scheme, parsed.netloc, parsed.path)
            in self._no_go_token
        ):
            r.headers["X-Esri-Authorization"] = f"Bearer {self.api_key}"
        r.register_hook("response", self.add_api_token)
        return r
