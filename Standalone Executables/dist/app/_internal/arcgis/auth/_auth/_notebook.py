from requests.auth import AuthBase
from urllib import parse

from ._schain import SupportMultiAuth
from arcgis.auth.tools import parse_url


class EsriNotebookAuth(AuthBase, SupportMultiAuth):
    """authentication for notebook servers Keys"""

    _invalid_token_urls = None
    _token = None
    auth = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        token: str,
        referer: str = None,
        auth: AuthBase = None,
        **kwargs,
    ):
        self._token = token
        self.auth = auth
        if referer is None:
            self.referer = ""
        else:
            self.referer = referer
        self._no_go_token = set()
        self.verify_cert = kwargs.pop("verify_cert", True)

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token={self.token[:5]}...>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token={self.token[:5]}...>"

    # ----------------------------------------------------------------------
    @property
    def token(self) -> str:
        """
        Gets/Sets the API token

        :returns: String
        """
        return self._token

    # ----------------------------------------------------------------------
    @token.setter
    def token(self, token: str):
        """Gets/Sets the Notebook Token"""
        if self._token != token:
            self._token = token

    # ----------------------------------------------------------------------
    def add_token(self, r, **kwargs):
        """generates a server token using Portal token"""
        if (
            r.text.lower().find("invalid token") > -1
            or r.text.lower().find("token required") > -1
        ):
            parsed = parse_url(r.url)
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
        return r

    # ----------------------------------------------------------------------
    def __call__(self, r):
        parsed = parse_url(r.url)
        if self.auth:
            self.auth.__call__(r)
        if (
            not "X-Esri-Authorization" in r.headers
            and not (parsed.scheme, parsed.netloc, parsed.path) in self._no_go_token
        ):
            r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
            r.headers.pop("Referer", None)  # ['Referer'] = "http"  # , self.referer
        r.register_hook("response", self.add_token)
        return r
