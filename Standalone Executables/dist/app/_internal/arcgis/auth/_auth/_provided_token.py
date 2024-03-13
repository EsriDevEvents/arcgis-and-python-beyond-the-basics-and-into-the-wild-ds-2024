"""
Handles security where a user provides the token
"""
from requests.auth import AuthBase
from urllib import parse
from ._schain import SupportMultiAuth
from ..tools import parse_url, assemble_url


class EsriUserTokenAuth(AuthBase, SupportMultiAuth):
    """
    Authentication Using User Created Tokens

    """

    _invalid_token_urls = None
    token = None
    auth = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        token: str,
        referer: str = None,
        verify_cert: bool = True,
        **kwargs,
    ):
        if token is None:
            raise ValueError("A `token` must be provided")
        self.token = token
        self.verify_cert = verify_cert
        if referer is None:
            self.referer = ""
        else:
            self.referer = referer
        self.legacy = kwargs.pop("legacy", False)

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def handle_40x(self, r, **kwargs):
        """Handles Case where token is invalid"""

        if (r.status_code < 500 and r.status_code > 399) and str(r.text).lower().find(
            "invalid token"
        ) > -1:
            # Recreate the request without the token
            #
            parsed = parse_url(r.url)
            server_url = assemble_url(parsed)
            self._invalid_token_urls.add(server_url)
            r.content
            r.raw.release_conn()
            r.request.headers.pop("X-Esri-Authorization", None)
            _r = r.connection.send(r.request, **kwargs)
            _r.headers["referer"] = self.referer or "http"
            _r.history.append(r)
            return _r
        return r

    # ----------------------------------------------------------------------
    def __call__(self, r):
        if self._invalid_token_urls is None:
            self._invalid_token_urls = set()
        parsed = parse_url(r.url)
        server_url = assemble_url(parsed)
        if not server_url in self._invalid_token_urls:
            r.register_hook("response", self.handle_40x)
            if self.legacy == False:
                r.headers["X-Esri-Authorization"] = f"Bearer {self.token}"
                r.headers["referer"] = self.referer or ""
            elif self.legacy and r.method == "GET":
                r.prepare_url(url=r.url, params={"token": self.token})
                r.headers["referer"] = self.referer or ""
            elif self.legacy and r.method == "POST":
                data = parse.parse_qs(r.body)
                data["token"] = self.token
                r.prepare_body(data, None, None)
                r.headers["referer"] = self.referer or ""
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
