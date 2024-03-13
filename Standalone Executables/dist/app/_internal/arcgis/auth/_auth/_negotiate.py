from __future__ import annotations
from arcgis.auth.tools import LazyLoader
import base64
import hashlib
import logging
import socket
import struct
import typing
from urllib.parse import urlparse
import requests
from requests.auth import AuthBase
from requests.exceptions import HTTPError
from ._utils import _split_username

try:
    pywintypes = LazyLoader("pywintypes", strict=True)
    sspi = LazyLoader("sspi", strict=True)
    sspicon = LazyLoader("sspicon", strict=True)
    win32security = LazyLoader("win32security", strict=True)

    HAS_GSSAPI = True
except ImportError:
    HAS_GSSAPI = False
from ._utils import parse_url, assemble_url

from ._schain import SupportMultiAuth

_logger = logging.getLogger(__name__)

__all__ = ["EsriHttpNegotiateAuth"]


###########################################################################
class EsriHttpNegotiateAuth(AuthBase, SupportMultiAuth):
    """
    This class is dervived from the `requests_negotiate_sspi` package. It
    extends the `HttpNegotiateAuth` class inorder to handle the Esri security
    model.

    ================    ===============================================================
    **Parameter**       **Description**
    ----------------    ---------------------------------------------------------------
    username            Optional String. The username account with the domain (DOMAIN\\USERNAME).
    ----------------    ---------------------------------------------------------------
    password            Optional String. The username's password.
    ----------------    ---------------------------------------------------------------
    service             Optional String. Kerberos Service type for remote Service Principal Name.
    ----------------    ---------------------------------------------------------------
    host                Optional String. Host name for Service Principal Name.
    ----------------    ---------------------------------------------------------------
    delegate            Optional Boolean.  Indicates that the user's credentials are to be delegated to the server.
    ----------------    ---------------------------------------------------------------
    referer             Optional String. The referer for the Esri token.
    ----------------    ---------------------------------------------------------------
    verify_cert         Optional Boolean. When false, certificate errors are ignored. The default is True.
    ================    ===============================================================

    If username and password are not specified, the user's default credentials are used.
    This allows for single-sign-on to domain resources if the user is currently logged on
    with a domain account.

    """

    _auth_info = None
    _service = "HTTP"
    _host = None
    _delegate = False

    # ----------------------------------------------------------------------
    def __init__(
        self,
        username=None,
        password=None,
        service=None,
        host=None,
        delegate=False,
        *,
        referer: str = "http",
        verify_cert: bool = True,
        **kwargs,
    ):
        """Create a new Negotiate auth handler

        Args:
         username: Username.
         password: Password.
         domain: NT Domain name.
             Default: '.' for local account.
         service: Kerberos Service type for remote Service Principal Name.
             Default: 'HTTP'
         host: Host name for Service Principal Name.
             Default: Extracted from request URI
         delegate: Indicates that the user's credentials are to be delegated to the server.
             Default: False

         If username and password are not specified, the user's default credentials are used.
         This allows for single-sign-on to domain resources if the user is currently logged on
         with a domain account.
        """
        domain = None
        try:
            username, domain = _split_username(username)
        except:
            username, domain = username, None
        if HAS_GSSAPI == False:
            raise Exception(
                "The system does not have the required dependencies"
                " or is not a Windows based operating system."
            )
        if domain is None:
            domain = "."

        if username is not None and password is not None:
            self._auth_info = (username, domain, password)

        if service is not None:
            self._service = service

        if host is not None:
            self._host = host

        self._delegate = delegate
        self._server_log: dict[str, typing.Any] = {}
        self._referer: str = referer
        self._verify_cert: bool = verify_cert
        self._proxy: dict[str, typing.Any] = kwargs.pop("proxy", None)

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def generate_token(self, r, scheme, args):
        """
        Generates the Server Token

        Args:
         r: the requests.Response object
         schema: string. Http or HTTPS
         args: a list of optional arguments for the requests' send method.

        """
        parsed = parse_url(url=r.url)
        server_url = assemble_url(parsed)
        token_url: str = None
        if server_url in self._server_log:
            token_url: str = self._server_log[server_url]
        elif r.text.lower().find("token required") > -1:
            resp = requests.get(
                f"{server_url}/rest/info",
                params={"f": "json"},
                auth=self,
                headers={"referer": self._referer},
                verify=self._verify_cert,
                proxies=self._proxy,
            ).json()
            self._server_log[parsed.netloc] = resp["authInfo"]["tokenServicesUrl"]
            token_url: str = self._server_log[parsed.netloc]

        if token_url:
            r.content
            r.raw.release_conn()
            request = r.request.copy()

            # this is important for some web applications that store
            # authentication-related info in cookies
            if r.headers.get("set-cookie"):
                request.headers["Cookie"] = r.headers.get("set-cookie")
            postdata = {
                "request": "getToken",
                "serverURL": server_url,
                "referer": "http",
                "f": "json",
            }
            resp = requests.post(
                token_url,
                params=postdata,
                auth=self,
                headers={"referer": self._referer},
                verify=self._verify_cert,
                proxies=self._proxy,
            ).json()
            token_str = resp["token"]
            request.headers["X-Esri-Authorization"] = f"Bearer {token_str}"

            return request.copy()
        else:
            return r

    # ----------------------------------------------------------------------
    def _retry_using_http_Negotiate_auth(self, response, scheme, args):
        """performs the NTLM authorization"""
        if "Authorization" in response.request.headers:
            return response

        if self._host is None:
            targeturl = urlparse(response.request.url)
            self._host = targeturl.hostname
            try:
                self._host = socket.getaddrinfo(
                    self._host, None, 0, 0, 0, socket.AI_CANONNAME
                )[0][3]
            except socket.gaierror as e:
                _logger.info(
                    "Skipping canonicalization of name %s due to error: %s",
                    self._host,
                    e,
                )

        targetspn = "{}/{}".format(self._service, self._host)

        # We request mutual auth by default
        scflags = sspicon.ISC_REQ_MUTUAL_AUTH

        if self._delegate:
            scflags |= sspicon.ISC_REQ_DELEGATE

        # Set up SSPI connection structure
        pkg_info = win32security.QuerySecurityPackageInfo(scheme)
        clientauth = sspi.ClientAuth(
            scheme,
            targetspn=targetspn,
            auth_info=self._auth_info,
            scflags=scflags,
            datarep=sspicon.SECURITY_NETWORK_DREP,
        )
        sec_buffer = win32security.PySecBufferDescType()

        # Channel Binding Hash (aka Extended Protection for Authentication)
        # If this is a SSL connection, we need to hash the peer certificate, prepend the RFC5929 channel binding type,
        # and stuff it into a SEC_CHANNEL_BINDINGS structure.
        # This should be sent along in the initial handshake or Kerberos auth will fail.
        if hasattr(response, "peercert") and response.peercert is not None:
            md = hashlib.sha256()
            md.update(response.peercert)
            appdata = "tls-server-end-point:".encode("ASCII") + md.digest()
            cbtbuf = win32security.PySecBufferType(
                pkg_info["MaxToken"], sspicon.SECBUFFER_CHANNEL_BINDINGS
            )
            cbtbuf.Buffer = struct.pack(
                "LLLLLLLL{}s".format(len(appdata)),
                0,
                0,
                0,
                0,
                0,
                0,
                len(appdata),
                32,
                appdata,
            )
            sec_buffer.append(cbtbuf)

        content_length = int(
            response.request.headers.get("Content-Length", "0"), base=10
        )

        if hasattr(response.request.body, "seek"):
            if content_length > 0:
                response.request.body.seek(-content_length, 1)
            else:
                response.request.body.seek(0, 0)

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        response.content
        response.raw.release_conn()
        request = response.request.copy()

        # this is important for some web applications that store
        # authentication-related info in cookies
        if response.headers.get("set-cookie"):
            request.headers["Cookie"] = response.headers.get("set-cookie")

        # Send initial challenge auth header
        try:
            error, auth = clientauth.authorize(sec_buffer)
            request.headers["Authorization"] = "{} {}".format(
                scheme, base64.b64encode(auth[0].Buffer).decode("ASCII")
            )
            _logger.debug(
                "Sending Initial Context Token - error={} authenticated={}".format(
                    error, clientauth.authenticated
                )
            )
        except pywintypes.error as e:
            _logger.debug("Error calling {}: {}".format(e[1], e[2]), exc_info=e)
            return response

        # A streaming response breaks authentication.
        # This can be fixed by not streaming this request, which is safe
        # because the returned response3 will still have stream=True set if
        # specified in args. In addition, we expect this request to give us a
        # challenge and not the real content, so the content will be short
        # anyway.
        args_nostream = dict(args, stream=False)
        response2 = response.connection.send(request, **args_nostream)

        # Should get another 401 if we are doing challenge-response (NTLM)
        if response2.status_code != 401:
            # Kerberos may have succeeded; if so, finalize our auth context
            final = response2.headers.get("WWW-Authenticate")
            if final is not None:
                try:
                    # Sometimes Windows seems to forget to prepend 'Negotiate' to the success response,
                    # and we get just a bare chunk of base64 token. Not sure why.
                    final = final.replace(scheme, "", 1).lstrip()
                    tokenbuf = win32security.PySecBufferType(
                        pkg_info["MaxToken"], sspicon.SECBUFFER_TOKEN
                    )
                    tokenbuf.Buffer = base64.b64decode(final.encode("ASCII"))
                    sec_buffer.append(tokenbuf)
                    error, auth = clientauth.authorize(sec_buffer)
                    _logger.debug(
                        "Kerberos Authentication succeeded - error={} authenticated={}".format(
                            error, clientauth.authenticated
                        )
                    )
                except TypeError:
                    pass

            # Regardless of whether or not we finalized our auth context,
            # without a 401 we've got nothing to do. Update the history and return.
            response2.history.append(response)
            return response2

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        response2.content
        response2.raw.release_conn()
        request = response2.request.copy()

        # Keep passing the cookies along
        if response2.headers.get("set-cookie"):
            request.headers["Cookie"] = response2.headers.get("set-cookie")

        # Extract challenge message from server
        challenge = [
            val[len(scheme) + 1 :]
            for val in response2.headers.get("WWW-Authenticate", "").split(", ")
            if scheme in val
        ]
        if len(challenge) != 1:
            raise HTTPError(
                "Did not get exactly one {} challenge from server.".format(scheme)
            )

        # Add challenge to security buffer
        tokenbuf = win32security.PySecBufferType(
            pkg_info["MaxToken"], sspicon.SECBUFFER_TOKEN
        )
        tokenbuf.Buffer = base64.b64decode(challenge[0])
        sec_buffer.append(tokenbuf)
        _logger.debug("Got Challenge Token (NTLM)")

        # Perform next authorization step
        try:
            error, auth = clientauth.authorize(sec_buffer)
            request.headers["Authorization"] = "{} {}".format(
                scheme, base64.b64encode(auth[0].Buffer).decode("ASCII")
            )
            _logger.debug(
                "Sending Response - error={} authenticated={}".format(
                    error, clientauth.authenticated
                )
            )
        except pywintypes.error as e:
            _logger.debug("Error calling {}: {}".format(e[1], e[2]), exc_info=e)
            return response

        response3 = response2.connection.send(request, **args)

        if response3.text.lower().find("token required") > -1:
            request4 = self.generate_token(r=response3, scheme=scheme, args=args)
            response4 = response3.connection.send(request4, **args)
            if response4.status_code == 401:
                response4.request.headers.pop("Authorization")
            return self._retry_using_http_Negotiate_auth(
                response=response4, scheme=scheme, args=args
            )

        else:
            # Update the history and return
            response3.history.append(response)
            response3.history.append(response2)
            return response3

    # ----------------------------------------------------------------------
    def _response_hook(self, r, **kwargs):
        """hook logic"""
        if r.status_code == 401:
            for scheme in ("Negotiate", "NTLM"):
                if scheme.lower() in r.headers.get("WWW-Authenticate", "").lower():
                    return self._retry_using_http_Negotiate_auth(r, scheme, kwargs)
        elif r.text.lower().find("token required") > -1:
            for scheme in ("Negotiate", "NTLM"):
                if (
                    scheme.lower()
                    in r.headers.get("WWW-Authenticate", "Negotiate").lower()
                ):
                    request = self.generate_token(r, "NTLM", kwargs)
                    response4 = r.connection.send(request, **kwargs)
                    response4.history.append(r)
                    return response4

    # ----------------------------------------------------------------------
    def __call__(self, r):
        """call method used by requests to register the hook."""
        r.headers["Connection"] = "Keep-Alive"
        r.register_hook("response", self._response_hook)
        return r
