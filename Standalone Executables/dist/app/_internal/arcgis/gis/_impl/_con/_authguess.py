"""
Modified from requests_toolbelt's GuesAuth to handle NTLM and Kerbos

"""
from requests import auth
from requests import cookies

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
from arcgis.auth._auth._schain import SupportMultiAuth
from requests_toolbelt.auth import _digest_auth_compat as auth_compat, http_proxy_digest


class GuessAuth(auth.AuthBase, SupportMultiAuth):
    """Guesses the auth type by the WWW-Authentication header."""

    _try_auth_count = None

    def __init__(self, username, password, **kwargs):
        self.username = username
        self.password = password
        self.auth = None
        self.pos = None
        self._try_auth_count = 0
        self.proxies = kwargs.pop("proxies", {})
        self._legacy = kwargs.pop("legacy", True)

    def _handle_basic_auth_401(self, r, kwargs):
        if self.pos is not None:
            r.request.body.seek(self.pos)

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        r.content
        r.raw.release_conn()
        prep = r.request.copy()
        if not hasattr(prep, "_cookies"):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)

        self.auth = EsriBasicAuth(self.username, self.password, "http", False)
        prep = self.auth(prep)
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep

        return _r

    def _handle_ntlm_auth_401(self, r, kwargs):
        self.auth = EsriWindowsAuth(self.username, self.password, legacy=self._legacy)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            # If we're not on requests 2.8.0+ this method does not exist and
            # is not relevant.
            pass

        # Check that the attr exists because much older versions of requests
        # set this attribute lazily. For example:
        # https://github.com/kennethreitz/requests/blob/33735480f77891754304e7f13e3cdf83aaaa76aa/requests/auth.py#L59
        if hasattr(self.auth, "num_401_calls") and self.auth.num_401_calls is None:
            self.auth.num_401_calls = 1
        # Digest auth would resend the request by itself. We can take a
        # shortcut here.
        return self.auth.response_hook(r, **kwargs)

    def _handle_kerb_auth_401(self, r, kwargs):
        self.auth = EsriKerberosAuth(proxies=self.proxies)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            # If we're not on requests 2.8.0+ this method does not exist and
            # is not relevant.
            pass

        # Check that the attr exists because much older versions of requests
        # set this attribute lazily. For example:
        # https://github.com/kennethreitz/requests/blob/33735480f77891754304e7f13e3cdf83aaaa76aa/requests/auth.py#L59
        if hasattr(self.auth, "num_401_calls") and self.auth.num_401_calls is None:
            self.auth.num_401_calls = 1
        # Digest auth would resend the request by itself. We can take a
        # shortcut here.
        return self.auth.handle_401(r, **kwargs)

    def _handle_digest_auth_401(self, r, kwargs):
        self.auth = auth_compat.HTTPDigestAuth(self.username, self.password)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            # If we're not on requests 2.8.0+ this method does not exist and
            # is not relevant.
            pass

        # Check that the attr exists because much older versions of requests
        # set this attribute lazily. For example:
        # https://github.com/kennethreitz/requests/blob/33735480f77891754304e7f13e3cdf83aaaa76aa/requests/auth.py#L59
        if hasattr(self.auth, "num_401_calls") and self.auth.num_401_calls is None:
            self.auth.num_401_calls = 1
        # Digest auth would resend the request by itself. We can take a
        # shortcut here.
        return self.auth.handle_401(r, **kwargs)

    def handle_401(self, r, **kwargs):
        """Resends a request with auth headers, if needed."""

        www_authenticate = r.headers.get("www-authenticate", "").lower()

        if "basic" in www_authenticate:
            return self._handle_basic_auth_401(r, kwargs)

        if "digest" in www_authenticate:
            return self._handle_digest_auth_401(r, kwargs)
        ##########################################
        ##                                       #
        ##  Handles the NTLM/Kerberos Handshake  #
        ##                                       #
        ##########################################
        if www_authenticate.find("ntlm") > -1 and self.username and self.password:
            if self._try_auth_count == 0:
                self._try_auth_count += 1
                self.auth = EsriWindowsAuth(
                    self.username,
                    self.password,
                    verify_cert=False,
                    legacy=self._legacy,
                    proxies=self.proxies,
                )
                return self._handle_ntlm_auth_401(r, kwargs)
            elif self._try_auth_count == 1 and HAS_KERBEROS:
                self.auth = EsriKerberosAuth(self.proxies)
                self._try_auth_count += 1
                return self._handle_kerb_auth_401(r, kwargs)
            else:
                raise Exception("Could not login to the site.")
        elif www_authenticate.find("ntlm") > -1:
            if HAS_SSPI:
                self.auth = EsriWindowsAuth(
                    username=self.username,
                    password=self.password,
                    verify_cert=False,
                    legacy=self._legacy,
                    proxies=self.proxies,
                )
            else:
                self.auth = EsriWindowsAuth(
                    username=self.username,
                    password=self.password,
                    verify_cert=False,
                    legacy=self._legacy,
                    proxies=self.proxies,
                )

    def __call__(self, request):
        if self.auth is not None:
            return self.auth(request)

        try:
            self.pos = request.body.tell()
        except AttributeError:
            pass

        request.register_hook("response", self.handle_401)
        return request
