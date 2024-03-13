from urllib.parse import parse_qs
from getpass import getpass

from requests_oauthlib import OAuth1, OAuth2
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from requests.auth import AuthBase
import lxml.html

from ._schain import SupportMultiAuth
from ..tools._lazy import LazyLoader
from ..tools import parse_url, assemble_url

warnings = LazyLoader("warnings")
re = LazyLoader("re")
json = LazyLoader("json")
webbrowser = LazyLoader("webbrowser")
getpass = LazyLoader("getpass")
_dt = LazyLoader("datetime")
requests = LazyLoader("requests")


###########################################################################
class EsriOAuth2Auth(AuthBase, SupportMultiAuth):
    """
    Performs the OAuth Workflow for logging in to Enterprise
    """

    _token = None
    _token_url = None
    _client_id = None
    _client_secret = None
    _username = None
    _password = None
    _token_url = None
    _referer = None
    _expiration = None
    _create_time = None
    _refresh_token = None
    _invalid_token_urls = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str = None,
        username: str = None,
        password: str = None,
        referer: str = "http",
        expiration: int = 1440,
        proxies: dict = None,
        session: "Session" = None,
        **kwargs,
    ) -> None:
        """
        Initializer for Oauth2 Workflows
        """
        self._invalid_token_urls = set()
        self.baseurl = base_url
        self._client_id = client_id
        self._client_secret = client_secret
        self.legacy = kwargs.pop("legacy", False)
        self._username = username
        if self._username and password is None:
            password = getpass.getpass(f"Enter user {username} password:")
        self._password = password
        if session is None:
            self._session = requests.Session()
            self._session.headers["referer"] = referer
            self._session.verify = kwargs.pop("veriy", True)
        else:
            self._session = session
        if proxies:
            self._proxies = proxies
        else:
            self._proxies = proxies

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}, token=.....>"

    # ----------------------------------------------------------------------
    def _oauth_token(self):
        """performs the oauth2 when secret and client exist"""
        auth_url = "%s/oauth2/authorize" % self.baseurl
        tu = "%s/oauth2/token" % self.baseurl
        # handles the refreshing of the token
        if not (self._create_time is None) and (
            _dt.datetime.now()
            >= self._create_time + _dt.timedelta(minutes=self._expiration)
        ):
            self._token = None
        elif (
            not (self._create_time is None)
            and not (self._token is None)
            and (
                _dt.datetime.now()
                < self._create_time + _dt.timedelta(minutes=self._expiration)
            )
        ):
            return self._token
        # Handles token generation
        if (
            self._refresh_token is not None
            and self._client_id is not None
            and self._token is None
        ):  # Case 1: Refreshing a token
            parameters = {
                "client_id": self._client_id,
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
            }
            token_info = self._session.post(tu, data=parameters)
            self._token = token_info["access_token"]
            return self._token
        elif (
            self._client_id
            and self._client_secret
            and self._username
            and self._password
        ):
            oauth = OAuth2Session(
                client=BackendApplicationClient(client_id=self._client_id),
            )

            oauth.verify = False
            if self._proxies:
                oauth.proxies = self._proxies
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = oauth.fetch_token(
                    token_url=tu,
                    username=self._username,
                    password=self._password,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    include_client_id=True,
                    verify=False,
                    proxies=self._proxies,
                    expiration=26000,
                )
            if "expires_in" in res:
                self._create_time = _dt.datetime.fromtimestamp(
                    res["expires_at"]
                ) - _dt.timedelta(seconds=7200)
                self._expiration = res["expires_in"] / 60
                if "token" in res:
                    return res["token"]
                if "access_token" in res:
                    return res["access_token"]
        elif (
            self._client_id and self._client_secret
        ):  # case 2: has both client and secret keys
            client = BackendApplicationClient(client_id=self._client_id)
            oauth = OAuth2Session(
                client=client, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            if self._proxies:
                oauth.proxies = self._proxies
            oauth.verify = False
            res = oauth.fetch_token(
                token_url=tu,
                client_id=self._client_id,
                client_secret=self._client_secret,
                include_client_id=True,
                verify=False,
                proxies=self._proxies,
            )
            if "expires_in" in res:
                self._create_time = _dt.datetime.fromtimestamp(
                    res["expires_at"]
                ) - _dt.timedelta(seconds=7200)
                self._expiration = res["expires_in"] / 60
                if "token" in res:
                    return res["token"]
                if "access_token" in res:
                    return res["access_token"]
        elif (
            self._client_id and self._username is None and self._password is None
        ):  # case 3: client id only
            auth_url = "%s/oauth2/authorize" % self.baseurl
            tu = "%s/oauth2/token" % self.baseurl
            oauth = OAuth2Session(
                self._client_id, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            if self._proxies:
                oauth.proxies = self._proxies
            oauth.verify = False
            authorization_url, state = oauth.authorization_url(
                auth_url, **{"allow_verification": "false"}
            )
            print(
                "Please sign in to your GIS and paste the code that is obtained below."
            )
            print(
                "If a web browser does not automatically open, please navigate to the URL below yourself instead."
            )
            print("Opening web browser to navigate to: " + authorization_url)

            webbrowser.open_new(authorization_url)
            authorization_response = getpass.getpass(
                "Enter code obtained on signing in using SAML: "
            )

            self._create_time = _dt.datetime.now()
            token_info = oauth.fetch_token(
                tu,
                code=authorization_response,
                verify=False,
                proxies=self._proxies,
                include_client_id=True,
                authorization_response="authorization_code",
            )
            self._expiration = token_info["expires_in"] / 60 - 2
            self._refresh_token = token_info["refresh_token"]
            self._token = token_info["access_token"]
            return self._token
        elif self._client_id and not (
            self._username is None and self._password is None
        ):  # case 4: client id and username/password (SAML workflow)
            parameters = {
                "client_id": self._client_id,
                "response_type": "code",
                "expiration": -1,  # we want refresh_token to work for the life of the script
                "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
                "allow_verification": "false",
            }
            content = str(self._session.get(auth_url, params=parameters).content)

            pattern = re.compile("var oAuthInfo = ({.*?});", re.DOTALL)
            if len(pattern.findall(content)) == 0:
                pattern = re.compile("var oAuthInfo = ({.*?})", re.DOTALL)

            soup = lxml.html.fromstring(content)

            def _load_oauth_info(js_object):
                """converts the js oauth to dict"""
                try:
                    oauth_info = json.loads(js_object)
                except:
                    oauth_info = json.loads(js_object + "}")
                return oauth_info

            for script in soup.xpath("//script/text()"):
                script_code = str(script).strip()
                matches = pattern.search(script_code)
                if not matches is None:
                    js_object = matches.groups()[0]
                    try:
                        oauth_info = _load_oauth_info(js_object)
                    except Exception:
                        raise Exception(
                            (
                                "Could not login. Please validate your creden"
                                "tials or make sure the security question is set on the user."
                            )
                        )
                    break

            parameters = {
                "user_orgkey": "",
                "username": self._username,
                "password": self._password,
                "oauth_state": oauth_info["oauth_state"],
            }
            resp = self._session.post(
                "%s/oauth2/signin" % self.baseurl,
                data=parameters,
                verify=False,
                proxies=self._proxies,
                allow_redirects=False,
            )
            if resp.status_code == 302:
                url = resp.headers["Location"]
                if url.find("acceptTermsAndConditions") > -1:
                    r2 = self._session.post(
                        url, data={"acceptTermsAndConditions": True}
                    )
                    content = r2.text
                elif url.find("oauth2/approval") > -1:
                    r2 = self._session.get(url)
                    content = r2.text

            soup = lxml.html.fromstring(content)
            codes = [
                t[len("SUCCESS code=") :]
                for t in soup.xpath("//title//text()")
                if t.find("SUCCESS") > -1
            ]
            if len(codes) > 0:
                code = codes[0]

            oauth = OAuth2Session(
                self._client_id, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            if code is None:
                raise Exception("Could not generate a token.")
            self._create_time = _dt.datetime.now()
            token_info = oauth.fetch_token(
                tu,
                code=code,
                verify=False,
                include_client_id=True,
                authorization_response="authorization_code",
            )
            self._refresh_token = token_info["refresh_token"]
            self._token = token_info["access_token"]
            self._expiration = token_info["expires_in"] / 60 - 2

            return self._token
        return None

    # ----------------------------------------------------------------------
    @property
    def token(self) -> str:
        """
        Gets the Oauth token

        :returns: String
        """
        return self._oauth_token()

    # ----------------------------------------------------------------------
    def handle_40x(self, r, **kwargs):
        """Handles Case where token is invalid"""

        if (r.status_code < 500 and r.status_code > 399) or str(r.text).lower().find(
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
                r.headers["X-Esri-Authorization"] = f"Bearer {self._oauth_token()}"
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "GET":
                r.prepare_url(url=r.url, params={"token": self._oauth_token()})
                r.headers["referer"] = self._referer or ""
            elif self.legacy and r.method == "POST":
                data = parse_qs(r.body)
                data["token"] = self._oauth_token()
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
