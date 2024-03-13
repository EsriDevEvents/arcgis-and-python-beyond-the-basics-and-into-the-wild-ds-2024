"""
urllib parsing helpers to help figure out of the URL returns a file.
"""
import os
import re
import unicodedata
from urllib.parse import urlparse, urlsplit, urljoin
from urllib.parse import urlunsplit, unquote, quote

# Check https://regex101.com/r/A326u1/5 for reference
DOMAIN_FORMAT = re.compile(
    r"(?:^(\w{1,255}):(.{1,255})@|^)"  # http basic authentication [optional]
    r"(?:(?:(?=\S{0,253}(?:$|:))"  # check full domain length to be less than or equal to 253 (starting after http basic auth, stopping before port)
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"  # check for at least one subdomain (maximum length per subdomain: 63 characters), dashes in between allowed
    r"(?:[a-z0-9]{1,63})))"  # check for top level domain, no dashes allowed
    r"|localhost)"  # accept also "localhost" only
    r"(:[0-9]{1,5})?",  # port [optional]
    re.IGNORECASE,
)
SCHEME_FORMAT = re.compile(
    r"^(http|hxxp|ftp|fxp)s?$", re.IGNORECASE  # scheme: http(s) or ftp(s)
)


def validate_url(url: str):
    url = url.strip()

    if not url:
        raise Exception("No URL specified")

    if len(url) > 2048:
        raise Exception(
            "URL exceeds its maximum length of 2048 characters (given length={})".format(
                len(url)
            )
        )

    result = urlparse(url)
    scheme = result.scheme
    domain = result.netloc

    if not scheme:
        raise Exception("No URL scheme specified")

    if not re.fullmatch(SCHEME_FORMAT, scheme):
        raise Exception(
            "URL scheme must either be http(s) or ftp(s) (given scheme={})".format(
                scheme
            )
        )

    if not domain:
        raise Exception("No URL domain specified")

    if not re.fullmatch(DOMAIN_FORMAT, domain):
        raise Exception("URL domain malformed (domain={})".format(domain))

    return url


# --------------------------------------------------------------------------
def _normalize_url(url, charset="utf-8"):
    """Normalizes a URL. Based on http://code.google.com/p/url-normalize."""

    def _clean(string):
        string = str(unquote(string), "utf-8", "replace")
        return unicodedata.normalize("NFC", string).encode("utf-8")

    default_port = {
        "ftp": 21,
        "telnet": 23,
        "http": 80,
        "gopher": 70,
        "news": 119,
        "nntp": 119,
        "prospero": 191,
        "https": 443,
        "snews": 563,
        "snntp": 563,
    }

    # if there is no scheme use http as default scheme
    if url[0] not in ["/", "-"] and ":" not in url[:7]:
        url = "http://" + url

    # shebang urls support
    url = url.replace("#!", "?_escaped_fragment_=")

    # splitting url to useful parts
    scheme, auth, path, query, fragment = urlsplit(url.strip())
    (userinfo, host, port) = re.search("([^@]*@)?([^:]*):?(.*)", auth).groups()

    # Always provide the URI scheme in lowercase characters.
    scheme = scheme.lower()

    # Always provide the host, if any, in lowercase characters.
    host = host.lower()
    if host and host[-1] == ".":
        host = host[:-1]
    # take care about IDN domains
    host = host.decode(charset).encode("idna")  # IDN -> ACE

    # Only perform percent-encoding where it is essential.
    # Always use uppercase A-through-F characters when percent-encoding.
    # All portions of the URI must be utf-8 encoded NFC from Unicode strings
    path = quote(_clean(path), "~:/?#[]@!$&'()*+,;=")
    fragment = quote(_clean(fragment), "~")

    # note care must be taken to only encode & and = characters as values
    query = "&".join(
        [
            "=".join([quote(_clean(t), "~:/?#[]@!$'()*+,;=") for t in q.split("=", 1)])
            for q in query.split("&")
        ]
    )

    # Prevent dot-segments appearing in non-relative URI paths.
    if scheme in ["", "http", "https", "ftp", "file"]:
        output = []
        for part in path.split("/"):
            if part == "":
                if not output:
                    output.append(part)
            elif part == ".":
                pass
            elif part == "..":
                if len(output) > 1:
                    output.pop()
            else:
                output.append(part)
        if part in ["", ".", ".."]:
            output.append("")
        path = "/".join(output)

    # For schemes that define a default authority, use an empty authority if
    # the default is desired.
    if userinfo in ["@", ":@"]:
        userinfo = ""

    # For schemes that define an empty path to be equivalent to a path of "/",
    # use "/".
    if path == "" and scheme in ["http", "https", "ftp", "file"]:
        path = "/"

    # For schemes that define a port, use an empty port if the default is
    # desired
    if port and scheme in list(default_port.keys()):
        if port.isdigit():
            port = str(int(port))
            if int(port) == default_port[scheme]:
                port = ""

    # Put it all back together again
    auth = (userinfo or "") + host
    if port:
        auth += ":" + port
    if url.endswith("#") and query == "" and fragment == "":
        path += "#"
    return urlunsplit((scheme, auth, path, query, fragment))


# --------------------------------------------------------------------------
def _parse_hostname(url, include_port=False):
    """Parses the hostname out of a URL."""
    parsed_url = urlparse((url))
    return parsed_url.netloc if include_port else parsed_url.hostname


# --------------------------------------------------------------------------
def _is_http_url(url):
    if url:
        return urlparse(url).scheme in ["http", "https"]
    return False


# --------------------------------------------------------------------------
def _unpack(obj_or_seq, key=None, flatten=False):
    """Turns a list of single item dicts in a list of the dict's values."""

    # The trivial case (passed in None, return None)
    if not obj_or_seq:
        return None

    # We assume it's a sequence
    new_list = []
    for obj in obj_or_seq:
        value = _unpack_obj(obj, key, flatten)
        new_list.extend(value)

    return new_list


# --------------------------------------------------------------------------
def _unpack_obj(obj, key=None, flatten=False):
    try:
        if key:
            value = [obj.get(key)]
        else:
            value = list(obj.values())
    except AttributeError:
        value = [obj]

    # Flatten any lists if directed to do so
    if value and flatten:
        value = [item for sublist in value for item in sublist]

    return value


# --------------------------------------------------------------------------
def _remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)


# --------------------------------------------------------------------------
def _tostr(obj):
    if not obj:
        return ""
    if isinstance(obj, list):
        return ", ".join(map(_tostr, obj))
    return str(obj)


# --------------------------------------------------------------------------
def _filename_from_url(url):
    """:return: detected filename or None"""
    fname = os.path.basename(urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0 or len(fname.strip(" \n\t.").split(".")) == 1:
        return None
    return fname


# --------------------------------------------------------------------------
def _get_file_name(s: dict) -> str:
    """stips the filename from content-disposition using regex"""
    fname = re.findall("filename\*=([^;]+)", s, flags=re.IGNORECASE)
    if not fname:
        fname = re.findall("filename=([^;]+)", s, flags=re.IGNORECASE)
    if "utf-8''" in fname[0].lower():
        fname = re.sub("utf-8''", "", fname[0], flags=re.IGNORECASE)
        try:
            if type(fname) == str:
                fname = unquote(fname)
            else:
                fname = unquote(fname).decode("utf-8")
        except:
            fname = unquote(fname).encode("utf-8")
    else:
        fname = fname[0]
    # clean space and double quotes
    return fname.strip().strip('"')


# --------------------------------------------------------------------------
def _filename_from_headers(headers: dict) -> str:
    """
    Detect filename from Content-Disposition headers if present.


    :param: headers as dict, list or string
    :return: filename from content-disposition header or None
    """
    if type(headers) == str:
        headers = headers.splitlines()
    if type(headers) == list:
        headers = dict([x.split(":", 1) for x in headers])
    cdisp = headers.get("Content-Disposition")
    if not cdisp:
        return None
    return _get_file_name(cdisp)
