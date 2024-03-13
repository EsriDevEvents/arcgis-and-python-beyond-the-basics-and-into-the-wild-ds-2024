from __future__ import annotations
import re
import urllib.parse as urllib_parse
from functools import lru_cache
from requests import Response
from typing import Any
from .._error import EsriHttpResponseError

__all__ = [
    "parse_url",
    "_split_username",
    "assemble_url",
    "check_response_for_error",
]


# @lru_cache(maxsize=255)
def check_response_for_error(resp: Response, raise_error: bool = False) -> str | None:
    """
    Checks the payload for `error` keyword.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    resp                requests.Response. The response object to examin.  First the
                        response object is checked to see if it's JSON, then it examines the
                        response for `error` keywords.
    ---------------     --------------------------------------------------------------------
    raise_error         boolean. If True and an error is found, the  EsriHttpResponseError
                        is raised, else the message is returned as a string.
    ===============     ====================================================================

    :returns: str

    :raise: EsriHttpResponseError
    """
    data: dict[str, Any]
    try:
        data = resp.json()
    except Exception as ex:
        return str(ex)
    if "error" in data and "message" in data["error"] and raise_error:
        raise EsriHttpResponseError(data["message"]["error"])
    elif "error" in data and "message" in data["error"] and raise_error == False:
        return data["error"]["message"]
    return None


@lru_cache(maxsize=255)
def parse_url(url: str) -> object:
    """
    Parses a URL string into it's pieces.

    :returns: Named Tuple
    """
    return urllib_parse.urlparse(url)


@lru_cache(maxsize=255)
def assemble_url(parsed: object) -> str:
    """
    creates the URL from a parsed URL
    """
    if parsed.port:
        netloc: str = parsed.netloc.split(":")[0]
        server_url = (
            f'{parsed.scheme}://{netloc}:{parsed.port}/{parsed.path[1:].split("/")[0]}'
        )
    else:
        server_url = (
            f'{parsed.scheme}://{parsed.netloc}/{parsed.path[1:].split("/")[0]}'
        )
    return server_url


@lru_cache(maxsize=255)
def _split_username(username: str) -> list[str]:
    regex = r"(\S*)?(@|#|//|\\\\|(?<!/)/(?!/)|\\)(\S*)"
    matches = re.finditer(regex, username, re.IGNORECASE | re.DOTALL)
    tokens = []
    for match in matches:
        for group in match.groups():
            tokens.append(group)

    if tokens[1] in ["//", "\\", "/"]:
        uname = tokens[2]
        dom = tokens[0]
    elif tokens[1] == "@":
        uname = tokens[0]
        dom = tokens[2]

    return [uname, dom]
