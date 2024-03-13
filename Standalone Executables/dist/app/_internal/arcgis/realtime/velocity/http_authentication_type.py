from dataclasses import dataclass, asdict, field
from typing import Dict, Union, Optional, Any, ClassVar


@dataclass
class _HttpAuthenticationType:
    _auth_type: ClassVar[str]

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        raise NotImplementedError


@dataclass
class NoAuth(_HttpAuthenticationType):
    """This dataclass is used to specify the no HTTP authentication scenario."""

    _auth_type: ClassVar[str] = "noauth"

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {f"{feed_or_source_name}.httpAuthenticationType": self._auth_type}


@dataclass
class BasicAuth(_HttpAuthenticationType):
    """
    This dataclass is used to specify a Basic HTTP Authentication scenario using username and password.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    username               String. Username for basic authentication.
    ------------------     --------------------------------------------------------------------
    password               String. Password for basic authentication.
    ==================     ====================================================================
    """

    _auth_type: ClassVar[str] = "basicauth"

    username: str
    password: str

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {
            f"{feed_or_source_name}.httpAuthenticationType": self._auth_type,
            f"{feed_or_source_name}.username": self.username,
            f"{feed_or_source_name}.password": self.password,
        }


@dataclass
class CertificateAuth(_HttpAuthenticationType):
    """
    This dataclass is used to specify a Basic HTTP Authentication scenario using username and password.

    =======================     ====================================================================
    **Parameter**                **Description**
    -----------------------     --------------------------------------------------------------------
    pfx_file_http_location      String. HTTP path of the PFX file.
    -----------------------     --------------------------------------------------------------------
    password                    String. Password for certificate authentication.
    =======================     ====================================================================
    """

    _auth_type: ClassVar[str] = "certificateauth"

    pfx_file_http_location: str
    password: str

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {
            f"{feed_or_source_name}.httpAuthenticationType": self._auth_type,
            f"{feed_or_source_name}.password": self.password,
            f"{feed_or_source_name}.pfxLocation": self.pfx_file_http_location,
        }
