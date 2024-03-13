from dataclasses import field, dataclass
from typing import Dict, ClassVar


@dataclass
class _KafkaAuthenticationType:
    _auth_type: ClassVar[str]

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        raise NotImplementedError


@dataclass
class NoAuth(_KafkaAuthenticationType):
    """This dataclass is used to specify that no authentication is needed to connect to a Kafka broker."""

    _auth_type: ClassVar[str] = "none"

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {f"{feed_or_source_name}.authenticationType": self._auth_type}


@dataclass
class SASLPlain(_KafkaAuthenticationType):
    """
    This dataclass is used to specify a SASL/Plain Authentication scenario using username and password for connecting
    to a Kafka broker.

    ==================     =============================================================================================
    **Parameter**           **Description**
    ------------------     ---------------------------------------------------------------------------------------------
    username               String. Username for basic authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    password               String. Password for basic authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    use_ssl                bool. When disabled, ArcGIS Velocity will connect via PLAINTEXT. The default value is True.
    ==================     =============================================================================================
    """

    _auth_type: ClassVar[str] = "saslPlain"

    username: str
    password: str
    use_ssl: bool = field(default=True)

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {
            f"{feed_or_source_name}.authenticationType": self._auth_type,
            f"{feed_or_source_name}.username": self.username,
            f"{feed_or_source_name}.password": self.password,
            f"{feed_or_source_name}.useSSL": self.use_ssl,
        }


@dataclass
class SaslScramSha256(_KafkaAuthenticationType):
    """
    This dataclass is used to specify a SASL/SCRAM-SHA-256 Authentication scenario using username and password for
    connecting to a Kafka broker.

    ==================     =============================================================================================
    **Parameter**           **Description**
    ------------------     ---------------------------------------------------------------------------------------------
    username               String. Username for authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    password               String. Password for authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    use_ssl                bool. When disabled, ArcGIS Velocity will connect via PLAINTEXT. The default value is True.
    ==================     =============================================================================================
    """

    _auth_type: ClassVar[str] = "saslSCRAMSha256"

    username: str
    password: str
    use_ssl: bool = field(default=True)

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {
            f"{feed_or_source_name}.authenticationType": self._auth_type,
            f"{feed_or_source_name}.username": self.username,
            f"{feed_or_source_name}.password": self.password,
            f"{feed_or_source_name}.useSSL": self.use_ssl,
        }


@dataclass
class SaslScramSha512(_KafkaAuthenticationType):
    """
    This dataclass is used to specify a SASL/SCRAM-SHA-512 Authentication scenario using username and password for
    connecting to a Kafka broker.

    ==================     =============================================================================================
    **Parameter**           **Description**
    ------------------     ---------------------------------------------------------------------------------------------
    username               String. Username for authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    password               String. Password for authentication.
    ------------------     ---------------------------------------------------------------------------------------------
    use_ssl                bool. When disabled, ArcGIS Velocity will connect via PLAINTEXT. The default value is True.
    ==================     =============================================================================================
    """

    _auth_type: ClassVar[str] = "saslSCRAMSha512"

    username: str
    password: str
    use_ssl: bool = field(default=True)

    def _build(self, feed_or_source_name: str) -> Dict[str, str]:
        return {
            f"{feed_or_source_name}.authenticationType": self._auth_type,
            f"{feed_or_source_name}.username": self.username,
            f"{feed_or_source_name}.password": self.password,
            f"{feed_or_source_name}.useSSL": self.use_ssl,
        }
