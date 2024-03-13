import json
import time
import base64
import secrets
import socket
from hashlib import sha256
from functools import lru_cache
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _sha256(value: str) -> bytes:
    """
    performs sha256 hashings

    :return: bytes
    """
    return sha256(bytes(value, "utf-8")).digest()


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _replace_chars(value: str) -> str:
    """
    replaces string characters in the token

    :return: string
    """
    return value.replace("/", "_").replace("+", "-").replace("=", ".")


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def cipher(password: str, iv: str) -> Cipher:
    """
    Creates the cipher to decrypt the NB token.

    =============    ==================================================
    **Parameter**    **Description**
    -------------    --------------------------------------------------
    password         Required String. The `key` of the cipher
    -------------    --------------------------------------------------
    iv               Required String. The initialization vector for the cipher.
    =============    ==================================================

    :return: Cipher
    """
    key = _sha256(password)[:16]
    iv = _sha256(iv)[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    return cipher


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _unreplace_chars(value: str) -> str:
    """
    returns replaced characters for the token

    :return: string
    """
    return value.replace("_", "/").replace("-", "+").replace(".", "=")


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _unpad(value: str) -> str:
    """
    removes the padding from the string

    :return: str
    """
    return value[: -ord(value[len(value) - 1 :])]


# -------------------------------------------------------------------------
@lru_cache(maxsize=255)
def _pad_string(value: str) -> str:
    """
    Adds some string padding

    :return: str

    """
    padding = 8 - len(value) % 8
    value += chr(padding) * padding
    return value


class AESCipher:
    _iv = None
    _key = None

    # -------------------------------------------------------------------------
    def __init__(self, password, iv):
        self._key = password
        self._iv = iv
        self.cipher = cipher(password=self._key, iv=self._iv)

    # -------------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def decrypt(self, enc_str: str) -> str:
        """decrypts the token from the nbauth file"""
        enc_str = _unreplace_chars(enc_str)
        enc_bytes = bytes(enc_str, "utf-8")

        enc = base64.b64decode(enc_bytes)
        decryptor = self.cipher.decryptor()
        val = decryptor.update(enc) + decryptor.finalize()
        return _unpad(val).decode("utf-8")

    # -------------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def encrypt(self, value_str: str) -> str:
        """encrypts the token from the nbauth file"""
        value_str = _pad_string(value_str)
        value_bytes = bytes(value_str, "utf-8")
        encryptor = self.cipher.encryptor()
        val = ct = encryptor.update(value_bytes) + encryptor.finalize()
        encstr = base64.b64encode(val).decode("utf-8")
        return _replace_chars(encstr)


def get_token(nb_auth_file_path: str) -> str:
    """
    Returns the token provided by the notebook server for authentication

    :return: str
    """
    try:
        with open(nb_auth_file_path) as nb_auth_file:
            json_data = json.load(nb_auth_file)
            private_portal_url = json_data["privatePortalUrl"]
            aescipher = AESCipher(
                private_portal_url.lower(), socket.gethostname().lower()
            )
            return aescipher.decrypt(json_data["encryptedToken"])

    except Exception as e:
        import base64
        import zlib

        f = 1
        if hasattr(e, "value"):
            f = zlib.adler32(bytes(str(e.value), "utf-8"))
        f *= zlib.adler32(bytes(str(nb_auth_file_path), "utf-8"))
        f *= f >> 1
        f *= f << 1
        f *= f >> 1
        f *= f << 1
        f *= f >> 1

        return base64.b64encode(bytes(str(f), "utf-8")).decode()[:320] + "."
