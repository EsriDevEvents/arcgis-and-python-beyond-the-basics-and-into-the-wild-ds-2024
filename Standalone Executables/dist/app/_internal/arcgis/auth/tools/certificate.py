"""
Tools to assist users to work with PKI Certificates
"""
import os
import tempfile
import cryptography
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    NoEncryption,
)

_crypto_version = [
    int(i) if i.isdigit() else i for i in cryptography.__version__.split(".")
]


# ----------------------------------------------------------------------
def pfx_to_pem(pfx_path, pfx_password, folder=None, use_openssl=False):
    """Decrypts the .pfx file to be used with requests.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    pfx_path            Required string.  File pathname to .pfx file to parse.
    ---------------     --------------------------------------------------------------------
    pfx_password        Required string.  Password to open .pfx file to extract key/cert.
    ---------------     --------------------------------------------------------------------
    folder              Optional String.  The save location of the certificate files.  The
                        default is the tempfile.gettempdir() directory.
    ---------------     --------------------------------------------------------------------
    user_openssl        Optional Boolean. If True, OpenPySSL is used to convert the pfx to pem instead of cryptography.
    ===============     ====================================================================

    :return: Tuple
       File path to key_file located in a tempfile location
       File path to cert_file located in a tempfile location
    """
    if folder is None:
        folder = tempfile.gettempdir()
    elif folder and not os.path.isdir(folder):
        raise Exception("Folder location does not exist.")
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False, dir=folder)
    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False, dir=folder)
    if use_openssl:
        try:
            import OpenSSL.crypto

            k = open(key_file.name, "wb")
            c = open(cert_file.name, "wb")
            try:
                pfx = open(pfx_path, "rb").read()
                p12 = OpenSSL.crypto.load_pkcs12(pfx, pfx_password)
            except OpenSSL.crypto.Error:
                raise RuntimeError("Invalid PFX password.  Unable to parse file.")
            k.write(
                OpenSSL.crypto.dump_privatekey(
                    OpenSSL.crypto.FILETYPE_PEM, p12.get_privatekey()
                )
            )
            c.write(
                OpenSSL.crypto.dump_certificate(
                    OpenSSL.crypto.FILETYPE_PEM, p12.get_certificate()
                )
            )
            k.close()
            c.close()
        except ImportError as e:
            raise e
        except Exception as ex:
            raise ex
    else:
        _default_backend = None
        if _crypto_version < [3, 0]:
            from cryptography.hazmat.backends import default_backend

            _default_backend = default_backend()

        with open(pfx_path, "rb") as f:
            (
                private_key,
                certificate,
                additional_certificates,
            ) = pkcs12.load_key_and_certificates(
                f.read(), str.encode(pfx_password), backend=_default_backend
            )
        cert_bytes = certificate.public_bytes(Encoding.PEM)
        pk_bytes = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        k = open(key_file.name, "wb")
        c = open(cert_file.name, "wb")

        k.write(pk_bytes)
        c.write(cert_bytes)
        k.close()
        c.close()
        del k
        del c
    key_file.close()
    cert_file.close()
    return cert_file.name, key_file.name  # certificate/key
