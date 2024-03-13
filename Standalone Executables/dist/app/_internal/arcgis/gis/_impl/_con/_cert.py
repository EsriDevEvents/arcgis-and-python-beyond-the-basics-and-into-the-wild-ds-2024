import contextlib
import OpenSSL.crypto
import os
import ssl
import tempfile


# ----------------------------------------------------------------------
def pfx_to_pem(pfx_path, pfx_password):
    """Decrypts the .pfx file to be used with requests.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    pfx_path            Required string.  File pathname to .pfx file to parse.
    ---------------     --------------------------------------------------------------------
    pfx_password        Required string.  Password to open .pfx file to extract key/cert.
    ===============     ====================================================================

    :return:
       File path to key_file located in a tempfile location
       File path to cert_file located in a tempfile location
    """
    try:
        import OpenSSL.crypto
    except:
        raise RuntimeError(
            "OpenSSL.crypto library is not installed.  You must install this in order "
            + "to use a PFX for connecting to a PKI protected portal."
        )
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
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
    return key_file.name, cert_file.name


# if __name__ == "__main__":
# cert_file=r"./pki_cert_test.pfx"
# password="1234"
# print(pfx_to_pem(cert_file, password))
