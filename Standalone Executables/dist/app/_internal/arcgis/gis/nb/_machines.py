import os
from typing import Optional

from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


########################################################################
class MachineManager(object):
    """
    This resource provides the name and URL of the ArcGIS Notebook
    Server machine in the site. An object of this
    class can be created using :attr:`~arcgis.gis.nb.NotebookServer.machine` property of the
    :class:`~arcgis.gis.nb.NotebookServer` class

    """

    _url = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< MachineManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< MachineManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def list(self):
        """
        returns all :class:`~arcgis.gis.nb.Machine` instances
        """
        res = []
        for m in self.properties.machines:
            url = self._url + "/{m}".format(m=m.machineName)
            res.append(Machine(url=url, gis=self._gis))
        return res


########################################################################
class Machine(object):
    """
    This resource provides information about the machine in your ArcGIS
    Notebook Server site.  You can update some of these properties
    using the Edit Machine operation.
    """

    _url = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< Machine @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< Machine @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get/Set the properties on the ArcGIS Notebook Server machine.

        Set operation allows you to update properties on the ArcGIS
        Notebook Server machine.

        ArcGIS Notebook Server uses port 11443 for communication. When you
        create a site, this is assigned as the default. You must ensure that
        your firewall allows communication through port 11443.
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, value):
        """
        See main ``properties`` property docstring.
        """
        import json

        url = self._url + "/edit"
        params = {"f": "json"}
        params.update(dict(self.properties))
        if isinstance(value, dict):
            params.update(value)
        else:
            raise ValueError("value must be a dictionary")
        try:
            self._con.post(url, params)
        except json.JSONDecodeError:
            pass

    # ----------------------------------------------------------------------
    @property
    def hardware(self):
        """
        This resource displays hardware information for the machine in your
        ArcGIS Notebook Server site. It updates the information when it
        detects any change to the configuration of your machine, as well
        as each time the machine is restarted.

        :return: Dict
        """
        url = self._url + "/hardware"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        Returns the Machine's Status

        :return: Dict

        """
        url = self._url + "/status"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def create_self_signed_cert(
        self,
        alias: str,
        keysize: str,
        common_name: str,
        org_unit: str,
        organization: str,
        city: str,
        state: str,
        country: str,
        keyalg: str = "RSA",
        sigalg: str = "SHA1withRSA",
        validity: int = 90,
        san: Optional[str] = None,
    ):
        """
        Use this operation to create a self-signed certificate or as a
        starting point for getting a production-ready CA-signed certificate.
        ArcGIS Notebook Server will generate a certificate for you and store
        it in its keystore. The certificate generated should only be used in
        development and staging environments.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        alias                  Required String. A unique name that easily identifies the certificate.
        ------------------     --------------------------------------------------------------------
        keyalg	               Optional String. The algorithm used to generate the key pairs. The
                               default is RSA.
        ------------------     --------------------------------------------------------------------
        keysize                Required String. Specifies the size in bits to use when generating
                               the cryptographic keys used to create the certificate. The larger
                               the key size, the harder it is to break the encryption; however, the
                               time to decrypt encrypted data increases with key size. For DSA, the
                               key size can be between 512 and 1,024. For RSA, the recommended key
                               size is 2,048 or greater.
        ------------------     --------------------------------------------------------------------
        sigalg                 Optional String. Use the default (SHA1withRSA). If your organization
                               has specific security restrictions, then one of the following
                               algorithms can be used for DSA: SHA256withRSA, SHA384withRSA,
                               SHA512withRSA, SHA1withDSA.
        ------------------     --------------------------------------------------------------------
        common_name            Required String. Use the domain name of your server name as the
                               common name. If your server will be accessed on the Internet through
                               the URL ``https://www.notebookserver.com:11443/arcgis/``, use
                               ``www.notebookserver.com`` as the common name.If your server will only
                               be accessible on your local area network (LAN) through the URL
                               ``https://notebookserver.domain.com:11443/arcgis/``, use notebookserver
                               as the common name.
        ------------------     --------------------------------------------------------------------
        org_unit	       Required String. The name of your organizational unit, for example,
                               GIS Department.
        ------------------     --------------------------------------------------------------------
        organization	       Required String. The name of your organization, for example, Esri.
        ------------------     --------------------------------------------------------------------
        city                   Required String. The name of the city or locality, for example,
                               Redlands.
        ------------------     --------------------------------------------------------------------
        state                  Required String. The full name of your state or province, for
                               example, California.
        ------------------     --------------------------------------------------------------------
        country                Required String. The abbreviated code for your country, for
                               example, US.
        ------------------     --------------------------------------------------------------------
        validity               Required Integer. The total time in days during which this
                               certificate will be valid, for example, 365. The default is 90.
        ------------------     --------------------------------------------------------------------
        san                    Optional String. The subject alternative name (SAN) is an optional
                               parameter that defines alternatives to the common name (CN)
                               specified in the SSL certificate. There cannot be any spaces in the
                               SAN parameter value. If no SAN is defined, a website can only be
                               accessed (without SSL certificate errors) by using the common name
                               in the URL. If a SAN is defined and a DNS name is present, the
                               website can only be accessed by what is listed in the SAN. Multiple
                               DNS names can be specified if desired. For example, the URLs
                               ``https://www.esri.com``, ``https://esri``, and ``https://10.60.1.16`` can be
                               used to access the same site if the SSL certificate is created
                               using the following SAN parameter
                               value: ``DNS:www.esri.com,DNS:esri,IP:10.60.1.16``
        ==================     ====================================================================

        :return: Boolean

        """
        url = self._url + "/sslCertificates/generate"
        params = {
            "f": "json",
            "alias": alias,
            "keyalg": keyalg,
            "keysize": keysize,
            "sigalg": sigalg,
            "commonName": common_name,
            "organizationalUnit": org_unit,
            "organization": organization,
            "city": city,
            "state": state,
            "country": country,
            "validity": validity,
            "san": san,
        }
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        import json

        res = False
        try:
            res = self._con.post(url, params, try_json=False)
        except json.JSONDecodeError:
            return True
        return res

    # ----------------------------------------------------------------------
    def unregister(self):
        """
        Removes this machine from the site.  This server machine will no
        longer participate in the site or run any of the GIS services.  All
        resources that were acquired by the server machine (memory, files,
        and so forth) will be released.

        Typically, you should only invoke this operation if the machine
        is going to be shut down for extended periods of time, or if it
        is being upgraded.

        Once a machine has been unregistered, you can create a new site
        or join an existing site.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {"f": "json"}
        uURL = self._url + "/unregister"
        res = self._con.post(path=uURL, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def ssl_certificates(self):
        """
        Gets the list of all the certificates (self-signed and CA-signed)
        created for the server machine. The server securely stores these
        certificates inside a key store within the configuration store.
        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates"
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def ssl_certificate(self, certificate: str):
        """
        Provides the self-signed certificate object.

        .. note::
            Even though a self-signed certificate can be used to enable SSL, it
            is recommended that you use a self-signed certificate only on staging
            or development servers.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        certificate            Required string. The name of the certificate in the key store to
                               grab information from.
        ==================     ====================================================================

        :return:
            The certificate object.

        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates/{cert}".format(cert=certificate)
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def delete_certificate(self, certificate: str):
        """
        Deletes a SSL certificate using the certificate alias.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        certificate            Required string. The name of the certificate to delete
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates/{cert}/delete".format(cert=certificate)
        res = self._con.get(path=url, params=params)
        if isinstance(res, dict) and "status" in res:
            return res["status"]
        else:
            return res

    # ----------------------------------------------------------------------
    def export_certificate(self, certificate: str):
        """
        Downloads an SSL certificate. The file returned by the
        server is an X.509 certificate. The downloaded certificate can then
        be imported into a client that is making HTTP requests.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        certificate            Required string. The name of the certificate in the key store.
        ==================     ====================================================================

        :return:
            The SSL certificate object.

        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates/%s/export" % certificate
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def generate_CSR(self, certificate: str):
        """
        Generates a certificate signing request (CSR) for a
        self-signed certificate. A CSR is required by a CA to create a
        digitally signed version of your certificate.  Supply the certificate
        object that was created with method ssl_certificate.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        certificate            Required string. The name of the certificate in the key store.
        ==================     ====================================================================

        :return:
           The CSR.
        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates/{cert}/generateCSR".format(cert=certificate)
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def import_CA_signed_certificate(
        self, certificate: str, ca_signed_certificate: str
    ):
        """
        Imports a certificate authority (CA)-signed SSL certificate into the key store.


        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        certificate                Required string. The name of the certificate in the key store.
        ----------------------     --------------------------------------------------------------------
        ca_signed_certificate      Required string. The multi-part POST parameter containing the
                                   signed certificate file.
        ======================     ====================================================================

        :return:
            A boolean indicating success (True) or failure (False).

        """
        params = {"f": "json"}
        url = self._url + "/sslCertificates/{cert}/importCASignedCertificate".format(
            cert=certificate
        )
        files = {"caSignedCertificate": ca_signed_certificate}
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def import_existing_server_certificate(
        self, alias: str, cert_password: str, cert_file: str
    ):
        """
        Imports an existing server certificate, stored in
        the PKCS #12 format, into the keystore.
        If the certificate is a CA-signed certificate, you must first
        import the CA root or intermediate certificate using the
        importRootCertificate operation.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        alias                  Required string. A unique name for the certificate that easily
                               identifies it.
        ------------------     --------------------------------------------------------------------
        cert_password          Required string. The password to unlock the file containing the certificate.
        ------------------     --------------------------------------------------------------------
        cert_file              Required string. The multi-part POST parameter containing the certificate file.
        ==================     ====================================================================


        :return:
            A boolean indicating success (True) or failure (False).

        """
        url = self._url + "/sslCertificates/importExistingServerCertificate"
        params = {"f": "json", "alias": alias, "certPassword": cert_password}
        files = {"certFile": cert_file}
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def import_root_certificate(self, alias: str, root_CA_certificate: str):
        """
        Imports a certificate authority's (CA) root and intermediate
        certificates into the keystore.

        To create a production quality CA-signed certificate, you need to
        add the CA's certificates into the keystore that enables the SSL
        mechanism to trust the CA (and the certificates it is signed).
        While most of the popular CA's certificates are already available
        in the keystore, you can use this operation if you have a custom
        CA or specific intermediate certificates.

        ===================     ====================================================================
        **Parameter**            **Description**
        -------------------     --------------------------------------------------------------------
        alias                   Required string. The name of the certificate.
        -------------------     --------------------------------------------------------------------
        root_CA_certificate     Required string. The multi-part POST parameter containing the certificate file.
        ===================     ====================================================================


        :return:
           A boolean indicating success (True) or failure (False).

        """
        url = self._url + "/sslCertificates/importRootOrIntermediate"
        params = {"f": "json", "alias": alias}
        files = {"rootCACertificate": root_CA_certificate}
        return self._con.post(path=url, postdata=params, files=files)
