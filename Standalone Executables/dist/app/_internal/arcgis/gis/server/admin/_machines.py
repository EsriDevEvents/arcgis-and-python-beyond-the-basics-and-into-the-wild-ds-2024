"""
This resource represents a collection of all the server machines that
have been registered with the site. It other words, it represents
the total computing power of your site. A site will continue to run
as long as there is one server machine online.
For a server machine to start hosting GIS services, it must be
grouped (or clustered). When you create a new site, a cluster called
'default' is created for you.
The list of server machines in your site can be dynamic. You can
register additional server machines when you need to increase the
computing power of your site or unregister them if you no longer
need them.
"""
from __future__ import absolute_import
from __future__ import print_function
import json
from .._common import BaseServer
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import GIS
from arcgis.gis._impl._con import Connection
from typing import Optional


########################################################################
class MachineManager(BaseServer):
    """
    This resource represents a collection of all the server machines that
    have been registered with the site. In other words, it represents the
    total computing power of your site. A site will continue to run as long
    as there is at least one server machine online.

    For a server machine to start hosting GIS services, it must be in a cluster
    (note that clusters have been deprecated, see
    http://server.arcgis.com/en/server/latest/administer/windows/about-single-cluster-mode.htm ).
    When you create a new site, a cluster called 'default' (deployed with
    singleClusterMode set to true) is created for you.

    The list of server machines in your site can be dynamic. You can
    register additional server machines when you need to increase the
    computing power of your site, or unregister them if you no longer need
    them.
    """

    _machines = None
    _json_dict = None
    _con = None
    _url = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The admin URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine information.
                               Default is False.
        ==================     ====================================================================

        """

        super(MachineManager, self).__init__(gis=gis, url=url)
        self._url = url
        self._con = gis
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def _init(self, connection: Connection = None) -> dict:
        """Loads the properties into the class."""
        if connection is None:
            connection = self._con
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not attr.startswith("_")
        ]
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                if "machines" in result:
                    self._machines = []
                    for m in result["machines"]:
                        self._machines.append(
                            Machine(
                                url=self._url + "/%s" % m["machineName"], gis=self._con
                            )
                        )
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def list(self) -> list:
        """

        :return:
             A list of :class:`machines <arcgis.gis.server.Machine>` that are part of the server configuration.


        """
        if self._machines is None:
            self._init()
        return self._machines

    # ----------------------------------------------------------------------
    def get(self, machine_name: str) -> "Machine":
        """
        Provides the machine object for a given machine.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        machine_name           Required string. The name of the server. Example: machines_obj.get("SERVER.DOMAIN.COM")
        ==================     ====================================================================

        :return:
            :class:`~arcgis.gis.server.Machine` object

        """
        url = self._url + "/%s" % machine_name
        return Machine(url=url, gis=self._con)

    # ----------------------------------------------------------------------
    def register(self, name: str, admin_url: str) -> bool:
        """
        For a server machine to participate in a site, it needs to be
        registered with the site. The server machine must have ArcGIS
        Server software installed and authorized.

        Registering machines this way is a "pull" approach to growing
        the site and is a convenient way when a large number of machines
        need to be added to a site. A server machine can also
        choose to join a site.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The name of the server machine.
        ------------------     --------------------------------------------------------------------
        admin_url              Required string. The URL where the Administrator API is running on
                               the server machine. Example: http: //<machineName>:6080/arcgis/admin
        ==================     ====================================================================


        :return:
           A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json", "machineName": name, "adminURL": admin_url}
        url = "%s/register" % self._url
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def rename(self, name: str, new_name: str) -> bool:
        """
        You must use this operation if one of the registered machines
        has undergone a name change. This operation updates any
        references to the former machine configuration.

        By default, when the server is restarted, it is capable of
        identifying a name change and repairing itself and all its
        references. This operation is a manual call to handle the
        machine name change.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required string. The former name of the server machine that is
                               registered with the site.
        ------------------     --------------------------------------------------------------------
        new_name               Required string. The new name of the server machine.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json", "machineName": name, "newMachineName": new_name}
        url = self._url + "/rename"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class Machine(BaseServer):
    """
    A resource to provide administrative tools for managing this machine
    and the required SSL Certificate.

    .. note::
        The Machine

        A server machine represents a machine on which ArcGIS Server
        software has been installed and licensed. A site is made up of one
        or more machines that must be registered with the site.  The site's
        machines work together to host GIS services and data, and provide
        administrative capabilities for the site. Each server machine is
        capable of performing all these administrative tasks and hence a
        site can be thought of as a distributed peer-to-peer network of
        machines. The server machine communicates with its peers over a
        range of TCP and UDP ports that can be configured using the edit operation
        (https://developers.arcgis.com/rest/enterprise-administration/server/editmachine.htm ).


    .. note::
        SSL Certificates

        A certificate represents a key pair that has been digitally signed
        and acknowledged by a Certifying Authority (CA). It is the most
        fundamental component in enabling SSL on your server. Before you
        enable SSL on your server, you need to generate a certificate and
        get it signed by a trusted CA.

        The Generate Certificate
        (https://developers.arcgis.com/rest/enterprise-administration/server/generatecertificate.htm )
        operation creates a new self-signed certificate and adds it to
        the keystore. For your convenience, the server is capable of generating
        self-signed certificates that can be used during development or
        staging. However, it is critical that you obtain CA-signed
        certificates when standing up a production server. Even though
        a self-signed certificate can be used to enable SSL, it is recommended
        that you use these only on staging or development servers.

        In order to get a certificate signed by a CA, you need to generate
        a CSR (certificate signing request) and then submit it to your CA.
        The CA will sign your certificate request which can then be
        imported into the server by using the import CA signed certificate
        operation.


    """

    _appServerMaxHeapSize = None
    _webServerSSLEnabled = None
    _webServerMaxHeapSize = None
    _platform = None
    _adminURL = None
    _machineName = None
    _ServerStartTime = None
    _webServerCertificateAlias = None
    _socMaxHeapSize = None
    _synchronize = None
    _configuredState = None
    _ports = None
    _json = None
    _json_dict = None
    _con = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        connection = gis
        super(Machine, self).__init__(gis=connection, url=url)
        self._url = url
        self._con = connection
        self._currentURL = url
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def hardware(self) -> dict:
        """
        This resource displays hardware information for the machine in your
        ArcGIS Server site. It updates the information when it detects any
        change to the configuration of your machine, as well as each time
        the machine is restarted.

        :return: Dict
        """
        url = self._url + "/hardware"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def status(self) -> dict:
        """
        Gets the status/state of this machine.
        """
        uURL = self._url + "/status"
        params = {
            "f": "json",
        }
        return self._con.get(path=uURL, params=params)

    # ----------------------------------------------------------------------
    def start(self) -> bool:
        """
        Starts this server machine. Starting the machine enables its
        ability to host GIS services.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {"f": "json"}
        uURL = self._url + "/start"
        res = self._con.post(path=uURL, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    def synchronize(self) -> bool:
        """
        On occasion, one or more machines in a server site might be
        unavailable due to network issues or because they are down
        (intentionally or unintentionally). Once these machines become
        available again, they will need to synchronize with the site to
        pick up any changes made to the site during that downtime. This
        is done automatically by the site, but it is only a one-time
        attempt. If there are any issues with this synchronizing effort,
        a SEVERE message is logged.

        This operation allows administrators to manually synchronize specific
        machines with the site. Synchronizing a machine with the site will
        reconfigure the machine and redeploy all services. This will take a
        few minutes. During this time, all administrative operations on the
        site will be blocked.

        :return: Boolean

        """
        url = self._url + "/synchronizeWithSite"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def stop(self) -> bool:
        """
        Stops this server machine. Stopping the machine disables its
        ability to host GIS services.


        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {"f": "json"}
        uURL = self._url + "/stop"
        res = self._con.post(path=uURL, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def unregister(self) -> bool:
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
    def ssl_certificates(self) -> dict:
        """
        Gets the list of all the certificates (self-signed and CA-signed)
        created for the server machine. The server securely stores these
        certificates inside a key store within the configuration store.
        """
        params = {"f": "json"}
        url = self._url + "/sslcertificates"
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
        url = self._url + "/sslcertificates/{cert}".format(cert=certificate)
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def delete_certificate(self, certificate: str) -> bool:
        """
        Deletes a SSL certificate using the certificate alias.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        certificate            Required string. The name of the certificate to delete
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json", "csrfPreventToken": self._con.token}
        url = self._url + "/sslcertificates/{cert}/delete".format(cert=certificate)
        res = self._con.post(url, params)
        if isinstance(res, dict) and "status" in res:
            return res["status"]
        else:
            return res

    # ----------------------------------------------------------------------
    def export_certificate(self, certificate: str) -> str:
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
        url = self._url + "/sslcertificates/%s/export" % certificate
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def generate_CSR(self, certificate: str) -> dict:
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
        url = self._url + "/sslcertificates/{cert}/generateCSR".format(cert=certificate)
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def import_CA_signed_certificate(
        self, certificate: str, ca_signed_certificate: str
    ) -> bool:
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
        url = self._url + "/sslcertificates/{cert}/importCASignedCertificate".format(
            cert=certificate
        )
        files = {"caSignedCertificate": ca_signed_certificate}
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def import_existing_server_certificate(
        self, alias: str, cert_password: str, cert_file: str
    ) -> bool:
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
        url = self._url + "/sslcertificates/importExistingServerCertificate"
        params = {"f": "json", "alias": alias, "certPassword": cert_password}
        files = {"certFile": cert_file}
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def import_root_certificate(self, alias: str, root_CA_certificate: str) -> dict:
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
        url = self._url + "/sslcertificates/importRootOrIntermediate"
        params = {"f": "json", "alias": alias}
        files = {"rootCACertificate": root_CA_certificate}
        return self._con.post(path=url, postdata=params, files=files)
