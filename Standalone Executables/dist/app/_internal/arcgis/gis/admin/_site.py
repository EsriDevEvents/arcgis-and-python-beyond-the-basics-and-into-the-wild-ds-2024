from typing import Optional
from ._base import BasePortalAdmin


########################################################################
class Site(BasePortalAdmin):
    """
    Site is the root resources used after a local GIS is installed. Here
    administrators can create, export, import, and join sites.
    """

    _url = None
    _con = None
    _pa = None
    _gis = None
    _properties = None
    _json = None
    _json_dict = None

    # ----------------------------------------------------------------------
    def __init__(self, url, portaladmin, **kwargs):
        """Constructor"""
        super(Site, self).__init__(url=url, gis=portaladmin._gis)
        initialize = kwargs.pop("initialize", False)
        self._url = url
        self._pa = portaladmin
        self._gis = portaladmin._gis
        self._con = portaladmin._con
        if initialize:
            self._init()

    # ----------------------------------------------------------------------
    @staticmethod
    def create(
        con,
        url: str,
        username: str,
        password: str,
        full_name: str,
        email: str,
        content_store: str,
        description: str = "",
        question_idx: Optional[int] = None,
        question_ans: Optional[str] = None,
        license_file: Optional[str] = None,
        user_license: Optional[str] = None,
    ):
        """
        The create site operation initializes and configures Portal for
        ArcGIS for use. It must be the first operation invoked after
        installation. Creating a new site involves:
          - Creating the initial administrator account
          - Creating a new database administrator account (which is same as
            the initial administrator account)
          - Creating token shared keys
          - Registering directories
        This operation is time consuming, as the database is initialized
        and populated with default templates and content. If the database
        directory is not empty, this operation attempts to migrate the
        database to the current version while keeping its data intact. At
        the end of this operation, the web server that hosts the API is
        restarted.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        con                             Required Connection. The connection object.
        ---------------------------     --------------------------------------------------------------------
        url                             Required string. The portal administration url
                                        Ex: https://mysite.com/<web adaptor>/portaladmin
        ---------------------------     --------------------------------------------------------------------
        username                        Required string. The initial admin account name
        ---------------------------     --------------------------------------------------------------------
        password                        Required string. The password for initial admin account
        ---------------------------     --------------------------------------------------------------------
        full_name                       Required string. The full name of the admin account
        ---------------------------     --------------------------------------------------------------------
        email                           Required string. The account email address
        ---------------------------     --------------------------------------------------------------------
        content_store                   Required string. JSON string including the path to the location of
                                        the site's content.
        ---------------------------     --------------------------------------------------------------------
        description                     Optional string. The optional description for the account
        ---------------------------     --------------------------------------------------------------------
        question_idx                    Optional integer. The index of the secret question to retrieve a
                                        forgotten password
        ---------------------------     --------------------------------------------------------------------
        question_ans                    Optional string. The answer to the secret question
        ---------------------------     --------------------------------------------------------------------
        license_file                    Optional string. The portal license file. Starting at 10.7, you will
                                        obtain your portal license file - which contains information
                                        regarding your user types, apps, and app bundles-from My Esri. For
                                        more information, see Obtain a portal license file.
        ---------------------------     --------------------------------------------------------------------
        user_license                    The user type for the initial administrator account. The values
                                        listed below are the user types that are compatible with the
                                        Administrator role.

                                        Values: creatorUT, GISProfessionalBasicUT,
                                                GISProfessionalStdUT, GISProfessionalAdvUT

        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/createNewSite" % url
        params = {
            "f": "json",
            "username": username,
            "password": password,
            "fullName": full_name,
            "email": email,
            "description": description,
            "contentStore": content_store,
        }
        if question_idx and question_ans:
            params["securityQuestionIdx"] = question_idx
            params["securityQuestionAns"] = question_ans
        if user_license:
            params["userLicenseTypeId"] = user_license
        if license_file:
            license_file = {"file": license_file}
        return con.post(url, params, files=license_file)

    # ----------------------------------------------------------------------
    def export_site(self, location: str):
        """
        This operation exports the portal site configuration to a location
        you specify. The exported file includes the following information:
          Content directory - the content directory contains the data
           associated with every item in the portal
          Database dump file - a plain-text file that contains the SQL
           commands required to reconstruct the portal database
          Configuration store connection file - a JSON file that contains
           the database connection information

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The path to the folder accessible to the portal
                                        where the exported site configuration will be written.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        .. code-block:: python

            USAGE: Export Portal Site to a location the Portal server has access to.  ** This can be a
                   lengthy operation.

            from arcgis.gis import GIS
            gis = GIS("https://yourportal.com/portal", "portaladmin", "password")
            sitemgr = gis.admin.site
            response = sitemgr.export_site(r'c:\\temp')
            print(response)

            # Output
            {'status': 'success', 'location': 'C:\\Temp\\June-9-2018-5-22-29-PM-EDT-FULL.portalsite'}

        """
        url = "%s/exportSite" % self._url
        params = {"f": "json", "location": location}
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def import_site(self, location: str):
        """
        The importSite operation lets you restore your site from a backup
        site configuration file that you created using the exportSite
        operation. It imports the site configuration file into the
        currently running portal site.
        The importSite operation will replace all site configurations with
        information included in the backup site configuration file. See the
        export_site operation documentation for details on what the backup
        file includes. The importSite operation also updates the portal
        content index.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. A file path to an exported configuration.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = "%s/importSite" % self._url
        if url.find(":7443") == -1:
            raise ValueError(
                "You must access portal not using the web adaptor (port 7443)"
            )
        params = {"f": "json", "location": location}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    def join(self, admin_url: str, username: str, password: str):
        """
        The joinSite operation connects a portal machine to an existing
        site. You must provide an account with administrative privileges to
        the site for the operation to be successful.
        When an attempt is made to join a site, the site validates the
        administrative credentials, then returns connection information
        about its configuration store back to the portal machine. The portal
        machine then uses the connection information to work with the
        configuration store.
        If this is the first portal machine in your site, use the Create
        Site operation instead.
        The join operation:
         - Registers a machine to an existing site (active machine)
         - Creates a snapshot of the database of the active machine
         - Updates the token shared key
         - Updates Web Adaptor configurations
        Sets up replication to keep the database of both machines in sync
        The operation is time-consuming as the database is configured on
        the machine and all configurations are applied from the active
        machine. After the operation is complete, the web server that hosts
        the API will be restarted.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        admin_url                       Required string. The admin URL of the existing portal site to which
                                        a machine will be joined
        ---------------------------     --------------------------------------------------------------------
        username                        Required string. The username for the initial administrator account
                                        of the existing portal site.
        ---------------------------     --------------------------------------------------------------------
        password                        Required string. The password for the initial administrator account
                                        of the existing portal site.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = "%s/joinSite" % self._url
        params = {
            "f": "json",
            "machineAdminUrl": admin_url,
            "username": username,
            "password": password,
        }
        return self._con.post(path=url, postdata=params)
