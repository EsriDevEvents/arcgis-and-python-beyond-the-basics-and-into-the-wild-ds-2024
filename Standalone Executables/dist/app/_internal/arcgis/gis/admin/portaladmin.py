"""
Entry point to working with local enterprise GIS functions
"""
from datetime import datetime
from typing import Optional
from ...gis._impl._con import Connection
from ...gis import GIS, Item, User
from ._resources import PortalResourceManager
from ._base import BasePortalAdmin
from ...apps.tracker._location_tracking import LocationTrackingManager


########################################################################
class PortalAdminManager(BasePortalAdmin):
    """
    This is the root resource for administering your portal. Starting from
    this root, all of the portal's environment is organized into a
    hierarchy of resources and operations. A version number is returned as
    a part of this resource. After installation, the portal can be
    configured using the Create Site operation. Once initialized, the
    portal environment is available through System and Security resources.

    ================    =================================================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------------------------
    url                 web address to portaladmin rest API (ends with: portal//sharing/rest/)
    ----------------    ---------------------------------------------------------------------------------
    gis                 GIS object containing Administrative credentials
    ----------------    ---------------------------------------------------------------------------------
    initialize          Optional if True, properties of REST endpoint are loaded on creation of object.
                        False (default) means they are loaded when needed.
    ================    =================================================================================

    """

    _logs = None
    _federation = None
    _system = None
    _security = None
    _machines = None
    _site = None
    _url = None
    _gis = None
    _idp = None
    _ux = None
    _sp = None
    _metadata = None
    _collaborations = None
    _servers = None
    _pp = None
    _license = None
    _livingatlas = None
    _category_schema = None
    _whm = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None, **kwargs):
        """initializer"""
        if kwargs.pop("is_admin", True):
            super(PortalAdminManager, self).__init__(url=url, gis=gis, **kwargs)
            initialize = kwargs.pop("initialize", False)
            if isinstance(gis, Connection):
                self._con = gis
            elif isinstance(gis, GIS):
                self._gis = gis
                self._con = gis._con
            else:
                raise ValueError("connection must be of type GIS or Connection")
            try:
                self.resources = PortalResourceManager(gis=self._gis)
            except:
                pass
            if initialize:
                self._init(self._gis)
        else:
            super(PortalAdminManager, self).__init__(
                url=url, gis=gis, is_admin=False, initialize=False
            )
            if isinstance(gis, Connection):
                self._con = gis
            elif isinstance(gis, GIS):
                self._gis = gis
                self._con = gis._con

    # ----------------------------------------------------------------------
    @property
    def ux(self):
        """
        returns a UX/UI manager with properties such as description, featured_content, name, etc.

        :return:
            :class:`~arcgis.gis.admin.UX` object

        """
        if self._ux is None:
            from ._ux import UX

            self._ux = UX(gis=self._gis)
        return self._ux

    # ----------------------------------------------------------------------
    @property
    def collaborations(self):
        """
        The collaborations resource lists all collaborations in which a
        portal participates

        :return:
            :class:`~arcgis.gis.admin.CollaborationManager` object

        """
        if self._collaborations is None:
            from ._collaboration import CollaborationManager

            self._collaborations = CollaborationManager(gis=self._gis)
        return self._collaborations

    # ----------------------------------------------------------------------
    @property
    def category_schema(self):
        """
        This resource allows for the setting and manipulating of category schemas.

        :return:
            :class:`~arcgis.gis.admin.CategoryManager` object

        """
        if self._category_schema is None:
            from ._catagoryschema import CategoryManager

            self._category_schema = CategoryManager(gis=self._gis)
        return self._category_schema

    # ----------------------------------------------------------------------
    @property
    def idp(self):
        """
        This resource allows for the setting and configuration of the identity provider

        :return:
            :class:`~arcgis.gis.admin.IdentityProviderManager` object

        """
        if self._idp is None:
            from ._idp import IdentityProviderManager

            self._idp = IdentityProviderManager(gis=self._gis)
        return self._idp

    # ----------------------------------------------------------------------
    @property
    def location_tracking(self):
        """

        The manager for Location Tracking. See :class:`~arcgis.apps.tracker.LocationTrackingManager`.

        :return:
            :class:`~arcgis.apps.tracker.LocationTrackingManager` object

        """
        return LocationTrackingManager(self._gis)

    # ----------------------------------------------------------------------
    @property
    def social_providers(self):
        """
        This resource allows for the setting and configuration of the social providers
        for a GIS.

        :return:
            :class:`~arcgis.gis.admin.SocialProviders` object

        """
        if self._sp is None:
            from ._socialproviders import SocialProviders

            self._sp = SocialProviders(gis=self._gis)
        return self._sp

    @property
    def info(self) -> dict:
        """
        Returns the current version and build number of the Enterprise system

        :returns: dict
        """
        if self._gis.version >= [10, 3]:
            url = "%s/portaladmin/info" % self._gis._portal.url
            params = {"f": "json"}
            return self._gis._con.get(url, params)
        return None

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """
        returns a set of tools to work with ArcGIS Enterprise metadata
        settings.

        :return:
            :class:`~arcgis.gis.admin.MetadataManager` object

        """
        if self._metadata is None:
            from ._metadata import MetadataManager

            self._metadata = MetadataManager(gis=self._gis)
        return self._metadata

    # ----------------------------------------------------------------------
    @property
    def servers(self):
        """returns a server manager object

        :return:
            :class:`~arcgis.gis.server.ServerManager`

        """
        if self._servers is None:
            from ..server import ServerManager

            self._servers = ServerManager(gis=self._gis)
        return self._servers

    # ----------------------------------------------------------------------
    def scheduled_tasks(
        self,
        item: Optional[Item] = None,
        active: Optional[bool] = None,
        user: Optional[User] = None,
        types: Optional[str] = None,
    ):
        """
        This property allows `org_admins` to be able to see all scheduled tasks on the enterprise

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        item              Optional Item. The item to query tasks about.
        ----------------  -------------------------------------------------------------------------------
        active            Optional Bool. Queries tasks based on active status.
        ----------------  -------------------------------------------------------------------------------
        user              Optional User. Search for tasks for a single user.
        ----------------  -------------------------------------------------------------------------------
        types             Optional String. The type of notebook execution for the item.  This can be
                          `ExecuteNotebook`, or `UpdateInsightsWorkbook`.
        ================  ===============================================================================


        :return: List of Tasks

        """
        _tasks = []
        num = 100
        url = f"{self._gis._portal.resturl}portals/self/allScheduledTasks"
        params = {"f": "json", "start": 1, "num": num}
        if item:
            params["itemId"] = item.itemid
        if not active is None:
            params["active"] = active
        if user:
            params["userFilter"] = user.username
        if types:
            params["types"] = types
        res = self._con.get(url, params)
        start = res["nextStart"]
        _tasks.extend(res["tasks"])
        while start != -1:
            params["start"] = start
            params["num"] = num
            res = self._con.get(url, params)
            if len(res["tasks"]) == 0:
                break
            _tasks.extend(res["tasks"])
            start = res["nextStart"]
        return _tasks

    # ----------------------------------------------------------------------
    @property
    def machines(self):
        """
        This resource lists all the portal machines in a site. Each portal
        machine has a status that indicates whether the machine is ready
        to accept requests.


        :return:
            :class:`~arcgis.gis.admin.Machines` object

        """
        if self._machines is None:
            from ._machines import Machines

            # url root needs to be administrator site root
            url = "%s/portaladmin/machines" % self._gis._portal.url
            self._machines = Machines(url=url, gis=self._gis, portaladmin=self)
        return self._machines

    # ----------------------------------------------------------------------
    @property
    def security(self):
        """
        accesses the controls for the security of a local portal site

        :return:
            :class:`~arcgis.gis.admin.Security` object

        """

        if self._security is None:
            from ._security import Security

            url = "%s/portaladmin/security" % self._gis._portal.url
            self._security = Security(url=url, gis=self._gis)
        return self._security

    # ----------------------------------------------------------------------
    @property
    def site(self):
        """
        Site is the root resources used after a local GIS is installed. Here
        administrators can create, export, import, and join sites.

        :return:
            :class:`~arcgis.gis.admin.Site` object

        """
        if self._site is None:
            from ._site import Site

            self._site = Site(url=self._url, portaladmin=self)
        return self._site

    # ----------------------------------------------------------------------
    @property
    def logs(self):
        """
        returns a class to work with the portal logs

        :return:
            :class:`~arcgis.gis.admin.Logs` object

        """
        if self._logs is None:
            from ._logs import Logs

            url = "%s/portaladmin/logs" % self._gis._portal.url
            self._logs = Logs(url=url, gis=self._gis)
        return self._logs

    # ----------------------------------------------------------------------
    @property
    def federation(self):
        """
        provides access into the federation settings of a server.


        :return:
            :class:`~arcgis.gis.admin.Federation` object

        """
        if self._federation is None:
            from ._federation import Federation

            url = "%s/portaladmin/federation" % self._gis._portal.url
            self._federation = Federation(url=url, gis=self._gis)
        return self._federation

    # ----------------------------------------------------------------------
    @property
    def system(self):
        """
        This resource provides access to the ArcGIS Web Adaptor
        configuration, portal directories, database management server,
        indexing capabilities, license information, and the properties of
        your portal.


        :return:
            :class:`~arcgis.gis.admin.System` object

        """
        if self._system is None:
            from ._system import System

            url = "%s/portaladmin/system" % self._gis._portal.url
            self._system = System(url=url, gis=self._gis)
        return self._system

    # ----------------------------------------------------------------------
    @property
    def password_policy(self):
        """tools to manage a Site's password policy

        :return:
            :class:`~arcgis.gis.admin.PasswordPolicy` object

        """
        if self._pp is None:
            from ._security import PasswordPolicy

            url = "%s/portals/self/securityPolicy" % (self._gis._portal.resturl)
            self._pp = PasswordPolicy(url=url, gis=self._gis)
        return self._pp

    # ----------------------------------------------------------------------
    @property
    def license(self):
        """
        provides a set of tools to access and manage user licenses and
        entitlements.

        :return:
            :class:`~arcgis.gis.admin.LicenseManager` object

        """
        if self._license is None:
            from ._license import LicenseManager

            url = f"{self._gis._portal.resturl}/portals/self/purchases"
            self._license = LicenseManager(url=url, gis=self._gis)
        return self._license

    # ----------------------------------------------------------------------
    @property
    def living_atlas(self):
        """
        provides a set of tools to manage and setup Living Atlas content.

        :return:
            :class:`~arcgis.gis.admin.LivingAtlas` object

        """
        if self._livingatlas is None:
            from ._livingatlas import LivingAtlas

            url = self._url + "/system/content/livingatlas"
            self._livingatlas = LivingAtlas(url=url, gis=self._gis)
        return self._livingatlas

    # ----------------------------------------------------------------------
    @property
    def webhooks(self):
        """Provides access to Portal's WebHook Manager

        :return:
            :class:`~arcgis.gis.admin.WebhookManager` object

        """
        if self._whm is None and self._gis.version >= [6, 4]:
            from ._wh import WebhookManager

            url = self._gis._portal.resturl + "portals/self/webhooks"
            self._whm = WebhookManager(url=url, gis=self._gis)
        return self._whm

    # ----------------------------------------------------------------------
    @property
    def mode(self) -> dict:
        """
        Gets/Set the mode of the ArcGIS Enterprise deployment.  When obtaining
        the mode, it returns information about the current state of the system.

        ================  ===============================================================================
        **Key**           **Description**
        ----------------  -------------------------------------------------------------------------------
        read_only         Required Boolean.  A boolean that specifies whether the Enterprise portal is
                          in read-only mode. Read-only mode will block requests to modify or create any
                          data, including content, users, groups, or site settings.
                          The default value is false.
        ----------------  -------------------------------------------------------------------------------
        message           Optional String. Sets a custom message to be displayed whenever an attempt to
                          modify or update content or site settings is made through the API.
        ================  ===============================================================================

        .. code-block:: python

            >>> gis.admin.mode({'read_only' : False})
            >>> assert gis.admin.mode['isReadOnly'] == False

        """
        url = "%s/portaladmin/mode" % self._gis._portal.url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @mode.setter
    def mode(self, mode: dict):
        """
        See main ``mode`` property docstring.
        """
        url = "%s/portaladmin/mode/update" % self._gis._portal.url
        if mode is None:
            mode = {"read_only": False, "message": ""}
        params = {"f": "json", "isReadOnly": mode.pop("read_only", False)}
        if "message" in mode:
            params["description"] = mode.pop("message", "")
        res = self._con.post(url, params)
        if "status" in res and res["status"] != "success":
            raise RuntimeError(res)

    # ----------------------------------------------------------------------
    def history(
        self, start_date: datetime, num: int = 100, save_folder: Optional[str] = None
    ):
        """
        Returns a CSV file containing the login history from a start_date to the present.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        start_date        Required datetime.datetime object. The beginning date.
        ----------------  -------------------------------------------------------------------------------
        num               Optional Integer. The maximum number of records to return.
        ----------------  -------------------------------------------------------------------------------
        save_folder       Optional String. The save location of the CSV file.
        ================  ===============================================================================

        :return: string

        """
        if self._gis.properties.isPortal:
            raise NotImplementedError("Login history only available in ArcGIS Online")
        else:
            if self._gis.version >= [6.4]:
                import tempfile, json
                from arcgis._impl.common._utils import _date_handler

                if save_folder is None:
                    save_folder = tempfile.gettempdir()
                url = "{url}portals/self/history".format(url=self._gis._portal.resturl)
                params = {
                    "f": "csv",
                    "num": num,
                    "all": True,
                    "fromDate": json.dumps(start_date, default=_date_handler),
                }
                return self._gis._con.post(
                    url, params, file_name="history.csv", out_folder=save_folder
                )
