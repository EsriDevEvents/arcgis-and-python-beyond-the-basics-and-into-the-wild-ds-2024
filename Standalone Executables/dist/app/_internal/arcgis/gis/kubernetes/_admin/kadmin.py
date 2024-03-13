from __future__ import annotations
from typing import Optional
from urllib.request import HTTPError
from arcgis._impl.common._isd import InsensitiveDict
from ._base import _BaseKube
from ._logs import LogManager
from ._datastores import DataStores
from ._overview import Overview
from ._usage import UsageStatistics
from ._mode import Mode
from ._system import SystemManager
from ._jobs import JobManager
from arcgis.gis.admin._license import LicenseManager
from arcgis.gis import Item, User
from arcgis.apps.tracker._location_tracking import LocationTrackingManager


class KubernetesAdmin(_BaseKube):
    """
    Kubernetes Administration Class. This class is not meant to be initialized
    directly, but instead is returned by the _admin_ property on the
    :class:`gis <arcgis.gis.GIS>` object when logged in as an administrator.

    .. code-block:: python

        #Usage Example:

        >>> gis = GIS(profile="your_kubernetes_admin_profile")
        >>> kube_admin = gis.admin
        >>> kube_admin

        <KubernetesAdmin at https://kubenetes.example.com/arcgis/admin>
    """

    _url = None
    _gis = None
    _con = None
    _log = None
    _ds = None
    _sm = None
    _sp = None
    _mode = None
    _catalog = None
    _idp = None
    _whm = None
    _security = None
    _services = None
    _license = None
    _metadata = None
    _uploads = None
    _properties = None
    _organizations = None
    _category_schema = None
    _jobs = None
    _collaborations = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """class initializer"""
        super(KubernetesAdmin, self)
        self._url = url
        self._gis = gis
        self._con = gis._con
        self._init(gis._con)

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = InsensitiveDict(result)
            else:
                self._json_dict = {}
                self._properties = InsensitiveDict({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = InsensitiveDict({})

    # ----------------------------------------------------------------------
    @property
    def overview(self) -> Overview:
        """
        Provides access to the :class:`~arcgis.gis.kubernetes.overview`
        resource to access persisted cache or real-time information.

        :return: :class:`~arcgis.gis.kubernetes.Overview` object

        """
        url = f"{self._url}/overview"
        return Overview(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def usage(self) -> UsageStatistics:
        """
        Provides access to the metrics viewer and metrics API tools.

        :return:
            An :class:`~arcgis.gis.kubernetes.UsageStatistics` object.

        """
        url = f"{self._url}/usagestatistics"
        return UsageStatistics(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def logs(self) -> LogManager:
        """Accesses a :class:`~arcgis.gis.kubernetes.LogManager` object to
        manage and query the Kubernetes logs."""
        if self._log is None:
            url = f"{self._url}/logs"
            self._log = LogManager(url, gis=self._gis)
        return self._log

    # ----------------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        """Provides access to a :class:`~arcgis.gis.kubernetes.Mode` object
        to help manage `service deployment modes <https://enterprise-k8s.arcgis.com/en/latest/administer/service-modes.htm>`_.
        """
        if self._mode is None:
            self._mode = Mode(url=f"{self._url}/mode", gis=self._gis)
        return self._mode

    # ----------------------------------------------------------------------
    @property
    def datastores(self) -> DataStores:
        """
        Provides access to the _Datastore Manager_, allowing the administrator
        to manage registered datastores.

        :return: A Kubernetes :class:`~arcgis.gis.kubernetes.DataStores` object.
        """
        if self._ds is None:
            url = self._url + "/data"
            self._ds = DataStores(url=url, gis=self._gis)
        return self._ds

    # ----------------------------------------------------------------------
    @property
    def system(self) -> SystemManager:
        """
        This is a collection of system-wide resources for your deployment
        such as the configuration store, licenses, and deployment-wide
        security.

        :return:
            :class:`~arcgis.gis.kubernetes.SystemManager` object.

        """
        if self._sm is None:
            url = self._url + "/system"
            self._sm = SystemManager(url=url, gis=self._gis)
        return self._sm

    # ----------------------------------------------------------------------
    @property
    def jobs(self) -> JobManager:
        """
        This resource is a collection of the asynchronous
        :class:`jobs <arcgis.gis.kubernetes.Job>` created in your
        deployment. When operations that support asynchronous executions are
        run with the async option enabled, a new job entry is created that
        can be queried for its current status and messages.
        """
        if self._jobs is None:
            url = self._url + "/jobs"
            self._jobs = JobManager(url=url, gis=self._gis)
        return self._jobs

    # ----------------------------------------------------------------------
    @property
    def license(self) -> LicenseManager:
        """
        Provides access to the :class:`~arcgis.gis.kubernetes.LicenseManager`,
        and its set of tools to access and manage user licenses and
        entitlements.
        """
        if self._license is None:
            url = self._gis._portal.resturl + "portals/self/purchases"
            self._license = LicenseManager(url=url, gis=self._gis)
        return self._license

    # ----------------------------------------------------------------------
    @property
    def category_schema(self):
        """This resource allows for the setting and manipulating of catagory schemas.

        :returns:
            :class:`~arcgis.gis.admin.CategoryManager` object
        """
        if self._category_schema is None:
            from arcgis.gis.admin._catagoryschema import CategoryManager

            self._category_schema = CategoryManager(gis=self._gis)
        return self._category_schema

    # ----------------------------------------------------------------------
    @property
    def _idp(self):
        """
        This resource allows for the setting and configuration of the identity provider
        """
        if self._idp is None:
            from arcgis.gis.admin._idp import IdentityProviderManager

            self._idp = IdentityProviderManager(gis=self._gis)
        return self._idp

    # ----------------------------------------------------------------------
    def scheduled_tasks(
        self,
        item: Optional[Item] = None,
        active: Optional[bool] = None,
        user: Optional[User] = None,
        types: Optional[str] = None,
    ):
        """
        This method allows organization admins to see all scheduled tasks
        on the organization.

        ================  ===============================================================================
        **Parameter**     **Description**
        ----------------  -------------------------------------------------------------------------------
        item              Optional Item. The item to query tasks about.
        ----------------  -------------------------------------------------------------------------------
        active            Optional Bool. Queries tasks based on active status.
        ----------------  -------------------------------------------------------------------------------
        user              Optional User. Search for tasks for a single user.
        ----------------  -------------------------------------------------------------------------------
        types             Optional String. The type of notebook execution for the item:

                          * `ExecuteNotebook`
                          * `UpdateInsightsWorkbook`
        ================  ===============================================================================


        :return: List of :class:`Tasks <arcgis.gis.tasks.Task>`.

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
    def _location_tracking(self):
        """
        The manager for Location Sharing. See :class:`~arcgis.apps.tracker.LocationTrackingManager'.
        """
        return LocationTrackingManager(self._gis)

    # ----------------------------------------------------------------------
    @property
    def social_providers(self):
        """
        Accesses the :class:`~arcgis.gis.admin.SocialProviders` resource to
        allow for the setting and configuration of the social providers
        for the organization.
        """
        if self._sp is None:
            from arcgis.gis.admin._socialproviders import SocialProviders

            self._sp = SocialProviders(gis=self._gis)
        return self._sp

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """
        Accesses the :class:`~arcgis.gis.admin.MetadataManager` which
        provides a set of tools to work with the organization's metadata
        settings.
        """
        if self._metadata is None:
            from arcgis.gis.admin._metadata import MetadataManager

            self._metadata = MetadataManager(gis=self._gis)
        return self._metadata

    # ----------------------------------------------------------------------
    @property
    def organizations(self):
        """Provides access to the :class:`~arcgis.gis.kubernetes.KubeOrganizations`
        object to work with the organization's settings."""
        if self._organizations is None:
            from ._organizations import KubeOrganizations

            url = f"{self._url}/orgs"
            self._organizations = KubeOrganizations(url=url, gis=self._gis)
        return self._organizations

    # ----------------------------------------------------------------------
    @property
    def services(self):
        """Provides access to the Kubernetes
        :class:`~arcgis.gis.kubernetes.ServicesManager` object for the site
        """
        if self._services is None:
            from ._services import ServicesManager

            url = f"{self._url}/services"
            self._services = ServicesManager(url, gis=self._gis)
        return self._services

    # ----------------------------------------------------------------------
    @property
    def services_catalog(self):
        """Provides access to the kubernetes
        :class:`~arcgis.gis.kubernetes._server.KubeServiceDirectory` work with the
        services on the site.
        """
        if self._catalog is None:
            from arcgis.gis.kubernetes._server import KubeServiceDirectory

            url = f"{self._url.replace('/admin', '/rest')}/services"
            self._catalog = KubeServiceDirectory(url, gis=self._gis)
        return self._catalog

    # ----------------------------------------------------------------------
    @property
    def uploads(self):
        """Gets the :class:`~arcgis.gis.kubernetes.Uploads` object to work
        with the site uploads."""
        if self._uploads is None:
            from ._uploads import Uploads

            url = self._url + "/uploads"
            self._uploads = Uploads(url=url, gis=self._con, initialize=True)
        return self._uploads

    # ----------------------------------------------------------------------
    @property
    def security(self) -> "KubeSecurity":
        """
        Gets a :class:`~arcgis.gis.kubernetes.KubeSecurity` object to work
        with the site's security settings

        :return:
            :class:`~arcgis.gis.kubernetes.KubeSecurity` object.
        """
        if self._security is None:
            from arcgis.gis.kubernetes._admin._security import KubeSecurity

            url = self._url + "/security"
            self._security = KubeSecurity(url=url, gis=self._gis)
        return self._security

    # ----------------------------------------------------------------------
    @property
    def collaborations(self):
        """
        The collaborations resource lists all collaborations in which a
        portal participates
        """
        if self._collaborations is None:
            from arcgis.gis.admin._collaboration import CollaborationManager

            self._collaborations = CollaborationManager(gis=self._gis)
        return self._collaborations

    # ----------------------------------------------------------------------
    @property
    def webhooks(self):
        """Provides access to the organiztion's
        :class:`Webhook Manager <arcgis.gis.admin.WebHookManager>`."""
        if self._whm is None and self._gis.version >= [6, 4]:
            from arcgis.gis.admin._wh import WebhookManager

            url = self._gis._portal.resturl + "portals/self/webhooks"
            self._whm = WebhookManager(url=url, gis=self._gis)
        return self._whm
