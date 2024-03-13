"""
Entry point to working with local enterprise GIS functions
"""
from __future__ import annotations
import json
import tempfile

from datetime import datetime
from typing import Optional
from .._impl._con import Connection
from ...gis import GIS, Item, User
from ._resources import PortalResourceManager
from ._base import BasePortalAdmin
from ...apps.tracker._location_tracking import LocationTrackingManager
from ._dsmgr import DataStoreMetricsManager
from arcgis.auth.tools import LazyLoader

_pd = LazyLoader("pandas")

_utils = LazyLoader("arcgis._impl.common._utils")


########################################################################
class AGOLAdminManager(object):
    """
    This is the root resource for administering your online GIS. Starting from
    this root, all of the GIS's environment is organized into a
    hierarchy of resources and operations.

    Parameter:
    :param gis: GIS object containing Administrative credentials
    :param ux: the UX object (optional)
    :param metadata: the metadata manager object (optional)
    :param collaborations: the CollaborationManager object (optional)
    """

    _con = None
    _gis = None
    _ux = None
    _idp = None
    _pp = None
    _credits = None
    _metadata = None
    _collaborations = None
    _ur = None
    _sp = None
    _license = None
    _usage = None
    _category_schema = None
    _certificates = None
    _servers = None
    _dmm = None

    # ----------------------------------------------------------------------
    def __init__(self, gis, ux=None, metadata=None, collaborations=None):
        """initializer"""
        self._gis = gis
        self._con = gis._con
        self._ux = ux
        self._collaborations = collaborations
        self._metadata = metadata
        self.resources = PortalResourceManager(gis=self._gis)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (
            type(self).__name__,
            self._gis._portal.resturl,
        )

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (
            type(self).__name__,
            self._gis._portal.resturl,
        )

    # ----------------------------------------------------------------------
    @property
    def ux(self):
        """returns a UX/UI manager

        :return:
            :class:`~arcgis.gis.admin.UX` object

        """
        if self._ux is None:
            from ._ux import UX

            self._ux = UX(gis=self._gis)
        return self._ux

    # ----------------------------------------------------------------------
    @property
    def datastore_metrics(self) -> DataStoreMetricsManager:
        """
        Provides administrators information about the datastore on ArcGIS Online.

         :return:
            :class:`~arcgis.gis.admin._dsmgr.DataStoreMetricsManager` object
        """
        if self._dmm is None:
            self._dmm = DataStoreMetricsManager(gis=self._gis)
        return self._dmm

    # ----------------------------------------------------------------------
    @property
    def _user_experience_program(self):
        """
        ArcGIS Online works continuously to improve our products and one of
        the best ways to find out what needs improvement is through
        customer feedback. The Esri User Experience Improvement program
        (EUEI) allows your organization to contribute to the design and
        development of ArcGIS Online. The program collects information
        about the usage of ArcGIS Online including hardware and browser
        characteristics, without interrupting work. The program is
        completely optional and anonymous; none of the information
        collected is used to identify or contact members of your
        organization.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required boolean. True means that the organization will be enrolled
                            in the Esri User Experience Improvement Program. False means the
                            organization will not be part of the program.
        ===============     ====================================================================
        """
        return self._gis.properties["eueiEnabled"]

    # ----------------------------------------------------------------------
    @_user_experience_program.setter
    def _user_experience_program(self, value: bool):
        """
        See main ``_user_experience_program`` property docstring.
        """
        if value != self._user_experience_program:
            self._gis.update_properties(
                {"clearEmptyFields": True, "eueiEnabled": value}
            )
            self._gis._get_properties(True)

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
        This resource allows for the setting and manipulating of category
        schemas.

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

        The manager for Location Tracking. See :class:`~arcgis.apps.tracker.LocationTrackingManager`

        :return:
            :class:`~arcgis.apps.tracker.LocationTrackingManager` object

        """
        return LocationTrackingManager(self._gis)

    @property
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    @property
    def credits(self):
        """
        Manages the credits on a ArcGIS Online

        :return:
            :class:`~arcgis.gis.admin.CreditManager` object

        """
        if self._credits is None:
            from ._creditmanagement import CreditManager

            self._credits = CreditManager(gis=self._gis)
        return self._credits

    # ----------------------------------------------------------------------
    @property
    def metadata(self):
        """
        resources to work with metadata on GIS

        :return:
            :class:`~arcgis.gis.admin.MetadataManager` object

        """
        if self._metadata is None:
            from ._metadata import MetadataManager

            self._metadata = MetadataManager(gis=self._gis)
        return self._metadata

    # ----------------------------------------------------------------------
    @property
    def password_policy(self):
        """tools to manage a Site's password policy"""
        if self._pp is None:
            from ._security import PasswordPolicy

            url = "%s/portals/self/securityPolicy" % (self._gis._portal.resturl)
            self._pp = PasswordPolicy(url=url, gis=self._gis)
        return self._pp

    # ----------------------------------------------------------------------
    @property
    def usage_reports(self):
        """
        provides access to the usage reports of the ArcGIS Online organization

        :return:
            :class:`~arcgis.gis.admin.AGOLUsageReports` object

        """
        if self._ur is None:
            from ._usage import AGOLUsageReports

            url = "%sportals/%s/usage" % (
                self._gis._portal.resturl,
                self._gis.properties.id,
            )
            self._ur = AGOLUsageReports(url=url, gis=self._gis)
        return self._ur

    # ----------------------------------------------------------------------
    @property
    def license(self):
        """
        provides a set of tools to access and manage user licenses and
        entitlements.
        """
        if self._license is None:
            from ._license import LicenseManager

            url = self._gis._portal.resturl + "portals/self/purchases"
            self._license = LicenseManager(url=url, gis=self._gis)
        return self._license

    # ----------------------------------------------------------------------
    @property
    def urls(self):
        """
        returns the URLs to the Hosting and Tile Server for ArcGIS Online
        """
        res = self._gis._con.get(
            path="%s/portals/%s/urls"
            % (self._gis._portal.resturl, self._gis.properties.id),
            params={"f": "json"},
        )
        return res

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
    def history(
        self,
        start_date: datetime,
        to_date: Optional[datetime] = None,
        num: int = 100,
        all_events: bool = True,
        event_ids: Optional[str] = None,
        event_types: Optional[str] = None,
        actors: Optional[str] = None,
        owners: Optional[str] = None,
        actions: Optional[str] = None,
        ips: Optional[str] = None,
        sort_order: str = "asc",
        data_format: str = "csv",
        save_folder: Optional[str] = None,
    ):
        """
        Returns a CSV file or Pandas's DataFrame containing the login history from a start_date to the present.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        start_date        Required datetime.datetime object. The beginning date to start with.
        ----------------  -------------------------------------------------------------------------------
        to_date           Optional datetime.datetime object. The ending date.  If not provided, the query
                          will attempt to obtain all records till the current date.
        ----------------  -------------------------------------------------------------------------------
        num               Optional Integer. The maximum number of records to return.  The maximum value
                          is 10,000 set by the ArcGIS REST API.  If the value of -1 is provided it will
                          attempt to get all records for the date range.  The default is **100**.
        ----------------  -------------------------------------------------------------------------------
        all_events        Optional Boolean. If `True`, all types of events are included.  If `False`, only
                          actions targeted by the organization are included.  When exporting as `csv` this
                          parameter is `True`.
        ----------------  -------------------------------------------------------------------------------
        event_id          Optional String. Filter events by specific target user name or target ID in a batch result set.
                          It can be the ID of an item, a group, a role, a collaboration, an identity
                          provider, and so on.
        ----------------  -------------------------------------------------------------------------------
        event_types       Optional String.  Filter events by a comma-separated list of target types in a
                          batch result set.

                          Values: a (organization), c (collaboration), cp (collaboration participate),
                                  cpg (collaboration participate group), cw (collaboration workspace),
                                  cwp (collaboration workspace participate), g (group), i (item),
                                  idp (identity provider), inv (invitation), r (role), u (user)
        ----------------  -------------------------------------------------------------------------------
        actors            Optional String. Comma seperated list of usernames.
        ----------------  -------------------------------------------------------------------------------
        owners            Optional String. Filter events by a comma-separated list of user names who own
                          the action targets in a batch result set.
        ----------------  -------------------------------------------------------------------------------
        actions           Optional String. Comma seperated list of actions to query for.

                          Values: `add`, `addusers`, `create`, `delete`, `removeusers`, `share`, `unshare`,
                          `update`, `failedlogin`, `login`, and `updateUsers`.
        ----------------  -------------------------------------------------------------------------------
        ips               Optional String. Filter events by a comma-separated list of IP addresses in a batch result set.
        ----------------  -------------------------------------------------------------------------------
        sort_order        Optional String.  Describes whether the results return in ascending or
                          descending chronological order. The default is ascending.

                          Values: `asc` or `desc`
        ----------------  -------------------------------------------------------------------------------
        data_format       Optional String.  The way the data is returned to the user.  The response can
                          be a `df`, `csv`, or 'raw'.  'df' returns a DataFrame, 'csv' returns a comma
                          seperated file, and 'raw' returns the JSON string as a dictionary.

                          Values: `df`, `csv`, 'raw'
        ----------------  -------------------------------------------------------------------------------
        save_folder       Optional String. The save location of the CSV file.
        ================  ===============================================================================

        :return: string or pd.DataFrame or dict

        """
        _date_handler = _utils._date_handler
        if save_folder is None:
            save_folder = tempfile.gettempdir()
        if num == 0:
            raise ValueError("`num` cannot be zero.")
        url = "{url}portals/self/history".format(url=self._gis._portal.resturl)
        params = {
            "f": data_format,
            "num": num,
            #'start' : "",
            "all": json.dumps(all_events),
            "id": event_ids,
            "types": event_types,
            "actors": actors,
            "owners": owners,
            "actions": actions,
            "fromDate": json.dumps(start_date, default=_date_handler),
            "sortOrder": sort_order,
            "ips": ips,
        }

        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        if to_date:
            params["toDate"] = json.dumps(to_date, default=_date_handler)

        if data_format == "csv":
            params["f"] = "csv"
            params["num"] = 10000
            params["all"] = "true"
            return self._gis._con.post(
                url,
                params,
                file_name="history.csv",
                out_folder=save_folder,
                try_json=False,
            )
        elif data_format in ["df"]:
            if event_ids or event_types or actors or owners or actions:
                params["all"] = "true"
            params["f"] = "json"
            data = []

            res = self._gis._con.get(url, params)
            data.extend(res["items"])
            while len(res["items"]) > 0 and "nextKey" in res:
                params["start"] = res["nextKey"]
                res = self._gis._con.post(url, params)
                data.extend(res["items"])
                if num > 0 and len(data) >= num:
                    data = data[:num]
                    break
            return _pd.DataFrame(data)
        elif data_format in ["raw", "json"]:
            if event_ids or event_types or actors or owners or actions:
                params["all"] = "true"
            params["f"] = "json"
            data = []

            res = self._gis._con.get(url, params)
            data.extend(res["items"])
            while len(res["items"]) > 0 and "nextKey" in res:
                params["start"] = res["nextKey"]
                res = self._gis._con.get(url, params)
                new_data = res["items"]
                if len(new_data) == 0:
                    break
                data.extend(new_data)
                if num > 0 and len(data) >= num:
                    data = data[:num]
                    break
            return data

    # ----------------------------------------------------------------------
    @property
    def certificates(self):
        """
        Provides access to managing the organization's certificates.

        :return:
            :class:`~arcgis.gis._impl.CertificateManager` object

        """
        if self._certificates is None:
            from .._impl import CertificateManager

            self._certificates = CertificateManager(gis=self._gis)
        return self._certificates

    # ----------------------------------------------------------------------
    @property
    def servers(self):
        """
        Provides access to managing the services hosted on ArcGIS Online

        :return:
            :class:`~arcgis.gis.agoserver.AGOLServersManager`

        """
        if self._servers is None:
            from arcgis.gis.agoserver import AGOLServersManager

            self._servers = AGOLServersManager(gis=self._gis)
        return self._servers
