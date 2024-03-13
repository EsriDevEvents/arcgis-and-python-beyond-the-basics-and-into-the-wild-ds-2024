"""
This is the ArcGIS Notebook Server API Framework
"""

import os
import copy
import warnings
from urllib.parse import urlparse

from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap

from ._logs import LogManager
from ._system import SystemManager
from ._security import SecurityManager
from ._machines import MachineManager
from ._nbm import NotebookManager


########################################################################
class NotebookServer(object):
    """
    Provides access to the ArcGIS Notebook Server administration API.
    """

    _da = None
    _gis = None
    _url = None
    _properties = None
    _logs = None
    _system = None
    _machine = None
    _notebook = None
    _security = None
    _services = None
    _version = None
    _sitemanager = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        if url.lower().endswith("/admin") == False:
            url += "/admin"
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
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
        return "< NotebookServer @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< NotebookServer @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """Properties of the object"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def data_access(self) -> "NotebookDataAccess":
        """Provides access to managing files stored on notebook server.

        :return:
            :class:`~arcgis.gis.nb._dataaccess.NotebookDataAccess` object

        """
        if self._da is None:
            from ._dataaccess import NotebookDataAccess

            url = self._url + "/dataaccess"
            self._da = NotebookDataAccess(url, self._gis)
        return self._da

    # ----------------------------------------------------------------------
    @property
    def version(self):
        """
        Returns the notebook server version

        :return: List
        """
        if self._version is None:
            self._version = [int(i) for i in self.properties.version.split(".")]
        return self._version

    # ----------------------------------------------------------------------
    @property
    def site(self):
        """
        Provides access to the notebook server's site management operations

        :return: :class:`~arcgis.gis.nb.SiteManager`
        """
        if self._sitemanager is None:
            from ._site import SiteManager

            self._sitemanager = SiteManager(url=self._url, notebook=self, gis=self._gis)
        return self._sitemanager

    # ----------------------------------------------------------------------
    @property
    def info(self):
        """
        Returns information about the server site itself

        :return: PropertyMap

        """
        url = self._url + "/info"
        params = {"f": "json"}
        res = self._gis._con.get(url, params)
        return PropertyMap(res)

    # ----------------------------------------------------------------------
    @property
    def health_check(self) -> bool:
        """

        The `health_check` verifies that your ArcGIS Notebook Server site
        has been created, and that its Docker environment has been
        correctly configured.

        **This is only avaible if the site can be accessed around the web adapter**

        :return: Boolean

        """
        netloc = urlparse(self._url).netloc
        if netloc.find(":11443") > -1:
            url = "https://{base}/arcgis/rest/info/healthcheck".format(base=netloc)
        else:
            url = "https://{base}:11443/arcgis/rest/info/healthcheck".format(
                base=netloc
            )
        params = {"f": "json"}
        verify_original = copy.deepcopy(self._gis._con._verify_cert)
        if self._gis._con._verify_cert:
            self._gis._con._verify_cert = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._gis._con.get(url, params)
        self._gis._con._verify_cert = verify_original
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    @property
    def logs(self):
        """
        Provides access to the notebook server's logging system

        :return:
            :class:`~arcgis.gis.nb.LogManager`

        """
        if self._logs is None:
            url = self._url + "/logs"
            self._logs = LogManager(url=url, gis=self._gis)
        return self._logs

    # ----------------------------------------------------------------------
    @property
    def system(self):
        """
        returns access to the system properties of the ArcGIS Notebook Server

        :return:
            :class:`~arcgis.gis.nb.SystemManager`

        """
        if self._system is None:
            url = self._url + "/system"
            self._system = SystemManager(url=url, gis=self._gis)
        return self._system

    # ----------------------------------------------------------------------
    @property
    def machine(self):
        """
        Provides access to managing the registered machines with ArcGIS
        Notebook Server

        :return: :class:`~arcgis.gis.nb.MachineManager`

        """
        if self._machine is None:
            url = self._url + "/machines"
            self._machine = MachineManager(url=url, gis=self._gis)
        return self._machine

    # ----------------------------------------------------------------------
    @property
    def security(self):
        """
        Provides access to managing the ArcGIS Notebook Server's security
        settings.

        :return: :class:`~arcgis.gis.nb.SecurityManager`

        """
        if self._security is None:
            url = self._url + "/security"
            self._security = SecurityManager(url=url, gis=self._gis)
        return self._security

    # ----------------------------------------------------------------------
    @property
    def notebooks(self):
        """
        Provides access to managing the ArcGIS Notebook Server's
        Notebooks

        :return: :class:`~arcgis.gis.nb.NotebookManager`
        """
        if self._notebook is None:
            url = self._url + "/notebooks"
            self._notebook = NotebookManager(url=url, gis=self._gis, nbs=self)
        return self._notebook

    # ----------------------------------------------------------------------
    @property
    def url(self):
        """The URL of the notebook server."""
        return self._url

    # ----------------------------------------------------------------------
    @property
    def services(self):
        """
        Provices access to managing notebook created geoprocessing tools


        :return:
            :class:`~arcgis.gis.nb._services.NBServicesManager`

        """
        if self._services is None:
            from arcgis.gis.nb._services import NBServicesManager

            url = self._url + "/services"
            self._services = NBServicesManager(url, self._gis, nbs=self)
        return self._services
