from __future__ import annotations
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
from ._base import BaseMissionServer
from ._logs import LogManager
from ._system import SystemManager
from ._machines import MachineManager
from ._security import SecurityManager
from .api import MissionCatalog


###########################################################################
class MissionServer(BaseMissionServer):
    """
    A Mission Server Instance.
    """

    _url = None
    _gis = None
    _con = None
    _properties = None
    _machinemgr = None
    _securitymgr = None
    _security = None
    _system = None
    _logs = None
    _machine = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis=None):
        self.catalog = MissionCatalog(gis=gis)
        if url.lower().find("/admin") == -1:
            if url.endswith("/"):
                url = url[:-1]
            url += "/admin"
        self._url = url
        super().__init__(url, gis)
        if gis is None:
            from arcgis import env

            gis = env.active_gis
        if gis is None:
            raise ValueError("A GIS could not be obtained.")
        self._gis = gis
        self._con = self._gis._con

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
    def logs(self):
        """
        Provides access to the Mission server's logging system

        :return:
            :class:`~arcgis.gis.mission._logs.LogManager` object

        """
        if self._logs is None:
            url = self._url + "/logs"
            self._logs = LogManager(url=url, gis=self._gis)
        return self._logs

    # ----------------------------------------------------------------------
    @property
    def system(self):
        """
        returns access to the system properties of the ArcGIS Mission Server

        :return:
            :class:`~arcgis.gis.mission._system.SystemManager`

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
        Mission Server

        :return:
            :class:`~arcgis.gis.mission._machines.MachineManager`

        """
        if self._machine is None:
            url = self._url + "/machines"
            self._machine = MachineManager(url=url, gis=self._gis)
        return self._machine

    # ----------------------------------------------------------------------
    @property
    def security(self):
        """
        Provides access to managing the ArcGIS Mission Server's security
        settings.

        :return:
            :class:`~arcgis.gis.mission._security.SecurityManager`

        """
        if self._security is None:
            url = self._url + "/security"
            self._security = SecurityManager(url=url, gis=self._gis)
        return self._security
