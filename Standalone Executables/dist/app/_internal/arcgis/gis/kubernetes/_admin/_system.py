from __future__ import annotations
import json
from arcgis.gis.kubernetes._admin._base import _BaseKube
from ._deployment import DeploymentManager
from ._upgrades import UpgradeManager
from ._recovery import RecoveryManager
from ._content import LanguageManager, ExternalContentManager
from ._architecture import ArchitectureManager
from ._tasks import TaskManager
from ._adaptors import WebAdaptorManager
from ._license import LicenseManager
from typing import List, Dict, Any, Tuple, Optional


class Server(_BaseKube):
    """Represents a single kubernetes service"""

    # ----------------------------------------------------------------------
    def update(self, server_json):
        """
        Updates the server JSON

        :return: Boolean

        """
        url = f"{self._url}/edit"
        params = {"f": "json", "serverJson": server_json}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """returns the status of the server"""
        url = f"{self._url}/status"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def stop(self):
        """returns the status of the server"""
        url = f"{self._url}/stop"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def start(self):
        """returns the status of the server"""
        url = f"{self._url}/start"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


###########################################################################
class ServerDefaults(_BaseKube):
    """Represents the server default values"""

    @property
    def properties(self):
        url = self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    @properties.setter
    def properties(self, value):
        if self.properties != value:
            url = self._url + "/edit"
            params = {"f": "json", "propertyJson": value}
            self._con.post(url, params)


###########################################################################
class ServerManager(_BaseKube):
    """Manages the Registered Servers"""

    _gis = None
    _con = None
    _properties = None

    # ----------------------------------------------------------------------
    @property
    def list(self):
        """
        Allows for the modification of Servers

        :return: List of Server objects

        """
        servers = []
        if "servers" in self.properties:
            for server in self.properties.servers:
                url = f"{self._url}/{server.id}"
                servers.append(Server(url, self._gis))
        return servers

    # ----------------------------------------------------------------------
    @property
    def defaults(self):
        """
        Returns the default properties for each server type.

        :return: list of `ServerDefault`

        """
        d = []
        url = f"{self._url}/properties"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "properties" in res:
            for i in res["properties"]:
                purl = f"{url}/{i['id']}"
                d.append(ServerDefaults(url=purl, gis=self._gis))
        return d


###########################################################################
class Indexer(_BaseKube):
    """
    This resource contains connection information to the default indexing service.
    """

    def reconfigure(self) -> bool:
        """
        This operation recreates the index service metadata, schema, and data in the event it becomes corrupted.

        :returns: Boolean
        """
        params = {"f": "json"}
        url = f"{self._url}/reconfigure"
        res = self._con.post(url, params)
        return res.get("status", "failed") == "success"

    @property
    def status(self):
        """
        `status` allows you to view the status of the indexing service. You
        can view the number of users, groups, and search items in both the
        database (store) and the index. If the database and index do not
        match, indexing is either in progress or there is a problem with
        the index. It is recommended that you reindex to correct any
        issues. If indexing is in progress, you can monitor the status by
        refreshing the page.

        :return: dict

        """
        params = {"f": "json"}
        url = f"{self._url}/status"
        return self._con.get(url, params)

    def reindex(self, mode, includes=None):
        """
        The operation allows you to generate or update the indexes for content, such as users, groups, and items stored in the database store.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        mode                Required String. The mode in which the indexer should run.
                            Values: USER_MODE, GROUP_MODE, SEARCH_MODE, or FULL_MODE
        ---------------     --------------------------------------------------------------------
        includes            Optional String. A comma separated list of elements to include in
                            the index. This is useful if you want to only index certain items
                            or user accounts.
        ===============     ====================================================================

        :return: Boolean

        """
        url = f"{self._url}/reindex"
        params = {"f": "json", "mode": mode, "includes": includes}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


class Container:
    """
    A single representation of a registered container.
    """

    _properties = None
    _url = None
    _con = None
    _gis = None

    def __init__(self, url: str, gis: "GIS"):
        self._url = url
        self._gis = gis
        self._con = gis._con

    def properties(self):
        """returns the properties of the endpoint"""
        return self._con.get(self._url, {"f": "json"})

    def edit(self, value: dict) -> dict:
        """
        Allows certain container registry properties to be updated after your organization has been configured.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required dict. A dictionary of registry properties.
        ===============     ====================================================================



        :returns: bool
        """
        url = f"{self._url}/edit"
        params = {"f": "json", "containerRegistryJson": value}
        return self._con.post(url, params).get("status", "failed") == "success"


###########################################################################
class SystemManager(_BaseKube):
    """
    The system resource is a collection of system-wide resources for a
    deployment, such as the configuration store and deployment-wide
    security.
    """

    _recovery = None
    _indexer = None
    _sm = None
    _deployments = None
    _upgrades = None
    _license = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, initialize=False):
        """Constructor


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object..
        ==================     ====================================================================

        """
        super(SystemManager, self).__init__(gis=gis, url=url)
        self._url = url
        self._gis = gis
        self._con = gis._con
        if initialize:
            self._init(gis._con)

    # ----------------------------------------------------------------------
    @property
    def containers(self) -> list:
        """
        Returns the registered containers.

        :returns: list
        """
        url = f"{self._url}/containerregistries"
        params = {"f": "json"}
        return [
            Container(url=f"{url}/{container['id']}", gis=self._gis)
            for container in self._con.get(url, params).get("containerRegistries", [])
        ]

    # ----------------------------------------------------------------------
    @property
    def deployments(self) -> DeploymentManager:
        """Manages the deployment settings for enterprise"""
        url = f"{self._url}/deployments"
        if self._deployments is None:
            self._deployments = DeploymentManager(url=url, gis=self._gis)
        return self._deployments

    # ----------------------------------------------------------------------
    @property
    def upgrades(self) -> UpgradeManager:
        """
        The upgrades resource provides access to child operations and
        resources that are used to manage the release and patch updates that
        can be applied to an ArcGIS Enterprise on Kubernetes deployment.
        During the upgrade process, this resource returns detailed,
        real-time messages regarding the status and progress of the
        upgrade. While an upgrade is in progress, child operations and
        resources for this resource will be inaccessible. Once completed,
        all child endpoints will be operational, and the JSON view of the
        resource will contain a log of the job messages and the completion
        status.
        """
        url = f"{self._url}/upgrades"
        if self._upgrades is None:
            self._upgrades = UpgradeManager(url=url, gis=self._gis)
        return self._upgrades

    # ----------------------------------------------------------------------
    @property
    def recovery(self) -> RecoveryManager:
        """
        This resource allows an administrator the ability to manage
        disaster recovery settings.

        :return: RecoveryManager
        """
        if self._recovery is None:
            url = f"{self._url}/disasterrecovery"
            self._recovery = RecoveryManager(url=url, gis=self._gis)
        return self._recovery

    # ----------------------------------------------------------------------
    @property
    def web_adaptors(self) -> WebAdaptorManager:
        """
        The webadaptors resource lists the ArcGIS Enterprise on Kubernetes Web Adaptor configured your deployment. The web adaptor can be configured using the config operation.
        """
        url = f"{self._url}/webadaptors"
        return WebAdaptorManager(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def licenses(self) -> List[Dict[str, Any]]:
        """
        The licenses resource lists the current license level of ArcGIS Server and all authorized extensions.

        :return: List[Dict[str, Any]]
        """
        url = f"{self._url}/licenses"
        return LicenseManager(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def external_content(self) -> ExternalContentManager:
        """ """
        return ExternalContentManager(url=f"{self._url}/content", gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def language(self) -> LanguageManager:
        """
        The content resource provides access to the languages resource.
        The languages resource provides a list of current languages for an
        organization.

        :return: LanguageManager

        """
        return LanguageManager(url=f"{self._url}/content", gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def tasks(self) -> TaskManager:
        """
        This resource returns a list of tasks (CleanGPJobs, BackupRetentionCleaner, CreateBackup) that exist within your deployment.
        """
        url = f"{self._url}/tasks"
        return TaskManager(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def architecture_profiles(self) -> ArchitectureManager:
        """
        This resource returns the architecture profile that is set when an organization is configured and provides access to all three architecture
        profile resources: development, standard-availability, and enhanced-availability.
        """
        url = f"{self._url}/architectureprofiles"
        return ArchitectureManager(url=url, gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Gets/Sets the system properties resource list system properties
        that have been modified to control the portal's environment.

        :return: dict
        """
        url = f"{self._url}/properties"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, value):
        """
        Gets/Sets the system properties resource list system properties
        that have been modified to control the portal's environment.

        :return: dict
        """
        url = f"{self._url}/properties/update"
        params = {"f": "json", "properties": value}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def indexer(self):
        """
        Allows user to manage the site's indexer

        :return: `Indexer`
        """
        if self._indexer is None:
            url = f"{self._url}/indexer"
            self._indexer = Indexer(url=url, gis=self._gis)
        return self._indexer

    # ----------------------------------------------------------------------
    @property
    def servers(self) -> ServerManager:
        """Returns a manager to work with ArcGIS Servers registerd with Kubernetes"""
        if self._sm is None:
            self._sm = ServerManager(url=f"{self._url}/servers", gis=self._gis)
        return self._sm
