from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS
from typing import Dict, Any, Optional, List


class Deployment:
    """
    This represents a single deployment microservice.
    """

    _url = None
    _gis = None

    def __init__(self, url: str, gis: GIS) -> None:
        self._url = url
        self._gis = gis
        self._con = gis._con

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Returns the properties of the `Deployment`

        :return: Dict[str, Any]
        """
        self._con.get(self._url, {"f": "json"})

    def edit(self, props: Dict[str, Any]) -> bool:
        """
        `edit` allows you to edit the scaling (replicas) and resource
        allocation (resources) for a specific microservice within your
        deployment.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        props                  Required Dict[str, Any]. he microservice properties, represented as a dictionary.
        ==================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = f"{self._url}/edit"
        params = {"f": "json", "deploymentJson": props}
        res = self._con.post(url, params)
        return res.get("status", "failed") == "success"

    def refresh(self) -> bool:
        """
        The `refresh` operation can be used to troubleshoot microservices
        and pods that may be unresponsive or not functioning as expected.
        Performing this operation will restart the corresponding pods and
        recreate the microservice.

        :return: Boolean. True if successful else False.
        """
        url = f"{self._url}/refresh"
        params = {"f": "json"}
        res = self._con.post(url, params)
        return res.get("status", "failed") == "success"

    def status(self) -> Dict[str, Any]:
        """
        Returns the status of the current deployment.

        :return: Dict[str, Any]
        """
        url = f"{self._url}/status"
        params = {"f": "json"}
        return self._con.get(url, params)


class DeploymentProperty:
    _url = None
    _gis = None

    def __init__(self, url: str, gis: GIS) -> None:
        self._url = url
        self._gis = gis
        self._con = gis._con

    @property
    def properties(self) -> Dict[str, Any]:
        params = {"f": "json"}
        return self._con.get(self._url, params)

    @property
    def get(self, template_id: str) -> Dict[str, Any]:
        """
        Gets a default template based on an ID

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        template_id            Required String.  The unique ID of the property template.
        ==================     ====================================================================

        :return: Dict[str, Any]
        """
        url = f"{self._url}/{template_id}"
        params = {"f": "json"}
        return self._con.get(url, params)

    def edit(self, template_id: str, props: Dict[str, Any]) -> bool:
        """
        'edit' modifies the default scaling and resource allocation
        properties of a specific microservice within an organization. Once
        the default properties have been updated, all newly published
        microservices that match the type, provider, and mode of the
        default template will have the updated properties assigned to them.
        Preexisting deployments will need to have their properties
        individually updated using the edit operation.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        template_id            Required String.  The unique ID of the property template.
        ------------------     --------------------------------------------------------------------
        props                  Required Dict[Str, Any]. A dictionary representing the default
                               property template that specifies the scaling (min & max) and
                               resource allocations (memoryMin, cpuMin, memoryMax, cpuMax) for a
                               particular microservice type.
        ==================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = f"{self._url}/{template_id}/edit"
        params = {
            "f": "json",
            "propertyJson": props,
        }
        return self._con.post(url, params).get("status", "failed") == "success"


class DeploymentManager(_BaseKube):
    """
    The `DeploymentManager` is a list of all the microservices that are
    used to host or run your services, as well as non-service items such as
    the ArcGIS Enterprise Admin API, Portal Sharing, and ingress controller
    microservices. Each microservice listed here may correspond to one or
    more pods in ArcGIS Enterprise on Kubernetes. This resource provides
    access to the edit operation that sets scaling and resource
    allocations, as well as the find deployments operation that can be
    used to search and filter microservices.
    """

    _dp = None
    _con = None
    _gis = None
    _url = None

    def __init__(
        self,
        url: str,
        gis: GIS = None,
    ) -> None:
        """class initializer"""
        super()
        self._url = url
        self._gis = gis

    # ---------------------------------------------------------------------
    def search(
        self,
        name: Optional[str] = None,
        filter_type: Optional[str] = None,
        filter_id: Optional[str] = None,
        provider: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Deployment]:
        """
        `find` queries and returns a `List[Deployment]` of microservies
        within ArcGIS Enterprise on Kubernetes. The results can be
        fine-tuned by specifying the name, type, ID, provider, and mode of
        a microservice. These filters are options; if no filter is applied,
        all microservices are returned by the operation.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Optional String. The name of the microservice.
        ------------------     --------------------------------------------------------------------
        filter_type            Optional String. The microservice type.

                               Allowed Values: `FeatureServer`, `GeometryServer`, `GPServer`,
                               `GPSyncServer`, `MapServer`, `TileServer`, `System`, `InMemoryStore`,
                               `ObjectStore`, `SpatiotemporalIndexStore`, `QueueServer`,
                               `RelationalStore`.
        ------------------     --------------------------------------------------------------------
        filter_id              Optional String. The microservice ID.
        ------------------     --------------------------------------------------------------------
        provider               Optioal String. The microservice provider. Only microservices
                               related to an ArcGIS service type will have a provider type. A
                               provider type of Undefined is used for non-service related
                               microservices (Admin API, Portal Sharing, ingress controller,
                               etc.).

                               Values: `SDS`, `ArcObjects11`, `DMaps`, `Undefined`, `Postgres`,
                               `Tiles`, `Ignite`, `MinIO`, `Elasticsearch`, `RabbitMQ`
        ------------------     --------------------------------------------------------------------
        mode                   Optional String. The microservice mode. A mode type of Undefined is
                               used when the microservices is system related (Admin API, Portal
                               Sharing, ingress controller, etc.). Only microservices related to
                               an ArcGIS service type use either the Dedicated or Shared value for
                               this parameter.

                               Values: `Shared`, `Dedicated`, `Undefined`, `Primary`, `Standby`,
                               `Coordinator`
        ==================     ====================================================================

        :return: List[Deployment]

        """
        url = f"{self._url}/findDeployment"
        params = {
            "filterName": name,
            "filterType": filter_type,
            "filterId": filter_id,
            "filterProvider": provider,
            "filterMode": mode,
            "f": "json",
        }
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        return [
            Deployment(url + "/%s" % deploy["deploymentId"], gis=self._gis)
            for deploy in self._con.post(url, params).get("filteredDeployments", [])
            if "deploymentId" in deploy
        ]

    def get(self, deployment_id: str) -> Deployment:
        """
        Returns a Single Deployment by Id.

        :return: Deployment
        """
        url = f"{self._url}/{deployment_id}"
        return Deployment(url=url, gis=self._gis)

    @property
    def deployment_properties(self) -> DeploymentProperty:
        """
        Provides administartors with tools to with the deployment property templates.

        :return: DeploymentProperty
        """
        if self._dp is None:
            url = f"{self._url}/properties"
            self._dp = DeploymentProperty(url=url, gis=self._gis)
        return self._dp
