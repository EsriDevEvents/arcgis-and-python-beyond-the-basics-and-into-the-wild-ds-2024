"""
This resource is a collection of all the clusters created within your
site. The Create Cluster operation lets you define a new cluster
configuration.
"""
from __future__ import absolute_import
from __future__ import print_function
import json
from .._common import BaseServer
from .parameters import ClusterProtocol
from arcgis.gis import GIS
from typing import Optional


########################################################################
class Clusters(BaseServer):
    """
    This resource is a collection of all the clusters created within your
    site. The Create Cluster operation lets you define a new cluster
    configuration.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The administration URL for the ArcGIS Server.
    ---------------     --------------------------------------------------------------------
    gis                 Required Server object. Connection object.
    ---------------     --------------------------------------------------------------------
    initialize          Optional boolean. If true, information loaded at object
    ===============     ====================================================================

    """

    _con = None
    _json_dict = None
    _json = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor"""
        super(Clusters, self).__init__(gis=gis, url=url)
        self._con = gis
        self._url = url
        if url.lower().endswith("/clusters"):
            self._url = url
        else:
            self._url = url + "/clusters"
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def create_cluster(
        self,
        cluster_name: str,
        machine_names: Optional[str] = None,
        port: Optional[str] = None,
    ) -> dict:
        """
        Creating a new cluster involves defining a clustering protocol that
        will be shared by all server machines participating in the cluster.
        All server machines that are added to the cluster must be
        registered with the site. The clustering protocol and the initial
        list of server machines are optional. In this case, the server
        picks the default clustering protocol and selects the port numbers
        such that they do not conflict with other configured ports on the
        server machine. Once a cluster has been created you can add more
        machines (to increase the compute power) or remove them (to reduce
        the compute power) dynamically.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cluster_name        Require string. The name of the cluster. This must be a unique name
                            within a site
        ---------------     --------------------------------------------------------------------
        machine_names       Optional string. An optional comma-separated list of server machines
                            to be added to this cluster.
        ---------------     --------------------------------------------------------------------
        port                Optional string. A TCP port number that will be used by all the
                            server machines to communicate with each other when using the TCP
                            clustering protocol. This is the default clustering protocol. If
                            this parameter is missing, a suitable default will be used.
        ===============     ====================================================================


        :return: dict

        """
        if port is None:
            port = ""
        if machine_names is None:
            machine_names = ""
        url = self._url + "/create"
        params = {
            "f": "json",
            "clusterName": cluster_name,
            "machineNames": machine_names,
            "tcpClusterPort": port,
        }
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def get_machines(self) -> dict:
        """
        This operation lists all the server machines that don't participate
        in any cluster and are available to be added to a cluster.
        The list would be empty if all registered server machines already
        participate in some cluster.
        """
        url = self._url + "/getAvailableMachines"
        params = {"f": "json"}
        return self._con.get(path=url, postdata=params)


########################################################################
class Cluster(BaseServer):
    """
    A Cluster is a group of server machines that host a collection of GIS
    services. Grouping server machines into a cluster allows you to treat
    them as a single unit to which you can publish GIS services.A cluster
    with more than one server machine provides a level of fault tolerance
    to the services. At the same time, having more than one machine
    increases the computing power of your cluster, hence increasing the
    overall throughput.
    A cluster is dynamic with respect to the list of server machines. New
    server machines can be added to increase computing power without
    affecting the already running GIS services. You can also remove
    machines from a cluster and re-assign them to another cluster.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    url                 Required string. The administration URL for the ArcGIS Server.
    ---------------     --------------------------------------------------------------------
    gis                 Required Server object. Connection object.
    ---------------     --------------------------------------------------------------------
    initialize          Optional boolean. If true, information loaded at object
    ===============     ====================================================================


    """

    _con = None
    _json_dict = None
    _json = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor"""
        super(Cluster, self).__init__(gis=gis, url=url)
        self._con = gis
        self._url = url
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    @property
    def clusters(self) -> list:
        """returns the cluster object for each server"""
        if "clusters" in self.properties:
            Cs = []
            for c in self._clusters:
                url = self._url + "/%s" % c["clusterName"]
                Cs.append(Cluster(url=url, gis=self._con, initialize=True))
            return Cs
        return []

    # ----------------------------------------------------------------------
    def start(self) -> bool:
        """
        Starts the cluster.  Starting a cluster involves starting all the
        server machines within the cluster and the GIS services that are
        deployed to the cluster. This operation attempts to start all the
        server machines. If one or more of them cannot be reached, this
        operation reports an error.
        """
        params = {"f": "json"}
        url = self._url + "/start"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def stop(self) -> bool:
        """
        Stops a cluster. This also stops all the GIS services that are
        hosted on the cluster. This operation attempts to stop all the
        server machines within the cluster. If one or more machines cannot
        be reached, then this operation reports an error.
        """
        params = {"f": "json"}
        url = self._url + "/stop"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes the cluster configuration. All the server machines in the
        cluster will be stopped and returned to the pool of registered
        machines. The GIS services that were deployed on the cluster are
        also stopped. Deleting a cluster does not delete your GIS services.
        """
        params = {"f": "json"}
        url = self._url + "/delete"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["statis"] == "success"
        return res

    # ----------------------------------------------------------------------
    def cluster_services(self) -> dict:
        """
        This resource lists all the services that are currently deployed to
        the cluster (of machines). A service deployed to a cluster runs on
        every server machine that is participating in the cluster.

        This resource was added at ArcGIS 10.1 Service Pack 1.
        """
        params = {"f": "json"}
        url = self._url + "/services"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def cluster_machines(self) -> dict:
        """
        This resource lists all the server machines that are currently
        participating in the cluster. Each server machine listing is
        accompanied by its status indicating whether the server machine is
        running or stopped.
        The list of server machines participating in a cluster is dynamic
        as machines can be added or removed.
        """
        url = self._url + "/machines"
        params = {"f": "json"}
        return self._con.get(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def add_machines(self, names: str) -> dict:
        """
        Adds new server machines to the cluster. The server machines need
        to be registered with the site prior to this operation. When a
        server machine is added to the cluster, it pulls all the GIS
        services that were deployed to cluster and prepares to run them.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        names               Required string. A comma-separated list of machine names. The
                            machines must be registered prior to completing this operation.
        ===============     ====================================================================

        :return: dict

        """
        url = self._url + "/machines/add"
        params = {"f": "json", "machineNames": names}
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def remove_machines(self, names: str) -> dict:
        """
        Removes server machines from the cluster. The server machines are
        returned back to the pool of registered server machines.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        names               Required string. A comma-separated list of machine names. The
                            machines must be registered prior to completing this operation.
        ===============     ====================================================================

        :return: dict
        """
        url = self._url + "/machines/remove"
        params = {"f": "json", "machineNames": names}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def edit_protocol(self, cpo: ClusterProtocol) -> dict:
        """
        Updates the Cluster Protocol. This will cause the cluster to be
        restarted with updated protocol configuration.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        cpo                 Required ClusterProtocal object (CPO). The CPO is a configuration
                            object used to assist users in configuring protocols on ArcGIS Server.
        ===============     ====================================================================

        :return: dict


        """
        if isinstance(cpo, ClusterProtocol):
            value = str(cpo.value["tcpClusterPort"])
        elif isinstance(cpo, dict):
            value = json.dumps(cpo)
        else:
            raise AttributeError("Invalid Input, must be a ClusterProtocal Object")
        url = self._url + "/editProtocol"
        params = {"f": "json", "tcpClusterPort": value}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res
