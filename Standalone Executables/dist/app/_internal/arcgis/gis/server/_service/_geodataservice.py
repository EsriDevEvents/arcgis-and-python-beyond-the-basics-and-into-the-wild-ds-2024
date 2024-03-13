"""
"""
from __future__ import absolute_import
from .._common._base import BaseServer


########################################################################
class GeoData(BaseServer):
    """
    Represents a single geodata service layer
    """

    _url = None
    _con = None
    _json_dict = None
    _replicasResource = None
    _defaultWorkingVersion = None
    _workspaceType = None
    _replicas = None
    _serviceDescription = None
    _versions = None

    # ----------------------------------------------------------------------
    @property
    def replicasResource(self):
        """returns a list of replices"""
        if self._replicasResource is None:
            self._replicasResource = {}
            if isinstance(self.replicas, list):
                for replica in self.replicas:
                    self._replicasResource["replicaName"] = replica.name
                    self._replicasResource["replicaID"] = replica.guid
        return self._replicasResource

    # ----------------------------------------------------------------------
    def unRegisterReplica(self, replicaGUID):
        """unRegisterReplica operation is performed on a Geodata Service
        resource (POST only). This operation unregisters a replica on the
        geodata service. Unregistering a replica is only supported when
        logged in as an admin user. You can provide arguments to the
        unRegisterReplica operation.
        Inputs:
            replicaID - The ID of the replica. The ID of a replica can be
                        found by accessing the Geodata Service Replicas
                        resource."""

        url = self._url + "/unRegisterReplica"
        params = {"f": "json", "replicaID": replicaGUID}
        return self._con.post(path=url, postdata=params)
