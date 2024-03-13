from __future__ import annotations
from urllib.request import HTTPError
from ._base import _BaseKube
from arcgis.gis import GIS
from arcgis.gis._impl._con import Connection
from arcgis._impl.common._mixins import PropertyMap
from typing import Dict, Any, List, Tuple, Optional


###########################################################################
class Overview(_BaseKube):
    """ """

    _con = None
    _gis = None
    _url = None

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS) -> None:
        """initializer"""
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def _init(self, connection: Optional[Connection] = None) -> None:
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json", "resource": "all"}
        try:
            result = connection.get(path=self._url, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except HTTPError as err:
            raise RuntimeError(err)
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ---------------------------------------------------------------------
    def get(self, resource: Optional[str] = None) -> Dict[str, Any]:
        """
        The `get` returns the persisted cache or real-time
        information, such as health or status information, for the overview
        resource types, and is what is called when the Overview page of the
        Enterprise Manager is updated. Whether the information is cached or
        returned in real-time depends on the updateIntervalMin property
        returned by the config resource, which specifies the interval (in
        minutes) from which information for each resource type is pulled
        and cached.

        Resource types that have an updateIntervalMin value of 0 will not
        have their information cached and, instead, will have real-time
        information returned when the resource is called. The update
        interval can me modified through the update operation.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        resource            Optional String.  Specifies the resource type (criticalLogs,
                            dataStores, systemServices, utilityServices) that will have their
                            information returned. Using all as the input value will return
                            information for all resource types.

                            The default is `None`. When `None`, all resources will be returned.
        ===============     ====================================================================

        :return: Dict[str, Any]

        """
        if resource is None:
            resource = "all"
        url = self._url
        params = {"f": "json", "resources": resource}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def config(self) -> Dict[str, Any]:
        """
        Gets/sets a dictionary of resource types that correspond with the
        `Overview` class.  It contains the update interval for each
        property.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        resource            Required Dictionary. A dictionary object containing the `id`, `type`,
                            and `updateIntervalMin` for an overview resource type, returned by
                            the config resource. The accepted values for `updateIntervalMin`
                            (0-60) can be modified to update the interval (in minutes) of which
                            the resource type will have it's information pulled and cached. If
                            set to 0, information for the resource type will not be cached and
                            will, instead, have it's real-time information returned when the
                            overview resource is called. The default values for each resource
                            type are listed below:

                               + criticalLogs: 0
                               + systemServices: 1
                               + utilityServices: 1
                               + dataStores: 2

        ===============     ====================================================================

        :return: Dict[str, Any]

        """
        url = f"{self._url}/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """
        See main ``config`` property docstring
        """
        url = f"{self._url}/config/config"
        params = {"f": "json", "resourceConfigJson": value}
        return self._con.get(url, params)
