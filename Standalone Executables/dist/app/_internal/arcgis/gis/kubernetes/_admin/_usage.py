from __future__ import annotations
from arcgis.gis import GIS
from arcgis._impl.common._isd import InsensitiveDict
from typing import Dict, Any, Union, Optional


class UsageStatistics:
    """
    Provides access to the metrics viewer and metrics API tools. As an
    administrator, you can use these tools to monitor GIS service usage in
    your organization for feature services, map services, tiled map
    services, geocode services, and geometry services. Information that
    can be gathered from service usage statistics include:

    - Historical service usage data
    - Peak and off-peak periods of service usage
    - Slowdown in response times or throughput



    """

    _gis = None
    _url = None
    _properties = None

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS) -> None:
        """initializer"""
        self._url = url
        self._gis = gis

    # ---------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """returns the properties of the resource"""
        if self._properties is None:
            self._properties = InsensitiveDict(
                self._gis._con.get(self._url, {"f": "json"})
            )
        return self._properties

    # ---------------------------------------------------------------------
    def update_credentials(self, resource: str, username: str, password: str) -> bool:
        """
        Updates the credentials for the metrics viewer and metrics API.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        resource            Required String.  Specifies whether the updated credentials will be
                            applied to the metrics viewer (`grafana`) or the metrics
                            API (`prometheus`).

                            Values: `grafana` or `prometheus`
        ---------------     --------------------------------------------------------------------
        username            Required String. The new username for the specified metrics resource.
        ---------------     --------------------------------------------------------------------
        password            Required String. The new password for the metrics resource. The new
                            password must be a minimum of eight characters and must contain at
                            least one letter (A-Z, a-z), one number (0-9), and one special
                            character.
        ===============     ====================================================================

        :return: boolean
        """
        url = f"{self._url}/update/credentials"
        try:
            params = {
                "f": "json",
                "resource": resource,
                "username": username,
                "password": password,
            }
            resp = self._gis._con.post(url, params)
            return resp.get("status", "failed") == "success"
        except Exception as e:
            print(e)
            raise
        return False
