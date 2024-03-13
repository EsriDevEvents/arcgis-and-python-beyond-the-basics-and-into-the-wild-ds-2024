from __future__ import annotations
from arcgis.gis import GIS
from typing import TypeVar

T = TypeVar("T")
V = TypeVar("V")


class InstancePreference:
    """
    Provides information about the available instances for notebook containers.

    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The base url for the InstancePreference endpoints.
    ----------------  -------------------------------------------------------------------------------
    gis               Required GIS. The ArcGIS Online connection object.
    ================  ===============================================================================

    """

    def __init__(self, url: str, gis: GIS):
        """init"""
        self._url = url
        self._gis = gis

    @property
    def available(self) -> dict[T, V]:
        """
        Returns Information on the available notebook instances.

        :returns: dict[T,V]
        """
        url = f"{self._url}/availableInstanceTypes"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    @property
    def instances(self) -> dict[T, V]:
        """
        Returns a dictionary containing the available instance types for the system

        :return: dict[T,V]
        """
        url = f"{self._url}"
        params = {"f": "json"}
        return self._gis._con.get(url, params)
