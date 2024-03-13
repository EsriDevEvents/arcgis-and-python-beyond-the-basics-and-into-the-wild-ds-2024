from __future__ import annotations
from arcgis.gis import GIS
from typing import TypeVar

T = TypeVar("T")
V = TypeVar("V")


class RuntimeManager:
    """
    Provides information about the Runtimes in the Notebook Server

    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The base url for the RuntimeManager endpoints.
    ----------------  -------------------------------------------------------------------------------
    gis               Required GIS. The ArcGIS Online connection object.
    ================  ===============================================================================

    """

    def __init__(self, url: str, gis: GIS):
        """initializer"""
        self._url = url
        self._gis = gis

    @property
    def properties(self) -> dict[T, V]:
        """
        Returns the runtimes on the GIS

        :returns: dict[T,V]
        """
        params = {"f": "json"}
        return self._gis._con.get(self._url, params)

    def list(self) -> list[dict[T, V]]:
        """
        returns a list of runtimes on the system

        :returns: list[dict[T,V]]
        """
        params = {"f": "json"}
        return self._gis._con.get(self._url, params).get("runtimes", [])

    def manifest(self, id: str) -> dict[T, V]:
        """
        returns a dictionary containing all the libraries for that runtime

        :returns: dict[T,V]
        """
        url = f"{self._url}/{id}/manifest"
        params = {"f": "json"}
        return self._gis._con.get(url, params)
