from __future__ import annotations
from arcgis.gis import GIS
from typing import Any, TypeVar

__all__ = ["Container", "ContainerManager"]


K = TypeVar("K")
V = TypeVar("V")


class Container:
    """
    Represents a Single Notebook Container

    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The url for the container.
    ----------------  -------------------------------------------------------------------------------
    gis               Required GIS. The ArcGIS Online connection object.
    ================  ===============================================================================


    """

    _properties = None

    def __init__(self, url: str, gis: GIS):
        """initalizer"""
        self._url = url
        self._gis = gis

    @property
    def properties(self):
        if self._properties is None:
            url = f"{self._url}"
            params = {"f": "json"}
            self._properties = self._gis._con.get(url, params)
        return self._properties

    def terminate(self) -> bool:
        """stops the current container"""
        url = f"{self._url}/terminateContainer"
        params = {"f": "json"}
        return self._gis._con.post(url, params)

    @property
    def notebooks(self) -> list[dict[str, Any]]:
        """returns a list of notebooks running in the current container"""
        url = f"{self._url}/notebooks"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    def close(self, notebook_id: str) -> bool:
        """closes a notebook"""
        url = f"{self._url}/notebooks/{notebook_id}/closeNotebook"
        params = {"f": "json"}
        return self._gis._con.post(url, params)


class ContainerManager:
    """
    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The base url for the ContainerManager endpoints.
    ----------------  -------------------------------------------------------------------------------
    gis               Required GIS. The ArcGIS Online connection object.
    ================  ===============================================================================

    """

    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    def list(self) -> list[dict[str, Any]]:
        """Returns a list of containers"""
        url = f"{self._url}"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    def get(self, id: str) -> Container:
        """Gets an instance of a container"""
        return Container(url=f"{self._url}/{id}", gis=self._gis)

    def start(self, runtime: str, instance_type: str | None = None) -> dict[K, V]:
        """starts a container"""
        url = f"{self._url}/startContainer"
        params = {
            "f": "json",
            "notebookRuntimeId": runtime,
        }
        if instance_type:
            params["instanceTypeName"] = instance_type
        res = self._gis._con.post(url, params)
        return res
