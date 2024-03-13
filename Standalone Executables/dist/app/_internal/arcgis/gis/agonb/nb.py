from __future__ import annotations
from arcgis.gis import GIS
from .containers import ContainerManager
from .instpref import InstancePreference
from .runtime import RuntimeManager
from .snapshot import SnapshotManager
from .notebook import NotebookManager
from typing import TypeVar

__all__ = ["AGOLNotebookManager"]

K = TypeVar("K")
V = TypeVar("V")


class AGOLNotebookManager:
    _container = None
    _istpref = None
    _runtimes = None
    _snapshot = None
    _nbm = None

    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    @property
    def containers(self) -> ContainerManager:
        """
        Provides the ability to manage containers and the notebooks within them
        :returns: ContainerManager
        """
        if self._container is None:
            self._container = ContainerManager(
                url=f"{self._url}/system/containers", gis=self._gis
            )
        return self._container

    @property
    def instance_preferences(self) -> InstancePreference:
        """
        Provides information about the available instances for notebooks

        :returns: InstancePreference
        """
        if self._istpref is None:
            url = f"{self._url}/notebooks/instancePreferences"
            self._istpref = InstancePreference(url=url, gis=self._gis)
        return self._istpref

    @property
    def _machines(self) -> dict[K, V]:
        """Returns information about the machines running on notebook server"""
        url = f"{self._url}/machines"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    @property
    def runtimes(self) -> RuntimeManager:
        """Provides information about the available runtimes on the notebook server"""
        if self._runtimes is None:
            url = f"{self._url}/notebooks/runtimes"
            self._runtimes = RuntimeManager(url=url, gis=self._gis)
        return self._runtimes

    @property
    def snaphots(self) -> SnapshotManager:
        """
        Returns tools to work with snapshots on notebooks

        :returns: SnapshotManager
        """
        if self._snapshot is None:
            url = f"{self._url}/notebooks/snapshots"
            self._snapshot = SnapshotManager(url=url, gis=self._gis)
        return self._snapshot

    @property
    def notebooksmanager(self) -> NotebookManager:
        """
        Manages the run and execution of notebooks

        :returns: NotebookManager
        """
        if self._nbm is None:
            url = f"{self._url}/notebooks"
            self._nbm = NotebookManager(url=url, gis=self._gis, nbs=self)
        return self._nbm
