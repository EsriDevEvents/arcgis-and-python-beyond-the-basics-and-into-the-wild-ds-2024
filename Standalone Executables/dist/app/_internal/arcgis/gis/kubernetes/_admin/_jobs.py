from __future__ import annotations
from typing import Any
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS


class Job(_BaseKube):
    """A Kubernetes asynchronous opertion"""

    _properties = None

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> dict[str, Any]:
        """
        returns the object properties
        """
        self._properties = None
        if self._properties is None:
            self._init()
        return self._properties


###########################################################################
class JobManager(_BaseKube):
    """
    Provides access to the jobs resources defined on the ArcGIS
    Enterprise.
    """

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ---------------------------------------------------------------------
    def job(self, job_id):
        """
        This resource returns the progress and status messages of an asynchronous
        job. Updated progress information can be acquired by periodically querying this operation.
        """
        url = f"{self._url}/{job_id}"
        params = {"f": "json"}
        return self._con.get(url, params)
