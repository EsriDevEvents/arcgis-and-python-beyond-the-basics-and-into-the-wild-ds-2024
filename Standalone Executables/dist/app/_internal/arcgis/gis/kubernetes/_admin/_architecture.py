from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS
from typing import Any, Optional


###########################################################################
class ArchitectureManager(_BaseKube):
    """
    Provides access to the architecture resources defined on the ArcGIS
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
    @property
    def development(self) -> dict[str, Any]:
        """
        The development architecture profile is designed for use in
        nonproduction environments, including those for testing and
        evaluation, and requires the least amount of hardware and
        resources. This profile prioritizes replicated pods for publishing
        tools and the private ingress controller, setting replicas for
        both pods at 2. All other pod replicas are set as 1.

        :return: Dict[str, Any]
        """
        url = f"{self._url}/development"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    @property
    def standard(self) -> dict[str, Any]:
        """
        The standard-availability architecture profile is designed for use
        in production environments and those wanting to minimize unplanned
        downtime with redundancy across many pods. As a high-availability
        pod, standard-availability provides continued use and availability
        in the even of a failure, and requires less hardware than
        enhanced-availability.

        :return: Dict[str, Any]

        """
        url = f"{self._url}/standard-availability"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    @property
    def enhanced(self) -> dict[str, Any]:
        """
        The enhanced-availability architecture profile is designed for use
        in business or mission-critical production environments. This
        profile is designed for the highest level of availability, as it
        includes increased and expanded redundancy across critical pods. As
        a high-availability profile, enhanced-availability provides continued
        use and availability in the event of a failure. However, of the
        available profiles, the hardware requirements are the highest.

        :return: Dict[str, Any]

        """
        url = f"{self._url}/enhanced-availability"
        params = {"f": "json"}
        return self._con.get(url, params)
