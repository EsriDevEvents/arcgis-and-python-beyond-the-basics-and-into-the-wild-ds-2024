from __future__ import annotations
from arcgis.auth.tools import LazyLoader

from functools import lru_cache
from typing import Any

arcgis = LazyLoader("arcgis")
requests = LazyLoader("requests")

__all__ = ["service_properties"]


@lru_cache(maxsize=255)
def service_properties(
    url: str | None = None,
    gis: arcgis.gis.GIS | None = None,
) -> dict[str, Any] | None:
    """
    Returns the Service JSON for the GeoEnrichment Service

    :returns: dict[str,Any] or None
    """
    if gis is None:
        gis = arcgis.env.active_gis
    if url is None:
        url = (
            f"{gis.properties['helperServices']['geoenrichment']['url']}/Geoenrichment"
        )
    session: arcgis.auth.EsriSession = gis._con._session
    response: requests.Response = session.get(
        url,
        params={
            "f": "json",
        },
    )
    response.raise_for_status()
    return response.json()
