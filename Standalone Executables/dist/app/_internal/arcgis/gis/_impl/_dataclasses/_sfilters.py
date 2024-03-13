from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from arcgis.geometry import (
    Point,
    Polygon,
    Polyline,
    Envelope,
    MultiPoint,
    SpatialReference,
)


class SpatialRelationship(Enum):
    """
    ==================  ===============================================================================
    **Parameter**        **Description**
    ------------------  -------------------------------------------------------------------------------
    INTERSECTS          Query Geometry Intersects Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    ENVELOPEINTERSECTS  Envelope of Query Geometry Intersects Envelope of Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    INDEXINTERSECTS     Query Geometry Intersects Index entry for Target Geometry (Primary Index Filter).
    ------------------  -------------------------------------------------------------------------------
    TOUCHES             Query Geometry Touches Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    OVERLAPS            Query Geometry Overlaps Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    CROSSES             Query Geometry Crosses Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    WITHIN              Query Geometry is Within Target Geometry.
    ------------------  -------------------------------------------------------------------------------
    CONTAINS            Query Geometry Contains Target Geometry.
    ==================  ===============================================================================
    """

    INTERSECTS = "esriSpatialRelIntersects"
    CONTAINS = "esriSpatialRelContains"
    CROSSES = "esriSpatialRelCrosses"
    ENVELOPEINTERSECTS = "esriSpatialRelEnvelopeIntersects"
    INDEXINTERSECTS = "esriSpatialRelIndexIntersects"
    OVERLAPS = "esriSpatialRelOverlaps"
    TOUCHES = "esriSpatialRelTouches"
    WITHIN = "esriSpatialRelWithin"


@dataclass
class SpatialFilter:
    """Creates a spatial filter that can be used in query and create view operations."""

    geometry: Point | Polygon | Polyline | Envelope | MultiPoint
    spatial_rel: SpatialRelationship = SpatialRelationship.INTERSECTS
    sr: SpatialReference | None = None

    def as_json(self):
        """Converts the SpatialFilter to a JSON representation"""
        gt = {
            "point": "Point",
            "multipoint": "Multipoint",
            "polygon": "Polygon",
            "polyline": "Polyline",
            "envelope": "Envelope",
        }
        spatial_filter = {
            "geometry": self.geometry,
            "geometryType": "esriGeometry" + gt[str(self.geometry.type).lower()],
            "spatialRel": self.spatial_rel.value,
        }

        if self.sr is None:
            if "spatialReference" in self.geometry:
                sr = self.geometry["spatialReference"]

        else:
            spatial_filter["inSR"] = sr
        return spatial_filter
