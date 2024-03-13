from __future__ import annotations
from arcgis.gis import Item, Group, GIS
from arcgis._impl.common._clone import (
    CloneNode,
    _ItemDefinition,
    _TextItemDefinition,
)


class BaseCloneDefinition(CloneNode):
    """
    The base cloning module that allows users to extend the cloning API to
    meet there cloning workflows.
    """

    def clone(self) -> Item:
        """
        Override the clone operation in order performs the cloning logic
        """
        raise NotImplementedError("clone is not implemented")


class BaseCloneItemDefinition(_ItemDefinition):
    """
    Represents the definition of an item within ArcGIS Online or Portal.
    """

    def clone(self) -> Item:
        """
        Override the clone operation in order performs the cloning logic
        """
        raise NotImplementedError("clone is not implemented")


class BaseCloneTextItemDefinition(_TextItemDefinition):
    """
    Represents the definition of a text based item within ArcGIS Online or Portal.
    """

    def clone(self) -> Item:
        """
        Override the clone operation in order performs the cloning logic
        """
        raise NotImplementedError("clone is not implemented")


class BaseCloneGroup:
    """
    The base cloning module that allows users to extend the cloning API to
    meet there cloning workflows.
    """

    _group_source: Group = None
    _include_items: bool = None
    _gis: GIS = None

    def __init__(
        self,
        *,
        gis: GIS | None = None,
    ) -> None:
        """initializer"""
        if gis is None:
            import arcgis

            gis = arcgis.env.active_gis
        self._gis: GIS = gis

    def clone(self, **kwags) -> Group:
        """
        Override the clone operation in order performs the cloning logic
        """
        raise NotImplementedError("clone is not implemented")
