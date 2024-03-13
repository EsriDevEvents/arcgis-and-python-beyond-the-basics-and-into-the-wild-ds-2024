from __future__ import annotations
from arcgis.auth.tools._lazy import LazyLoader
from dataclasses import dataclass, field

from ._sfilters import SpatialFilter

arcgis = LazyLoader("arcgis")


@dataclass
class ViewLayerDefParameter:
    """
    When creating views, an optional definition query can be provided to limit what users
    of the view can see with the service view.

    ==================  ===============================================================================
    **Parameter**        **Description**
    ------------------  -------------------------------------------------------------------------------
    layer               Required FeatureLayer.  The layer to apply the layer definition to.
    ------------------  -------------------------------------------------------------------------------
    query_definition    Optional String.  The where clause to limit the layer with.
    ------------------  -------------------------------------------------------------------------------
    spatial_filter      Optional :class:`~arcgis.gis._impl._dataclasses.SpatialFilter`. A spatial
                        filter that can limit the data a user sees.
    ------------------  -------------------------------------------------------------------------------
    fields              Optional list[dict].  An array of field/visible fields that shows or hides a
                        field.  If this parameter is not given, all the fields are shown.

                        .. code-block:: python

                            [
                                {"name":"STATE_CITY","visible":True},
                                {"name":"TYPE","visible":False},
                            ]

    ==================  ===============================================================================

    """

    layer: arcgis.features.FeatureLayer
    query_definition: str | None = None
    spatial_filter: SpatialFilter | None = None
    fields: list[dict] | None = None
    _dict_data: dict | None = field(init=False)

    def __str__(self) -> str:
        return "<ViewLayerDefParameter>"

    def __repr__(self) -> str:
        return "<ViewLayerDefParameter>"

    def __post_init__(self):
        self._dict_data = self._create_dict()

    def _create_dict(self) -> dict:
        data = {}
        if self.query_definition:
            data["viewDefinitionQuery"] = self.query_definition
        if self.spatial_filter:
            data["viewLayerDefinition"] = {"filter": self.spatial_filter.as_json()}
        if self.fields:
            data["fields"] = self.fields
        self._dict_data = data
        return data

    @classmethod
    def fromlayer(
        self,
        layer: arcgis.features.managers.FeatureLayerManager
        | arcgis.features.FeatureLayer,
    ) -> "ViewLayerDefParameter":
        """Creates a view layer definition parameter object from a layer."""
        from arcgis.features.managers import FeatureLayerManager
        from arcgis.features import FeatureLayer, Table

        if isinstance(layer, (FeatureLayer, Table)):
            fl = layer
            adminlayer = layer.manager
        else:
            adminlayer = layer
            fl = FeatureLayer(
                url=adminlayer.url.replace(r"/admin/", r"/"), gis=layer._gis
            )
        props = dict(adminlayer.properties)
        gfilter = (
            props.get("adminLayerInfo", {})
            .get("viewLayerDefinition", {})
            .get("table", {})
            .get("filter", None)
        )
        if gfilter:
            from arcgis.gis._impl._dataclasses import SpatialRelationship, SpatialFilter

            g = arcgis.geometry.Geometry(gfilter["value"]["geometry"])
            sf = SpatialFilter(
                geometry=g,
                sr=g["spatialReference"],
                spatial_rel=SpatialRelationship._value2member_map_[gfilter["operator"]],
            )
        else:
            sf = None
        fields = [
            {"name": fld["name"], "visible": fld.get("visible", True)}
            for fld in props["fields"]
        ]
        return ViewLayerDefParameter(
            layer=fl,
            query_definition=props.get("viewDefinitionQuery", None),
            spatial_filter=sf,
            fields=fields,
        )

    def as_json(self) -> dict:
        """returns the view as a dictionary"""
        return self._create_dict()
