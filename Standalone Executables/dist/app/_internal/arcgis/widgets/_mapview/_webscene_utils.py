DEFAULT_WEBSCENE_TEXT_PROPERTY = {
    "operationalLayers": [],
    "baseMap": {
        "id": "basemap",
        "title": "Topographic Vector",
        "baseMapLayers": [
            {
                "id": "world-hillshade-layer",
                "url": "//services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
                "layerType": "ArcGISTiledMapServiceLayer",
                "title": "World Hillshade",
                "showLegend": False,
                "visibility": True,
                "opacity": 1,
            },
            {
                "id": "topo-vector-base-layer",
                "styleUrl": "//cdn.arcgis.com/sharing/rest/content/items/7dc6cea0b1764a1f9af2e679f642f0f5/resources/styles/root.json",
                "layerType": "VectorTileLayer",
                "title": "World Topo",
                "visibility": True,
                "opacity": 1,
            },
        ],
        "elevationLayers": [
            {
                "url": "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer",
                "id": "globalElevation",
                "listMode": "hide",
                "layerType": "ArcGISTiledElevationServiceLayer",
                "title": "Elevation",
            }
        ],
    },
    "ground": {
        "layers": [
            {
                "url": "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer",
                "id": "globalElevation",
                "listMode": "hide",
                "layerType": "ArcGISTiledElevationServiceLayer",
                "title": "Elevation",
            }
        ]
    },
    "viewingMode": "global",
    "spatialReference": {"latestWkid": 3857, "wkid": 102100},
    "version": "1.11",
    "authoringApp": "PortalMycontentCreate",
    "authoringAppVersion": "6.2.0.0",
}
