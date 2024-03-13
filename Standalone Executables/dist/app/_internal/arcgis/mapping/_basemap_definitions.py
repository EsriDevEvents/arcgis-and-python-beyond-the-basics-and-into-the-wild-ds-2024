# Note that these may change and need to be updated periodically
# https://developers.arcgis.com/javascript/latest/api-reference/esri-Map.html#basemap -> NO API KEY MAPS
# https://devtopia.esri.com/WebGIS/arcgis-js-api/blob/4master/esri/support/basemapDefinitions.ts -> for the json structure

basemap_dict = {
    "satellite": [
        {
            "id": "satellite-base-layer",
            "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Imagery",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        }
    ],
    "hybrid": [
        {
            "id": "hybrid-base-layer",
            "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Imagery",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "hybrid-reference-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/30d6b8271e1849cd9c3042060001f425/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "Hybrid Reference Layer",
            "isReference": True,
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
    ],
    "terrain": [
        {
            "id": "terrain-base-layer",
            "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Terrain Base",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "terrain-reference-layer",
            "url": "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Reference_Overlay/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Reference Overlay",
            "isReference": True,
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
    ],
    "oceans": [
        {
            "id": "oceans-base-layer",
            "url": "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Ocean Base",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "oceans-reference-layer",
            "url": "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Reference/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Ocean Reference",
            "isReference": True,
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
    ],
    "osm": [
        {
            "id": "osm-base-layer",
            "layerType": "OpenStreetMap",
            "title": "Open Street Map",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        }
    ],
    "dark-gray-vector": [
        {
            "id": "dark-gray-base-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/5e9b3685f4c24d8781073dd928ebda50/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "Dark Gray Base",
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "dark-gray-reference-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/747cb7a5329c478cbe6981076cc879c5/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "Dark Gray Reference",
            "isReference": True,
            "visibility": True,
            "opacity": 1,
        },
    ],
    "gray-vector": [
        {
            "id": "gray-base-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/291da5eab3a0412593b66d384379f89f/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "Light Gray Base",
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "gray-reference-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/1768e8369a214dfab4e2167d5c5f2454/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "Light Gray Reference",
            "isReference": True,
            "visibility": True,
            "opacity": 1,
        },
    ],
    "streets-vector": [
        {
            "id": "streets-vector-base-layer",
            "styleUrl": "##cdn.arcgis.com/sharing/rest/content/items/de26a3cf4cc9451298ea173c4b324736/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "World Streets",
            "visibility": True,
            "opacity": 1,
        }
    ],
    "topo-vector": [
        {
            "id": "world-hillshade-layer",
            "url": "https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
            "layerType": "ArcGISTiledMapServiceLayer",
            "title": "World Hillshade",
            "showLegend": False,
            "visibility": True,
            "opacity": 1,
        },
        {
            "id": "topo-vector-base-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/7dc6cea0b1764a1f9af2e679f642f0f5/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "World Topo",
            "visibility": True,
            "opacity": 1,
        },
    ],
    "streets-night-vector": [
        {
            "id": "streets-night-vector-base-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/86f556a2d1fd468181855a35e344567f/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "World Streets Night",
            "visibility": True,
            "opacity": 1,
        }
    ],
    "streets-relief-vector": [
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
            "id": "streets-relief-vector-base-layer",
            "styleUrl": "//www.arcgis.com/sharing/rest/content/items/b266e6d17fc345b498345613930fbd76/resources/styles/root.json",
            "title": "World Streets Relief",
            "layerType": "VectorTileLayer",
            "visibility": True,
            "opacity": 1,
        },
    ],
    "streets-navigation-vector": [
        {
            "id": "streets-navigation-vector-base-layer",
            "styleUrl": "https://www.arcgis.com/sharing/rest/content/items/63c47b7177f946b49902c24129b87252/resources/styles/root.json",
            "layerType": "VectorTileLayer",
            "title": "World Streets Navigation",
            "visibility": True,
            "opacity": 1,
        }
    ],
    "arcgis-imagery": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Imagery",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/World_Imagery/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Imagery:Labels",
            "title": "Hybrid Reference Layer",
            "isReference": True,
        },
    ],
    "arcgis-imagery-standard": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Imagery",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/World_Imagery/MapServer",
        }
    ],
    "arcgis-imagery-labels": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Imagery:Labels",
            "title": "Hybrid Reference Layer",
            "isReference": True,
        }
    ],
    "arcgis-light-gray": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:LightGray:Base",
            "title": "Light Gray Canvas Base",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:LightGray:Labels",
            "title": "Light Gray Canvas Labels",
            "isReference": True,
        },
    ],
    "arcgis-dark-gray": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:DarkGray:Base",
            "title": "Dark Gray Canvas Base",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:DarkGray:Labels",
            "title": "Dark Gray Canvas Labels",
            "isReference": True,
        },
    ],
    "arcgis-navigation": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Navigation",
            "title": "World Navigation Map",
        }
    ],
    "arcgis-navigation-night": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:NavigationNight",
            "title": "World Navigation Map (Dark Mode)",
        }
    ],
    "arcgis-streets": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Streets",
            "title": "World Street Map",
        }
    ],
    "arcgis-streets-night": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:StreetsNight",
            "title": "World Street Map (Night)",
        }
    ],
    "arcgis-streets-relief": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:StreetsRelief:Base",
            "title": "World Street Map (with Relief)",
        },
    ],
    "arcgis-topographic": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Topographic:Base",
            "title": "World Topographic Map",
        },
    ],
    "arcgis-oceans": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Ocean Base",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Oceans:Labels",
            "title": "World Ocean Reference",
            "isReference": True,
        },
    ],
    "osm-standard": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:Standard",
            "title": "OpenStreetMap",
        }
    ],
    "osm-standard-relief": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:StandardRelief:Base",
            "layerType": "VectorTileLayer",
            "title": "OpenStreetMap Relief Base",
        },
    ],
    "osm-streets": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:Streets",
            "title": "OpenStreetMap (Streets)",
        }
    ],
    "osm-streets-relief": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:StreetsRelief:Base",
            "layerType": "VectorTileLayer",
            "title": "OpenStreetMap Relief Base",
        },
    ],
    "osm-light-gray": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:LightGray:Base",
            "title": "OSM (Light Gray Base)",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:LightGray:Labels",
            "title": "OSM (Light Gray Reference)",
            "isReference": True,
        },
    ],
    "osm-dark-gray": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:DarkGray:Base",
            "title": "OSM (Dark Gray Base)",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/OSM:DarkGray:Labels",
            "title": "OSM (Dark Gray Reference)",
            "isReference": True,
        },
    ],
    "arcgis-terrain": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Terrain:Base",
            "title": "World Terrain Base",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Terrain:Detail",
            "title": "World Terrain Reference",
            "isReference": True,
        },
    ],
    "arcgis-community": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Community",
            "title": "Community",
        }
    ],
    "arcgis-charted-territory": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:ChartedTerritory:Base",
            "title": "Charted Territory",
        },
    ],
    "arcgis-colored-pencil": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:ColoredPencil",
            "title": "Colored Pencil",
        }
    ],
    "arcgis-nova": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Nova",
            "title": "Nova",
        }
    ],
    "arcgis-modern-antique": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        },
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:ModernAntique:Base",
            "title": "Modern Antique",
        },
    ],
    "arcgis-midcentury": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Midcentury",
            "title": "Mid-Century",
        }
    ],
    "arcgis-newspaper": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemaps-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:Newspaper",
            "title": "Newspaper",
        }
    ],
    "arcgis-hillshade-light": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer",
        }
    ],
    "arcgis-hillshade-dark": [
        {
            "layerType": "ArcGISTiledMapServiceLayer",
            "showLegend": False,
            "title": "World Hillshade (Dark)",
            "url": "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade_Dark/MapServer",
        }
    ],
    "arcgis-human-geography": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemapsdev-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:HumanGeography",
            "title": "HumanGeography",
        }
    ],
    "arcgis-human-geography-dark": [
        {
            "layerType": "VectorTileLayer",
            "styleUrl": "https://basemapsdev-api.arcgis.com/arcgis/rest/services/styles/ArcGIS:HumanGeographyDark",
            "title": "HumanGeographyDark",
        }
    ],
}
