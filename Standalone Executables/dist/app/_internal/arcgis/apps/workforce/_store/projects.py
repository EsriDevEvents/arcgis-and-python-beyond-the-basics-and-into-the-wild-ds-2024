""" Defines store functions for working with Projects.
"""
import arcgis
from arcgis.gis import Item
from arcgis.apps import workforce
from arcgis.apps.workforce._store._definitions import *
from arcgis.apps.workforce.exceptions import WorkforceError
from warnings import warn
import concurrent.futures
import os


def get_project(project_id, gis):
    """Loads and returns a workforce project.
    :param gis: An authenticated arcigs.gis.GIS object.
    :param project_id: The project's id. Version 1 - id of the project item. Version 2 - id of the feature service
    :returns: workforce.Project
    """
    item = Item(gis, project_id)
    return workforce.Project(item)


def create_project(title, summary=None, major_version=None, gis=None):
    """
    Creates a new Workforce Project

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    title                  :class:`String`.
                           The title of the Project to create (must be unique to the organization)
    ------------------     --------------------------------------------------------------------
    summary                :class:`String`.
                           The summary of the Project
    ------------------     --------------------------------------------------------------------
    major_version          Optional :class:`Int`
                           The version of the Project to create. 1 represents the original
                           Workforce Project which does not support offline. 2 represents the newer
                           Workforce Project which supports offline among other things. Defaults
                           to 2 in GIS 8.2 and higher
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS`.
                           The authenticated GIS to use.
                           Defaults to the active GIS if None is provided.
    ==================     ====================================================================

    Returns a :class:`arcgis.apps.workforce.Project`
    """

    if gis is None:
        gis = arcgis.env.active_gis

    if major_version is None:
        if gis.version < [8, 2]:
            major_version = 1
        else:
            major_version = 2

    if major_version == 1:
        return _v1_create_project(gis, summary, title)

    elif major_version == 2:
        if gis.version < [7, 1]:
            raise ValueError(
                "Offline Workforce Projects not supported at GIS lower than 7.1"
            )
        return _v2_create_project(gis, summary, title)

    else:
        raise WorkforceError("Invalid Project Version Specified")


def _get_basemap(gis):
    """
    Helper method to get the basemap to use. Tries to use the org defined basemap.
    :param gis:
    :return:
    """
    default_raster_basemap = {
        "baseMapLayers": [
            {
                "id": "World_Topo_Map",
                "layerType": "ArcGISTiledMapServiceLayer",
                "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer",
                "visibility": True,
                "opacity": 1,
                "title": "World_Topo_Map",
            }
        ],
        "title": "Topographic",
    }
    try:
        if gis.properties["useVectorBasemaps"]:
            bm_group = gis.groups.search(
                gis.properties["vectorBasemapGalleryGroupQuery"], outside_org=True
            )[0]
            query = 'group:{} AND name: "0220_Navigation_Title"'.format(bm_group.id)
            items = gis.content.search(query, outside_org=True)
            if items:
                return items[0].get_data()["baseMap"]
        else:
            return default_raster_basemap
    except:
        return default_raster_basemap


def _create_application_properties():
    """
    Helper method to build the application properties
    :return:
    """
    return {
        "viewing": {
            "search": {
                "enabled": True,
                "disablePlaceFinder": True,
                "layers": [
                    {
                        "id": "Workers_0",
                        "field": {
                            "name": "name",
                            "exactMatch": True,
                            "type": "esriFieldTypeString",
                        },
                    }
                ],
            }
        }
    }


def _build_operational_layers(
    item, popup_def=None, visibility=True, layer_index=0, capabilities=None
):
    """
    Helper method to build operational layers
    :param item: The item containing the layer (at index 0)
    :param popup_def: The popup definition of the layer
    :param visibility: Is this layer visible in the map
    :return: The operational layer dictionary
    """
    layer = item.layers[layer_index]
    op_layer = {
        "layerType": "ArcGISFeatureLayer",
        "opacity": 1,
        "itemId": item.id,
        "id": "{}_0".format(layer.properties["name"]),
        "title": layer.properties["name"],
        "url": layer.url,
        "visibility": visibility,
    }
    if capabilities:
        op_layer["capabilities"] = capabilities
    if popup_def:
        op_layer["popupInfo"] = popup_def
    return op_layer


def _build_table(item, table_index):
    """
    Helper method to build table
    :param item: The item containing the layer (at index 0)
    :param table_index: The index for the table in the item
    """
    table = item.tables[table_index]
    op_table = {
        "capabilities": "Create,Delete,Query,Update,Editing,Sync",
        "url": table.url,
        "id": "{}_0".format(table.properties["name"]),
        "title": table.properties["name"],
        "itemId": item.id,
    }
    return op_table


def _v2_create_project(gis, summary, title):
    """Creates project following version 2 Workforce schema"""
    for f in gis.users.me.folders:
        if f["title"].lower() == title.lower():
            raise WorkforceError("A folder named '{}' already exists.".format(title))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_folder = executor.submit(gis.content.create_folder, title)
        future_group = executor.submit(
            gis.groups.create,
            title,
            "workforce",
            access="private",
            is_invitation_only=True,
            is_view_only=True,
            sort_field="title",
            sort_order="asc",
        )
    future_folder.result()
    group = future_group.result()
    group.protected = True
    folder_name = title
    group_id = group.id
    workforce_service_name = "workforce_{}".format(group_id)
    workforce_service_item = _v2_create_service_with_layers(
        gis,
        folder_name,
        workforce_service_name,
        assignment_layer_definition_v2,
        worker_layer_definition_v2,
        dispatcher_table_definition_v2,
        assignment_type_table_definition_v2,
        app_integration_table_definition_v2,
        title,
    )

    # create webmaps
    my_path = os.path.abspath(os.path.dirname(__file__))
    thumbnail = os.path.join(
        my_path, "/".join(("resources", "default-project-thumbnail.png"))
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        workers_webmap_future = executor.submit(
            _v2_create_worker_webmap,
            gis,
            folder_name,
            workforce_service_item,
            assignment_layer_popup_definition_v2,
            worker_layer_popup_definition_v2,
            title,
            summary,
            thumbnail,
        )
        dispatchers_webmap_future = executor.submit(
            _v2_create_dispatcher_webmap,
            gis,
            folder_name,
            workforce_service_item,
            assignment_layer_popup_definition_v2,
            worker_layer_popup_definition_v2,
            title,
            summary,
            thumbnail,
        )
    workers_webmap = workers_webmap_future.result()
    dispatchers_webmap = dispatchers_webmap_future.result()

    project_items = [workforce_service_item, workers_webmap, dispatchers_webmap]
    # share and protect items
    for i in project_items:
        i.share(groups=[group])
        i.protect()

    # set fs item properties / thumbnail
    workforce_service_item.update(
        thumbnail=thumbnail,
        item_properties={
            "snippet": summary,
            "properties": {
                "workforceProjectGroupId": group_id,
                "workforceProjectVersion": "2.0.0",
                "workforceDispatcherMapId": dispatchers_webmap.id,
                "workforceWorkerMapId": workers_webmap.id,
            },
        },
    )

    # manually add the owner as the first dispatcher
    user_id_field_name = "userId"
    for field in workforce_service_item.tables[0].properties.fields:
        if field["name"].lower() == "userid":
            user_id_field_name = field["name"]
            break

    workforce_service_item.tables[0].edit_features(
        adds=[
            arcgis.features.Feature(
                attributes={
                    "name": gis.users.me.fullName,
                    user_id_field_name: gis.users.me.username,
                }
            )
        ]
    )

    # add relationships
    try:
        workers_webmap.add_relationship(
            workforce_service_item, "WorkforceMap2FeatureService"
        )
        dispatchers_webmap.add_relationship(
            workforce_service_item, "WorkforceMap2FeatureService"
        )
    except Exception:
        warn(
            "Relationship not added. This version of ArcGIS may not support the WorkforceMap2FeatureService relationship"
        )

    # create the Project to return, add navigator as default integration
    project = arcgis.apps.workforce.Project(workforce_service_item)
    project.integrations.add(
        integration_id="arcgis-navigator",
        prompt="Navigate to Assignment",
        url_template="https://navigator.arcgis.app?stop=${assignment.latitude},${assignment.longitude}&stopname=${assignment.location}&callback=https://workforce.arcgis.app&callbackprompt=Workforce",
    )

    return project


def _v1_create_project_item(
    gis,
    folder_name,
    assignments_item,
    dispatchers_item,
    workers_item,
    tracks_item,
    dispatchers_webmap,
    workers_webmap,
    title,
    summary,
    folder_id,
    group_id,
):
    """
    Creates the project item
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the project
    :param assignments_item: The assignments item
    :param dispatchers_item: THe dispatchers item
    :param workers_item: The workers item
    :param tracks_item: The tracks item
    :param dispatchers_webmap: The dispatcher webmap item
    :param workers_webmap: The workers webmap item
    :param title: The title of the project
    :param summary: The summary of the project
    :param folder_id: The folder id
    :param group_id: The group id
    :return: The project item
    """
    item_properties = {
        "title": title,
        "snippet": summary,
        "tags": "workforce",
        "type": "Workforce Project",
        "typeKeywords": "Workforce Project",
    }
    project_data = {
        "workerWebMapId": workers_webmap.id,
        "dispatcherWebMapId": dispatchers_webmap.id,
        "dispatchers": {
            "serviceItemId": dispatchers_item.id,
            "url": dispatchers_item.layers[0].url,
        },
        "assignments": {
            "serviceItemId": assignments_item.id,
            "url": assignments_item.layers[0].url,
        },
        "workers": {
            "serviceItemId": workers_item.id,
            "url": workers_item.layers[0].url,
        },
        "tracks": {
            "serviceItemId": tracks_item.id,
            "url": tracks_item.layers[0].url,
            "enabled": False,
            "updateInterval": 30,
        },
        "assignmentIntegrations": [
            {
                "id": "default-navigator",
                "prompt": "Navigate to Assignment",
                "urlTemplate": "https://navigator.arcgis.app?stop=${assignment.latitude},${assignment.longitude}&stopname=${assignment.location}&callback=https://workforce.arcgis.app&callbackprompt=Workforce",
            }
        ],
        "version": "1.3.0",
        "groupId": group_id,
        "folderId": folder_id,
    }
    item_properties["text"] = json.dumps(project_data)
    item = gis.content.add(item_properties, folder=folder_name)
    return item


def _v2_create_worker_webmap(
    gis,
    folder_name,
    workforce_service_item,
    assignments_popup_def,
    workers_popup_def,
    title,
    summary,
    thumbnail,
):
    """
    Creates the worker webmap
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the webmap
    :param workforce_service_item: The workforce service item
    :param assignments_popup_def: The assignments popup definition
    :param workers_popup_def: The workers popup definition
    :param title: The title of the project
    :return: The worker webmap item
    """
    extent = _get_default_extent(gis)
    array_extent = [[extent["xmin"], extent["ymin"]], [extent["xmax"], extent["ymax"]]]
    item_properties = {
        "title": "{}".format(title),
        "snippet": summary,
        "extent": array_extent,
        "type": "Web Map",
        "typeKeywords": "ArcGIS Online,Explorer Web Map,Map,Offline,Online Map,Web Map,Workforce Worker,Data Editing",
        "properties": {"workforceFeatureServiceId": workforce_service_item.id},
    }

    webmap_data = {
        "baseMap": _get_basemap(gis),
        "operationalLayers": [],
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        "authoringApp": "ArcGISPythonAPI",
        "authoringAppVersion": str(arcgis.__version__),
        "tables": [],
        "applicationProperties": {
            "viewing": {
                "search": {
                    "enabled": True,
                    "disablePlaceFinder": False,
                    "layers": [
                        {
                            "id": "Workers_0",
                            "field": {
                                "name": "name",
                                "exactMatch": False,
                                "type": "esriFieldTypeString",
                            },
                        }
                    ],
                }
            }
        },
        "version": "2.11",
    }
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workforce_service_item, assignments_popup_def, layer_index=0
        )
    )
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workforce_service_item, workers_popup_def, layer_index=1
        )
    )
    webmap_data["tables"].append(_build_table(workforce_service_item, table_index=0))
    webmap_data["tables"].append(_build_table(workforce_service_item, table_index=1))
    webmap_data["tables"].append(_build_table(workforce_service_item, table_index=2))
    item_properties["text"] = json.dumps(webmap_data)

    item = gis.content.add(item_properties, folder=folder_name, thumbnail=thumbnail)
    return item


def _v2_create_dispatcher_webmap(
    gis,
    folder_name,
    workforce_service_item,
    assignments_popup_def,
    workers_popup_def,
    title,
    summary,
    thumbnail,
):
    """
    Creates the dispatcher webmap
    :param gis: An authenticated GIS
    :param folder_name: The folder in which to store the webmap
    :param workforce_service_item: The workforce service item item
    :param assignments_popup_def: The assignments popup definition
    :param workers_popup_def: The workers popup definition
    :param title: The title of the project
    :return: The dispatcher webmap item
    """
    extent = _get_default_extent(gis)
    array_extent = [[extent["xmin"], extent["ymin"]], [extent["xmax"], extent["ymax"]]]
    item_properties = {
        "title": "{} Dispatcher Map".format(title),
        "snippet": summary,
        "extent": array_extent,
        "type": "Web Map",
        "typeKeywords": "ArcGIS Online,Explorer Web Map,Map,Offline,Online Map,Web Map,Workforce Dispatcher",
        "properties": {"workforceFeatureServiceId": workforce_service_item.id},
    }

    webmap_data = {
        "baseMap": _get_basemap(gis),
        "operationalLayers": [],
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        "authoringApp": "ArcGISPythonAPI",
        "authoringAppVersion": str(arcgis.__version__),
        "version": "2.11",
    }
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workforce_service_item,
            assignments_popup_def,
            layer_index=0,
            capabilities="Query,Sync",
        )
    )
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workforce_service_item,
            workers_popup_def,
            layer_index=1,
            capabilities="Query,Sync",
        )
    )
    item_properties["text"] = json.dumps(webmap_data)
    item = gis.content.add(item_properties, folder=folder_name, thumbnail=thumbnail)
    return item


def _v2_create_service(
    gis, service_name, folder_name, spatial_ref, item_properties=None
):
    """
    Helper method that creates a workforce service
    :param gis: An authenticated GIS
    :param service_name: The name of the service
    :param folder_name: The name of the folder in which to place the service
    :param spatial_ref: The spatial reference to use
    :return: The service item
    """
    capabilities = ["Query", "Editing", "Create", "Update", "Delete", "Sync"]
    if not gis.properties.isPortal:
        capabilities.append("ChangeTracking")
    create_params = {
        "name": service_name,
        "capabilities": ",".join(capabilities),
        "spatialReference": spatial_ref,
        "syncRowsMovedOutsideFilter": True,
        "preserveLayerIds": True,
    }

    return gis.content.create_service(
        service_name,
        folder=folder_name,
        create_params=create_params,
        item_properties=item_properties,
        service_type="featureService",
    )


def _v2_create_service_with_layers(
    gis,
    folder_name,
    service_name,
    assignments_layer_def,
    workers_layer_def,
    dispatchers_table_def,
    assignment_type_table_def,
    integration_table_def,
    title,
):
    """
    Creates a service, adds, and layer and optionally enables attachments
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the service
    :param service_name: The name of the service
    :param assignments_layer_def: The assignments layer definition (dictionary)
    :param workers_layer_def: The workers layer definition (dictionary)
    :param dispatchers_table_def: The dispatchers table definition (dictionary)
    :param assignment_type_table_def: The assignment type table definition (dictionary)
    :param integration_table_def: The integration table definition (dictionary)
    :return: The service item
    """
    default_extent = gis.properties["defaultExtent"]
    spatial_reference = default_extent["spatialReference"]
    layer_defs = [assignments_layer_def, workers_layer_def]
    table_defs = [
        dispatchers_table_def,
        assignment_type_table_def,
        integration_table_def,
    ]
    if gis.content.is_service_name_available(service_name, "featureService"):
        item_properties = {
            "title": title,
            "tags": "workforce",
            "typeKeywords": "Workforce Project",
        }
        item = _v2_create_service(
            gis, service_name, folder_name, spatial_reference, item_properties
        )
        feature_layer_collection = arcgis.features.FeatureLayerCollection.fromitem(item)
    else:
        raise Exception("Service name already exists.")

    for layer_def in layer_defs:
        layer_def["extent"] = default_extent

    feature_layer_collection.manager.add_to_definition(
        {"layers": layer_defs, "tables": table_defs}
    )

    # enabled editor tracking
    feature_layer_collection.manager.update_definition(
        {
            "editorTrackingInfo": {
                "enableEditorTracking": True,
                "enableOwnershipAccessControl": False,
                "allowOthersToUpdate": True,
                "allowOthersToDelete": True,
            }
        }
    )
    return item


def _get_default_extent(gis):
    """
    Helper method to get the extent used by the org or use default extent
    :param gis: An authenticated GIS
    :return: A dictionary representing an extent
    """
    default_extent = gis.properties["defaultExtent"]
    try:
        extent = arcgis.geometry.project(
            [arcgis.geometry.Envelope(default_extent)],
            default_extent["spatialReference"]["wkid"],
            4326,
        )[0]
    except:
        extent = {
            "spatialReference": {"wkid": 4326},
            "xmin": -180,
            "ymin": -90,
            "xmax": 180,
            "ymax": 90,
        }
    return extent


def _v1_create_project(gis, summary, title):
    for f in gis.users.me.folders:
        if f["title"].lower() == title.lower():
            raise WorkforceError("A folder named '{}' already exists.".format(title))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_folder = executor.submit(gis.content.create_folder, title)
        future_group = executor.submit(
            gis.groups.create,
            title,
            "workforce",
            access="private",
            is_invitation_only=True,
            is_view_only=True,
            sort_field="title",
            sort_order="asc",
        )
    folder = future_folder.result()
    group = future_group.result()
    folder_name = title
    folder_id = folder["id"]
    group_id = group.id
    # create services
    worker_service_name = "workers_{}".format(group_id)
    dispatcher_service_name = "dispatchers_{}".format(group_id)
    assignment_service_name = "assignments_{}".format(group_id)
    tracking_service_name = "location_{}".format(group_id)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        workers_item_future = executor.submit(
            _v1_create_service_with_layer,
            gis,
            folder_name,
            worker_service_name,
            worker_layer_definition_v1,
        )
        dispatchers_item_future = executor.submit(
            _v1_create_service_with_layer,
            gis,
            folder_name,
            dispatcher_service_name,
            dispatcher_layer_definition_v1,
        )
        assignments_item_future = executor.submit(
            _v1_create_service_with_layer,
            gis,
            folder_name,
            assignment_service_name,
            assignment_layer_definition_v1,
            attachments=True,
        )
        tracks_item_future = executor.submit(
            _v1_create_service_with_location_tracking_layer,
            gis,
            folder_name,
            tracking_service_name,
            tracking_layer_definition_v1,
        )
    workers_item = workers_item_future.result()
    dispatchers_item = dispatchers_item_future.result()
    assignments_item = assignments_item_future.result()
    tracks_item = tracks_item_future.result()

    # create webmaps
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        workers_webmap_future = executor.submit(
            _v1_create_worker_webmap,
            gis,
            folder_name,
            assignments_item,
            workers_item,
            tracks_item,
            assignment_layer_popup_definition_v1,
            worker_layer_popup_definition_v1,
            title,
        )
        dispatchers_webmap_future = executor.submit(
            _v1_create_dispatcher_webmap,
            gis,
            folder_name,
            assignments_item,
            workers_item,
            assignment_layer_popup_definition_v1,
            worker_layer_popup_definition_v1,
            title,
        )
    workers_webmap = workers_webmap_future.result()
    dispatchers_webmap = dispatchers_webmap_future.result()

    # create the project
    project_item = _v1_create_project_item(
        gis,
        folder_name,
        assignments_item,
        dispatchers_item,
        workers_item,
        tracks_item,
        dispatchers_webmap,
        workers_webmap,
        title,
        summary,
        folder_id,
        group_id,
    )
    project_items = [
        workers_item,
        dispatchers_item,
        assignments_item,
        tracks_item,
        workers_webmap,
        dispatchers_webmap,
        project_item,
    ]
    # share and protect items
    for i in project_items:
        i.share(groups=[group])
        i.protect()

    my_path = os.path.abspath(os.path.dirname(__file__))
    thumbnail = os.path.join(
        my_path, "/".join(("resources", "default-project-thumbnail.png"))
    )
    project_item.update(thumbnail=thumbnail)

    # manually add the owner as the first dispatcher
    user_id_field_name = "userId"
    for field in dispatchers_item.layers[0].properties.fields:
        if field["name"].lower() == "userid":
            user_id_field_name = field["name"]
            break

    dispatchers_item.layers[0].edit_features(
        adds=[
            arcgis.features.Feature(
                attributes={
                    "name": gis.users.me.fullName,
                    user_id_field_name: gis.users.me.username,
                }
            )
        ]
    )

    # create the Project to return
    project = arcgis.apps.workforce.Project(project_item)
    return project


def _v1_create_worker_webmap(
    gis,
    folder_name,
    assignments_item,
    workers_item,
    tracking_item,
    assignments_popup_def,
    workers_popup_def,
    title,
):
    """
    Creates the worker webmap
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the webmap
    :param assignments_item: The assignments item
    :param workers_item: The workers item
    :param tracking_item: The tracking item
    :param assignments_popup_def: The assignments popup definition
    :param workers_popup_def: The workers popup definition
    :param title: The title of the project
    :return: The worker webmap item
    """
    extent = _get_default_extent(gis)
    array_extent = [[extent["xmin"], extent["ymin"]], [extent["xmax"], extent["ymax"]]]
    item_properties = {
        "title": "{}_workers".format(title),
        "tags": "workforce-worker",
        "extent": array_extent,
        "type": "Web Map",
        "typeKeywords": "ArcGIS Online,Explorer Web Map,Map,Online Map,Web Map,Workforce Project",
    }

    webmap_data = {
        "baseMap": _get_basemap(gis),
        "operationalLayers": [],
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        "authoringApp": "ArcGISPythonAPI",
        "authoringAppVersion": str(arcgis.__version__),
        "applicationProperties": {
            "viewing": {
                "search": {
                    "enabled": True,
                    "disablePlaceFinder": False,
                    "layers": [
                        {
                            "id": "Workers_0",
                            "field": {
                                "name": "name",
                                "exactMatch": False,
                                "type": "esriFieldTypeString",
                            },
                        }
                    ],
                }
            }
        },
        "version": "2.11",
    }

    webmap_data["operationalLayers"].append(
        _build_operational_layers(tracking_item, visibility=False, capabilities="Query")
    )
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            assignments_item, assignments_popup_def, capabilities="Query,Sync"
        )
    )
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workers_item, workers_popup_def, capabilities="Query,Sync"
        )
    )
    item_properties["text"] = json.dumps(webmap_data)

    item = gis.content.add(item_properties, folder=folder_name)
    return item


def _v1_create_dispatcher_webmap(
    gis,
    folder_name,
    assignments_item,
    workers_item,
    assignments_popup_def,
    workers_popup_def,
    title,
):
    """
    Creates the dispatcher webmap
    :param gis: An authenticated GIS
    :param folder_name: The folder in which to store the webmap
    :param assignments_item: The assignments item
    :param workers_item: The workers item
    :param assignments_popup_def: The assignments popup definition
    :param workers_popup_def: The workers popup definition
    :param title: The title of the project
    :return: The dispatcher webmap item
    """
    extent = _get_default_extent(gis)
    array_extent = [[extent["xmin"], extent["ymin"]], [extent["xmax"], extent["ymax"]]]
    item_properties = {
        "title": "{}_dispatchers".format(title),
        "tags": "workforce-dispatcher",
        "extent": array_extent,
        "type": "Web Map",
        "typeKeywords": "ArcGIS Online,Explorer Web Map,Map,Offline,Online Map,Web Map,Workforce Project",
    }

    webmap_data = {
        "baseMap": _get_basemap(gis),
        "operationalLayers": [],
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        "authoringApp": "ArcGISPythonAPI",
        "authoringAppVersion": str(arcgis.__version__),
        "version": "2.11",
    }
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            assignments_item, assignments_popup_def, capabilities="Query,Sync"
        )
    )
    webmap_data["operationalLayers"].append(
        _build_operational_layers(
            workers_item, workers_popup_def, capabilities="Query,Sync"
        )
    )
    item_properties["text"] = json.dumps(webmap_data)
    item = gis.content.add(item_properties, folder=folder_name)
    return item


def _v1_create_service_with_layer(
    gis, folder_name, service_name, layer_definition, attachments=False, sync=True
):
    """
    Creates a service, adds, and layer and optionally enables attachments
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the service
    :param service_name: The name of the service
    :param layer_definition: The definition of the layer (dictionary)
    :param attachments: Should attachments be enabled on the service
    :return: The service item
    """
    default_extent = gis.properties["defaultExtent"]
    spatial_reference = default_extent["spatialReference"]
    layer_definition["extent"] = default_extent
    if gis.content.is_service_name_available(service_name, "featureService"):
        item = _v1_create_service(
            gis, service_name, folder_name, spatial_reference, sync=sync
        )
        item.update({"tags": "workforce"})
        feature_layer_collection = arcgis.features.FeatureLayerCollection.fromitem(item)

        # add layer
        feature_layer_collection.manager.add_to_definition(
            {"layers": [layer_definition]}
        )

        # enabled editor tracking
        feature_layer_collection.manager.update_definition(
            {
                "editorTrackingInfo": {
                    "enableEditorTracking": True,
                    "enableOwnershipAccessControl": False,
                    "allowOthersToUpdate": True,
                    "allowOthersToDelete": True,
                }
            }
        )

        # enable attachments
        if attachments:
            feature_layer_collection.layers[0].manager.update_definition(
                {"hasAttachments": True}
            )
        return item
    else:
        raise Exception("Service name already exists.")


def _v1_create_service_with_location_tracking_layer(
    gis, folder_name, service_name, layer_definition
):
    """
    Creates the location tracking service, adds a layer and optionally enables attachments
    :param gis: An authenticated GIS
    :param folder_name: The name of the folder in which to place the service
    :param service_name: The name of the service
    :param layer_definition: The definition of the layer (dictionary)
    :return: The location tracking item
    """
    item = _v1_create_service_with_layer(
        gis, folder_name, service_name, layer_definition, sync=False
    )
    item.layers[0].manager.update_definition(
        {
            "timeInfo": {
                "startTimeField": item.layers[0].properties["editFieldsInfo"][
                    "creationDateField"
                ],
                "timeReference": {"timeZone": "UTC", "respectDaylightSaving": False},
                "timeInterval": 0,
                "exportOptions": {
                    "useTime": True,
                    "timeDataCumulative": True,
                    "TimeOffset": 0,
                    "timeOffsetUnits": "esriTimeUnitsCenturies",
                },
                "hasLiveData": True,
            }
        }
    )
    item.update(
        {
            "tags": "workforce, Location Tracking",
            "typeKeywords": "Collector,Data,Feature Service Template,Layer,Layer Template,Location Tracking,Platform,Service template,Template",
        }
    )
    return item


def _v1_create_service(gis, service_name, folder_name, spatial_ref, sync=True):
    """
    Helper method that creates a workforce service
    :param gis: An authenticated GIS
    :param service_name: The name of the service
    :param folder_name: The name of the folder in which to place the service
    :param spatial_ref: The spatial reference to use
    :return: The service item
    """
    create_params = {
        "name": service_name,
        "allowGeometryUpdates": True,
        "units": "esriMeters",
        "spatialReference": spatial_ref,
        "capabilities": "Query,Editing,Create,Update,Delete",
    }
    if sync:
        create_params["syncEnabled"] = True
        create_params["syncCapabilities"] = {
            "supportsAsync": True,
            "supportsRegisteringExistingData": True,
            "supportsSyncDirectionControl": True,
            "supportsPerLayerSync": True,
            "supportsPerReplicaSync": True,
            "supportsSyncModelNone": True,
            "supportsRollbackOnFailure": True,
        }
        create_params["capabilities"] = "Query,Editing,Create,Update,Delete,Sync"

    return gis.content.create_service(
        service_name,
        folder=folder_name,
        create_params=create_params,
        service_type="featureService",
    )
