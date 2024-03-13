"""
Functions for calling the Raster Analysis Tools. The RasterAnalysisTools service is used by ArcGIS Server to provide distributed raster analysis.

The Hosted Imagery & Raster Analysis capabilities are available both on ArcGIS Enterprise and ArcGIS Online.
Refer https://doc.arcgis.com/en/arcgis-online/analyze/perform-raster-analysis.htm for more details on performing Analysis using ArcGIS Online.
Refer https://enterprise.arcgis.com/en/portal/latest/use/perform-raster-analysis.htm for more details on performing Analysis using ArcGIS Enterprise.
"""
from __future__ import annotations
from typing import Any, Optional, Union

from arcgis.geometry import SpatialReference
from arcgis.features.layer import FeatureLayer

import json as _json
import arcgis as _arcgis
import string as _string
import random as _random
import collections
from arcgis.gis import GIS, Item
from arcgis.raster._util import _set_context, _id_generator
from arcgis.raster import ImageryLayer
from .._impl.common._deprecate import deprecated


def get_datastores(gis: Optional[GIS] = None):
    """
    Returns a helper object to manage raster analytics datastores in the GIS.
    If a gis isn't specified, returns datastore manager of :attr:`~arcgis.env.active_gis`
    """
    gis = _arcgis.env.active_gis if gis is None else gis

    for ds in gis._datastores:
        if "RasterAnalytics" in ds._server["serverFunction"]:
            return ds

    return None


def is_supported(gis: Optional[GIS] = None):
    """
    Returns True if the GIS supports raster analytics. If a gis isn't specified,
    checks if :attr:`~arcgis.env.active_gis` supports raster analytics
    """
    gis = _arcgis.env.active_gis if gis is None else gis
    if "rasterAnalytics" in gis.properties.helperServices:
        return True
    else:
        return False


def _id_generator(size=6, chars=_string.ascii_uppercase + _string.digits):
    return "".join(_random.choice(chars) for _ in range(size))


def _create_output_image_service(gis, output_name, task, folder=None):
    ok = gis.content.is_service_name_available(output_name, "Image Service")
    if not ok:
        raise RuntimeError(
            "An Image Service by this name already exists: " + output_name
        )

    create_parameters = {
        "name": output_name,
        "description": "",
        "capabilities": "Image",
        "properties": {"path": "@", "description": "", "copyright": ""},
    }

    output_service = gis.content.create_service(
        output_name,
        create_params=create_parameters,
        service_type="imageService",
        folder=folder,
    )
    description = "Image Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": "Analysis Image Service generated from " + task,
    }
    output_service.update(item_properties)
    return output_service


def _create_output_feature_service(
    gis,
    output_name,
    output_service_name="Analysis feature service",
    task="GeoAnalytics",
    folder=None,
):
    ok = gis.content.is_service_name_available(output_name, "Feature Service")
    if not ok:
        raise RuntimeError(
            "A Feature Service by this name already exists: " + output_name
        )

    createParameters = {
        "currentVersion": 10.2,
        "serviceDescription": "",
        "hasVersionedData": False,
        "supportsDisconnectedEditing": False,
        "hasStaticData": True,
        "maxRecordCount": 2000,
        "supportedQueryFormats": "JSON",
        "capabilities": "Query",
        "description": "",
        "copyrightText": "",
        "allowGeometryUpdates": False,
        "syncEnabled": False,
        "editorTrackingInfo": {
            "enableEditorTracking": False,
            "enableOwnershipAccessControl": False,
            "allowOthersToUpdate": True,
            "allowOthersToDelete": True,
        },
        "xssPreventionInfo": {
            "xssPreventionEnabled": True,
            "xssPreventionRule": "InputOnly",
            "xssInputRule": "rejectInvalid",
        },
        "tables": [],
        "name": output_service_name.replace(" ", "_"),
    }

    output_service = gis.content.create_service(
        output_name,
        create_params=createParameters,
        service_type="featureService",
        folder=folder,
    )
    description = "Feature Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": output_service_name,
    }
    output_service.update(item_properties)
    return output_service


def _flow_direction_analytics_converter(
    raster_function,
    output_name=None,
    other_outputs=None,
    gis=None,
    future=False,
    **kwargs,
):
    input_surface_raster = (
        forceFlow
    ) = flowDirectionType = output_flow_direction_raster = output_drop_name = None

    input_surface_raster = raster_function["rasterFunctionArguments"][
        "in_surface_raster"
    ]
    if "force_flow" in raster_function["rasterFunctionArguments"].keys():
        forceFlow = raster_function["rasterFunctionArguments"]["force_flow"]
    if "flow_direction_type" in raster_function["rasterFunctionArguments"].keys():
        flowDirectionType = raster_function["rasterFunctionArguments"][
            "flow_direction_type"
        ]
    output_flow_direction_raster = output_name
    if "out_drop_raster" in other_outputs.keys():
        output_drop_name = "out_drop_raster" + "_" + _id_generator()
    return _flow_direction(
        input_surface_raster,
        forceFlow,
        flowDirectionType,
        output_flow_direction_raster,
        output_drop_name,
        gis=gis,
        future=future,
        **kwargs,
    )


def _calculate_travel_cost_analytics_converter(
    raster_function,
    output_name=None,
    other_outputs=None,
    gis=None,
    future=False,
    **kwargs,
):
    input_source = None
    input_cost_raster = None
    input_surface_raster = None
    maximum_distance = None
    input_horizonal_raster = None
    horizontal_factor = None
    input_vertical_raster = None
    vertical_factor = None
    source_cost_multiplier = None
    source_start_cost = None
    source_resistance_rate = None
    source_capacity = None
    source_direction = None
    allocation_field = None
    output_backlink_name = None
    output_allocation_name = None
    output_distance_name = None

    if raster_function["rasterFunctionArguments"]["in_source_data"] is not None:
        input_source = raster_function["rasterFunctionArguments"]["in_source_data"]
    if "in_cost_raster" in raster_function["rasterFunctionArguments"].keys():
        input_cost_raster = raster_function["rasterFunctionArguments"]["in_cost_raster"]
    if "in_surface_raster" in raster_function["rasterFunctionArguments"].keys():
        input_surface_raster = raster_function["rasterFunctionArguments"][
            "in_surface_raster"
        ]
    if "maximum_distance" in raster_function["rasterFunctionArguments"].keys():
        maximum_distance = raster_function["rasterFunctionArguments"][
            "maximum_distance"
        ]
    if "in_horizontal_raster" in raster_function["rasterFunctionArguments"].keys():
        input_horizonal_raster = raster_function["rasterFunctionArguments"][
            "in_horizontal_raster"
        ]
    if "horizontal_factor" in raster_function["rasterFunctionArguments"].keys():
        horizontal_factor = raster_function["rasterFunctionArguments"][
            "horizontal_factor"
        ]
    if "in_vertical_raster" in raster_function["rasterFunctionArguments"].keys():
        input_vertical_raster = raster_function["rasterFunctionArguments"][
            "in_vertical_raster"
        ]
    if "vertical_factor" in raster_function["rasterFunctionArguments"].keys():
        vertical_factor = raster_function["rasterFunctionArguments"]["vertical_factor"]
    if "source_cost_multiplier" in raster_function["rasterFunctionArguments"].keys():
        source_cost_multiplier = raster_function["rasterFunctionArguments"][
            "source_cost_multiplier"
        ]
    if "source_start_cost" in raster_function["rasterFunctionArguments"].keys():
        source_start_cost = raster_function["rasterFunctionArguments"][
            "source_start_cost"
        ]
    if "source_resistance_rate" in raster_function["rasterFunctionArguments"].keys():
        source_resistance_rate = raster_function["rasterFunctionArguments"][
            "source_resistance_rate"
        ]
    if "source_capacity" in raster_function["rasterFunctionArguments"].keys():
        source_capacity = raster_function["rasterFunctionArguments"]["source_capacity"]
    if "source_direction" in raster_function["rasterFunctionArguments"].keys():
        source_direction = raster_function["rasterFunctionArguments"][
            "source_direction"
        ]
    if "allocation_field" in raster_function["rasterFunctionArguments"].keys():
        allocation_field = raster_function["rasterFunctionArguments"][
            "allocation_field"
        ]
    output_distance_name = output_name

    if "out_backlink_raster" in other_outputs.keys():
        if other_outputs["out_backlink_raster"] is True:
            output_backlink_name = "out_backlink" + "_" + _id_generator()

    if "out_allocation_raster" in other_outputs.keys():
        if other_outputs["out_allocation_raster"] is True:
            output_allocation_name = "out_allocation" + "_" + _id_generator()

    return _calculate_travel_cost(
        input_source,
        input_cost_raster,
        input_surface_raster,
        maximum_distance,
        input_horizonal_raster,
        horizontal_factor,
        input_vertical_raster,
        vertical_factor,
        source_cost_multiplier,
        source_start_cost,
        source_resistance_rate,
        source_capacity,
        source_direction,
        allocation_field,
        output_distance_name,
        output_backlink_name,
        output_allocation_name,
        gis=gis,
        future=future,
        **kwargs,
    )


def _calculate_distance_analytics_converter(
    raster_function,
    output_name=None,
    other_outputs=None,
    gis=None,
    future=False,
    **kwargs,
):
    input_source = None
    maximum_distance = None
    output_cell_size = None
    allocation_field = None
    distance_method = None
    output_allocation_name = None
    output_direction_name = None
    output_distance_name = None
    output_back_direction_name = None
    in_barrier_data = None

    if raster_function["rasterFunctionArguments"]["in_source_data"] is not None:
        input_source = raster_function["rasterFunctionArguments"]["in_source_data"]
    if "maximum_distance" in raster_function["rasterFunctionArguments"].keys():
        maximum_distance = raster_function["rasterFunctionArguments"][
            "maximum_distance"
        ]
    if "allocation_field" in raster_function["rasterFunctionArguments"].keys():
        allocation_field = raster_function["rasterFunctionArguments"][
            "allocation_field"
        ]
    if "output_cell_size" in raster_function["rasterFunctionArguments"].keys():
        output_cell_size = raster_function["rasterFunctionArguments"][
            "output_cell_size"
        ]
    if "distance_method" in raster_function["rasterFunctionArguments"].keys():
        distance_method = raster_function["rasterFunctionArguments"]["distance_method"]
    if "in_barrier_data" in raster_function["rasterFunctionArguments"].keys():
        in_barrier_data = raster_function["rasterFunctionArguments"]["in_barrier_data"]

    output_distance_name = output_name

    if "out_direction_raster" in other_outputs.keys():
        if other_outputs["out_direction_raster"] is True:
            output_direction_name = "out_direction" + "_" + _id_generator()

    if "out_allocation_raster" in other_outputs.keys():
        if other_outputs["out_allocation_raster"] is True:
            output_allocation_name = "out_allocation" + "_" + _id_generator()

    if "out_back_direction_raster" in other_outputs.keys():
        if other_outputs["out_back_direction_raster"] is True:
            output_back_direction_name = "out_back_direction" + "_" + _id_generator()

    return _calculate_distance(
        input_source,
        maximum_distance,
        output_cell_size,
        allocation_field,
        output_distance_name,
        output_direction_name,
        output_allocation_name,
        in_barrier_data,
        output_back_direction_name,
        distance_method,
        gis=gis,
        future=future,
        **kwargs,
    )


def _distance_accumulation_analytics_converter(
    raster_function, output_name=None, other_outputs=None, gis=None, **kwargs
):
    in_source_data = None
    in_barrier_data = None
    in_surface_raster = None
    in_cost_raster = None
    in_vertical_raster = None
    vertical_factor = None
    in_horizontal_raster = None
    horizontal_factor = None
    source_initial_accumulation = None
    source_maximum_accumulation = None
    source_cost_multiplier = None
    source_direction = None
    distance_method = None
    output_back_direction_raster_name = None
    output_source_direction_raster_name = None
    output_source_location_raster_name = None

    if "in_source_data" in raster_function["rasterFunctionArguments"].keys():
        in_source_data = raster_function["rasterFunctionArguments"]["in_source_data"]
    if "in_barrier_data" in raster_function["rasterFunctionArguments"].keys():
        in_barrier_data = raster_function["rasterFunctionArguments"]["in_barrier_data"]
    if "in_surface_raster" in raster_function["rasterFunctionArguments"].keys():
        in_surface_raster = raster_function["rasterFunctionArguments"][
            "in_surface_raster"
        ]
    if "in_cost_raster" in raster_function["rasterFunctionArguments"].keys():
        in_cost_raster = raster_function["rasterFunctionArguments"]["in_cost_raster"]
    if "in_vertical_raster" in raster_function["rasterFunctionArguments"].keys():
        in_vertical_raster = raster_function["rasterFunctionArguments"][
            "in_vertical_raster"
        ]
    if "vertical_factor" in raster_function["rasterFunctionArguments"].keys():
        vertical_factor = raster_function["rasterFunctionArguments"]["vertical_factor"]
    if "in_horizontal_raster" in raster_function["rasterFunctionArguments"].keys():
        in_horizontal_raster = raster_function["rasterFunctionArguments"][
            "in_horizontal_raster"
        ]
    if "horizontal_factor" in raster_function["rasterFunctionArguments"].keys():
        horizontal_factor = raster_function["rasterFunctionArguments"][
            "horizontal_factor"
        ]
    if (
        "source_initial_accumulation"
        in raster_function["rasterFunctionArguments"].keys()
    ):
        source_initial_accumulation = raster_function["rasterFunctionArguments"][
            "source_initial_accumulation"
        ]
    if (
        "source_maximum_accumulation"
        in raster_function["rasterFunctionArguments"].keys()
    ):
        source_maximum_accumulation = raster_function["rasterFunctionArguments"][
            "source_maximum_accumulation"
        ]
    if "source_cost_multiplier" in raster_function["rasterFunctionArguments"].keys():
        source_cost_multiplier = raster_function["rasterFunctionArguments"][
            "source_cost_multiplier"
        ]
    if "source_direction" in raster_function["rasterFunctionArguments"].keys():
        source_direction = raster_function["rasterFunctionArguments"][
            "source_direction"
        ]
    if "distance_method" in raster_function["rasterFunctionArguments"].keys():
        distance_method = raster_function["rasterFunctionArguments"]["distance_method"]

    output_distance_accumulation_raster_name = output_name

    if "output_back_direction_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_back_direction_raster_name"], bool):
            if other_outputs["output_back_direction_raster_name"] is True:
                output_back_direction_raster_name = (
                    "output_back_direction_raster" + "_" + _id_generator()
                )
        else:
            output_back_direction_raster_name = other_outputs[
                "output_back_direction_raster_name"
            ]

    if "output_source_direction_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_source_direction_raster_name"], bool):
            if other_outputs["output_source_direction_raster_name"] is True:
                output_source_direction_raster_name = (
                    "output_source_direction_raster" + "_" + _id_generator()
                )
        else:
            output_source_direction_raster_name = other_outputs[
                "output_source_direction_raster_name"
            ]

    if "output_source_location_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_source_location_raster_name"], bool):
            if other_outputs["output_source_location_raster_name"] is True:
                output_source_location_raster_name = (
                    "output_source_location_raster" + "_" + _id_generator()
                )
        else:
            output_source_location_raster_name = other_outputs[
                "output_source_location_raster_name"
            ]

    return _distance_accumulation(
        input_source_raster_or_features=in_source_data,
        input_barrier_raster_or_features=in_barrier_data,
        input_surface_raster=in_surface_raster,
        input_cost_raster=in_cost_raster,
        input_vertical_raster=in_vertical_raster,
        vertical_factor=vertical_factor,
        input_horizontal_raster=in_horizontal_raster,
        horizontal_factor=horizontal_factor,
        source_initial_accumulation=source_initial_accumulation,
        source_maximum_accumulation=source_maximum_accumulation,
        source_cost_multiplier=source_cost_multiplier,
        source_direction=source_direction,
        distance_method=distance_method,
        output_distance_accumulation_raster_name=output_distance_accumulation_raster_name,
        output_back_direction_raster_name=output_back_direction_raster_name,
        output_source_direction_raster_name=output_source_direction_raster_name,
        output_source_location_raster_name=output_source_location_raster_name,
        gis=gis,
        **kwargs,
    )


def _distance_allocation_analytics_converter(
    raster_function, output_name=None, other_outputs=None, gis=None, **kwargs
):
    in_source_data = None
    in_barrier_data = None
    in_surface_raster = None
    in_cost_raster = None
    in_vertical_raster = None
    vertical_factor = None
    in_horizontal_raster = None
    horizontal_factor = None
    source_initial_accumulation = None
    source_maximum_accumulation = None
    source_cost_multiplier = None
    source_direction = None
    distance_method = None
    output_back_direction_raster_name = None
    output_source_direction_raster_name = None
    output_source_location_raster_name = None
    output_distance_accumulation_raster_name = None

    if "in_source_data" in raster_function["rasterFunctionArguments"].keys():
        in_source_data = raster_function["rasterFunctionArguments"]["in_source_data"]
    if "in_barrier_data" in raster_function["rasterFunctionArguments"].keys():
        in_barrier_data = raster_function["rasterFunctionArguments"]["in_barrier_data"]
    if "in_surface_raster" in raster_function["rasterFunctionArguments"].keys():
        in_surface_raster = raster_function["rasterFunctionArguments"][
            "in_surface_raster"
        ]
    if "in_cost_raster" in raster_function["rasterFunctionArguments"].keys():
        in_cost_raster = raster_function["rasterFunctionArguments"]["in_cost_raster"]
    if "in_vertical_raster" in raster_function["rasterFunctionArguments"].keys():
        in_vertical_raster = raster_function["rasterFunctionArguments"][
            "in_vertical_raster"
        ]
    if "vertical_factor" in raster_function["rasterFunctionArguments"].keys():
        vertical_factor = raster_function["rasterFunctionArguments"]["vertical_factor"]
    if "in_horizontal_raster" in raster_function["rasterFunctionArguments"].keys():
        in_horizontal_raster = raster_function["rasterFunctionArguments"][
            "in_horizontal_raster"
        ]
    if "horizontal_factor" in raster_function["rasterFunctionArguments"].keys():
        horizontal_factor = raster_function["rasterFunctionArguments"][
            "horizontal_factor"
        ]
    if (
        "source_initial_accumulation"
        in raster_function["rasterFunctionArguments"].keys()
    ):
        source_initial_accumulation = raster_function["rasterFunctionArguments"][
            "source_initial_accumulation"
        ]
    if (
        "source_maximum_accumulation"
        in raster_function["rasterFunctionArguments"].keys()
    ):
        source_maximum_accumulation = raster_function["rasterFunctionArguments"][
            "source_maximum_accumulation"
        ]
    if "source_cost_multiplier" in raster_function["rasterFunctionArguments"].keys():
        source_cost_multiplier = raster_function["rasterFunctionArguments"][
            "source_cost_multiplier"
        ]
    if "source_direction" in raster_function["rasterFunctionArguments"].keys():
        source_direction = raster_function["rasterFunctionArguments"][
            "source_direction"
        ]
    if "distance_method" in raster_function["rasterFunctionArguments"].keys():
        distance_method = raster_function["rasterFunctionArguments"]["distance_method"]

    output_distance_allocation_raster_name = output_name

    if "output_distance_accumulation_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_distance_accumulation_raster_name"], bool):
            if other_outputs["output_distance_accumulation_raster_name"] is True:
                output_distance_accumulation_raster_name = (
                    "output_distance_accumulation_raster" + "_" + _id_generator()
                )
        else:
            output_distance_accumulation_raster_name = other_outputs[
                "output_distance_accumulation_raster_name"
            ]

    if "output_back_direction_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_back_direction_raster_name"], bool):
            if other_outputs["output_back_direction_raster_name"] is True:
                output_back_direction_raster_name = (
                    "output_back_direction_raster" + "_" + _id_generator()
                )
        else:
            output_back_direction_raster_name = other_outputs[
                "output_back_direction_raster_name"
            ]

    if "output_source_direction_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_source_direction_raster_name"], bool):
            if other_outputs["output_source_direction_raster_name"] is True:
                output_source_direction_raster_name = (
                    "output_source_direction_raster" + "_" + _id_generator()
                )
        else:
            output_source_direction_raster_name = other_outputs[
                "output_source_direction_raster_name"
            ]

    if "output_source_location_raster_name" in other_outputs.keys():
        if isinstance(other_outputs["output_source_location_raster_name"], bool):
            if other_outputs["output_source_location_raster_name"] is True:
                output_source_location_raster_name = (
                    "output_source_location_raster" + "_" + _id_generator()
                )
        else:
            output_source_location_raster_name = other_outputs[
                "output_source_location_raster_name"
            ]

    return _distance_allocation(
        input_source_raster_or_features=in_source_data,
        input_barrier_raster_or_features=in_barrier_data,
        input_surface_raster=in_surface_raster,
        input_cost_raster=in_cost_raster,
        input_vertical_raster=in_vertical_raster,
        vertical_factor=vertical_factor,
        input_horizontal_raster=in_horizontal_raster,
        horizontal_factor=horizontal_factor,
        source_initial_accumulation=source_initial_accumulation,
        source_maximum_accumulation=source_maximum_accumulation,
        source_cost_multiplier=source_cost_multiplier,
        source_direction=source_direction,
        distance_method=distance_method,
        output_distance_allocation_raster_name=output_distance_allocation_raster_name,
        output_distance_accumulation_raster_name=output_distance_accumulation_raster_name,
        output_back_direction_raster_name=output_back_direction_raster_name,
        output_source_direction_raster_name=output_source_direction_raster_name,
        output_source_location_raster_name=output_source_location_raster_name,
        gis=gis,
        **kwargs,
    )


def _return_output(num_returns, output_dict, return_value_names):
    if num_returns == 1:
        return output_dict[return_value_names[0]]

    else:
        ret_names = []
        for return_value in return_value_names:
            ret_names.append(return_value)
        NamedTuple = collections.namedtuple("FunctionOutput", ret_names)
        function_output = NamedTuple(**output_dict)
        return function_output


def _set_output_raster(output_name, task, gis, output_properties=None):
    output_service = None
    output_raster = None

    if task == "GenerateRaster":
        task_name = "GeneratedRasterProduct"
    else:
        task_name = task

    folder = None
    folderId = None

    if output_properties is not None:
        if "folder" in output_properties:
            folder = output_properties["folder"]
    if folder is not None:
        if isinstance(folder, dict):
            if "id" in folder:
                folderId = folder["id"]
                folder = folder["title"]
        else:
            owner = gis.properties.user.username
            folderId = gis._portal.get_folder_id(owner, folder)
        if folderId is None:
            folder_dict = gis.content.create_folder(folder, owner)
            folder = folder_dict["title"]
            folderId = folder_dict["id"]

    if output_name is None:
        output_name = str(task_name) + "_" + _id_generator()
        output_service = _create_output_image_service(
            gis, output_name, task, folder=folder
        )
        output_raster = {
            "serviceProperties": {
                "name": output_service.name,
                "serviceUrl": output_service.url,
            },
            "itemProperties": {"itemId": output_service.itemid},
        }
    elif isinstance(output_name, str):
        output_service = _create_output_image_service(
            gis, output_name, task, folder=folder
        )
        output_raster = {
            "serviceProperties": {
                "name": output_service.name,
                "serviceUrl": output_service.url,
            },
            "itemProperties": {"itemId": output_service.itemid},
        }
    elif isinstance(output_name, _arcgis.gis.Item):
        output_service = output_name
        output_raster = {"itemProperties": {"itemId": output_service.itemid}}
    else:
        raise TypeError("output_raster should be a string (service name) or Item")

    if folderId is not None:
        output_raster["itemProperties"].update({"folderId": folderId})
    output_raster = _json.dumps(output_raster)
    return output_raster, output_service


def _save_ra(
    raster_function,
    output_name=None,
    other_outputs=None,
    gis=None,
    future=False,
    **kwargs,
):
    if raster_function["rasterFunctionArguments"]["toolName"] == "FlowDirection_sa":
        return _flow_direction_analytics_converter(
            raster_function,
            output_name=output_name,
            other_outputs=other_outputs,
            gis=gis,
            future=future,
            **kwargs,
        )
    if (
        raster_function["rasterFunctionArguments"]["toolName"]
        == "CalculateTravelCost_sa"
    ):
        return _calculate_travel_cost_analytics_converter(
            raster_function,
            output_name=output_name,
            other_outputs=other_outputs,
            gis=gis,
            future=future,
            **kwargs,
        )
    if raster_function["rasterFunctionArguments"]["toolName"] == "CalculateDistance_sa":
        return _calculate_distance_analytics_converter(
            raster_function,
            output_name=output_name,
            other_outputs=other_outputs,
            gis=gis,
            future=future,
            **kwargs,
        )
    if (
        raster_function["rasterFunctionArguments"]["toolName"]
        == "DistanceAccumulation_sa"
    ):
        return _distance_accumulation_analytics_converter(
            raster_function,
            output_name=output_name,
            other_outputs=other_outputs,
            gis=gis,
            **kwargs,
        )
    if (
        raster_function["rasterFunctionArguments"]["toolName"]
        == "DistanceAllocation_sa"
    ):
        return _distance_allocation_analytics_converter(
            raster_function,
            output_name=output_name,
            other_outputs=other_outputs,
            gis=gis,
            **kwargs,
        )


def _build_param_dictionary(
    gis,
    params,
    input_rasters,
    raster_type_name,
    raster_type_params=None,
    image_collection_properties=None,
    use_input_rasters_by_ref=False,
):
    inputRasterSpecified = False
    # input rasters
    if isinstance(input_rasters, list):
        # extract the IDs of all the input items
        # and then convert the list to JSON
        item_id_list = []
        url_list = []
        uri_list = []
        for item in input_rasters:
            if isinstance(item, Item):
                item_id_list.append(item.itemid)
            elif isinstance(item, str):
                if "http:" in item or "https:" in item:
                    url_list.append(item)
                else:
                    uri_list.append(item)

        if len(item_id_list) > 0:
            params["inputRasters"] = {"itemIds": item_id_list}
            inputRasterSpecified = True
        elif len(url_list) > 0:
            params["inputRasters"] = {"urls": url_list}
            inputRasterSpecified = True
        elif len(uri_list) > 0:
            params["inputRasters"] = {"uris": uri_list}
            inputRasterSpecified = True
    elif isinstance(input_rasters, str):
        # the input_rasters is a folder name; try and extract the folderID
        owner = gis.properties.user.username
        folderId = gis._portal.get_folder_id(owner, input_rasters)
        if folderId is None:
            if "http:" in input_rasters or "https:" in input_rasters:
                params["inputRasters"] = {"url": input_rasters}
            else:
                params["inputRasters"] = {"uri": input_rasters}
        else:
            params["inputRasters"] = {"folderId": folderId}
        inputRasterSpecified = True

    if inputRasterSpecified is False:
        raise RuntimeError(
            "Input raster list to be added to the collection must be specified"
        )
    else:
        if use_input_rasters_by_ref:
            params["inputRasters"].update({"byref": True})

    # raster_type
    if not isinstance(raster_type_name, str):
        raise RuntimeError("Invalid input raster_type parameter")

    elevation_set = 0
    if raster_type_params is not None:
        for element in raster_type_params.keys():
            if element.lower() == "constantz":
                value = raster_type_params[element]
                del raster_type_params[element]
                raster_type_params.update({"ConstantZ": value})

                elevation_set = 1
                break
            elif element.lower() == "averagezdem":
                value = raster_type_params[element]
                del raster_type_params[element]
                raster_type_params.update({"averagezdem": value})
                elevation_set = 1
                break

        if elevation_set == 0:
            if "orthomappingElevation" in gis.properties.helperServices.keys():
                raster_type_params["averagezdem"] = gis.properties.helperServices[
                    "orthomappingElevation"
                ]
            else:
                raster_type_params["averagezdem"] = {
                    "url": "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer"
                }
    else:
        if "orthomappingElevation" in gis.properties.helperServices.keys():
            raster_type_params = {
                "averagezdem": gis.properties.helperServices["orthomappingElevation"]
            }
        else:
            raster_type_params = {
                "averagezdem": {
                    "url": "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer"
                }
            }

    params["rasterType"] = {
        "rasterTypeName": raster_type_name,
        "rasterTypeParameters": raster_type_params,
    }
    if image_collection_properties is not None:
        if "rasterType" in params:
            params["rasterType"].update(
                {"imageCollectionProps": image_collection_properties}
            )

    params["rasterType"] = _json.dumps(params["rasterType"])
    return


###################################################################################################
###################################################################################################
def _set_image_collection_param(gis, params, image_collection):
    if isinstance(image_collection, str):
        # doesnotexist = gis.content.is_service_name_available(image_collection, "Image Service")
        # if doesnotexist:
        # raise RuntimeError("The input image collection does not exist")
        if "http:" in image_collection or "https:" in image_collection:
            params["imageCollection"] = _json.dumps({"url": image_collection})
        else:
            params["imageCollection"] = _json.dumps({"uri": image_collection})
    elif isinstance(image_collection, Item):
        params["imageCollection"] = _json.dumps({"itemId": image_collection.itemid})
    else:
        raise TypeError("image_collection should be a string (url or uri) or Item")

    return


# def monitor_vegetation(input_raster,
#                        method_to_use='NDVI',
#                        nir_band=1,
#                        red_band=2,
#                        options={},
#                        output_name=None,
#                        gis=None):
#     """
#
#     :param input_raster: multiband raster layer. Make sure the input raster has the appropriate bands available.
#
#     :param method_to_use: one of NDVI, GEMI, GVI, PVI, SAVI, MSAVI2, TSAVI, SULTAN.
#          the method used to create the vegetation index layer. The different vegetation indexes can help highlight
#          certain features, or help reduce various noise.
#
#         * GEMI - Global Environmental Monitoring Index - GEMI is a nonlinear vegetation index for global environmental
#             monitoring from satellite imagery. It is similar to NDVI, but it is less sensitive to atmospheric
#             effects. It is affected by bare soil; therefore, it is not recommended for use in areas of sparse or
#             moderately dense vegetation.
#         * GVI - Green Vegetation Index - Landsat TM - GVI was originally designed from Landsat MSS imagery but has been
#             modified for use with Landsat TM imagery. It is also known as the Landsat TM Tasseled Cap green
#             vegetation index. This monitoring index can also be used with imagery whose bands share the same
#             spectral characteristics.
#         * MSAVI2 - Modified Soil Adjusted Vegetation Index - MSAVI2 is a vegetation index that tries to minimize bare soil
#             influences of the SAVI method.
#         * NDVI - Normalized Difference Vegetation Index - NDVI is a standardized index allowing you to generate an image
#             displaying greenness, relative biomass. This index takes advantage of the contrast of the
#             characteristics of two bands from a multispectral raster dataset; the chlorophyll pigment absorptions
#             in the red band and the high reflectivity of plant materials in the near-infrared (NIR) band.
#         * PVI - Perpendicular Vegetation Index - PVI is similar to a difference vegetation index; however, it is sensitive
#             to atmospheric variations. When using this method to compare different images, it should only be used on
#             images that have been atmospherically corrected. This information can be provided by your data vendor.
#         * SAVI - Soil-Adjusted Vegetation Index - SAVI is a vegetation index that attempts to minimize soil brightness
#             influences using a soil-brightness correction factor. This is often used in arid regions where
#             vegetative cover is low.
#         * SULTAN - Sultan's Formula - The Sultan's Formula process takes a six-band 8-bit image and applied a specific
#             algorithm to it to produce a three-band 8-bit image. The resulting image highlights rock formations
#             called ophiolites on coastlines. This formula was designed based on the TM and ETM bands of a Landsat 5
#             or 7 scene.
#         * TSAVI - Transformed Soil-Adjusted Vegetation Index - Transformed-SAVI is a vegetation index that attempts to
#             minimize soil brightness influences by assuming the soil line has an arbitrary slope and intercept.
#
#     :param nir_band: the band indexes for the near-infrared (NIR) band.
#     :param red_band: the band indexes for the Red band.
#     :param options: additional parameters such as slope, intercept
#         * intercept is the value of near infrared (NIR) when the reflection value of the red (Red) band is 0 for the particular soil lines.
#         (a = NIR - sRed) , when Red is 0.
#         This parameter is only valid for Transformed Soil-Adjusted Vegetation Index.
#
#         * slope - Slope of soil line
#         The slope of the soil line. The slope is the approximate linear relationship between the NIR and red bands on a scatterplot.
#         This parameter is only valid for Transformed Soil-Adjusted Vegetation Index.
#
#         *
#     :param output_name:
#     :param gis:
#     :return:
#     """
#     NDVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 1, "BandIndexes": "1 2"}}
#
#     GEMI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 5, "BandIndexes": "1 2 3 4 5 6"}}
#
#     GVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 7, "BandIndexes": "1 2"}}
#
#     MSAVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 4, "BandIndexes": "1 2"}}
#
#     PVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 6, "BandIndexes": "1 2 111 222"}}
#
#     SAVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 2, "BandIndexes": "1 2 111"}}
#
#     SULTAN
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 8, "BandIndexes": "1 2 3 4 5 6"}}
#
#     TSAVI
#     {"rasterFunction": "BandArithmetic", "rasterFunctionArguments": {"Method": 3, "BandIndexes": "1 2 111 222 333"}}
#
#     raster_function = {"rasterFunction":"BandArithmetic","rasterFunctionArguments":{"Method":1,"BandIndexes":"1 2"}}
#
#     function_args = {'Raster': _layer_input(input_raster)}
#
#     return generate_raster(raster_function, function_args, output_name=output_name, gis=gis)


def generate_raster(
    raster_function,
    function_arguments: Optional[dict[str, Any]] = None,
    output_raster_properties: Optional[dict[str, Any]] = None,
    output_name: Optional[str] = None,
    process_as_multidimensional: Optional[bool] = None,
    build_transpose: Optional[bool] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_generate_raster/ra_generate_raster.png

    Function allows you to execute raster analysis on a distributed server deployment.

    ====================================     ==================================================================================================================
    **Parameter**                             **Description**
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    raster_function                          Required, Raster function to perform analysis on the input raster dataset.
                                             The value can be a string keyword for predefined raster functions such as
                                             NDVI, a JSON object that describes a raster function chain with a built-in
                                             functions that are known to the server, or the contents of a raster
                                             function template file (*.rft.xml).

                                             Please refer to the complete list of Raster Analysis functions to execute
                                             on a distributed server:
                                             `Raster function objects <https://developers.arcgis.com/documentation/common-data-types/raster-function-objects.htm>`_
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    function_arguments                       Optional, The dict to specify the raster function arguments' value.
                                             It is optional because the argument value can also be defined in the function template.
                                             The function_arguments parameter supports the RasterInfo argument for all raster functions.
                                             The information stored in RasterInfo allows you to specify the output raster dataset's
                                             properties such as cell size, extent, and nodata..

                                             Example:

                                                {"Raster": {"url": <image service url>}, "ResamplingType": 1}

                                             For specifying input Raster alone, portal Item can be passed. (i.e, parameter with name "Raster")
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    output_raster_properties                 Optional dict, can be used to set the output raster's key metadata properties.

                                             Example:

                                                {"SensorName": "Landsat 8", "CloudCover": 20}
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    process_as_multidimensional              Optional bool, Process as multidimensional if set to True,
                                             if the input is multidimensional raster.
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    build_transpose                          Optional bool, if set to true, transforms the output multidimensional
                                             raster. Valid only if ``process_as_multidimensional`` is set to True.
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    context                                  Optional dict, context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                | {"extent": {"xmin": -122.68,
                                                | "ymin": 45.53,
                                                | "xmax": -122.45,
                                                | "ymax": 45.6,
                                                | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                   {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                   {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                   {"mask": {"url": "<image_service_url>"}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                   {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                   {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                   {"parallelProcessingFactor": "60%"}

                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: BILINEAR, NEAREST, CUBIC.

                                               Example:

                                                   {'resamplingMethod': "NEAREST"}

    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     ------------------------------------------------------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter on ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ==================================================================================================================

    :return:
    output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: Performs raster analysis on a distributed server deployment.

        generate_raster_op = generate_raster(raster_function={"rasterFunction":"Grayscale"},
                                             function_arguments = {"Raster": {"url":<url>}
                                             output_name="output_name",
                                             process_as_multidimensional=True
                                             gis=gis,
                                             tiles_only=True)
    """
    gis = _arcgis.env.active_gis if gis is None else gis

    if context is None:
        context = {}
    if process_as_multidimensional is not None:
        context.update({"processAsMultidimensional": process_as_multidimensional})
    if build_transpose is not None:
        context.update({"buildTranspose": build_transpose})

    return gis._tools.rasteranalysis.generate_raster(
        raster_function=raster_function,
        function_arguments=function_arguments,
        output_raster_properties=output_raster_properties,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


def convert_feature_to_raster(
    input_feature: FeatureLayer,
    output_cell_size: dict[str, Any],
    value_field: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_convert_feature_to_raster/ra_convert_feature_to_raster.png

    Creates a new ImageryLayer from an existing feature layer.
    Any feature layer containing point, line, or polygon features can be converted to an ImageryLayer.

    The cell center is used to decide the value of the output raster pixel. The input field type determines
    the type of output raster. If the field is integer, the output raster will be integer;
    if it is floating point, the output will be floating point.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_feature                            Required feature layer. The input feature layer to convert to a raster dataset.
    ------------------------------------     --------------------------------------------------------------------
    output_cell_size                         Required dict. The cell size and unit for the output imagery layer.

                                             The available units are Feet, Miles, Meters, and Kilometers.

                                             Example

                                                {"distance":60,"units":"meters"}
    ------------------------------------     --------------------------------------------------------------------
    value_field                              Optional string.  The field that will be used to assign values to the
                                             output raster.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.

                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                   {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                   {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                   {"mask": {"url": "<image_service_url>"}}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                   {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                   {"parallelProcessingFactor": "60%"}

    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return:
    output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: Convert the feature layer into an imagery layer.

        feature_layer = gis.content.search("feature_layer_item", item_type="Feature Layer Collection")[0].layers[0]

        raster = conver_feature_to_raster(input_feature=feature_layer,
                                          output_cell_size={"distance": 60,"units": "meters"},
                                          gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.convert_feature_to_raster(
        input_feature=input_feature,
        output_cell_size=output_cell_size,
        output_name=output_name,
        value_field=value_field,
        context=context,
        future=future,
        **kwargs,
    )


def copy_raster(
    input_raster,
    output_cellsize: Optional[dict[str, Any]] = None,
    resampling_method: str = "NEAREST",
    clip_setting: Optional[str] = None,
    output_name: Optional[str] = None,
    process_as_multidimensional: Optional[bool] = None,
    build_transpose: Optional[bool] = None,
    context: Optional[dict[str, Any]] = None,
    raster_type_name: Optional[str] = None,
    raster_type_params: Optional[dict[str, Any]] = None,
    source_mosaic_dataset: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_copy_raster/ra_copy_raster.png 

    The Copy Raster task takes single raster input and generates the output image using parallel processing.
    The input raster can be clipped, resampled, and reprojected based on the setting.

    The function can also create hosted imagery layers in ArcGIS Enterprise and ArcGIS Online from local raster datasets by uploading the data to the server.
    Multiple images are mosaicked into a single dataset to create one layer.
    
    For this functionality to work in ArcGIS Online, Azure library packages for Python (Azure SDK for Python - azure-storage-blob: 12.1<= version <=12.17)
    needs to be pre-installed. Refer https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package

    ================================     ====================================================================
    **Parameter**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_raster                         Required raster layer or string or list. The input raster layer to be copied to.
                                         Path to a local raster dataset(s) can also be given to create hosted imagery 
                                         layers on enterprise and AGOL.
    --------------------------------     --------------------------------------------------------------------
    output_cellsize                      Required dict. The cell size and unit for the output imagery layer.
                                         The available units are Feet, Miles, Meters, and Kilometers.

                                         Example:

                                            {"distance":60,"units":meters}
    --------------------------------     --------------------------------------------------------------------
    resampling_method                    Optional string.  The field that will be used to assign values to the
                                         output raster.
    --------------------------------     --------------------------------------------------------------------
    clip_setting                         Optional string.  The field that will be used to assign values to the
                                         output raster.
    --------------------------------     --------------------------------------------------------------------
    output_name                          Optional String. If not provided, an Image Service is created by the method 
                                         and used as the output raster.

                                         You can pass in an existing Image Service Item from your GIS to use 
                                         that instead.

                                         Alternatively, you can pass in the name of the output Image Service 
                                         that should be created by this method to be used as the output for 
                                         the tool.
                                         A RuntimeError is raised if a service by that name already exists
    --------------------------------     --------------------------------------------------------------------
    process_as_multidimensional          Optional bool, Process as multidimensional if set to True, 
                                         if the input is multidimensional raster.
    --------------------------------     --------------------------------------------------------------------
    build_transpose                      Optional bool, if set to true, transforms the output multidimensional 
                                         raster. Valid only if ``process_as_multidimensional`` is set to True.
    --------------------------------     --------------------------------------------------------------------
    context                              context contains additional settings that affect task execution.

                                         context parameter overwrites values set through arcgis.env parameter
                                         
                                         This function has the following settings:

                                         - Output Spatial Reference (outSR): The output raster will be projected into the output spatial reference.
                                                
                                           Example:

                                                {"outSR": {spatial reference}}


                                         - Upload Properties (upload_properties): ``upload_properties`` key can be used to control specific upload parameters when trying to create hosted imagery layers in ArcGIS Online from local raster datasets.

                                           Available options:

                                           - ``maxUploadConcurrency``: Optional integer. Maximum number of parallel connections \
                                             to use for large uploads (when individual file/blob size exceeds 64MB). \
                                             This is the **max_concurrency** parameter of the `BlobClient.upload_blob() <https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#upload-blob-data--blob-type--blobtype-blockblob---blockblob----length-none--metadata-none----kwargs->`__ method. \
                                             (The default is 6)
                                           - ``maxWorkerThreads``: Optional integer. Maximum number of threads to execute asynchronously \
                                             when uploading multiple files. This is the **max_workers** parameter of the `ThreadPoolExecutor() <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`__ class. \
                                             (The default is None)
                                           - ``displayProgress``: Optional boolean. If set to True, a progress bar will be \
                                             displayed for tracking the progress of the uploads to user's rasterstore. \
                                             (The default is False)

                                           Example:

                                                | {"upload_properties": {"maxUploadConcurrency":8,
                                                |                       "maxWorkerThreads":20,
                                                |                       "displayProgress":True}
                                                | }

                                         The context parameter can also be used to specify whether to
                                         build footprints, pixel value that represents the NoData,
                                         resamplingMethod etc.


                                         Example:

                                            | {"buildFootprints":True,                                            
                                            | "footprintsArguments":{"method":"RADIOMETRY","minValue":1,"maxValue":5,
                                            |                        "shrinkDistance":50,"skipOverviews":True,"updateBoundary":True,
                                            |                        "maintainEdge":False,"simplification":None,"numVertices":20,
                                            |                        "minThinnessRatio":0.05,"maxSliverSize":20,"requestSize":2000,
                                            |                        "minRegionSize":100},
                                            | "defineNodata":True,                                            
                                            | "noDataArguments":{"noDataValues":[500],"numberOfBand":99,"compositeValue":True},                                            
                                            | "buildOverview":True}
    --------------------------------     --------------------------------------------------------------------
    raster_type_name                     Optional string. The name of the raster type to use for adding data to
                                         the mosaic dataset.


                                         Choice list:

                                             | [
                                             | "Aerial", "ASTER", "DMCII", "DubaiSat-2", "GeoEye-1", "GF-1 PMS", "GF-1 WFV",
                                             | "GF-2 PMS", "GRIB", "HDF", "IKONOS", "Jilin-1", "KOMPSAT-2", "KOMPSAT-3",
                                             | "Landsat 1-5 MSS", "Landsat 4-5 TM", "Landsat 7 ETM+", "Landsat 8", "Landsat 9"
                                             | "NetCDF", "PlanetScope", "Pleiades-1", "Pleiades NEO", "QuickBird", "RapidEye",
                                             | "Raster Dataset", "ScannedAerial", "Sentinel-2", "SkySat", "SPOT 5", "SPOT 6",
                                             | "SPOT 7", "Superview-1", "Tiled Imagery Layer", "UAV/UAS", "WordView-1",
                                             | "WordView-2", "WordView-3", "WordView-4", "ZY3-SASMAC", "ZY3-CRESDA"
                                             | ]

                                         If an existing mosaic dataset is being published as an 
                                         imagery layer using the ``source_mosaic_dataset`` parameter, the
                                         ``raster_type_name`` parameter can be set to None as it is not required.

                                         Example:

                                            "QuickBird"
    --------------------------------     --------------------------------------------------------------------
    raster_type_params                   Optional dict. Additional ``raster_type`` specific parameters.
        
                                         The process of add rasters to the image collection can be \
                                         controlled by specifying additional raster type arguments.

                                         The raster type parameters argument is a dictionary.

                                         The dictionary can contain productType, processingTemplate, \
                                         pansharpenType, Filter, pansharpenWeights, ConstantZ, \
                                         dem, zoffset, CorrectGeoid, ZFactor, StretchType, \
                                         ScaleFactor, ValidRange

                                         Please check the table below (Supported Raster Types), \
                                         for more details about the product types, \
                                         processing templates, pansharpen weights for each raster type. 

                                         - Possible values for pansharpenType - ["Mean", "IHS", "Brovey", "Esri", "Mean", "Gram-Schmidt"]
                                         - Possible values for filter - [None, "Sharpen", "SharpenMore"]
                                         - Value for StretchType dictionary can be as follows:

                                           - "None"
                                           - "MinMax; <min>; <max>"
                                           - "PercentMinMax; <MinPercent>; <MaxPercent>"
                                           - "StdDev; <NumberOfStandardDeviation>"
                                           Example: {"StretchType": "MinMax; <min>; <max>"}

                                         - Value for ValidRange dictionary can be as follows:

                                           - "<MaskMinValue>, <MaskMaxValue>"
                                           Example: {"ValidRange": "10, 200"}

                                         Example:

                                            {"productType":"All","processingTemplate":"Pansharpen",
                                            "pansharpenType":"Gram-Schmidt","filter":"SharpenMore",
                                            "pansharpenWeights":"0.85 0.7 0.35 1","constantZ":-9999}
    --------------------------------     --------------------------------------------------------------------
    source_mosaic_dataset                Optional string. Path to the existing mosaic dataset to be published 
                                         as an imagery layer.

                                         To publish an existing mosaic dataset, specify the path to the input 
                                         data of the mosaic in the ``input_raster`` parameter. 
                                         The data will be uploaded to ArcGIS Online.

                                         ``raster_type_name`` parameter can be set to None as it is not required to 
                                         publish an imagery layer from a mosaic dataset.

                                         .. note::
                                                Option available only on ArcGIS online

                                         Example:

                                            "./data/temp_uploaded.gdb/test"
    --------------------------------     --------------------------------------------------------------------
    gis                                  Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                         is used.
    --------------------------------     --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will 
                                         be a GPJob object and results will be returned asynchronously.
    --------------------------------     --------------------------------------------------------------------
    folder                               Keyword only parameter. Optional str or dict. Creates a folder in the 
                                         portal, if it does not exist, with the given folder name and persists 
                                         the output in this folder. The dictionary returned by the 
                                         :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                         Example:

                                               	| {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    --------------------------------     --------------------------------------------------------------------
    tiles_only                           Keyword only parameter. Optional boolean. 
                                         On AGOL, the default output image service for this function would 
                                         be a Tiled Imagery Layer. 

                                         To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                         Function will not honor tiles_only parameter on enterprise and 
                                         will generate Dynamic Imagery Layer by default. 
    ================================     ====================================================================

    :return:
    output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: This example creates a tiled imagery layer in ArcGIS Online. 
        # (To create dynamic imagery layer set the tiles_only keyword argument to False)

        copy_raster_op = copy_raster(input_raster=r"C:\data\input_raster.tif",
                                     output_name="output_name",
                                     raster_type_name="Raster Dataset",
                                     gis=gis,
                                     tiles_only=True)

        # Usage Example 2: This example creates a tiled imagery layer in AGOL from the datasets detected in the input folder. 
        # (To create dynamic imagery layer set the tiles_only keyword argument to False)

        copy_raster_op = copy_raster(input_raster=r"C:\data",
                                     output_name="output_name",
                                     raster_type_name="Raster Dataset",
                                     gis=gis,
                                     tiles_only=True)
        
        # Usage Example 3: This example creates an imagery layer using Landsat 1-5 MSS Raster Type with a Level 1 product and a Multispectral template.

        copy_raster_op_landsat15 = copy_raster(input_raster=r"C:\data\Landsat15MSS",
                                               output_name="landsat15_op",
                                               raster_type_name="Landsat 1-5 MSS",
                                               raster_type_params={"productType": "Level1", "processingTemplate": "Multispectral"},
                                               gis=gis)

        # Usage Example 4: This example creates an imagery layer using Sentinel-2 Raster Type with a Level 1 product and a Multispectral template.

        copy_raster_op_sentinel2 = copy_raster(input_raster=r"C:\data\Sentinel2",
                                               output_name="sentinel2_op",
                                               raster_type_name="Sentinel-2",
                                               raster_type_params={"productType": "Level1", "processingTemplate": "Multispectral-10m"},
                                               gis=gis)

        # Usage Example 5: This example creates an imagery layer using Pleiades-1 Raster Type with an Ortho product and a Pansharpen template.

        copy_raster_op_pleiades1 = copy_raster(input_raster=r"C:\data\Pleiades1",
                                               output_name="pleiades1_op",
                                               raster_type_name="Pleiades-1",
                                               raster_type_params={"productType":"ORTHO","processingTemplate":"Pansharpen"},
                                               gis=gis)

        # Usage Example 6: This example creates an imagery layer using SPOT 6 Raster Type with an Ortho Reflectance product and a Pansharpen Reflectance template.

        copy_raster_op_spot6 = copy_raster(input_raster=r"C:\data\SPOT6",
                                          output_name="spot6_op",
                                          raster_type_name="SPOT 6",
                                          raster_type_params={"productType": "ORTHO REFLECTANCE", "processingTemplate": "Pansharpen Reflectance"},
                                          gis=gis)

        # Usage Example 7: This example creates an imagery layer using WorldView-2 Raster Type with a Pansharpen and Multispectral template.

        copy_raster_op_worldview2 = copy_raster(input_raster=r"C:\data\WorldView",
                                                output_name="worldview2_op",
                                                raster_type_name="WorldView-2",
                                                raster_type_params= {"productType": "Basic", "processingTemplate": "Pansharpen and Multispectral"},
                                                gis=gis)    

        # Usage Example 8: This example creates an imagery layer from Tiled Imagery Layers.

        copy_raster_op_til = copy_raster(input_raster=[tile_lyr_input1, tile_lyr_input2],
                                         output_name="til_op",
                                         raster_type_name="Tiled Imagery Layer", 
                                         gis = gis)

        # Usage Example 9: This example copies the imagery layer and creates a new imagery layer item in the GIS. 
        
        raster = gis.content.search("raster_lyr")[0].layers[0]
        copy_raster_op = copy_raster(input_raster=raster,
                                     output_name="output_name",
                                     gis=gis)

        # Usage Example 10: This example publishes an existing mosaic dataset as a tiled imagery layer in ArcGIS Online.
        # (To create dynamic imagery layer set the tiles_only keyword argument to False)

        # Specify the actual source data path referenced by the uploaded mosaic dataset using input_raster parameter.
        # The data would be uploaded to the ArcGIS Online's user store from this path.
        
        landsat_mosaic = copy_raster(input_raster=[r"C:\data\landsat_data_folder"],
                                     output_name="mosaic_datset_op",
                                     context={"upload_properties":{"displayProgress":True}},
                                     source_mosaic_dataset=r"C:\data\md.gdb\landsat",
                                     gis = gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if context is None:
        context = {}
    if process_as_multidimensional is not None:
        context.update({"processAsMultidimensional": process_as_multidimensional})
    if build_transpose is not None:
        context.update({"buildTranspose": build_transpose})

    return gis._tools.rasteranalysis.copy_raster(
        input_raster=input_raster,
        output_cellsize=output_cellsize,
        resampling_method=resampling_method,
        clip_setting=clip_setting,
        output_name=output_name,
        context=context,
        future=future,
        raster_type_name=raster_type_name,
        raster_type_params=raster_type_params,
        md_to_upload=source_mosaic_dataset,
        **kwargs,
    )


def summarize_raster_within(
    input_zone_layer: FeatureLayer,
    input_raster_layer_to_summarize,
    zone_field: str = "Value",
    statistic_type: str = "Mean",
    ignore_missing_values: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    process_as_multidimensional: bool = False,
    percentile_value: int = 90,
    percentile_interpolation_type: str = "AUTO_DETECT",
    circular_calculation: bool = False,
    circular_wrap_value: float = 360,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_summarize_raster_within/ra_summarize_raster_within.png 

    Summarizes a raster based on areas (zones) defined by the first input layer (input_zone_layer).

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_zone_layer                         Required layer - area layer to summarize a raster layer within defined boundaries.

                                             The layer that defines the boundaries of the areas, or zones, that will be summarized.

                                             The layer can be a raster or feature data. For rasters, the zones are defined by all locations in the input that
    ------------------------------------     --------------------------------------------------------------------
    input_raster_layer_to_summarize          Required  - raster layer to summarize.

                                             The raster cells in this layer will be summarized by the areas (zones) that they fall within.
    ------------------------------------     --------------------------------------------------------------------
    zone_field                               Required string -  field to define the boundaries. This is the attribute of the layer that will be used
                                             to define the boundaries of the areas. For example, suppose the first input layer defines the management unit
                                             boundaries, with attributes that define the region, the district, and the parcel ID of each unit. You also have
                                             a raster layer defining a biodiversity index for each location. With the field you select, you can decide to
                                             calculate the average biodiversity at local, district, or regional levels.

                                             Default: "Value"
    ------------------------------------     --------------------------------------------------------------------
    statistic_type                           Optional string - statistic to calculate.
                                             You can calculate statistics of any numerical attribute of the points, lines, or areas within the input area
                                             layer. 
                                             
                                             statistic_type can be one of the following:
                                             ['Mean', 'Majority', 'Maximum', 'Median', 'Minimum', 'Minority', 'Range', 'STD', 'SUM', 'Variety', 'Percentile']

                                             - Mean: Calculates the average of all cells in the value raster that belongs to \
                                             the same zone as the output cell. This is the default.

                                             - Majority: Determines the majority value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Maximum: Determines the largest value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Median: Finds the median value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Minimum: Finds the smallest value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Minority: Determines the minority value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Range: Finds the range of all cells in the value that belongs to \
                                             the same zone as the output zone.

                                             - Sum: Adds the total value of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - STD: Finds the standard deviation of all cells in the value raster that belongs to \
                                             the same zone as the output cell.

                                             - Variety: Finds the variety of all cells in the value raster that belong to \
                                             the same zone as the output cell.

                                             - Percentile: Finds a percentile of all cells in the value raster that \
                                             belong to the same zone as the output cell. The 90th percentile \
                                             is calculated by default. You can specify other values (from 0 to 100) \
                                             using the percentile_value parameter.

                                             If the input_raster_layer_to_summarize is floating-point type, the zonal calculations 
                                             for Majority, Median, Mean, and Variety cannot be computed.
    ------------------------------------     --------------------------------------------------------------------
    ignore_missing_values                    Optional bool, If you choose to ignore missing values, only the cells that 
                                             have a value in the layer to be summarized will be
                                             used in determining the output value for that area. 
                                             Otherwise, if there are missing values anywhere in an area,
                                             it is deemed that there is insufficient information to perform 
                                             statistical calculations for all the cells in
                                             that zone, and that area will receive a null (NoData) value in the output.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster. 
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution. 

                                             context parameter overwrites values set through arcgis.env parameter
                                         
                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.
                                            
                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.
                                                
                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.
                                                        
                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                    {"mask": {"url": "<image_service_url>"}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                   {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: Bilinear, Nearest, Cubic.

                                               Example:

                                                   {'resamplingMethod': "Nearest"}
    ------------------------------------     --------------------------------------------------------------------
    process_as_multidimensional              Optional bool, Process as multidimensional if set to True, 
                                             if the input is multidimensional raster.
                                             
                                             - True - Statistics will be calculated from the current slice of a \
                                             multidimensional image service. This is the default.
                                             
                                             - False - Statistics will be calculated for all dimensions \
                                             (such as time or depth) of a multidimensional image service.
                                             Parameter available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    percentile_value                         Optional Double, The percentile to calculate. The default is 90, for the 90th percentile. 
                                             The values can range from 0 to 100. The 0th percentile is essentially 
                                             equivalent to the Minimum statistic, and the 100th percentile is equivalent to Maximum. 
                                             A value of 50 will produce essentially the same result as the Median statistic.
                             
                                             This parameter is honoured only available if the statistics_type parameter is 
                                             set to Percentile.
                                             Parameter available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    percentile_interpolation_type            Optional str. Specifies the method of interpolation to be used when the 
                                             specified percentile value lies between two input cell values.

                                             - AUTO_DETECT - If the input value raster has integer pixel type, the NEAREST method is used. If the input value raster has floating point pixel type, then the LINEAR method is used. This is the default.
                                             - NEAREST - Nearest value to the desired percentile. In this case, the output pixel type is same as that of the input value raster.
                                             - LINEAR - Weighted average of two surrounding values from the desired percentile. In this case, the output pixel type is floating point.

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    circular_calculation                     Optional bool. Denotes whether the statistics calculations will be arithmetic or circular.

                                             - False - Calculates arithmetic statistics. This is the default.
                                             - True - Calculates circular statistics that are appropriate for cyclic quantities, such as compass direction in degrees, daytimes, and fractional parts of real numbers.

                                             | Parameter available in ArcGIS Image Server 11 and higher.
    ------------------------------------     --------------------------------------------------------------------
    circular_wrap_value                      Optional float. The possible highest value (upper bound) in the cyclic data.
                                             It is a positive number, and the default is 360. This value also represents the same quantity
                                             as the possible lowest value (lower bound).
                                             This parameter is honored only if the circular_calculation parameter is set to True.

                                             Parameter available in ArcGIS Image Server 11 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean. 
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer. 

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default. 
    ====================================     ====================================================================

    :return:
    output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: Summarizes raster based on the input zone layer.

        zone_layer = gis.content.search("zone_lyr")[0].layers[0]
        raster_inp = gis.content.search("raster_inp_lyr")[0].layers[0]

        summarize1 = summarize_raster_within(input_zone_layer=zone_layer,
                                             input_raster_layer_to_summarize=raster_inp,
                                             zone_field="Value",
                                             gis=gis)

        # Usage Example 2: Summarizes raster based on the multidimensional input zone layer.

        multi_zone_layer = gis.content.search("multi_zone_lyr")[0].layers[0]
        raster_inp = gis.content.search("raster_inp_lyr")[0].layers[0]

        summarize2 = summarize_raster_within(input_zone_layer=multi_zone_layer,
                                             input_raster_layer_to_summarize=raster_inp,
                                             zone_field="Value",
                                             process_as_multidimensional=True
                                             gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.summarize_raster_within(
        input_zone_layer=input_zone_layer,
        input_raster_layer_to_summarize=input_raster_layer_to_summarize,
        zone_field=zone_field,
        output_name=output_name,
        statistic_type=statistic_type,
        ignore_missing_values=ignore_missing_values,
        context=context,
        process_as_multidimensional=process_as_multidimensional,
        percentile_value=percentile_value,
        percentile_interpolation_type=percentile_interpolation_type,
        circular_calculation=circular_calculation,
        circular_wrap_value=circular_wrap_value,
        future=future,
        **kwargs,
    )


def convert_raster_to_feature(
    input_raster,
    field: str = "Value",
    output_type: str = "Polygon",
    simplify: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    create_multipart_features: bool = False,
    max_vertices_per_feature: Optional[int] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_convert_raster_to_feature/ra_convert_raster_to_feature.png

    Function converts imagery data to feature class vector data.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required Imagery Layer. The input raster that will be converted to a feature dataset.
    ------------------------------------     --------------------------------------------------------------------
    field                                    Optional string - field that specifies which value will be used for the conversion.
                                             It can be any integer or a string field.

                                             A field containing floating-point values can only be used if the output is to a point dataset.

                                             Default is ``Value``
    ------------------------------------     --------------------------------------------------------------------
    output_type                              Optional string.

                                             One of the following: ['Point', 'Line', 'Polygon']
    ------------------------------------     --------------------------------------------------------------------
    simplify                                 Optional bool, This option that specifies how the features should be smoothed. It is
                                             only available for line and polygon output.

                                             True, then the features will be smoothed out. This is the default.

                                             if False, then The features will follow exactly the cell boundaries of the raster dataset.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, a Feature layer is created by the method and used as the output.

                                             You can pass in an existing Feature Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Feature Service that should be created by this method
                                             to be used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}
    ------------------------------------     --------------------------------------------------------------------
    create_multipart_features                Optional boolean. Specifies whether the output polygons will consist of
                                             single-part or multipart features.

                                             - True: Specifies that multipart features will be created based on polygons that have the same value.

                                             - False: Specifies that individual features will be created for each polygon. This is the default.
    ------------------------------------     --------------------------------------------------------------------
    max_vertices_per_feature                 Optional int. The vertex limit used to subdivide a polygon into smaller polygons.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
    output_feature: Feature layer item

    .. code-block:: python

        # Usage Example 1: This example converts imagery layer to feature layer.

        raster = gis.content.search("raster_lyr")[0].layers[0]

        feature = convert_raster_to_feature(input_raster=raster,
                                            output_type='Polygon',
                                            gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis

    return gis._tools.rasteranalysis.convert_raster_to_feature(
        input_raster=input_raster,
        field=field,
        output_type=output_type,
        output_name=output_name,
        simplify_lines_or_polygons=simplify,
        context=context,
        create_multipart_features=create_multipart_features,
        max_vertices_per_feature=max_vertices_per_feature,
        future=future,
        **kwargs,
    )


def calculate_density(
    input_point_or_line_features: FeatureLayer,
    count_field: Optional[str] = None,
    search_distance: Optional[dict[str, Any]] = None,
    output_area_units: Optional[str] = None,
    output_cell_size: Optional[dict[str, Any]] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    input_barriers: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_calculate_density/ra_calculate_density.png

    Density analysis takes known quantities of some phenomenon and creates a density map by spreading
    these quantities across the map. You can use this function, for example, to show concentrations of
    lightning strikes or tornadoes, access to health care facilities, and population densities.

    This function creates a density map from point or line features by spreading known quantities of some
    phenomenon (represented as attributes of the points or lines) across the map. The result is a
    layer of areas classified from least dense to most dense.

    For point input, each point should represent the location of some event or incident, and the
    result layer represents a count of the incident per unit area. A larger density value in a new
    location means that there are more points near that location. In many cases, the result layer can
    be interpreted as a risk surface for future events. For example, if the input points represent
    locations of lightning strikes, the result layer can be interpreted as a risk surface for future
    lightning strikes.

    For line input, the line density surface represents the total amount of line that is near each
    location. The units of the calculated density values are the length of line-per-unit area.
    For example, if the lines represent rivers, the result layer will represent the total length
    of rivers that are within the search radius. This result can be used to identify areas that are
    hospitable to grazing animals.

    Other use cases of this tool include the following:

    *   Creating crime density maps to help police departments properly allocate resources to high crime
        areas.
    *   Calculating densities of hospitals within a county. The result layer will show areas with
        high and low accessibility to hospitals, and this information can be used to decide where
        new hospitals should be built.
    *   Identifying areas that are at high risk of forest fires based on historical locations of
        forest fires.
    *   Locating communities that are far from major highways in order to plan where new roads should
        be constructed.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_point_or_line_features             Required feature layer - The input point or line layer that will be used to calculate
                                             the density layer.
    ------------------------------------     --------------------------------------------------------------------
    count_field                              Optional string - count field

                                             Provide a field specifying the number of incidents at each location. For example, if you have points that
                                             represent cities, you can use a field representing the population of the city as the count field, and the
                                             resulting population density layer will calculate larger population densities near cities with
                                             larger populations.

                                             If the default choice of None is used, then each location will be assumed to represent a
                                             single count.

                                             Example:

                                                "myCountField"
    ------------------------------------     --------------------------------------------------------------------
    search_distance                          Optional dict, Enter a distance specifying how far to search to find point or line features when calculating density values.

                                             For example, if you provide a search distance of 10,000 meters, the density of any location in the output layer
                                             is calculated based on features that are within 10,000 meters of the location. Any location that does not have
                                             any incidents within 10,000 meters will receive a density value of zero.
                                             If no distance is provided, a default will be calculated that is based on the locations of the input features
                                             and the values in the count field (if a count field is provided).

                                             Example:

                                                {"distance":"60","units":"Meters"}
    ------------------------------------     --------------------------------------------------------------------
    output_area_units                        Optional string - Output area units
                                             Specify the output area unit. Density is count divided by area, and this parameter specifies the unit of the
                                             area in the density calculation. The available areal units are SQUARE_MILES and SQUARE_KILOMETERS.

                                             Example:

                                                "SQUARE_KILOMETERS"
    ------------------------------------     --------------------------------------------------------------------
    output_cell_size                         Optional dict - Output cell size
                                             Enter the cell size and unit for the output rasters.

                                             Example:

                                                {distance":"60","units":"Meters"}
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                    {"mask": {"url": "<image_service_url>"}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: Bilinear, Nearest, Cubic.

                                               Example:

                                                    {'resamplingMethod': "Nearest"}
    ------------------------------------     --------------------------------------------------------------------
    input_barriers                           Optional. The dataset that defines the barriers. The barriers can be
                                             a feature layer of polyline or polygon features.

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`~arcgis.gis.ContentManager.create_folder` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter on ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return:
    output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: Creates a density map within a specified distance from said location.

        sample_lyr = gis.content.search("sample_lyr", item_type="Feature Layer Collection")[0].layers[0]

        calculate_density_op = calculate_density(input_point_or_line_features=sample_lyr,
                                                search_distance={"distance":"10","units":"Meters"},
                                                gis=gis)

        # To view output inline, run:
        calculate_density_op.layers[0]
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.calculate_density(
        input_point_or_line_features=input_point_or_line_features,
        output_name=output_name,
        count_field=count_field,
        search_distance=search_distance,
        output_area_units=output_area_units,
        output_cell_size=output_cell_size,
        context=context,
        future=future,
        input_barriers=input_barriers,
        **kwargs,
    )


def create_viewshed(
    input_elevation_surface,
    input_observer_features: FeatureLayer,
    optimize_for: Optional[str] = None,
    maximum_viewing_distance: Optional[dict[str, Any]] = None,
    maximum_viewing_distance_field: Optional[str] = None,
    minimum_viewing_distance: Optional[dict[str, Any]] = None,
    minimum_viewing_distance_field: Optional[str] = None,
    viewing_distance_is_3d: Optional[bool] = None,
    observers_elevation: Optional[dict[str, Any]] = None,
    observers_elevation_field: Optional[str] = None,
    observers_height: Optional[dict[str, Any]] = None,
    observers_height_field: Optional[str] = None,
    target_height: Optional[dict[str, Any]] = None,
    target_height_field: Optional[str] = None,
    above_ground_level_output_name: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_create_viewshed/ra_create_viewshed.png

    Function  allows you to execute raster analysis on a distributed server deployment.

    ====================================     =============================================================================================================================================
    **Parameter**                             **Description**
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    input_elevation_surface                  Required Imagery Layer.
                                             The input elevation surface for calculating the viewshed.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    input_observer_features                  Required Feature Layer.
                                             The input observer locations features.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    optimize_for                             Optional string.
                                             Choose the optimization method to use for calculating the viewshed.

                                             This parameter offers two methods: SPEED and ACCURACY.

                                             Example:

                                                "ACCURACY"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    maximum_viewing_distance                 Optional dict. This is a cutoff distance where the computation of
                                             visible areas stops. Beyond this distance, it is unknown whether the
                                             analysis points and the other objects can see each other.

                                             Supported units: Meters | Kilometers | Feet | Yards | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    maximum_viewing_distance_field           Optional string. Provide a field that specifies the maximum viewing distance
                                             for each observer. You can use any numerical field from the input
                                             observer point features.

                                             The value contained in the field must be in the same unit as the
                                             XY unit of the input elevation surface.

                                             Example:

                                                "radius2"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    minimum_viewing_distance                 Optional dict. This is a distance where the computation of visible areas begins.

                                             Supported units: Meters | Kilometers | Feet | Yards | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    minimum_viewing_distance_field           Optional string. Provide a field that specifies the minimum viewing distance for each observer.

                                             You can use any numerical field from the input observer point features.
                                             The value contained in the field must be in the same unit as the XY unit of the input elevation surface.

                                             Example:

                                                 "radius1"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    viewing_distance_is_3d                   Optional bool. Specify whether the minimum_viewing_distance and maximum_viewing_distance
                                             input parameters are measured in a three-dimensional or two-dimensional way.

                                             - If True, the viewing distances are measured in 3D.

                                             - If False, the viewing distances are measured in 2D. This is the default.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    observers_elevation                      Optional dict. Specify the elevation of your observer locations.

                                             Supported units: Meters | Kilometers | Feet | Yards | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    observers_elevation_field                Optional string. Provide a field that specifies the elevation for the observers.

                                             You can use any numerical field from the input observer point features.
                                             The value contained in the field must be in the same unit as the Z unit
                                             of the input elevation surface.

                                             Example:

                                                "spot"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    observers_height                         Optional dict. The height above ground of your observer locations.

                                             Supported units: Meters | Kilometers | Feet | Yards | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    observers_height_field                   Optional string. Provide a field that specifies the height for the observers.
                                             You can use any numerical field from the input observer point features.

                                             The value contained in the field must be in the same unit as the
                                             Z unit of the input elevation surface.

                                             Example:

                                                "offseta"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    target_height                            Optional dict. Enter the height of structures, or people on the ground,
                                             used to establish visibility.

                                             Supported units: Meters | Kilometers | Feet | Yards | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    target_height_field                      Optional string. Provide a field that specifies the height for the targets.
                                             You can use any numerical field from the input observer point features.

                                             The value contained in the field must be in the same unit as the Z unit
                                             of the input elevation surface.

                                             Example:

                                                "offsetb"
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    above_ground_level_output_name           Optional. If not provided, an Image Service is created by the method and
                                             used as the above ground level output raster.

                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the above ground level output
                                             Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.

                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                    {"mask": {"url": "<image_service_url>"}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: Bilinear, Nearest, Cubic.

                                               Example:

                                                    {'resamplingMethod': "Nearest"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     =============================================================================================================================================

    :return:

    named tuple with name values being:

    - output_raster

    - output_above_ground_level_raster (generated if value specified for above_ground_level_output_name)

    .. code-block:: python

        # Usage Example 1: Creates viewshed with cut-off distance

        elevation = gis.content.search("elev_lyr")[0].layers[0]
        observer =  gis.content.search("obs_shp", item_type="Feature Layer Collection")[0].layers[0]

        create_viewshed_op = create_viewshed(input_elevation_surface=elevation,
                                             input_observer_features=observer,
                                             maximum_viewing_distance={"distance":"100","units":"Miles"},
                                             gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.create_viewshed(
        input_elevation_surface=input_elevation_surface,
        input_observer_features=input_observer_features,
        output_name=output_name,
        optimize_for=optimize_for,
        maximum_viewing_distance=maximum_viewing_distance,
        maximum_viewing_distance_field=maximum_viewing_distance_field,
        minimum_viewing_distance=minimum_viewing_distance,
        minimum_viewing_distance_field=minimum_viewing_distance_field,
        viewing_distance_is3D=viewing_distance_is_3d,
        observers_elevation=observers_elevation,
        observers_elevation_field=observers_elevation_field,
        observers_height=observers_height,
        observers_height_field=observers_height_field,
        target_height=target_height,
        target_height_field=target_height_field,
        above_ground_level_output_name=above_ground_level_output_name,
        context=context,
        future=future,
        **kwargs,
    )


def interpolate_points(
    input_point_features: FeatureLayer,
    interpolate_field: str,
    optimize_for: str = "BALANCE",
    transform_data: bool = False,
    size_of_local_models: Optional[int] = None,
    number_of_neighbors: Optional[int] = None,
    output_cell_size: Optional[dict[str, Any]] = None,
    output_prediction_error: bool = False,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_interpolate_points/ra_interpolate_points.png

    This tool allows you to predict values at new locations based on measurements from a collection of points. The tool
    takes point data with values at each point and returns a raster of predicted values:

    * An air quality management district has sensors that measure pollution levels. Interpolate Points can be used to
      predict pollution levels at locations that don't have sensors, such as locations with at-risk populations-
      schools or hospitals, for example.
    * Predict heavy metal concentrations in crops based on samples taken from individual plants.
    * Predict soil nutrient levels (nitrogen, phosphorus, potassium, and so on) and other indicators (such as electrical
      conductivity) in order to study their relationships to crop yield and prescribe precise amounts of fertilizer
      for each location in the field.
    * Meteorological applications include prediction of temperatures, rainfall, and associated variables (such as acid
      rain).

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_point_features                     Required point layer containing locations with known values
                                             The point layer that contains the points where the values have been measured.
    ------------------------------------     --------------------------------------------------------------------
    interpolate_field                        Required string -  field to interpolate.
                                             Choose the field whose values you wish to interpolate. The field must be numeric.

                                             Example:

                                                "myField"
    ------------------------------------     --------------------------------------------------------------------
    optimize_for                             Optional string.
                                             Choose your preference for speed versus accuracy.
                                             More accurate predictions take longer to calculate.

                                             This parameter alters the default values of several other
                                             parameters of Interpolate Points in order to optimize speed of calculation,
                                             accuracy of results, or a balance of
                                             the two.

                                             By default, the tool will optimize for balance.

                                             One of the following: ['SPEED', 'BALANCE', 'ACCURACY']

                                             - SPEED is optimized for performance.

                                             - BALANCE is performed with a balance between performance and accuracy. This is the default.

                                             - ACCURACY is optimized towards achieving the most accurate result, at the expense of some performance..

                                             Example:

                                                "ACCURACY"
    ------------------------------------     --------------------------------------------------------------------
    transform_data                           Optional bool - Choose whether to transform your data to the normal distribution.

                                             Interpolation is most accurate for data that follows a normal (bell-shaped) distribution. If your data does not
                                             appear to be normally distributed, you should perform a transformation.

                                             - False means no transformation will be applied. This is suitable for data that is naturally normally distributed. This is the default.

                                             - True applies a suitable transformation for data that is not normally distributed.
    ------------------------------------     --------------------------------------------------------------------
    size_of_local_models                     Optional int - Size of local models
                                             Interpolate Points works by building local interpolation models that are mixed together to create the final
                                             prediction map. This parameter controls how many points will be contained in each local model. Smaller values
                                             will make results more local and can reveal small-scale effects, but it may introduce some instability in the
                                             calculations. Larger values will be more stable, but some local effects may be missed.

                                             The value can range from 30 to 500, but typical values are between 50 and 200.
    ------------------------------------     --------------------------------------------------------------------
    number_of_neighbors                      Optional int - Number of Neighbors
                                             Predictions are calculated based on neighboring points. This parameter controls how many points will be used in
                                             the calculation. Using a larger number of neighbors will generally produce more accurate results, but the
                                             results take longer to calculate.

                                             This value can range from 1 to 64, but typical values are between 5 and 15.
    ------------------------------------     --------------------------------------------------------------------
    output_cell_size                         Optional dict. Specify the cell size to use for the output raster.

                                             Supported units: Meters | Kilometers | Feet | Miles

                                             Example:

                                                 {"distance":"60","units":"Meters"}
    ------------------------------------     --------------------------------------------------------------------
    output_prediction_error                  Optional bool. Choose whether you want to create a raster of standard errors for the predicted values.

                                             - True specifies that standard error raster will be generated.

                                             - False specifies that it will not be generated

                                             Standard errors are useful because they provide information about the reliability of the predicted values.
                                             A simple rule of thumb is that the true value will fall within two standard errors of the predicted value 95
                                             percent of the time. For example, suppose a new location gets a predicted value of 50 with a standard error of
                                             5. This means that this tool's best guess is that the true value at that location is 50, but it reasonably could
                                             be as low as 40 or as high as 60. To calculate this range of reasonable values, multiply the standard error by
                                             2, add this value to the predicted value to get the upper end of the range, and subtract it from the predicted
                                             value to get the lower end of the range.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.

                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                   | {"extent": {"xmin": -122.68,
                                                   | "ymin": 45.53,
                                                   | "xmax": -122.45,
                                                   | "ymax": 45.6,
                                                   | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                   {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                   {'snapRaster': {'url': '<image_service_url>'}}

                                             - Mask (mask): Only cells that fall within the analysis
                                               mask will be considered in the operation.

                                               Example:

                                                   {"mask": {"url": "<image_service_url>"}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                   {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return:

    named tuple with name values being :

    - output_raster (the output_raster item description is updated with the process_info)

    - process_info (if run in a non-Jupyter environment, use process_info.data to get the HTML data)

    - output_error_raster (if output_prediction_error is set to True).

    .. code-block:: python

        # Usage Example 1: Performs point interpolation on the feature layer.
        # In this example operation is optimized for SPEED

        layer = gis.content.search("layer_shp", item_type="Feature Layer Collection")[0].layers[0]

        interpolate_points_speed = interpolate_points(input_point_features=layer,
                                                      interpolate_field='field',
                                                      optimize_for="SPEED",
                                                      transform_data=True,
                                                      gis=gis)

        # Usage Example 2: Performs point interpolation on the feature layer.
        # In this example operation is optimized for ACCURACY

        interpolate_points_acc = interpolate_points(input_point_features=layer,
                                                    interpolate_field='field',
                                                    optimize_for="ACCURACY",
                                                    transform_data=True,
                                                    gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.interpolate_points(
        input_point_features=input_point_features,
        interpolate_field=interpolate_field,
        output_name=output_name,
        optimize_for=optimize_for,
        transform_data=transform_data,
        size_of_local_models=size_of_local_models,
        number_of_neighbors=number_of_neighbors,
        output_cell_size=output_cell_size,
        output_prediction_error=output_prediction_error,
        context=context,
        future=future,
        **kwargs,
    )


def classify(
    input_raster,
    input_classifier_definition: dict[str, Any],
    additional_input_raster=None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_classify/ra_classify.png

    The Classify function will create categories of pixels based on the input raster and
    the classifier definition dictionary that was generated from the train_classifier function.

    ================================     ====================================================================
    **Parameter**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_raster                         Required :class:`~arcgis.raster.ImageryLayer` object.
    --------------------------------     --------------------------------------------------------------------
    input_classifier_definition          Required dict or ECD Portal item.

                                         The classifier definition dictionary generated from the train_classifier function.

                                         Example:

                                             | {"EsriClassifierDefinitionFile":0,
                                             | "FileVersion":3,"NumberDefinitions":1,
                                             | "Definitions":[...]}
    --------------------------------     --------------------------------------------------------------------
    additional_input_raster              Optional :class:`~arcgis.raster.ImageryLayer` object. This can be a segmented raster.
    --------------------------------     --------------------------------------------------------------------
    output_name                          Optional String. If specified, an Imagery Layer of given name is
                                         created. Else, an Image Service is created by the method and used
                                         as the output raster. You can pass in an existing Image Service Item
                                         from your GIS to use that instead. Alternatively, you can pass in
                                         the name of the output Image Service that should be created by this
                                         method to be used as the output for the tool. A RuntimeError is raised
                                         if a service by that name already exists
    --------------------------------     --------------------------------------------------------------------
    context                              context contains additional settings that affect task execution.

                                         context parameter overwrites values set through arcgis.env parameter

                                         This function has the following settings:

                                         - Extent (extent): A bounding box that defines the analysis area.

                                           Example:

                                               | {"extent": {"xmin": -122.68,
                                               | "ymin": 45.53,
                                               | "xmax": -122.45,
                                               | "ymax": 45.6,
                                               | "spatialReference": {"wkid": 4326}}}

                                         - Output Spatial Reference (outSR): The output raster will be
                                           projected into the output spatial reference.

                                           Example:

                                                {"outSR": {spatial reference}}

                                         - Snap Raster (snapRaster): The output raster will have its
                                           cells aligned with the specified snap raster.

                                           Example:

                                                {'snapRaster': {'url': '<image_service_url>'}}

                                         - Cell Size (cellSize): The output raster will have the resolution
                                           specified by cell size.

                                           Example:

                                                {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                         - Parallel Processing Factor (parallelProcessingFactor): controls
                                           Raster Processing (CPU) service instances.

                                           Example:

                                           Syntax example with a specified number of processing instances:

                                                {"parallelProcessingFactor": "2"}

                                           Syntax example with a specified percentage of total
                                           processing instances:

                                                {"parallelProcessingFactor": "60%"}

                                         - Resampling Method (resamplingMethod): The output raster will be
                                           resampled to method specified.

                                           The supported values are: Bilinear, Nearest, Cubic.

                                           Example:

                                              {'resamplingMethod': "Nearest"}

                                         - processAsMultidimensional - Specifies whether to process the input as a multidimensional raster.

                                           - False - The input will not be processed as a multidimensional raster.
                                             If the input is multidimensional, only the slice that is currently
                                             displayed will be processed.

                                           - True - The input will be processed as a multidimensional raster and all
                                             slices will be processed to produce a new multidimensional raster.

                                           Example:

                                                {'processAsMultidimensional': True}
    --------------------------------     --------------------------------------------------------------------
    gis                                  Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                         is used.
    --------------------------------     --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                         results will be returned asynchronously.
    --------------------------------     --------------------------------------------------------------------
    folder                               Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                         not exist, with the given folder name and persists the output in this folder.
                                         The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                         Example:

                                            | {'username': 'user1',
                                            | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                            | 'title': 'trial'}
    --------------------------------     --------------------------------------------------------------------
    tiles_only                           Keyword only parameter. Optional boolean.
                                         In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                         To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                         Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ================================     ====================================================================

    :return:
       output_raster : Imagery Layer item

    .. code-block:: python

        # Usage Example 1: Classifies a raster based on an Esri Classifier Definition dictionary and raster data inputs.

        classifier = {
            'EsriClassifierDefinitionFile': 0,
            'FileVersion': 3,
            'NumberDefinitions': 1,
            'Definitions': [...],
            }

        raster = gis.content.search('image_tif')[0].layers[0]

        classify_op = classify(input_raster=raster,
                               input_classifier_definition=classifier,
                               gis=gis)

        # Usage Example 2: Performs classification such that the output raster is projected onto the specified spatial reference.

        classify_op = classify(input_raster=raster,
                               input_classifier_definition=classifier,
                               context={"outSR":{spatial reference}}
                               gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.classify(
        input_raster=input_raster,
        input_classifier_definition=input_classifier_definition,
        output_name=output_name,
        additional_input_raster=additional_input_raster,
        context=context,
        future=future,
        **kwargs,
    )


def segment(
    input_raster,
    spectral_detail: float = 15.5,
    spatial_detail: float = 15,
    minimum_segment_size_in_pixels: float = 20,
    band_indexes: list[int] = [0, 1, 2],
    remove_tiling_artifacts: bool = False,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_segment/ra_segment.png

    Groups together adjacent pixels having similar spectral and spatial characteristics into
    segments, known as objects.

    ================================     ====================================================================
    **Parameter**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_raster                         Required :class:`~arcgis.raster.ImageryLayer` object
    --------------------------------     --------------------------------------------------------------------
    spectral_detail                      Optional float. Default is 15.5.
                                         Set the level of importance given to the spectral differences of
                                         features in your imagery. Valid values range from 1.0 to 20.0. A high
                                         value is appropriate when you have features you want to classify
                                         separately but have somewhat similar spectral characteristics.
                                         Smaller values create spectrally smoother outputs.

                                         For example, setting a higher spectral detail value for a forested
                                         scene, will preserve greater discrimination between the different tree
                                         species, resulting in more segments.
    --------------------------------     --------------------------------------------------------------------
    spatial_detail                       Optional float. Default is 15.
                                         Set the level of importance given to the proximity between features
                                         in your imagery. Valid values range from 1 to 20. A high value is
                                         appropriate for a scene where your features of interest are small
                                         and clustered together. Smaller values create spatially smoother
                                         outputs.

                                         For example, in an urban scene, you could classify an impervious
                                         surface using a smaller spatial detail, or you could classify
                                         buildings and roads as separate classes using a higher spatial detail.
    --------------------------------     --------------------------------------------------------------------
    minimum_segment_size_in_pixels       Optional float. Default is 20.
                                         Merge segments smaller than this size with their best fitting
                                         neighbor segment. This is related to the minimum mapping unit for a
                                         mapping project. Units are in pixels.
    --------------------------------     --------------------------------------------------------------------
    band_indexes                         Optional List of integers. Default is [0,1,2]
                                         Define which 3 bands are used in segmentation. Choose the bands that
                                         visually discriminate your features of interest best.
    --------------------------------     --------------------------------------------------------------------
    remove_tiling_artifacts              Optional Bool. Default is False.
                                         If False, the tool will not run to remove tiling artifacts after
                                         segmentation. The result may seem blocky at some tiling boundaries.
    --------------------------------     --------------------------------------------------------------------
    output_name                          Optional String. If specified, an Imagery Layer of given name is
                                         created. Else, an Image Service is created by the method and used
                                         as the output raster. You can pass in an existing Image Service Item
                                         from your GIS to use that instead. Alternatively, you can pass in
                                         the name of the output Image Service that should be created by this
                                         method to be used as the output for the tool. A RuntimeError is raised
                                         if a service by that name already exists
    --------------------------------     --------------------------------------------------------------------
    context                              context contains additional settings that affect task execution.

                                         context parameter overwrites values set through arcgis.env parameter

                                         This function has the following settings:

                                         - Extent (extent): A bounding box that defines the analysis area.

                                           Example:

                                            | {"extent": {"xmin": -122.68,
                                            | "ymin": 45.53,
                                            | "xmax": -122.45,
                                            | "ymax": 45.6,
                                            | "spatialReference": {"wkid": 4326}}}

                                         - Output Spatial Reference (outSR): The output raster will be
                                           projected into the output spatial reference.

                                           Example:

                                                {"outSR": {spatial reference}}

                                         - Snap Raster (snapRaster): The output raster will have its
                                           cells aligned with the specified snap raster.

                                           Example:

                                                {'snapRaster': {'url': '<image_service_url>'}}

                                         - Cell Size (cellSize): The output raster will have the resolution
                                           specified by cell size.

                                           Example:

                                                {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                         - Parallel Processing Factor (parallelProcessingFactor): controls
                                           Raster Processing (CPU) service instances.

                                           Example:

                                           Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                           Syntax example with a specified percentage of total
                                           processing instances:

                                                    {"parallelProcessingFactor": "60%"}

                                         - Resampling Method (resamplingMethod): The output raster will be
                                           resampled to method specified.
                                           The supported values are: Bilinear, Nearest, Cubic.

                                           Example:

                                                {'resamplingMethod': "Nearest"}
    --------------------------------     --------------------------------------------------------------------
    gis                                  Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                         is used.
    --------------------------------     --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                         results will be returned asynchronously.
    --------------------------------     --------------------------------------------------------------------
    folder                               Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                         not exist, with the given folder name and persists the output in this folder.
                                         The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                         Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    --------------------------------     --------------------------------------------------------------------
    tiles_only                           Keyword only parameter. Optional boolean.
                                         In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                         To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                         Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ================================     ====================================================================

    :return:
       output_raster : Imagery Layer item

    .. code-block:: python

        # Usage Example 1: Performs segmentation on an input raster using nearest neighbor resampling method.

        img_layer = gis.content.search("imgLayer")[0].layers[0]

        segment_op = segment(input_raster=img_layer,
                             context={"resamplingMethod":"Nearest"},
                             gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.segment(
        input_raster=input_raster,
        output_name=output_name,
        spectral_detail=spectral_detail,
        spatial_detail=spatial_detail,
        minimum_segment_size_in_pixels=minimum_segment_size_in_pixels,
        band_indexes=band_indexes,
        remove_tiling_artifacts=remove_tiling_artifacts,
        context=context,
        future=future,
        **kwargs,
    )


def train_classifier(
    input_raster,
    input_training_sample_json,
    classifier_parameters,
    segmented_raster=None,
    segment_attributes="COLOR;MEAN",
    dimension_value_field=None,
    output_ecd_item_name=None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_train_classifier/ra_train_classifier.png

    The Train Classifier task is a service to train image classifiers and return an .ecs file in dictionary format.
    The .ecs file is used in the classify function.

    ================================     ====================================================================
    **Parameter**                         **Description**
    --------------------------------     --------------------------------------------------------------------
    input_raster                         Required :class:`~arcgis.raster.ImageryLayer` object
    --------------------------------     --------------------------------------------------------------------
    input_training_sample_json           Optional JSON. This is the dictionary representation of the training samples.
                                         To convert feature layer to JSON, perform:

                                             | query_result = <feature_layer>.query()
                                             | input_training_sample_json = query_result.to_json

                                         Set input_training_sample_json to None, for iso method.
    --------------------------------     --------------------------------------------------------------------
    classifier_parameters                Required dict. The classifier algorithm and parameters used in the supervised training.

                                         - Random trees example:
                                                | {
                                                |   "method":"rt",
                                                |   "params": {
                                                |     "maxNumTrees":50,
                                                |     "maxTreeDepth":30,
                                                |     "maxSampleClass":1000 }
                                                | }

                                         - Support Vector machine example:
                                                | {
                                                |   "method":"svm",
                                                |   "params":{"maxSampleClass":1000}
                                                | }

                                         - Maximum likelihood example:
                                                | {"method":"mlc"}

                                         - ISO example:
                                                | {"method":"iso",
                                                | "params":
                                                | {
                                                | "maxNumClasses": 20,
                                                | "maxIteration": 20,
                                                | "minNumSamples": 20,
                                                | "skipFactor": 10,
                                                | "maxNumMerge": 5,
                                                | "maxMergeDist": 0.5}
                                                | }

    --------------------------------     --------------------------------------------------------------------
    segmented_raster                     Required :class:`~arcgis.raster.ImageryLayer` object
    --------------------------------     --------------------------------------------------------------------
    segment_attributes                   Optional string. The string of segment attributes used in the training (separated by semicolon).
                                         It is the permutation of the following attributes: COLOR; MEAN; STD; COUNT; COMPACTNESS; RECTANGULARITY.

                                         Example:

                                            "COLOR; MEAN"
    --------------------------------     --------------------------------------------------------------------
    dimension_value_field                Contains dimension values in the input training sample feature class.

                                         This parameter is required to classify a time series of raster data
                                         using the change analysis raster output from the
                                         analyze_changes_using_ccdc and analyze_changes_using_landtrendr function.

                                         Parameter available in ArcGIS Image Server 10.9 and higher.
    --------------------------------     --------------------------------------------------------------------
    output_ecd_item_name                 Optional String or Item. The output ECD Portal item to be created.
                                         If an Item is provided, the given item will be overwritten with the ECD instead.
    --------------------------------     --------------------------------------------------------------------
    gis                                  Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                         is used.
    --------------------------------     --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                         results will be returned asynchronously.
    ================================     ====================================================================

    :return:
       Named Tuple

    .. code-block:: python

        # Usage Example 1: Trains image classifiers based on 'Maximum Likelihood Estimation' algorithm.

        raster = gis.content.search('my_inp_raster')[0].layers[0]
        segment = gis.content.search('my_segmented_raster')[0].layers[0]

        train = train_classifier(input_raster=raster,
                                 input_training_sample_json=<sample_json>,
                                 classifier_parameters={"method":"mlc"},
                                 segmented_raster=segment)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.train_classifier(
        input_raster=input_raster,
        input_training_sample_json=input_training_sample_json,
        classifier_parameters=classifier_parameters,
        segmented_raster=segmented_raster,
        segment_attributes=segment_attributes,
        dimension_value_field=dimension_value_field,
        future=future,
        output_ecd_item_name=output_ecd_item_name,
        **kwargs,
    )


###################################################################################################
## Create image collection
###################################################################################################


def create_image_collection(
    image_collection: Item,
    input_rasters: list,
    raster_type_name: str,
    raster_type_params: Optional[dict[str, Any]] = None,
    out_sr: Optional[Union[str, SpatialReference]] = None,
    context: Optional[dict[str, Any]] = None,
    source_mosaic_dataset: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/create_image_collection/create_image_collection.png

    Create a collection of images that will participate in the ortho-mapping project.
    Provides provision to use input rasters by reference 
    and to specify image collection properties through context parameter.

    The function can also create hosted imagery layers in ArcGIS Enterprise and ArcGIS Online from local raster datasets by uploading the data to the server.
    A collection can be created from multiple input rasters.
    
    For this functionality to work on ArcGIS Online, Azure library packages for Python (Azure SDK for Python - azure-storage-blob: 12.1<= version <=12.17)
    needs to be pre-installed. Refer https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package

    ======================               ====================================================================
    **Parameter**                         **Description**
    ----------------------               --------------------------------------------------------------------
    image_collection                     Required, the name of the image collection to create.
                  
                                         The image collection can be an existing image service, in \
                                         which the function will create a mosaic dataset and the existing \
                                         hosted image service will then point to the new mosaic dataset.

                                         If the image collection does not exist, a new multi-tenant \
                                         service will be created.

                                         This parameter can be the Item representing an existing image_collection \
                                         or it can be a string representing the name of the image_collection \
                                         (either existing or to be created.)
    ----------------------               --------------------------------------------------------------------
    input_rasters                        Required, the list of input rasters to be added to
                                         the image collection being created. This parameter can
                                         be any one of the following:

                                         - List of portal Items of the images
                                         - An image service URL
                                         - Shared data path (this path must be accessible by the server)
                                         - Name of a folder on the portal
                                         - Path to local raster dataset(s)
                                         - :class:`~arcgis.raster.RasterCollection` object created from a list of datastore\
                                            rasters or using the :meth:`~arcgis.raster.RasterCollection.from_stac_api` method.

                                         The function can create hosted imagery layers on enterprise and AGOL from 
                                         local raster datasets by uploading the data to the server.
    ----------------------               --------------------------------------------------------------------
    raster_type_name                     Required string. The name of the raster type to use for adding data to \
                                         the image collection.

                                         Choice list:

                                             | [
                                             | "Aerial", "ASTER", "DMCII", "DubaiSat-2", "GeoEye-1", "GF-1 PMS", "GF-1 WFV",
                                             | "GF-2 PMS", "GRIB", "HDF", "IKONOS", "Jilin-1", "KOMPSAT-2", "KOMPSAT-3",
                                             | "Landsat 1-5 MSS", "Landsat 4-5 TM", "Landsat 7 ETM+", "Landsat 8", "Landsat 9",
                                             | "NetCDF", "PlanetScope", "Pleiades-1", "Pleiades NEO", "QuickBird", "RapidEye",
                                             | "Raster Dataset", "ScannedAerial", "Sentinel-2", "SkySat", "SPOT 5", "SPOT 6",
                                             | "SPOT 7", "Superview-1", "Tiled Imagery Layer", "UAV/UAS", "WordView-1",
                                             | "WordView-2", "WordView-3", "WordView-4", "ZY3-SASMAC", "ZY3-CRESDA"
                                             | ]

                                         If an existing mosaic dataset is being published as a 
                                         dynamic imagery layer using the ``source_mosaic_dataset`` parameter, the
                                         ``raster_type_name`` parameter can be set to None as it is not required.


                                         Example:

                                            "QuickBird"
    ----------------------               --------------------------------------------------------------------
    raster_type_params                   Optional dict. Additional ``raster_type`` specific parameters.
        
                                         The process of add rasters to the image collection can be \
                                         controlled by specifying additional raster type arguments.

                                         The raster type parameters argument is a dictionary.

                                         The dictionary can contain productType, processingTemplate, \
                                         pansharpenType, Filter, pansharpenWeights, ConstantZ, \
                                         dem, zoffset, CorrectGeoid, ZFactor, StretchType, \
                                         ScaleFactor, ValidRange

                                         Please check the table below (Supported Raster Types), \
                                         for more details about the product types, \
                                         processing templates, pansharpen weights for each raster type. 

                                         - Possible values for pansharpenType - ["Mean", "IHS", "Brovey", "Esri", "Mean", "Gram-Schmidt"]
                                         - Possible values for filter - [None, "Sharpen", "SharpenMore"]
                                         - Value for StretchType dictionary can be as follows:

                                           - "None"
                                           - "MinMax; <min>; <max>"
                                           - "PercentMinMax; <MinPercent>; <MaxPercent>"
                                           - "StdDev; <NumberOfStandardDeviation>"
                                           Example: {"StretchType": "MinMax; <min>; <max>"}
                                         - Value for ValidRange dictionary can be as follows:

                                           - "<MaskMinValue>, <MaskMaxValue>"
                                           Example: {"ValidRange": "10, 200"}

                                         Example:

                                            {"productType":"All","processingTemplate":"Pansharpen",
                                            "pansharpenType":"Gram-Schmidt","filter":"SharpenMore",
                                            "pansharpenWeights":"0.85 0.7 0.35 1","constantZ":-9999}
    ----------------------               --------------------------------------------------------------------
    out_sr                               Optional integer. Additional parameters of the service.
                            
                                         The following additional parameters can be specified:

                                         - Spatial reference of the image_collection; The well-known ID of \
                                         the spatial reference or a spatial reference dictionary object for the \
                                         input geometries.

                                         If the raster type name is set to "UAV/UAS", the spatial reference of the
                                         output image collection will be determined by the raster type parameters defined.
    ----------------------               --------------------------------------------------------------------
    context                              Optional dict. The context parameter is used to provide additional input parameters.
    
                                         Syntax: {"image_collection_properties": {"imageCollectionType":"Satellite"},"byref":True}
                                        
                                         Use ``image_collection_properties`` key to set value for imageCollectionType.

                                         .. note::

                                            The "imageCollectionType" property is important for image collection that will later on be adjusted by orthomapping system service. 
                                            Based on the image collection type, the orthomapping system service will choose different algorithm for adjustment. 
                                            Therefore, if the image collection is created by reference, the requester should set this 
                                            property based on the type of images in the image collection using the following keywords. 
                                            If the imageCollectionType is not set, it defaults to "UAV/UAS"

                                         If ``byref`` is set to 'True', the data will not be uploaded. If it is not set, the default is 'False'

                                         ``upload_properties`` key can be used to control specific upload parameters when trying to \
                                             create hosted imagery layers in ArcGIS Online from local raster datasets.

                                         Available options:

                                         - ``maxUploadConcurrency``: Optional integer. Maximum number of parallel connections \
                                            to use for large uploads (when individual file/blob size exceeds 64MB). \
                                            This is the **max_concurrency** parameter of the `BlobClient.upload_blob() <https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#upload-blob-data--blob-type--blobtype-blockblob---blockblob----length-none--metadata-none----kwargs->`__ method. \
                                            (The default is 6)
                                         - ``maxWorkerThreads``: Optional integer. Maximum number of threads to execute asynchronously \
                                            when uploading multiple files. This is the **max_workers** parameter of the `ThreadPoolExecutor() <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`__ class. \
                                            (The default is None)
                                         - ``displayProgress``: Optional boolean. If set to True, a progress bar will be \
                                            displayed for tracking the progress of the uploads to user's rasterstore. \
                                            (The default is False)

                                            Example:

                                                | {"upload_properties":{"maxUploadConcurrency":8,
                                                |                       "maxWorkerThreads":20,
                                                |                       "displayProgress":True}
                                                | }

                                         The context parameter can also be used to specify whether to build overviews, \
                                         build footprints, to specify pixel value that represents the NoData etc.


                                         Example:

                                            | {"buildFootprints":True,                                            
                                            | "footprintsArguments":{"method":"RADIOMETRY","minValue":1,"maxValue":5,
                                            | "shrinkDistance":50,"skipOverviews":True,"updateBoundary":True,
                                            | "maintainEdge":False,"simplification":None,"numVertices":20,
                                            | "minThinnessRatio":0.05,"maxSliverSize":20,"requestSize":2000,
                                            | "minRegionSize":100},
                                            | "defineNodata":True,                                            
                                            | "noDataArguments":{"noDataValues":[500],"numberOfBand":99,"compositeValue":True},                                            
                                            | "buildOverview":True}

                                         The context parameter can be used to add new fields when creating \
                                         the image collection.


                                         Example:

                                            | {"fields": [{"name": "cloud_cover", "type": "Long"},
                                            | {"name": "cloud_shadow_count", "type": "Long"}]}
    ----------------------               --------------------------------------------------------------------
    source_mosaic_dataset                Optional string. Path to the existing mosaic dataset to be published 
                                         as a hosted dynamic imagery layer.

                                         To publish an existing mosaic dataset, specify the path to the input 
                                         data of the mosaic in the ``input_rasters`` parameter. 
                                         The data will be uploaded to ArcGIS Online.

                                         ``raster_type_name`` parameter can be set to None as it is not required to 
                                         publish an imagery layer from a mosaic dataset.

                                         .. note::
                                                Option available only on ArcGIS online

                                         Example:

                                            "./data/temp_uploaded.gdb/test"
    ----------------------               --------------------------------------------------------------------
    gis                                  Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified, the active GIS is used.
    ----------------------               --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                         results will be returned asynchronously.
    ----------------------               --------------------------------------------------------------------
    folder                               Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                         not exist, with the given folder name and persists the output in this folder.
                                         The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                         Example:

                                               	| {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ======================               ====================================================================

    :return: The imagery layer item

+-----------------------------------------------------------------------------------------------------------------------------------------------+
|     Supported Raster Types                                                                                                                    |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| **Raster Type**        | **pansharpenWeights**  | **productType**                              |**processingTemplate**                        |
+========================+========================+==============================================+==============================================+
| "QuickBird"            |"0.85, 0.7, 0.35, 1"    |"All" (default), "Basic",                     |"All Bands",                                  |
|                        |                        |"Orthorectified",                             |"Multispectral",                              |
|                        |                        |"Standard Orthoready",                        |"Multispectral AComp",                        |
|                        |                        |"Standard"                                    |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic AComp",                         |
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen AComp",                           |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
|                        |                        |                                              |                                              |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Orthorectified Acomp",                       |"Multispectral AComp" (default),              |
|                        |                        |"Standard AComp",                             |"Panchromatic AComp",                         |
|                        |                        |"Standard OrthoreadyAcomp"                    |"Pansharpen AComp"                            |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"DubaiSat-2"            |"0.166,0.167,0.167,0.5" | "All" (default), "Geo", "Radio"              |"All Bands", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic",                               | 
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"GF-1 PMS"              |"0.2501,0.3646,0,0.3853"| "All" (default), "LEVEL 1"                   |"All Bands", "Multispectral" ,                |
|                        |                        |                                              |"Panchromatic",                               | 
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"GF-1 WFV"              |                        | "All" (default), "LEVEL 1"                   |"Multispectral" (default)                     |
|                        |                        |                                              |                                              |
|                        |                        |                                              |                                              |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"GF-2 PMS"              |"0.2501,0.3646,0,0.3853"| "All" (default), "LEVEL 1"                   |"All Bands", "Multispectral" ,                |
|                        |                        |                                              |"Panchromatic",                               | 
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"KOMPSAT-2"             |"0.166,0.167,0.167,0.5" | "All" (default), "KOMPSAT-2 Scene 1A",       |"All Bands", "Multispectral",                 |
|                        |                        | "KOMPSAT-2 Scene 2A"                         |"Panchromatic",                               | 
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"KOMPSAT-3"             |"0.166,0.167,0.167,0.5" | "All" (default), "Level 1G", "Level 1O"      |"All Bands", "Multispectral" ,                |
|                        |                        | "Level 1R"                                   |"Panchromatic",                               | 
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"Landsat 9"             |"0.42, 0.51, 0.07, 0.0" |"All" (default)                               |"All Bands",                                  |
|                        |                        |                                              |"Brightness Temperature",                     |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce", "Spectral Indices",                |
|                        |                        |                                              |"Surface Reflectance", "Thermal",             |
|                        |                        |                                              |"Top of Atmosphere Reflectance",              |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        |                                              |"Surface Temperature"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level1"                                      |"All Bands",                                  |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce",                                    |
|                        |                        |                                              |"Thermal",                                    |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Reflectance"                         |"Surface Reflectance"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Spectral Indices"                            |"Spectral Indices"                            |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Top of Atmosphere Reflectance"               |"Top of Atmosphere Reflectance"               |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Brightness Temperature"                      |"Brightness Temperature"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Temperature"                         |"Surface Temperature"                         |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"Landsat 8"             |"0.42, 0.51, 0.07, 0.0" |"All" (default)                               |"All Bands",                                  |
|                        |                        |                                              |"Brightness Temperature",                     |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce", "Spectral Indices",                |
|                        |                        |                                              |"Surface Reflectance", "Thermal",             |
|                        |                        |                                              |"Top of Atmosphere Reflectance",              |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        |                                              |"Surface Temperature"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level1"                                      |"All Bands",                                  |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce",                                    |
|                        |                        |                                              |"Thermal",                                    |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Reflectance"                         |"Surface Reflectance"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Spectral Indices"                            |"Spectral Indices"                            |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Top of Atmosphere Reflectance"               |"Top of Atmosphere Reflectance"               |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Brightness Temperature"                      |"Brightness Temperature"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Temperature"                         |"Surface Temperature"                         |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"Landsat 7 ETM+"        |"0.11, 0.14, 0.14, 0.61"| "All" (default)                              |"All Bands",                                  |
|                        |                        |                                              |"Brightness Temperature",                     |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce", "Spectral Indices",                |
|                        |                        |                                              |"Surface Reflectance", "Thermal",             |
|                        |                        |                                              |"Top of Atmosphere Reflectance",              |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        |                                              |"Surface Temperature"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level1"                                      |"All Bands",                                  |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral",                 |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),"QA",|
|                        |                        |                                              |"SnowIce",                                    |
|                        |                        |                                              |"Thermal",                                    |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Reflectance"                         |"Surface Reflectance"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Spectral Indices"                            |"Spectral Indices"                            |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Top of Atmosphere Reflectance"               |"Top of Atmosphere Reflectance"               |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Brightness Temperature"                      |"Brightness Temperature"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Temperature"                         |"Surface Temperature"                         |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"Landsat 4-5 TM"        |                        | "All" (default)                              |"All Bands",                                  |
|                        |                        |                                              |"Brightness Temperature",                     |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral" (default),       |
|                        |                        |                                              |"Panchromatic", "Pansharpen",                 |
|                        |                        |                                              |"Pansharpen and Multispectral", "QA",         |
|                        |                        |                                              |"SnowIce", "Spectral Indices",                |
|                        |                        |                                              |"Surface Reflectance", "Thermal",             |
|                        |                        |                                              |"Top of Atmosphere Reflectance",              |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        |                                              |"Surface Temperature"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level1"                                      |"All Bands",                                  |
|                        |                        |                                              |"Cirrus", "Cloud","Landcover",                |
|                        |                        |                                              |"Multiband", "Multispectral" (default),       |
|                        |                        |                                              |"Panchromatic", "Pansharpen" ,                |
|                        |                        |                                              |"Pansharpen and Multispectral", "QA",         |
|                        |                        |                                              |"SnowIce",                                    |
|                        |                        |                                              |"Thermal",                                    |
|                        |                        |                                              |"Vegetation", "Water"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Reflectance"                         |"Surface Reflectance"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Spectral Indices"                            |"Spectral Indices"                            |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Top of Atmosphere Reflectance",              |"Top of Atmosphere Reflectance",              |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Brightness Temperature",                     |"Brightness Temperature",                     |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Surface Temperature"                         |"Surface Temperature"                         |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
|"Landsat 1-5 MSS"       |                        | "All" (default), "Level1"                    |"All Bands", "Multiband",                     |
|                        |                        |                                              |"Multispectral" (default),                    |
|                        |                        |                                              |"Pseudocolor", "Thermal", "QA"                |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+ 
| "Sentinel-2"           |"0.85, 0.7, 0.35, 1"    |"ALL" (default)                               |"All Bands",                                  |
|                        |                        |                                              |"BOA Reflectance-10m",                        |
|                        |                        |                                              |"BOA Reflectance-20m",                        |
|                        |                        |                                              |"BOA Reflectance-60m",                        |
|                        |                        |                                              |"Multispectral" (default),                    |
|                        |                        |                                              |"Multispectral-10m",                          |
|                        |                        |                                              |"Multispectral-20m",                          |
|                        |                        |                                              |"Multispectral-60m",                          |
|                        |                        |                                              |"True Color",                                 |
|                        |                        |                                              |"SCL-20m",                                    |
|                        |                        |                                              |"SCL-60m"                                     |
|                        |                        |                                              |"Aerosol Optical Thickness",                  |
|                        |                        |                                              |"Water Vapour",                               |
|                        |                        |                                              |"BOA Reflectance"                             |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level1"                                      |"Multispectral" (default),                    |
|                        |                        |                                              |"Multispectral-10m",                          |
|                        |                        |                                              |"Multispectral-20m",                          |
|                        |                        |                                              |"Multispectral-60m",                          |
|                        |                        |                                              |"True Color"                                  |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Level2"                                      |"BOA Reflectance-10m",                        |
|                        |                        |                                              |"BOA Reflectance-20m",                        |
|                        |                        |                                              |"BOA Reflectance-60m"                         |
|                        |                        |                                              |"Aerosol Optical Thickness",                  |
|                        |                        |                                              |"Water Vapour",                               |
|                        |                        |                                              |"BOA Reflectance"                             |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+ 
| "GeoEye-1"             |"0.41, 0.16, 0.13, 0.3" |"All" (default), "Basic",                     |"All Bands",                                  |
|                        |                        |"Geo", "GeoProfessional"                      |"Multispectral",                              |
|                        |                        |"Orthorectified",                             |"Multispectral AComp",                        |
|                        |                        |"Standard",                                   |"Panchromatic",                               |
|                        |                        |"Standard Orthoready",                        |"Panchromatic AComp",                         |
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen AComp",                           |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
|                        |                        |                                              |                                              |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Orthorectified Acomp",                       |"Multispectral AComp" (default),              |
|                        |                        |"Standard AComp",                             |"Panchromatic AComp",                         |
|                        |                        |"Standard OrthoreadyAcomp"                    |"Pansharpen AComp"                            |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "Pleiades-1"           |"0.9, 0.75, 0.5, 0.5"   |"All" (default), "ORTHO",                     |"All Bands",                                  |
|                        |                        |"PRIMARY", "PROJECTED"                        |"Multispectral",                              |
|                        |                        |                                              |"Multispectral Display",                      | 
|                        |                        |                                              |"Multispectral Reflectance",                  |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Pansharpen Display",                         |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO DISPLAY"                               |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO REFLECTANCE"                           |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY DISPLAY"                             |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY REFLECTANCE"                         |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED DISPLAY"                           |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Dispplay"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED REFLECTANCE"                       |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "Pleiades Neo"         |"0.45, 0.55, 0, 0"      |"All" (default), "ORTHO",                     |"All Bands",                                  |
|                        |                        |"PRIMARY", "PROJECTED"                        |"Multispectral",                              |
|                        |                        |                                              |"Multispectral Display",                      | 
|                        |                        |                                              |"Multispectral Reflectance",                  |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Pansharpen Display",                         |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO DISPLAY"                               |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO REFLECTANCE"                           |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY DISPLAY"                             |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY REFLECTANCE"                         |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED DISPLAY"                           |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Dispplay"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED REFLECTANCE"                       |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "SPOT 5"               |"0.166,0.167,0.167,0.5" |"All" (default), "SPOT Scene 1A",             |"All Bands",                                  |
|                        |                        |"SPOT Scene 2A", "SPOTView Ortho"             |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Pseudocolor"                                 |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "SPOT 6"               |"0.45, 0.55, 0, 0"      |"All" (default), "ORTHO",                     |"All Bands",                                  |
|                        |                        |"PRIMARY", "PROJECTED"                        |"Multispectral",                              |
|                        |                        |                                              |"Multispectral Display",                      | 
|                        |                        |                                              |"Multispectral Reflectance",                  |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Pansharpen Display",                         |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO DISPLAY"                               |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO REFLECTANCE"                           |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY DISPLAY"                             |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY REFLECTANCE"                         |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED DISPLAY"                           |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Dispplay"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED REFLECTANCE"                       |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "SPOT 7"               |"0.45, 0.55, 0, 0"      |"All" (default), "ORTHO",                     |"All Bands",                                  |
|                        |                        |"PRIMARY", "PROJECTED"                        |"Multispectral",                              |
|                        |                        |                                              |"Multispectral Display",                      | 
|                        |                        |                                              |"Multispectral Reflectance",                  |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Pansharpen Display",                         |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO DISPLAY"                               |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"ORTHO REFLECTANCE"                           |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY DISPLAY"                             |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Display"                          |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PRIMARY REFLECTANCE"                         |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED DISPLAY"                           |"Multispectral Display" (default),            |
|                        |                        |                                              |"Panchromatic Display",                       |
|                        |                        |                                              |"Pansharpen Dispplay"                         |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"PROJECTED REFLECTANCE"                       |"Multispectral Reflectance" (default),        |
|                        |                        |                                              |"Panchromatic Reflectance",                   |
|                        |                        |                                              |"Pansharpen Reflectance"                      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "IKONOS"               |"0.378, 0.211, 0, 0.411"|"All" (default), "Geo",                       |"All Bands",                                  |
|                        |                        |"GeoProfessional",                            |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "WorldView-1"          |                        |"All" (default), "Basic",                     |"Panchromatic" (default)                      |
|                        |                        |"Orthorectified", "Standard",                 |                                              |
|                        |                        |"Standard Orthoready"                         |                                              |
|                        |                        |                                              |                                              |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "WorldView-2"          |"0.39, 0.23, 0.21, 0.17"|"All" (default), "Basic",                     |"All Bands",                                  |
|                        |                        |"Orthorectified",                             |"Multispectral",                              |
|                        |                        |"Standard Orthoready",                        |"Multispectral AComp"                         |
|                        |                        |"Standard"                                    |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic AComp",                         |
|                        |                        |                                              |"Pansharpen AComp",                           |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Orthorectified Acomp",                       |"Multispectral AComp" (default),              |
|                        |                        |"Standard AComp",                             |"Panchromatic AComp",                         |
|                        |                        |"Standard OrthoreadyAcomp"                    |"Pansharpen AComp"                            |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "WorldView-3"          |"0.38, 0.25, 0.2, 0.16" |"All" (default), "Basic",                     |"All Bands",                                  |
|                        |                        |"Orthorectified",                             |"Multispectral",                              |
|                        |                        |"Standard Orthoready",                        |"Multispectral AComp"                         |
|                        |                        |"Standard"                                    |"Panchromatic" ,                              |
|                        |                        |                                              |"Panchromatic AComp",                         |
|                        |                        |                                              |"Pansharpen",                                 |
|                        |                        |                                              |"Pansharpen AComp",                           |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"SWIR"                                        |
|                        |                        +----------------------------------------------+----------------------------------------------+
|                        |                        |"Orthorectified Acomp",                       |"Multispectral AComp" (default),              |
|                        |                        |"Standard AComp",                             |"Panchromatic AComp",                         |
|                        |                        |"Standard OrthoreadyAcomp"                    |"Pansharpen AComp"                            |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "WorldView-4"          |"0.39, 0.23, 0.21, 0.17"|"All" (default), "Basic",                     |"All Bands"                                   |
|                        |                        |"Standard OrthoReady"                         |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic"                                |
|                        |                        |                                              |"Pansharpen" ,                                |
|                        |                        |                                              |"Pansharpen and Multispectral" (default),     |
|                        |                        |                                              |"Multispectral AComp",                        |
|                        |                        |                                              |"Panchromatic AComp",                         |
|                        |                        |                                              |"Pansharpen AComp"                            |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "ZY3-SASMAC"           |"0.329, 0.509, 0, 0.162"|"All" (default), "LEVEL1"                     |"All Bands"                                   |
|                        |                        |                                              |"Double Line Stereo",                         |
|                        |                        |                                              |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Panchromatic BWD",                           |
|                        |                        |                                              |"Panchromatic FWD",                           |
|                        |                        |                                              |"Panchromatic NAD",                           |
|                        |                        |                                              |"Pansharpen"                                  |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
|                        |                        |                                              |"Three Line Stereo"                           |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "Raster Dataset"       |                        |                                              |                                              |
|                        |                        |                                              |                                              |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "NetCDF"               |                        |"All" (default)                               |"Default"(default),                           |
|                        |                        |                                              |"Multiband Composite",                        |
|                        |                        |                                              |"Vector Field"                                |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "GRIB"                 |                        |"All" (default)                               |"Default"(default),                           |
|                        |                        |                                              |"Multiband Composite",                        |
|                        |                        |                                              |"Vector Field"                                |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "HDF"                  |                        |"All" (default)                               |"Default"(default),                           |
|                        |                        |                                              |"Multiband Composite",                        |
|                        |                        |                                              |"Vector Field"                                |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "ASTER"                |                        |"All" (default), "L1T",                       |"VNIR" (default),                             |
|                        |                        |"L1B", "Surface Reflectance"                  |"SWIR",                                       |
|                        |                        |"Surface Radiance",                           |"TIR",                                        |
|                        |                        |"Surface Emissivity"                          |"Multispectral" (default),                    |
|                        |                        |"Surface Kinetic Temperature"                 |"TIR - Surface Emissivity",                   |
|                        |                        |"DEM"                                         |"VNIR - Surface Reflectance",                 |
|                        |                        |"Registered Radiance"                         |"SWIR - Surface Reflectance",                 |
|                        |                        |                                              |"TIR - Surface kinetic Temperature",          |
|                        |                        |                                              |"VNIR - Surface Radiance",                    |
|                        |                        |                                              |"SWIF - Surface Radiance",                    |
|                        |                        |                                              |"TIR - Surface Radiance",                     |
|                        |                        |                                              |"VNIR - Registered Radiance",                 |
|                        |                        |                                              |"VNIR - Registered Radiance",                 |
|                        |                        |                                              |"SWIR - Registered Radiance",                 |
|                        |                        |                                              |"TIR - Registered Radiance",                  |
|                        |                        |                                              |"Global Digital Elevation Model (ASTGTM)",    |
|                        |                        |                                              |"Digital Elevation Model",                    |
|                        |                        |                                              |"All Bands"                                   |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "DMCii"                |                        |"All" (default), "L1R", "L1T"                 |"Multispectral" (default),                    |
|                        |                        |                                              |"ALL",                                        |
|                        |                        |                                              |"Pseudocolor"                                 |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "RapidEye"             |                        |"All", "Level1B",                             |"Multispectral" (default)                     |
|                        |                        |"Level3A", "Level3B"                          |                                              |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "SkySat"               |"0.41, 0.16, 0.13, 0.3" |"All" (default), "Basic", "Ortho"             |"All Bands",                                  |
|                        |                        |                                              |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Pansharpen" ,                                |
|                        |                        |                                              |"Skysat Pansharpen",                          |
|                        |                        |                                              |"Visual"                                      |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "Jilin-1"              |"0.2501, 0.3646, 0"     |"All" (default), "LEVEL1"                     |"All Bands",                                  |
|                        |                        |                                              |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Pansharpen" ,                                |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "PlanetScope"          |"0.166,0.167,0.167,0.5" |"All" (default), "Level1", "Level3"           |"Multispectral" (default)                     |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+
| "SuperView-1"          |"0.85, 0.7, 0.35, 1"    |"All" (default), "LEVEL1B", "LEVEL2A",        |"All Bands",                                  |
|                        |                        |"LEVEL3A"                                     |"Multispectral",                              |
|                        |                        |                                              |"Panchromatic",                               |
|                        |                        |                                              |"Pansharpen" ,                                |
|                        |                        |                                              |"Pansharpen and Multispectral" (default)      |
+------------------------+------------------------+----------------------------------------------+----------------------------------------------+

.. code-block:: python

        # Usage Example 1: create image collection using a list of image items.

        image_item_list = [<Item title:"YUN_0040.JPG" type:Image owner:admin>,
                           <Item title:"YUN_0041.JPG" type:Image owner:admin>,
                           <Item title:"YUN_0042.JPG" type:Image owner:admin>,
                           <Item title:"YUN_0043.JPG" type:Image owner:admin>,
                           <Item title:"YUN_0044.JPG" type:Image owner:admin>]
                            
        params = {"gps": [['YUN_0040.JPG', 34.0069887, -117.09279029999999],
                          ['YUN_0041.JPG', 34.0070131, -117.09311519972222],
                          ['YUN_0042.JPG', 34.0070381, -117.09346329972222],
                          ['YUN_0043.JPG', 34.00706339972222, -117.09381479999999],
                          ['YUN_0044.JPG', 34.0070879, -117.09416449999999]],
                  "cameraProperties":{"maker":"Yuneec",
                                      "model":"E90",
                                      "focallength":8,
                                      "columns":5472,
                                      "rows":3648,
                                      "pixelsize":0.0024},
                  "isAltitudeFlightHeight":"false",
                  "averagezdem": {"url": "https://rais.dev.geocloud.com/arcgis/rest/services/Hosted/WorldSRTM90m/ImageServer"}}

        img_coll_result = create_image_collection(image_collection="imageCollection",
                                                  input_rasters=image_item_list,
                                                  raster_type_name="UAV/UAS",
                                                  raster_type_params=params,
                                                  out_sr=32632)

        # Usage Example 2: This example creates an image collection from the datasets detected in the input folder.

        img_coll_result = create_image_collection(image_collection="imageCollection",
                                                  input_rasters=r"C:\data",
                                                  raster_type_name="Raster Dataset",
                                                  gis=gis)
        
        # Usage Example 3: This example creates an image collection using Landsat 1-5 MSS Raster Type with a Level 1 product and a Multispectral template.

        landsat15_collection = create_image_collection(image_collection="landsat15_collection",
                                                       input_rasters=r"C:\data\Landsat15MSS",
                                                       raster_type_name="Landsat 1-5 MSS",
                                                       raster_type_params={"productType": "Level1", "processingTemplate": "Multispectral"},
                                                       gis=gis)

        # Usage Example 4: This example creates an image collection using Sentinel-2 Raster Type with a Level 1 product and a Multispectral template.

        sentinel2_collection = create_image_collection(image_collection="sentinel2_collection",
                                                       input_rasters=r"C:\data\Sentinel2",
                                                       raster_type_name="Sentinel-2",
                                                       raster_type_params={"productType": "Level1", "processingTemplate": "Multispectral-10m"},
                                                       gis=gis)

        # Usage Example 5: This example creates an image collection using Pleiades-1 Raster Type with an Ortho product and a Pansharpened template.

        pleiades1_collection = create_image_collection(image_collection="pleiades1_collection",
                                                       input_rasters=r"C:\data\Pleiades1",
                                                       raster_type_name="Pleiades-1",
                                                       raster_type_params={"productType":"ORTHO","processingTemplate":"Pansharpen"},
                                                       gis=gis)

        # Usage Example 6: This example creates an image collection using SPOT 6 Raster Type with an Ortho Reflectance product and a Pansharpen Reflectance template.

        spot6_collection = create_image_collection(image_collection="spot6_collection",
                                                   input_rasters=r"C:\data\SPOT6",
                                                   raster_type_name="SPOT 6",
                                                   raster_type_params={"productType": "ORTHO REFLECTANCE", "processingTemplate": "Pansharpen Reflectance"},
                                                   gis=gis)

        # Usage Example 7: This example creates an image collection using WorldView-2 Raster Type with a Pansharpen and Multispectral template.

        worldview2_collection = create_image_collection(image_collection="worldview2_collection",
                                                        input_rasters=r"C:\data\WorldView",
                                                        raster_type_name="WorldView-2",
                                                        raster_type_params= {"productType": "Basic", "processingTemplate": "Pansharpen and Multispectral"},
                                                        gis=gis)

        # Usage Example 8: This example creates an image collection from Tiled Imagery Layers.

        tiled_layers_collection = create_image_collection(image_collection="tiled_collection",
                                                          input_rasters=[tile_lyr_input1, tile_lyr_input2],
                                                          raster_type_name="Tiled Imagery Layer",
                                                          gis=gis)
        
        # Usage Example 9: This example creates an image collection on ArcGIS Enterprise from datastore by reference.

        sentinel_by_ref_collection = create_image_collection(image_collection="sentinel_by_ref_multispectral",
                                                             input_rasters=["/rasterStores/LocalRasterStore/S2A_MSIL1C_20180624T184921_N0206_R113_T10SEJ_20180624T234856.SAFE",
                                                                            "/rasterStores/LocalRasterStore/S2B_MSIL1C_20180622T185919_N0206_R013_T10SEJ_20180622T205930.SAFE"
                                                                           ],
                                                             raster_type_name="Sentinel-2",
                                                             raster_type_params={"productType":"All","processingTemplate":"Multispectral"},
                                                             context={"image_collection_properties":{"imageCollectionType":"Satellite"},"byref":True},
                                                             gis = gis)

        # Usage Example 10: This example publishes an existing mosaic dataset as a dynamic imagery layer in ArcGIS Online. 

        # Specify the actual source data path referenced by the uploaded mosaic dataset using input_rasters parameter. The data would be uploaded 
        # to the ArcGIS Online's user store from this path. 

        landsat_mosaic  = create_image_collection(image_collection="landsat_image_collection",
                                                  input_rasters=[r"C:\data\landsat_data_folder"],
                                                  raster_type_name=None,
                                                  context={"upload_properties":{"displayProgress":True}},
                                                  source_mosaic_dataset=r"C:\data\md.gdb\landsat",
                                                  gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    # url = gis.properties.helperServices.rasterAnalytics.url
    return gis._tools.rasteranalysis.create_image_collection(
        image_collection=image_collection,
        input_rasters=input_rasters,
        raster_type_name=raster_type_name,
        raster_type_params=raster_type_params,
        out_sr=out_sr,
        context=context,
        future=future,
        md_to_upload=source_mosaic_dataset,
        **kwargs,
    )


###################################################################################################
## Add image
###################################################################################################
def add_image(
    image_collection: Item,
    input_rasters: list,
    raster_type_name: Optional[str] = None,
    raster_type_params: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/add_image/add_image.png 

    Add a collection of images to an existing image collection. It provides provision to use input rasters by reference 
    and to specify image collection properties through context parameter.

    It can be used when new data is available to be included in the same 
    orthomapping project. When new data is added to the image collection
    the entire image collection must be reset to the original state.

    ==================                   ====================================================================
    **Parameter**                         **Description**
    ------------------                   --------------------------------------------------------------------
    input_rasters                        Required list. The list of input rasters to be added to
                                         the image collection being created. This parameter can
                                         be any one of the following types:
    
                                         - List of portal Items of the images
                                         - An image service URL
                                         - Shared data path (this path must be accessible by the server)
                                         - Name of a folder on the portal
    ------------------                   --------------------------------------------------------------------
    image_collection                     Required item. The item representing the image collection to add ``input_rasters`` to.
                  
                                         The image collection must be an existing image collection.
                                         This is the output image collection (mosaic dataset) item or url or uri.
    ------------------                   --------------------------------------------------------------------
    raster_type_name                     Optional string. The name of the raster type to use for adding data to
                                         the image collection.


                                         Choice list:

                                             | [
                                             | "Aerial", "ASTER", "DMCII", "DubaiSat-2", "GeoEye-1", "GF-1 PMS", "GF-1 WFV",
                                             | "GF-2 PMS", "GRIB", "HDF", "IKONOS", "Jilin-1", "KOMPSAT-2", "KOMPSAT-3",
                                             | "Landsat 1-5 MSS", "Landsat 4-5 TM", "Landsat 7 ETM+", "Landsat 8", "Landsat 9",
                                             | "NetCDF", "PlanetScope", "Pleiades-1", "Pleiades NEO", "QuickBird", "RapidEye",
                                             | "Raster Dataset", "ScannedAerial", "Sentinel-2", "SkySat", "SPOT 5", "SPOT 6",
                                             | "SPOT 7", "Superview-1", "Tiled Imagery Layer", "UAV/UAS", "WordView-1",
                                             | "WordView-2", "WordView-3", "WordView-4", "ZY3-SASMAC", "ZY3-CRESDA"
                                             | ]
                                         

                                         Example:

                                            "QuickBird"
    ------------------                   --------------------------------------------------------------------
    raster_type_params                   Optional dict. Additional ``raster_type`` specific parameters.
        
                                         The process of add rasters to the image collection can be \
                                         controlled by specifying additional raster type arguments.

                                         The raster type parameters argument is a dictionary.
                                         
                                         Syntax:

                                             {"gps": [["image1.jpg", "10", "2", "300"], ["image2.jpg", "10", "3", "300"], ["image3.jpg", "10", "4", "300"]],
                                             "cameraProperties": {"Maker": "Canon", "Model": "5D Mark II", "FocalLength": 20, "PixelSize": 10, "x0": 0, "y0": 0, "columns": 4000, "rows": 3000},
                                             "constantZ": 300,"isAltitudeFlightHeight": "True","dem": {"url": ``https://...``}

                                         The dictionary can contain productType, processingTemplate, \
                                         pansharpenType, Filter, pansharpenWeights, ConstantZ, \
                                         dem, zoffset, CorrectGeoid, ZFactor, StretchType, \
                                         ScaleFactor, ValidRange

                                         Please check the table below (Supported Raster Types), \
                                         for more details about the product types, \
                                         processing templates, pansharpen weights for each raster type. 

                                         - Possible values for pansharpenType - ["Mean", "IHS", "Brovey", "Esri", "Mean", "Gram-Schmidt"]
                                         - Possible values for filter - [None, "Sharpen", "SharpenMore"]
                                         - Value for StretchType dictionary can be as follows:

                                           - "None"
                                           - "MinMax; <min>; <max>"
                                           - "PercentMinMax; <MinPercent>; <MaxPercent>"
                                           - "StdDev; <NumberOfStandardDeviation>"
                                           Example: {"StretchType": "MinMax; <min>; <max>"}
                                         - Value for ValidRange dictionary can be as follows:

                                           - "<MaskMinValue>, <MaskMaxValue>"
                                           Example: {"ValidRange": "10, 200"}

                                         Example:

                                            {"productType":"All","processingTemplate":"Pansharpen",
                                            "pansharpenType":"Gram-Schmidt","filter":"SharpenMore",
                                            "pansharpenWeights":"0.85 0.7 0.35 1","constantZ":-9999}
    ------------------                   --------------------------------------------------------------------
    context                              Optional dict. The context parameter is used to provide additional input parameters.

                                         Syntax:

                                            {"image_collection_properties": {"imageCollectionType":"Satellite"},"byref":'True'}
                                            
                                         Use ``image_collection_properties`` key to set value for imageCollectionType.


                                         .. note::

                                            The "imageCollectionType" property is important for image collection that will later on be adjusted by orthomapping system service.
                                            Based on the image collection type, the orthomapping system service will choose different algorithm for adjustment.
                                            Therefore, if the image collection is created by reference, the requester should set this
                                            property based on the type of images in the image collection using the following keywords.
                                            If the imageCollectionType is not set, it defaults to "UAV/UAS"
 
                                         If byref is set to 'True', the data will not be uploaded. If it is not set, the default is 'False'
    ------------------                   --------------------------------------------------------------------
    gis                                  Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------                   --------------------------------------------------------------------
    future                               Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                         results will be returned asynchronously.
    ==================                   ====================================================================

    :return: The imagery layer item

    .. code-block:: python

            # Usage Example: To add an image to an existing image collection.

            params = {"gps":[["YUN_0040.JPG",34.006989,-117.09279,725.13]],
                      "cameraProperties":{"maker":"Yuneec","model":"E90","focallength":8,"columns":5472,"rows":3648,"pixelsize":0.0024},
                      "isAltitudeFlightHeight":"false",
                      "averagezdem": {"url": "https://rais.dev.geocloud.com/arcgis/rest/services/Hosted/WorldSRTM90m/ImageServer"}}

            add_image(image_collection=image_collection, input_rasters=[image_item], raster_type_name="UAV/UAS", raster_type_params=params)
  
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.add_image(
        image_collection=image_collection,
        input_rasters=input_rasters,
        raster_type_name=raster_type_name,
        raster_type_params=raster_type_params,
        context=context,
        future=future,
        **kwargs,
    )


###################################################################################################
## Delete image
###################################################################################################
def delete_image(
    image_collection: Item,
    where: str,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/delete_image/delete_image.png

    ``delete_image`` allows users to remove existing images from the image collection (mosaic dataset).
    The function will only delete the raster item in the mosaic dataset and will not remove the
    source image.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection from which to delete images
                           This can be the 'itemID' of an exisiting portal item or a url
                           to an Image Service or a uri
    ------------------     --------------------------------------------------------------------
    where                  Required string. A SQL ``where`` clause for selecting the images
                           to be deleted from the image collection
    ------------------     --------------------------------------------------------------------
    gis                    Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------     --------------------------------------------------------------------
    future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                           results will be returned asynchronously.
    ==================     ====================================================================

    :return: The imagery layer url

    .. code-block:: python

         # Usage Example: To delete an existing image from the image collection.

         del_img = delete_image(image_collection=img_coll_item, where="OBJECTID=10")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.delete_image(
        image_collection=image_collection, where=where, future=future, **kwargs
    )


###################################################################################################
## Delete image collection
###################################################################################################
def delete_image_collection(
    image_collection: Item, *, gis: Optional[GIS] = None, future: bool = False, **kwargs
):
    """
    .. image:: _static/images/delete_image_collection/delete_image_collection.png

    Delete the image collection. This service tool will delete the image collection
    image service, that is, the portal-hosted image layer item. It will not delete
    the source images that the image collection references.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    image_collection       Required, the input image collection to delete.

                           The image_collection can be a portal Item or an image service URL or a URI.

                           The image_collection must exist.
    ------------------     --------------------------------------------------------------------
    gis                    Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------     --------------------------------------------------------------------
    future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                           results will be returned asynchronously.
    ==================     ====================================================================

    :return: Boolean value indicating whether the deletion was successful or not.

    .. code-block:: python

            # Usage Example: To delete an existing image collection.

            delete_flag = delete_image_collection(image_collection=image_collection_item)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.delete_image_collection(
        image_collection=image_collection, future=future, **kwargs
    )


def _flow_direction(
    input_surface_raster,
    force_flow=False,
    flow_direction_type="D8",
    output_flow_direction_name=None,
    output_drop_name=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Replaces cells of a raster corresponding to a mask
    with the values of the nearest neighbors.

    Parameters
    ----------
    input_surface_raster : The input raster representing a continuous surface.

    force_flow  : Boolean, Specifies if edge cells will always flow outward or follow normal flow rules.

    flow_direction_type : Specifies which flow direction type to use.
                          D8 - Use the D8 method. This is the default.
                          MFD - Use the Multi Flow Direction (MFD) method.
                          DINF - Use the D-Infinity method.

    output_drop_name : An optional output drop raster .
                       The drop raster returns the ratio of the maximum change in elevation from each cell
                       along the direction of flow to the path length between centers of cells, expressed in percentages.

    output_flow_direction_name : Optional. If not provided, an Image Service is created by the method and used as the output raster.
        You can pass in an existing Image Service Item from your GIS to use that instead.
        Alternatively, you can pass in the name of the output Image Service that should be created by this method to be used as the output for the tool.
        A RuntimeError is raised if a service by that name already exists

    gis: Keyword only parameter. Optional, the GIS on which this tool runs. If not specified, the active GIS is used.

    future: Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
            results will be returned asynchronously.

    Returns
    -------
    output_raster : Image layer item
    """

    # task = "FlowDirection"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.flow_direction(
        input_surface_raster=input_surface_raster,
        output_flow_direction_name=output_flow_direction_name,
        force_flow=force_flow,
        flow_direction_type=flow_direction_type,
        output_drop_name=output_drop_name,
        future=future,
        **kwargs,
    )


def _calculate_travel_cost(
    input_source,
    input_cost_raster=None,
    input_surface_raster=None,
    maximum_distance=None,
    input_horizonal_raster=None,
    horizontal_factor="BINARY",
    input_vertical_raster=None,
    vertical_factor="BINARY",
    source_cost_multiplier=None,
    source_start_cost=None,
    source_resistance_rate=None,
    source_capacity=None,
    source_direction="FROM_SOURCE",
    allocation_field=None,
    output_distance_name=None,
    output_backlink_name=None,
    output_allocation_name=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """

    Parameters
    ----------
    input_source : The layer that defines the sources to calculate the distance too. The layer
                   can be raster or feature.

    input_cost_raster  : A raster defining the impedance or cost to move planimetrically through each cell.

    input_surface_raster : A raster defining the elevation values at each cell location.

    maximum_distance : The maximum distance to calculate out to. If no distance is provided, a default will
                       be calculated that is based on the locations of the input sources.

    input_horizonal_raster : A raster defining the horizontal direction at each cell.

    horizontal_factor : The Horizontal Factor defines the relationship between the horizontal cost
                        factor and the horizontal relative moving angle.

    input_vertical_raster : A raster defining the vertical (z) value for each cell.

    vertical_factor : The Vertical Factor defines the relationship between the vertical cost factor and
                      the vertical relative moving angle (VRMA).

    source_cost_multiplier : Multiplier to apply to the cost values.

    source_start_cost : The starting cost from which to begin the cost calculations.

    source_resistance_rate : This parameter simulates the increase in the effort to overcome costs
                            as the accumulative cost increases.

    source_capacity : Defines the cost capacity for the traveler for a source.

    source_direction : Defines the direction of the traveler when applying horizontal and vertical factors,
                       the source resistance rate, and the source starting cost.

    allocation_field : A field on theinputSourceRasterOrFeatures layer that holds the values that define each source.

    output_backlink_name  : This is the output image service name that will be created.

    output_allocation_name : This is the output image service name that will be created.

    output_distance_name : Optional. If not provided, an Image Service is created by the method and used as the output raster.
        You can pass in an existing Image Service Item from your GIS to use that instead.
        Alternatively, you can pass in the name of the output Image Service that should be created by this method to be used as the output for the tool.
        A RuntimeError is raised if a service by that name already exists

    gis: Keyword only parameter. Optional, the GIS on which this tool runs. If not specified, the active GIS is used.

    future: Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
            results will be returned asynchronously.

    Returns
    -------
    output_raster : Image layer item
    """

    # task = "CalculateTravelCost"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.calculate_travel_cost(
        input_source_raster_or_features=input_source,
        output_name=output_distance_name,
        input_cost_raster=input_cost_raster,
        input_surface_raster=input_surface_raster,
        maximum_distance=maximum_distance,
        input_horizontal_raster=input_horizonal_raster,
        horizontal_factor=horizontal_factor,
        input_vertical_raster=input_vertical_raster,
        vertical_factor=vertical_factor,
        source_cost_multiplier=source_cost_multiplier,
        source_start_cost=source_start_cost,
        source_resistance_rate=source_resistance_rate,
        source_capacity=source_capacity,
        source_travel_direction=source_direction,
        output_backlink_name=output_backlink_name,
        output_allocation_name=output_allocation_name,
        allocation_field=allocation_field,
        future=future,
        **kwargs,
    )


@deprecated(
    deprecated_in="1.8.1",
    details="Please use arcgis.raster.analytics.optimal_region_connections instead. ",
)
def optimum_travel_cost_network(
    input_regions_raster,
    input_cost_raster,
    output_optimum_network_name: Optional[str] = None,
    output_neighbor_network_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_optimum_travel_cost_network/ra_optimum_travel_cost_network.png

    Calculates the optimum cost network from a set of input regions.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_regions_raster                     Required Imagery Layer object. The layer that defines the regions to find the optimum travel cost netork for.
                                             The layer can be raster or feature.
    ------------------------------------     --------------------------------------------------------------------
    input_cost_raster                        Required Imagery Layer object. A raster defining the impedance or cost to
                                             move planimetrically through each cell.
    ------------------------------------     --------------------------------------------------------------------
    output_optimum_network_name              Optional. If not provided, a feature layer is created by the method and used as the output.
                                             You can pass in an existing feature layer Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output feature layer  that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    output_neighbor_network_name             Optional. This is the name of the output neighbour network feature layer that will be created.
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
    output_raster : Imagery layer item
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.determine_optimum_travel_cost_network(
        input_regions_raster_or_features=input_regions_raster,
        input_cost_raster=input_cost_raster,
        output_optimum_network_name=output_optimum_network_name,
        output_neighbor_network_name=output_neighbor_network_name,
        context=context,
        future=future,
        **kwargs,
    )


def list_datastore_content(
    datastore: Union[str, list],
    filter: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """

    List the contents of the datastore registered with the server (fileShares, cloudStores, rasterStores).

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    datastore              Required string or list. fileshare, rasterstore or cloudstore datastore from which the contents are to be listed.
                           It can be a string specifying the datastore path example: "/fileShares/SensorData", "/cloudStores/testcloud",
                           "/rasterStores/rasterstore"
                           or it can be a Datastore object containing a fileshare, rasterstore  or a cloudstore path.

                           Example:

                              | ds=analytics.get_datastores()
                              | ds_items=ds.search()
                              | ds_items[1]
                              | ds_items[1] may be specified as input for datastore

                           It can also be a list of datastore paths or list of datastore object containing a fileshare,
                           rasterstore or cloudstore path.

                           In order to list the datastore items, one can specify just the name of the datastore

                           eg: fileShares
                           or
                           eg: cloudStore,rasterStore
    ------------------     --------------------------------------------------------------------
    filter                 Optional. To filter out the raster contents to be displayed
    ------------------     --------------------------------------------------------------------
    gis                    Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------     --------------------------------------------------------------------
    future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                           results will be returned asynchronously.
    ==================     ====================================================================

    :return:
        List of contents in the datastore

    .. code-block:: python

        # Usage Example 1: Returns the contents of the rasterstore registered with the server.

        datastore_op = list_datastore_content("/rasterStores/rasterstore", gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.list_datastore_content(
        data_store_name=datastore, filter=filter, future=future, **kwargs
    )


def build_footprints(
    image_collection,
    computation_method: str = "RADIOMETRY",
    value_range=None,
    context=None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Computes the extent of every raster in an image collection.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    image_collection                         Required. The input image collection.The image_collection can be a
                                             portal Item or an image service URL or a URI.
                                             The image_collection must exist.
    ------------------------------------     --------------------------------------------------------------------
    computation_method                       Optional string. Refine the footprints using one of the following methods:
                                             RADIOMETRY, GEOMETRY
                                             Default: RADIOMETRY
    ------------------------------------     --------------------------------------------------------------------
    value_range                              Optional. Parameter to specify the value range.
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
    The imagery layer url

    .. code-block:: python

        # Usage Example 1: Determines the extent of every raster in the given image collection.
        # 'GEOMETRY' computation method refines the footprints

        collection = gis.content.search("my_img_collection")[0]

        collection_url = build_footprints(image_collection=collection.url,
                                          computation_method="GEOMETRY",
                                          gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.build_footprints(
        image_collection=image_collection,
        computation_method=computation_method,
        value_range=value_range,
        context=context,
        future=future,
        **kwargs,
    )


def build_overview(
    image_collection,
    cell_size: Optional[float] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Generates overviews on an image collection.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    image_collection                         Required. The input image collection.The image_collection can be a
                                             portal Item or an image service URL or a URI.
                                             The image_collection must exist.
    ------------------------------------     --------------------------------------------------------------------
    cell_size                                optional float or int, to set the cell size for overview.
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
    The imagery layer url

    .. code-block:: python

        # Usage Example 1: This snippet is a quick implementation of the build_overview function.

        collection = gis.content.search("my_img_collection")[0]

        collection_url = build_overview(image_collection=collection.url,
                                        gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.build_overview(
        image_collection=image_collection,
        cell_size=cell_size,
        context=context,
        future=future,
        **kwargs,
    )


def calculate_statistics(
    image_collection,
    skip_factors: Optional[dict[str, int]] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis=None,
    future: bool = False,
    **kwargs,
):
    """
    Calculates statistics for an image collection

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    image_collection                         Required. The input image collection.The image_collection can be a
                                             portal Item or an image service URL or a URI.
                                             The image_collection must exist.
    ------------------------------------     --------------------------------------------------------------------
    skip_factors                             Optional dictionary, Controls the portion of the raster that is used when calculating the statistics.

                                             Example:

                                                | {"x":5,"y":5}
                                                | x value represents - the number of horizontal pixels between samples
                                                | y value represents - the number of vertical pixels between samples.
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}

                                             Function also supports following keys through context:
                                             ignoreValues, skipExisting, areaOfInterest
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
    The imagery layer url

    .. code-block:: python

        # Usage Example 1: This snippet calculates statistics for the given image collection.

        collection = gis.content.search("my_img_collection")[0]

        collection_url = calculate_statistics(image_collection=collection.url,
                                              gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.calculate_statistics(
        image_collection=image_collection,
        skip_factors=skip_factors,
        context=context,
        future=future,
        **kwargs,
    )


@deprecated(
    deprecated_in="1.8.1",
    details="Please use arcgis.raster.functions.gbl.distance_accumulation "
    "followed by arcgis.raster.analytics.optimal_path_as_line instead.",
)
def determine_travel_costpath_as_polyline(
    input_source_data,
    input_cost_raster,
    input_destination_data,
    path_type="BEST_SINGLE",
    output_polyline_name=None,
    destination_field=None,
    context=None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_determine_travel_costpath_as_polyline/ra_determine_travel_costpath_as_polyline.png 

    Calculates the least cost polyline path between sources and known destinations.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_source_data                        The layer that identifies the cells to determine the least 
                                             costly path from. This parameter can have either a raster input or 
                                             a feature input.
    ------------------------------------     --------------------------------------------------------------------
    input_cost_raster                        A raster defining the impedance or cost to move planimetrically through
                                             each cell.
    
                                             The value at each cell location represents the cost-per-unit distance for 
                                             moving through the cell. Each cell location value is multiplied by the 
                                             cell resolution while also compensating for diagonal movement to 
                                             obtain the total cost of passing through the cell. 
    
                                             The values of the cost raster can be an integer or a floating point, but they 
                                             cannot be negative or zero as you cannot have a negative or zero cost.
    ------------------------------------     --------------------------------------------------------------------
    input_destination_data                   The layer that defines the destinations used to calculate the distance. 
                                             This parameter can have either a raster input or a feature input.
    ------------------------------------     --------------------------------------------------------------------
    path_type                                A keyword defining the manner in which the values and zones on the 
                                             input destination data will be interpreted in the cost path calculations.

                                             A string describing the path type, which can either be BEST_SINGLE, 
                                             EACH_CELL, or EACH_ZONE.

                                             - BEST_SINGLE: For all cells on the input destination data, the \
                                             least-cost path is derived from the cell with the minimum of \
                                             the least-cost paths to source cells. This is the default.

                                             - EACH_CELL: For each cell with valid values on the input \
                                             destination data, at least-cost path is determined and saved \
                                             on the output raster. With this option, each cell of the input \
                                             destination data is treated separately, and a least-cost path \
                                             is determined for each from cell.

                                             - EACH_ZONE: For each zone on the input destination data, \
                                             a least-cost path is determined and saved on the output raster. \
                                             With this option, the least-cost path for each zone begins at \
                                             the cell with the lowest cost distance weighting in the zone.
    ------------------------------------     --------------------------------------------------------------------
    output_polyline_name                     Optional. If not provided, a feature layer is created by the method 
                                             and used as the output.

                                             You can pass in an existing feature layer Item from your GIS to use 
                                             that instead.

                                             Alternatively, you can pass in the name of the output feature layer  that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    destination_field                         The field used to obtain values for the destination locations.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs.
                                             If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:
                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
        The imagery layer url

    """

    # task = "DetermineTravelCostPathAsPolyline"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.determine_travel_costpath_as_polyline(
        input_source_raster_or_features=input_source_data,
        input_cost_raster=input_cost_raster,
        input_destination_raster_or_features=input_destination_data,
        output_polyline_name=output_polyline_name,
        path_type=path_type,
        destination_field=destination_field,
        context=context,
        future=future,
        **kwargs,
    )


def _calculate_distance(
    input_source_data,
    maximum_distance=None,
    output_cell_size=None,
    allocation_field=None,
    output_distance_name=None,
    output_direction_name=None,
    output_allocation_name=None,
    input_barrier_data=None,
    output_back_direction_name=None,
    distance_method="PLANAR",
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Calculates the Euclidean distance, direction, and allocation from a single source or set of sources.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_source_data                        The layer that defines the sources to calculate the distance to.
                                             The layer can be raster or feature. To use a raster input, it must
                                             be of integer type.
    ------------------------------------     --------------------------------------------------------------------
    maximum_distance                         Defines the threshold that the accumulative distance values
                                             cannot exceed. If an accumulative Euclidean distance value exceeds
                                             this value, the output value for the cell location will be NoData.
                                             The default distance is to the edge of the output raster.

                                             Supported units: Meters | Kilometers | Feet | Miles

                                             Example:

                                             {"distance":"60","units":"Meters"}
    ------------------------------------     --------------------------------------------------------------------
    output_cell_size                         Specify the cell size to use for the output raster.

                                             Supported units: Meters | Kilometers | Feet | Miles

                                             Example:
                                             {"distance":"60","units":"Meters"}
    ------------------------------------     --------------------------------------------------------------------
    allocation_field                         A field on the input_source_data layer that holds the values that
                                             defines each source.

                                             It can be an integer or a string field of the source dataset.

                                             The default for this parameter is 'Value'.
    ------------------------------------     --------------------------------------------------------------------
    output_distance_name                     Optional. This is the output distance imagery layer that will be
                                             created.

                                             If not provided, an imagery layer is created by the method
                                             and used as the output.
    ------------------------------------     --------------------------------------------------------------------
    output_direction_name                    Optional. This is the output direction imagery layer that will be
                                             created.

                                             If not provided, an imagery layer is created by the method
                                             and used as the output.

                                             The output direction raster is in degrees, and indicates the
                                             direction to return to the closest source from each cell center.
                                             The values on the direction raster are based on compass directions,
                                             with 0 degrees reserved for the source cells. Thus, a value of 90
                                             means 90 degrees to the East, 180 is to the South, 270 is to the west,
                                             and 360 is to the North.
    ------------------------------------     --------------------------------------------------------------------
    output_allocation_name                   Optional. This is the output allocation  imagery layer that will be
                                             created.

                                             If not provided, an imagery layer is created by the method
                                             and used as the output.

                                             This parameter calculates, for each cell, the nearest source based
                                             on Euclidean distance.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
        The imagery layer url

    """

    # task = "CalculateDistance"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.calculate_distance(
        input_source_raster_or_features=input_source_data,
        output_distance_name=output_distance_name,
        maximum_distance=maximum_distance,
        output_cell_size=output_cell_size,
        output_direction_name=output_direction_name,
        output_allocation_name=output_allocation_name,
        allocation_field=allocation_field,
        distance_method=distance_method,
        input_barrier_raster_or_features=input_barrier_data,
        output_back_direction_name=output_back_direction_name,
        future=future,
        **kwargs,
    )


def generate_multidimensional_anomaly(
    input_multidimensional_raster,
    variables: Optional[list] = None,
    method: str = "DIFFERENCE_FROM_MEAN",
    calculation_interval: Optional[str] = None,
    ignore_nodata: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    reference_mean_raster=None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Computes the anomaly for each slice in a multidimensional raster to generate a multidimensional dataset.
    An anomaly is the deviation of an observation from its standard or mean value.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            The input imagery layer object.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables for which anomalies will
                                             be calculated. If no variable is specified, all variables with a time
                                             dimension will be analyzed.
    ------------------------------------     --------------------------------------------------------------------
    method                                   Optional String. Specifies the method that will be used to calculate
                                             the anomaly.

                                             - DIFFERENCE_FROM_MEAN : The difference between a pixel value and the mean
                                               of that pixel's values across slices defined by the
                                               interval will be calculated. This is the default.

                                             - PERCENT_DIFFERENCE_FROM_MEAN : The percent difference between a pixel value
                                               and the mean of that pixel's values across
                                               slices defined by the interval will be calculated.

                                             - PERCENT_OF_MEAN : The percent of the mean will be calculated.

                                             - Z_SCORE : The z-score for each pixel will be calculated.
                                               A z-score of 0 indicates the pixel's value is identical to the mean.
                                               A z-score of 1 indicates the pixel's value is 1 standard deviation from the mean.
                                               If a z-score is 2, the pixel's value is 2 standard deviations from the mean, and so on.

                                             - DIFFERENCE_FROM_MEDIAN : The difference between a pixel value and the median of that
                                               pixel's values across slices defined by the interval will be calculated

                                             - PERCENT_DIFFERENCE_FROM_MEDIAN : The percent difference between a pixel value and the
                                               median of that pixel's values across slices defined by the interval will be calculated.

                                             - PERCENT_OF_MEDIAN : The percent of the median will be calculated.
    ------------------------------------     --------------------------------------------------------------------
    calculation_interval                     Optional String. Specifies the temporal interval that will be
                                             used to calculate the mean.

                                             - ALL : Calculates the mean across all slices for each pixel.

                                             - YEARLY : Calculates the yearly mean for each pixel.

                                             - RECURRING_MONTHLY : Calculates the monthly mean for each pixel.

                                             - RECURRING_WEEKLY : Calculates the weekly mean for each pixel.

                                             - RECURRING_DAILY : Calculates the daily mean for each pixel.

                                             - HOURLY : Calculates the hourly mean for each pixel.

                                             - EXTERNAL_RASTER : An existing raster dataset that contains the mean or median value for each pixel is referenced.
    ------------------------------------     --------------------------------------------------------------------
    ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored in
                                             the analysis.

                                             - True : The analysis will include all valid pixels along a given dimension and ignore any NoData pixels. This is the default.

                                             - False : The analysis will result in NoData if there are any NoData values for the pixel along the given dimension.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    reference_mean_raster                    Optional Imagery Layer object representing the reference mean raster. Available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    .. code-block:: python

        # Usage Example 1: This example generates an anomaly multidimensional raster for temperature data, comparing pixel values with the mean
        # pixel value across all slices.

        generate_anomaly = generate_multidimensional_anomaly(input_multidimensional_raster=multidimensional_lyr_input,
                                                             variables=["oceantemp"],
                                                             method="PERCENT_DIFFERENCE_FROM_MEAN",
                                                             temporal_interval="YEARLY",
                                                             output_name="temp_anomaly",
                                                             ignore_nodata=True,
                                                             gis=gis,
                                                             folder="generate_mdim_anomaly")


    :return: output_raster : Imagery Layer Item

    """

    # task = "GenerateMultidimensionalAnomaly"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.generate_multidimensional_anomaly(
        input_multidimensional_raster=input_multidimensional_raster,
        output_name=output_name,
        variables=variables,
        method=method,
        calculation_interval=calculation_interval,
        ignore_nodata=ignore_nodata,
        context=context,
        reference_mean_raster=reference_mean_raster,
        future=future,
        **kwargs,
    )


def build_multidimensional_transpose(
    input_multidimensional_raster,
    context: Optional[dict[str, Any]] = None,
    delete_transpose: bool = False,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Transposes a multidimensional raster dataset, which chunks the multidimensional data along each dimension
    to optimize performance when accessing pixel values across all slices.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    delete_tranpose                          Optional boolean. Specifies whether to delete an existing transpose.

                                             - True - The transpose, if it exists, will be deleted. No new transpose will be built.

                                             - False - The transpose will be built. If there is an existing transpose, it will be overwritten. This is the default.

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer URL

    .. code-block:: python

        # Usage Example 1: Build the transpose for a sea surface temperature CRF dataset.

        build_mdim_transpose_op = build_multidimensional_transpose(input_multidimensional_raster=multidimensional_lyr_input, gis=gis)

    """

    # task = "BuildMultidimensionalTranspose"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.build_multidimensional_transpose(
        input_multidimensional_raster=input_multidimensional_raster,
        context=context,
        delete_transpose=delete_transpose,
        future=future,
        **kwargs,
    )


def aggregate_multidimensional_raster(
    input_multidimensional_raster,
    dimension: Optional[str] = None,
    variables: Optional[list] = None,
    aggregation_method: str = "MEAN",
    aggregation_definition: str = "ALL",
    interval_keyword: Optional[str] = None,
    interval_value: Optional[str] = None,
    interval_unit: Optional[int] = None,
    interval_ranges: Optional[list[dict[str, str]]] = None,
    aggregation_function=None,
    ignore_nodata: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    dimensionless: bool = False,
    percentile_value: float = 90,
    percentile_interpolation_type: str = "NEAREST",
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Generates a multidimensional image service by aggregating existing multidimensional raster variables along a dimension.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Required String. The aggregation dimension. This is the dimension
                                             along which the variables will be aggregated.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables that will be aggregated
                                             along the given dimension. If no variable is specified, all variables
                                             with the selected dimension will be aggregated.

                                             For example, to aggregate your daily temperature data into monthly
                                             average values, specify temperature as the variable to be aggregated.
                                             If you do not specify any variables and you have both daily temperature
                                             and daily precipitation variables, both variables will be aggregated
                                             into monthly averages and the output multidimensional raster will include
                                             both variables.
    ------------------------------------     --------------------------------------------------------------------
    aggregation_method                       Optional String. Specifies the mathematical method that will be used
                                             to combine the aggregated slices in an interval.

                                             - MEAN : Calculates the mean of a pixel's values across all slices in the interval. This is the default.

                                             - MAXIMUM : Calculates the maximum value of a pixel across all slices in the interval.

                                             - MAJORITY : Calculates the value that occurred most frequently for a pixel across all slices in the interval.

                                             - MINIMUM : Calculates the minimum value of a pixel across all slices in the interval.

                                             - MINORITY : Calculates the value that occurred least frequently for a pixel across all slices in the interval.

                                             - MEDIAN : Calculates the median value of a pixel across all slices in the interval.

                                             - PERCENTILE: The percentile of values for a pixel will be calculated across all slices
                                               in the interval. The 90th percentile is calculated by default. You can specify other
                                               values (from 0 to 100) using the percentile_value parameter.

                                             - RANGE : Calculates the range of values for a pixel across all slices in the interval.

                                             - STD : Calculates the standard deviation of a pixel's values across all slices in the interval.

                                             - SUM : Calculates the sum of a pixel's values across all slices in the interval.

                                             - VARIETY : Calculates the number of unique values of a pixel across all slices in the interval.

                                             - CUSTOM : Calculates the value of a pixel based on a custom raster function.
                                               When the aggregation_method is set to CUSTOM, the aggregation_function parameter
                                               becomes available.
    ------------------------------------     --------------------------------------------------------------------
    aggregation_definition                   Optional String. Specifies the dimension interval for which the data
                                             will be aggregated.

                                             - ALL : The data values will be aggregated across all slices. This is the default.

                                             - INTERVAL_KEYWORD : The variable data will be aggregated using a commonly known interval.

                                             - INTERVAL_VALUE : The variable data will be aggregated using a user-specified interval and unit.

                                             - INTERVAL_RANGES : The variable data will be aggregated between specified pairs of values or dates.
    ------------------------------------     --------------------------------------------------------------------
    interval_keyword                         Optional String. Specifies the keyword interval that will be used
                                             when aggregating along the dimension. This parameter is required
                                             when the aggregation_definition parameter is set to INTERVAL_KEYWORD, and
                                             the aggregation must be across time.

                                             - HOURLY : The data values will be aggregated into hourly time steps,
                                               and the result will include every hour in the time series.

                                             - DAILY : The data values will be aggregated into daily time steps,
                                               and the result will include every day in the time series.

                                             - WEEKLY : The data values will be aggregated into weekly time steps,
                                               and the result will include every week in the time series.

                                             - DEKADLY : Divides each month into 3 periods of 10 days each
                                               (last period might have more or less than 10 days)
                                               and each month would output 3 slices.

                                             - PENTADLY : Divides each month into 6 periods of 5 days each
                                               (last period might have more or less than 5 days)
                                               and each month would output 6 slices.

                                             - MONTHLY : The data values will be aggregated into monthly time steps,
                                               and the result will include every month in the time series.

                                             - QUARTERLY : The data values will be aggregated into quarterly time steps,
                                               and the result will include every quarter in the time series.

                                             - YEARLY : The data values will be aggregated into yearly time steps,
                                               and the result will include every year in the time series.

                                             - RECURRING_DAILY : The data values will be aggregated into daily time steps,
                                               and the result includes each one aggregated value per day.
                                               The output will include, at most, 366 daily time slices

                                             - RECURRING_WEEKLY : The data values will be aggregated into weekly time steps,
                                               and the result will include one aggregated value per week.
                                               The output will include, at most, 53 weekly time slices.

                                             - RECURRING_MONTHLY : The data values will be aggregated into weekly time steps,
                                               and the result will include one aggregated value per month.
                                               The output will include, at most, 12 monthly time slices.

                                             - RECURRING_QUARTERLY : The data values will be aggregated into weekly time steps,
                                               and the result will include one aggregated value per quarter.
                                               The output will include, at most, 4 quarterly time slices.
    ------------------------------------     --------------------------------------------------------------------
    interval_value                           Optional String. The size of the interval that will be used for the
                                             aggregation. This parameter is required when the aggregation_definition
                                             parameter is set to INTERVAL_VALUE.

                                             For example, to aggregate 30 years of monthly temperature data into
                                             5-year increments, enter 5 as the interval_value, and specify
                                             interval_unit as YEARS.
    ------------------------------------     --------------------------------------------------------------------
    interval_unit                            Optional Integer. The unit that will be used for the interval value.
                                             This parameter is required when the dimension parameter is set to a
                                             time field and the aggregation_definition parameter is set to INTERVAL_VALUE.

                                             If you are aggregating over anything other than time, this option
                                             will not be available and the unit for the interval value will match
                                             the variable unit of the input multidimensional raster data.

                                             - HOURS : The data values will be aggregated into hourly time slices at the interval provided.
                                             - DAYS : The data values will be aggregated into daily time slices at the interval provided.
                                             - WEEKS : The data values will be aggregated into weekly time slices at the interval provided.
                                             - MONTHS : The data values will be aggregated into monthly time slices at the interval provided.
                                             - YEARS : The data values will be aggregated into yearly time slices at the interval provided.
    ------------------------------------     --------------------------------------------------------------------
    interval_ranges                          Optional List of dictionary objects. Interval ranges specified as list of dictionary objects
                                             that will be used to aggregate groups of values.

                                             This parameter is required when the aggregation_definition parameter is set to INTERVAL_RANGE.
                                             If dimension is StdTime, then the value must be specified in human readable time format (YYYY-MM-DDTHH:MM:SS).

                                             Syntax:

                                                 [{"minValue":"<min value>","maxValue":"<max value>"},
                                                 {"minValue":"<min value>","maxValue":"<max value>"}]

                                             Example:

                                                 [{"minValue":"2012-01-15T03:00:00","maxValue":"2012-01-15T09:00:00"},
                                                 {"minValue":"2012-01-15T12:00:00","maxValue":"2012-01-15T21:00:00"}]
    ------------------------------------     --------------------------------------------------------------------
    aggregation_function                     Optional RFT dict object or Raster Funtion Template item from portal.
                                             A custom raster function that will be used to compute the pixel values of the aggregated
                                             rasters.

                                             This parameter is required when the aggregation_method parameter
                                             is set to CUSTOM.
    ------------------------------------     --------------------------------------------------------------------
    ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored in
                                             the analysis.

                                             - True : The analysis will include all valid pixels along a given dimension and ignore any NoData pixels.
                                               This is the default.

                                             - False : The analysis will result in NoData if there are any NoData values for the pixel along the given dimension.
    ------------------------------------     --------------------------------------------------------------------
    dimensionless                            Optional Boolean. Specifies whether the layer will have dimension values.
                                             This parameter is only active if a single slice is selected to create a layer.

                                             - True : The layer will not have dimension values.

                                             - False : The layer will have dimension values. This is the default.

                                             Parameter available in ArcGIS Image Server 10.9.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    percentile_value                         Optional float. The percentile to calculate. The default is 90, indicating
                                             the 90th percentile.

                                             The values can range from 0 to 100. The 0th percentile is essentially equivalent
                                             to the minimum statistic, and the 100th percentile is equivalent to maximum.
                                             A value of 50 will produce essentially the same result as the median statistic.

                                             This option is only honored if the aggregation_method parameter is set to PERCENTILE.

                                             Parameter available in ArcGIS Image Server 10.9.1 and higher.

                                             Example:

                                                 90
    ------------------------------------     --------------------------------------------------------------------
    percentile_interpolation_type            Optional string. Specifies the method of percentile interpolation that will be used when there is an
                                             even number of values from the input raster to be calculated

                                             - NEAREST : The nearest available value to the desired percentile will be used.
                                               In this case, the output pixel type will be the same as that of the input
                                               value raster. This is the default

                                             - LINEAR  : The weighted average of the two surrounding values from the desired
                                               percentile will be used. In this case, the output pixel type will be floating point.

                                             Parameter available in ArcGIS Image Server 10.9.1 and higher.

                                             Example:

                                                 NEAREST
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer Item

    .. code-block:: python

        # Usage Example 1: This example aggregates temperature data into yearly data with the average temperature values.

        agg_multi_dim = aggregate_multidimensional_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                          variables=["temperature"],
                                                          dimension="StdTime",
                                                          aggregation_method="MAXIMUM",
                                                          aggregation_definition="INTERVAL_KEYWORD",
                                                          interval_keyword="YEARLY",
                                                          interval_value=None,
                                                          output_name="yearly_temp",
                                                          ignore_nodata=True,
                                                          gis=gis,
                                                          folder="aggregate_mdim_raster")

    .. code-block:: python

        # Usage Example 2: This example aggregates temperature data into hourly data with the average temperature values for multiple variables.

        agg_multi_dim = aggregate_multidimensional_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                          variables=["cceiling","ccover","gust","temperature"],
                                                          dimension="StdTime",
                                                          aggregation_method="MEAN",
                                                          aggregation_definition="INTERVAL_VALUE",
                                                          interval_value=3,
                                                          interval_unit="HOURS",
                                                          output_name="hourly_data",
                                                          ignore_nodata=True,
                                                          gis=gis,
                                                          folder={'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'aggregate_mdim_raster'})

    .. code-block:: python

        # Usage Example 3: This example aggregates temperature data using a custom aggregation function for multiple variables. This example uses aggregation function
        # uploaded as a Raster Function Template item on portal.

        agg_multi_dim = aggregate_multidimensional_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                          variables=["temperature"],
                                                          dimension="StdTime",
                                                          aggregation_method="CUSTOM",
                                                          aggregation_definition="INTERVAL_RANGES",
                                                          interval_ranges=[{"minValue":"2012-01-15T03:00:00","maxValue":"2012-01-15T09:00:00"},{"minValue":"2012-01-15T12:00:00","maxValue":"2012-01-15T21:00:00"}],
                                                          aggregation_function=rft_item,
                                                          output_name="temp_range4",
                                                          ignore_nodata=True,
                                                          gis=gis)

    """

    # task = "AggregateMultidimensionalRaster"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.aggregate_multidimensional_raster(
        input_multidimensional_raster=input_multidimensional_raster,
        output_name=output_name,
        dimension=dimension,
        aggregation_method=aggregation_method,
        variables=variables,
        aggregation_definition=aggregation_definition,
        interval_keyword=interval_keyword,
        interval_value=interval_value,
        interval_unit=interval_unit,
        interval_ranges=interval_ranges,
        aggregation_function=aggregation_function,
        ignore_nodata=ignore_nodata,
        dimensionless=dimensionless,
        percentile_value=percentile_value,
        percentile_interpolation_type=percentile_interpolation_type,
        context=context,
        future=future,
        **kwargs,
    )


def generate_trend_raster(
    input_multidimensional_raster,
    dimension: Optional[str] = None,
    variables: Optional[list] = None,
    trend_line_type: str = "LINEAR",
    frequency: Optional[int] = None,
    ignore_nodata: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    cycle_length: Optional[float] = None,
    cycle_unit: str = "YEARS",
    rmse: bool = True,
    r2: bool = False,
    slope_p_value: bool = False,
    seasonal_period: str = "DAYS",
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Estimates the trend for each pixel along a dimension for a given variable in a multidimensional raster.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Required String. The dimension along which a trend will be extracted
                                             for the variable or variables selected in the analysis.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables for which trends will be calculated.
                                             If no variable is specified, the first variable in the multidimensional
                                             raster will be analyzed.
    ------------------------------------     --------------------------------------------------------------------
    trend_line_type                          Optional String. Specifies the type of line to be used to fit to the
                                             pixel values along a dimension.

                                             - LINEAR : Fits the pixel values for a variable along a linear trend line. This is the default.

                                             - POLYNOMIAL : Fits the pixel values for a variable along a second order polynomial trend line.

                                             - HARMONIC : Fits the pixel values for a variable along a harmonic trend line.

                                             - MANN-KENDALL : Variable pixel values will be evaluated using the Mann-Kendall trend test. Option available in ArcGIS Image Server 10.9 and higher.

                                             - SEASONAL-KENDALL : Variable pixel values will be evaluated using the Seasonal-Kendall trend test. Option available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    frequency                                Optional Integer.

                                             If the line_type parameter is set to HARMONIC, the default value is 1 ,or one harmonic cycle per year.

                                             If the line_type parameter is set to POLYNOMIAL, the default value is 2, or second order polynomial.
    ------------------------------------     --------------------------------------------------------------------
    ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored in
                                             the analysis.

                                             - True : The analysis will include all valid pixels along a given dimension and ignore any NoData pixels.
                                               This is the default.

                                             - False : The analysis will result in NoData if there are any NoData values for the pixel along the given dimension.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                     {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                                Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    cycle_length                             Optional Float.
                                             The length of periodic variation to model. This parameter is required
                                             when the Trend Line Type is set to Harmonic. For example,
                                             leaf greenness often has one strong cycle of variation in a
                                             single year, so the cycle length is 1 year. Hourly temperature
                                             data has one strong cycle of variation throughout a single day,
                                             so the cycle length is 1 day. Available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    cycle_unit                               Optional String. Default is "YEARS". Specifies the time unit to be
                                             used for the length of harmonic cycle. Available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    rmse                                     Optional Boolean. Default value is True. Specifies whether the root
                                             mean square error (RMSE) of the trend fit line will be calculated.
                                             Parameter available  in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    r2                                       Optional Boolean. Default value is False. Specifies whether the
                                             R-squared goodness-of-fit statistic for the trend fit line will be calculated.
                                             Parameter available  in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    slope_p_value                            Optional Boolean. Default value is False. Specifies whether the
                                             p-value statistic for the slope coefficient of the trend line will be calculated.
                                             Parameter available in ArcGIS Image Server 10.8.1 and higher.
    ------------------------------------     --------------------------------------------------------------------
    seasonal_period                          Optional String. Specifies the seasonal period. Default - "DAYS"
                                             Possible Options - "DAYS", "MONTHS"

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer Item

    .. code-block:: python

        # Usage Example 1: This example aggregates temperature data into yearly data with the average temperature values.

        trend_coeff_multidim  = generate_trend_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                      variables=["NightLightData"],
                                                      dimension="StdTime",
                                                      trend_line_type='POLYNOMIAL',
                                                      frequency=2,
                                                      ignore_nodata=True,
                                                      output_name="polynomial_trend_coefficients",
                                                      gis=gis,
                                                      folder="generate_trend_raster")

    """

    # task = "GenerateTrendRaster"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.generate_trend_raster(
        input_multidimensional_raster=input_multidimensional_raster,
        output_name=output_name,
        dimension=dimension,
        variables=variables,
        trend_line_type=trend_line_type,
        frequency=frequency,
        ignore_nodata=ignore_nodata,
        context=context,
        cycle_length=cycle_length,
        cycle_unit=cycle_unit,
        rmse=rmse,
        r2=r2,
        slope_p_value=slope_p_value,
        seasonal_period=seasonal_period,
        future=future,
        **kwargs,
    )


def predict_using_trend_raster(
    input_multidimensional_raster,
    variables: Optional[list] = None,
    dimension_definition: str = "BY_VALUE",
    dimension_values: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval_value: float = 1,
    interval_unit: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Estimates the trend for each pixel along a dimension for a given variable in a multidimensional raster.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables that will be predicted in
                                             the analysis. If no variables are specified, all variables will be used.

    ------------------------------------     --------------------------------------------------------------------
    dimension_definition                     Required String. Specifies the method used to provide prediction dimension values.

                                             - BY_VALUE : The prediction will be calculated for a single dimension value or
                                               a list of dimension values defined by the dimension_values parameter.
                                               This is the default.

                                               For example, you want to predict yearly precipitation for the years
                                               2050, 2100, and 2150.

                                             - BY_INTERVAL - The prediction will be calculated for an interval of the dimension defined
                                               by a start and an end value. For example, you want to predict yearly precipitation for
                                               every year between 2050 and 2150.
    ------------------------------------     --------------------------------------------------------------------
    dimension_values                         Optional list. The dimension value or values to be used in the prediction.

                                             This parameter is required when dimension_def parameter is set to BY_VALUE.
    ------------------------------------     --------------------------------------------------------------------
    start                                    Optional String.The start date, height, or depth of the dimension interval to be used
                                             in the prediction.
    ------------------------------------     --------------------------------------------------------------------
    end                                      Optional String. The end date, height, or depth of the dimension interval to be used in the prediction.
    ------------------------------------     --------------------------------------------------------------------
    interval_value                           Optional Float. The number of steps between two dimension values to be included in the prediction. The default value is 1

                                             For example, to predict temperature values every five years, use a value of 5.
    ------------------------------------     --------------------------------------------------------------------
    interval_unit                            Optional String. Specifies the unit that will be used for the value interval.
                                             This parameter only applies when the dimension of analysis is a time dimension.

                                             - HOURS - The prediction will be calculated for each hour in the range of time described by the start, end, and interval_value parameters.

                                             - DAYS - The prediction will be calculated for each day in the range of time described by the start, end, and interval_value parameters.

                                             - WEEKS - The prediction will be calculated for each week in the range of time described by the start, end, and interval_value parameters.

                                             - MONTHS - The prediction will be calculated for each month in the range of time described by the start, end, and interval_value parameters.

                                             - YEARS - The prediction will be calculated for each year in the range of time described by the start, end, and interval_value parameters.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer Item

    .. code-block:: python

        # Usage Example 1: This example generates the forecasted precipitation and temperature for January 1, 2050, and January 1, 2100.

        predict_output = predict_using_trend_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                    variables=["temp","precip"],
                                                    dimension_definition='BY_VALUE',
                                                    dimension_values=["2050-01-01T00:00:00","2100-01-01T00:00:00"],
                                                    output_name="predicted_temp_precip",
                                                    gis=gis.
                                                    folder="predict_trend")

    .. code-block:: python

        # Usage Example 2: This example generates the forecasted NDVI values for each month in year 2025.

        predict_output = predict_using_trend_raster(input_multidimensional_raster=multidimensional_lyr_input,
                                                    variables=["NDVI"],
                                                    dimension_definition='BY_INTERVAL',
                                                    start="2025-01-01T00:00:00",
                                                    end="2025-12-31T00:00:00",
                                                    interval_value=1,
                                                    interval_unit="MONTHS",
                                                    output_name="predict_using_trend_raster",
                                                    gis=gis,
                                                    folder={'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'})

    """

    # task = "PredictUsingTrendRaster"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.predict_using_trend_raster(
        input_multidimensional_raster=input_multidimensional_raster,
        output_name=output_name,
        variables=variables,
        dimension_definition=dimension_definition,
        dimension_values=dimension_values,
        start=start,
        end=end,
        interval_value=interval_value,
        interval_unit=interval_unit,
        context=context,
        future=future,
        **kwargs,
    )


def find_argument_statistics(
    input_raster,
    dimension: Optional[list] = None,
    dimension_definition: str = "ALL",
    interval_keyword: Optional[str] = None,
    variables: Optional[list] = None,
    statistics_type: str = "ARGUMENT_MIN",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    multiple_occurrence_value: Optional[int] = None,
    ignore_nodata: bool = True,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    argument_value: Optional[int] = None,
    comparison: Optional[str] = "EQUAL_TO",
    occurrence: Optional[str] = "FIRST_OCCURRENCE",
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Extracts the dimension value at which a given statistic is attained for each pixel in a multidimensional raster.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required :class:`~arcgis.raster.ImageryLayer` object. The input raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Required String. The dimension from which the statistic will be
                                             extracted. If the input raster is not a multidimensional raster,
                                             this parameter is not required.
    ------------------------------------     --------------------------------------------------------------------
    dimension_definition                     Required String. Specifies the dimension interval for which the data will be analyzed.

                                             -  ALL : The data values will be analyzed across all slices. This is the default.

                                             -  INTERVAL_KEYWORD :The variable data will be analyzed using a commonly known interval.

                                             Example:

                                                'ALL'
    ------------------------------------     --------------------------------------------------------------------
    interval_keyword                         Required String. Specifies the keyword interval that will be used
                                             when analyzing along the dimension. This parameter is required when the
                                             dimension_definition parameter is set to INTERVAL_KEYWORD, and the
                                             analysis must be across time.

                                             Possible options:
                                             HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY,
                                             RECURRING_DAILY, RECURRING_WEEKLY, RECURRING_MONTHLY, RECURRING_QUARTERLY
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables to be analyzed. If the input
                                             raster is not multidimensional, the pixel values of the multiband raster
                                             are considered the variable. If the input raster is multidimensional and
                                             no variable is specified, all variables with the selected dimension will be analyzed.

                                             For example, to find the years in which temperature values were highest,
                                             specify temperature as the variable to be analyzed. If you do not specify any
                                             variables and you have both temperature and precipitation variables,
                                             both variables will be analyzed and the output multidimensional raster will
                                             include both variables.
    ------------------------------------     --------------------------------------------------------------------
    statistics_type                          Optional String. Specifies the statistic to extract from the variable or variables along the given dimension.

                                             - ARGUMENT_MIN : The dimension value at which the minimum variable value is reached will be extracted. This is the default.

                                             - ARGUMENT_MAX : The dimension value at which the maximum variable value is reached will be extracted.

                                             - ARGUMENT_MEDIAN : The dimension value at which the median variable value is reached will be extracted.

                                             - DURATION : The longest dimension duration for which the variable values fall between the minimum and maximum values.

                                             - ARGUMENT_VALUE: The dimension value at which the specified variable value is reached will be extracted.
    ------------------------------------     --------------------------------------------------------------------
    min_value                                Optional Float. The minimum variable value to be used to extract the duration.

                                             This parameter is required when the statistics_type parameter is set to DURATION.
    ------------------------------------     --------------------------------------------------------------------
    max_value                                Optional Float. The maximum variable value to be used to extract the duration.
    ------------------------------------     --------------------------------------------------------------------
    multiple_occurrence_value                Optional Integer. Specifies the pixel value to use to indicate that a given argument
                                             statistic was reached more than once in the input raster dataset. If not specified,
                                             the pixel value will be the value of the dimension the first time the argument
                                             statistic was reached.
    ------------------------------------     --------------------------------------------------------------------
    ignore_nodata                            Optional Boolean. Specifies whether NoData values are ignored in
                                             the analysis.

                                             - True : The analysis will include all valid pixels along a given dimension and ignore any NoData pixels.
                                               This is the default.

                                             - False : The analysis will result in NoData if there are any NoData values for the pixel along the given dimension.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    argument_value                           Optional Integer. The value at which a comparison will be made to extract
                                             the dimension value. This parameter is required when the statistics_type
                                             parameter is set to ARGUMENT_VALUE.

                                             |
                                             .. note::
                                                    This parameter is currently only available on ArcGIS online.
    ------------------------------------     --------------------------------------------------------------------
    comparison                               Optional String. Specifies the comparison type that will be used to
                                             extract the dimension value.

                                             - EQUAL_TO : The extracted dimension is equal to the specified value. This is the default.

                                             - GREATER_THAN : The extracted dimension is greater than the specified value.

                                             - SMALLER_THAN : The extracted dimension is smaller than the specified value.

                                             .. note::
                                                    This parameter is currently only available on ArcGIS online.
    ------------------------------------     --------------------------------------------------------------------
    occurrence                               Optional String. Specifies whether the value of the dimension will be returned the first
                                             time or last time the argument statistic is reached.

                                             - FIRST_OCCURRENCE : The value of the dimension will be returned the first time the argument statistic is reached. This is the default.

                                             - LAST_OCCURRENCE : The value of the dimension will be returned the last time the argument statistic is reached.

                                             .. note::
                                                    This parameter is currently only available on ArcGIS online.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer Item

    .. code-block:: python

        # Usage Example 1: This example finds the minimum precipitation and temperature values across a time series multidimensional raster.
        # If the same minimum value is found multiple times, the pixel value will be 99999.

        arg_stat_output = arcgis.raster.analytics.find_argument_statistics(input_raster=input_layer,
                                                                                  dimension="StdTime",
                                                                                  variables=["precip","temp"],
                                                                                  statistics_type='ARGUMENT_MIN',
                                                                                  multiple_occurrence_value=99999,
                                                                                  ignore_nodata=True,
                                                                                  output_name="arg_stat_output",
                                                                                  gis=gis,
                                                                                  folder="find_argument_statistics")

    .. code-block:: python

        # Usage Example 2: This example finds the longest time interval for which salinity fell between 10 and 15 units of measurement in the multidimensional raster.

        arg_stat_output = find_argument_statistics(input_raster=input_layer,
                                                   dimension="StdTime",
                                                   variables=["salinity"],
                                                   statistics_type='DURATION',
                                                   min_value=10,
                                                   max_value=15,
                                                   ignore_nodata=True,
                                                   output_name="arg_stat_output",
                                                   gis=gis,
                                                   folder={'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'})
    """
    # task = "FindArgumentStatistics"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.find_argument_statistics(
        input_raster=input_raster,
        output_name=output_name,
        dimension=dimension,
        dimension_definition=dimension_definition,
        interval_keyword=interval_keyword,
        variables=variables,
        statistics_type=statistics_type,
        min_value=min_value,
        max_value=max_value,
        multiple_occurrence_value=multiple_occurrence_value,
        ignore_nodata=ignore_nodata,
        context=context,
        future=future,
        argument_value=argument_value,
        comparison=comparison,
        occurrence=occurrence,
        **kwargs,
    )


def linear_spectral_unmixing(
    input_raster,
    input_spectral_profile: Union[dict, str],
    value_option: list[str] = [],
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Performs subpixel classification and calculates the fractional abundance of endmembers for individual pixels.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required :class:`~arcgis.raster.ImageryLayer` object. The input raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    input_spectral_profile                   Required Dict, String or ECD Portal item. The class spectral profile information.
    ------------------------------------     --------------------------------------------------------------------
    value_option                             Optional List of string(s). Specifies the options to define the output pixel values.

                                             - SUM_TO_ONE : Class values for each pixel are provided in decimal format with the sum
                                               of all classes equal to 1. For example, Class1 = 0.16; Class2 = 0.24; Class3 = 0.60.

                                             - NON_NEGATIVE : There will be no negative output values.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                                Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                                Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                                Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                                Example:

                                                Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                Syntax example with a specified percentage of total
                                                processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer Item

    .. code-block:: python

        # Usage Example 1: This example calculates the fractional abundance of classes from a classifier definition file (.ecd)
        # located in a datastore registered with the raster analytics server and generates a multiband raster.

        unmixing_output = linear_spectral_unmixing(input_raster=input_layer,
                                                   input_spectral_profile="/fileShares/Mdim/SpectralUnmixing_json.ecd",
                                                   output_name="linear_spectral_unmixing",
                                                   gis=gis,
                                                   folder="linear_spectral_unmixing")


    .. code-block:: python

        # Usage Example 2: This example calculates the fractional abundance of classes from a dictionary and generates a multiband raster.

        input_spectral_profile_dict = {"EsriEndmemberDefinitionFile":0,"FileVersion":1,"NumberEndmembers":3,"NumberBands":7,
                                       "Endmembers":[{"EndmemberID":1,"EndmemberName":"urban","SpectralProfile":[88,42,48,38,86,115,59]},
                                                     {"EndmemberID":2,"EndmemberName":"vegetation","SpectralProfile":[50,21,20,35,50,110,23]},
                                                     {"EndmemberID":3,"EndmemberName":"water","SpectralProfile":[51,20,14,9,7,116,4]}]}

        unmixing_outputs = arcgis.raster.analytics.linear_spectral_unmixing(input_raster=multidimensional_lyr_input,
                                                                            input_spectral_profile=input_spectral_profile_dict,
                                                                            value_option=["SUM_TO_ONE","NON_NEGATIVE"],
                                                                            output_name="linear_spectral_unmixing",
                                                                            gis=gis,
                                                                            folder={'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'})

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.linear_spectral_unmixing(
        input_raster=input_raster,
        output_name=output_name,
        input_spectral_profile=input_spectral_profile,
        value_option=value_option,
        context=context,
        future=future,
        **kwargs,
    )


def subset_multidimensional_raster(
    input_multidimensional_raster,
    variables: Optional[list] = None,
    dimension_definition: str = "ALL",
    dimension_ranges: Optional[list[dict, Any]] = None,
    dimension_values: Optional[list[dict[str, Any]]] = None,
    dimension: Optional[str] = None,
    start_of_first_iteration: Optional[str] = None,
    end_of_first_iteration: Optional[str] = None,
    iteration_step: Optional[float] = None,
    iteration_unit: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Subsets a multidimensional raster by slicing data along defined variables and dimensions.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional list. The variables that will be included in the output
                                             multidimensional raster. If no variable is specified, all of the
                                             variables will be used.
    ------------------------------------     --------------------------------------------------------------------
    dimension_definition                     Optional String. Specifies the method that will be used to slice the dimension.

                                             - ALL : The full range for each dimension will be used. This is the default.

                                             - BY_RANGES : The dimension will be sliced using a range or a list of ranges.

                                             - BY_ITERATION : The dimension will be sliced over a specified interval.

                                             - BY_VALUE : The dimension will be sliced using a list of dimension values.
    ------------------------------------     --------------------------------------------------------------------
    dimension_ranges                         Optional list of dicts.

                                             This slices the data based on the dimension name and the
                                             minimum and maximum values for the range.

                                             This parameter is required when the dimension_definition is set to BY_RANGE.

                                             If dimension is StdTime, then the min value and max value must be specified in
                                             human readable time format (YYYY-MM-DDTHH:MM:SS).

                                             dimension_values has to be specified as:

                                                 [{"dimension":"<dimension_name>",
                                                 "minValue":"<dimension_min_value>",
                                                 "maxValue":"<dimension_max_value>"},
                                                 {"dimension":"<dimension_name>",
                                                 "minValue":"<dimension_min_value>",
                                                 "maxValue":"<dimension_max_value>"}]

                                             Example:

                                                 [{"dimension":"StdTime",
                                                 "minValue":"2013-05-17T00:00:00",
                                                 "maxValue":"2013-05-17T03:00:00"},
                                                 {"dimension":"StdZ",
                                                 "minValue":"-5000",
                                                 "maxValue":"-4000"}]
    ------------------------------------     --------------------------------------------------------------------
    dimension_values                         Optional list of dicts.

                                             This slices the data based on the dimension name and the value specified.

                                             This parameter is required when the dimension_definition is set to BY_VALUE.

                                             If dimension is StdTime, then the value must be specified in
                                             human readable time format (YYYY-MM-DDTHH:MM:SS).

                                             dimension_values has to be specified as:

                                                 [{"dimension":"<dimension_name>", "value":"<dimension_value>"},
                                                 {"dimension":"<dimension_name>", "value":"<dimension_value>"}]

                                             Example:

                                                [{"dimension":"StdTime", "value":"2012-01-15T03:00:00"},
                                                {"dimension":" StdZ ", "value":"-4000"}]
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Optional string. The dimension along which the variables will be sliced.
    ------------------------------------     --------------------------------------------------------------------
    start_of_first_iteration                 Optional string. The beginning of the interval.
                                             This parameter is required when the dimension_definition is set to BY_ITERATION
    ------------------------------------     --------------------------------------------------------------------
    end_of_first_iteration                   Optional String.The end of the interval.
                                             This parameter is required when the dimension_definition is set to BY_ITERATION
    ------------------------------------     --------------------------------------------------------------------
    iteration_step                           Optional Float. The interval over which the data will be sliced.
                                             This parameter is required when the dimension_definition is set to BY_ITERATION
    ------------------------------------     --------------------------------------------------------------------
    iteration_unit                           Optional String. The iteration unit.

                                             This parameter is required when the dimension_definition is set to BY_ITERATION

                                             - HOURS - Uses hours as the specified unit of time.

                                             - DAYS - Uses days as the specified unit of time.

                                             - WEEKS - Uses weeks as the specified unit of time.

                                             - MONTHS - Uses months as the specified unit of time.

                                             - YEARS -Uses years as the specified unit of time.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: output_raster : Imagery layer item

    .. code-block:: python

        # Usage Example 1: This creates a new multidimensional image service with variables cceiling and ccover for StdTime  dimensions
        # values - 2012-01-15T03:00:00 and  2012-01-15T09:00:00

        subset_output = subset_multidimensional_raster(input_multidimensional_raster=input_multidimensional_lyr,
                                                       variables=["cceiling","ccover"],
                                                       dimension_definition='BY_VALUE',
                                                       dimension_values=[{"dimension":"StdTime", "value":"2012-01-15T03:00:00"},
                                                                         {"dimension":"StdTime", "value":"2012-01-15T09:00:00"}]
                                                       output_name="subset_op",
                                                       gis=gis,
                                                       folder="subset_multidimensional_raster")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.subset_multidimensional_raster(
        input_multidimensional_raster=input_multidimensional_raster,
        output_name=output_name,
        variables=variables,
        dimension_definition=dimension_definition,
        dimension_ranges=dimension_ranges,
        dimension_values=dimension_values,
        dimension=dimension,
        start_of_first_iteration=start_of_first_iteration,
        end_of_first_iteration=end_of_first_iteration,
        iteration_step=iteration_step,
        iteration_unit=iteration_unit,
        context=context,
        future=future,
        **kwargs,
    )


@deprecated(
    deprecated_in="1.8.1",
    details="Please use arcgis.raster.analytics.optimal_path_as_line instead. ",
)
def costpath_as_polyline(
    input_destination_data,
    input_cost_distance_raster,
    input_cost_backlink_raster,
    path_type="BEST_SINGLE",
    destination_field=None,
    output_polyline_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    .. image:: _static/images/ra_costpath_as_polyline/ra_costpath_as_polyline.png 

    Calculates the least cost polyline path between sources and known destinations.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_destination_data                   A raster or feature layer that identifies those cells from which the 
                                             least-cost path is determined to the least costly source.
                                             If the input is a raster, the input consists of cells that have valid 
                                             values (zero is a valid value), and the remaining cells must be 
                                             assigned NoData.
    ------------------------------------     --------------------------------------------------------------------
    input_cost_distance_raster               The cost distance raster to be used to determine the least-cost path 
                                             from the sources to the destinations.
                                             The cost distance raster is usually created with the Cost Distance, 
                                             Cost Allocation or Cost Back Link functions. The cost distance raster stores, 
                                             for each cell, the minimum accumulative cost distance over a cost surface 
                                             from each cell to a set of source cells.
    ------------------------------------     --------------------------------------------------------------------
    input_cost_backlink_raster               The name of a cost back link raster used to determine the path to return 
                                             to a source via the least-cost path.
                                             For each cell in the back link raster, a value identifies the neighbor 
                                             that is the next cell on the least accumulative cost path from the cell 
                                             to a single source cell or set of source cells.
    ------------------------------------     --------------------------------------------------------------------
    path_type                                A keyword defining the manner in which the values and zones on the 
                                             input destination data will be interpreted in the cost path calculations.
                                             A string describing the path type, which can either be BEST_SINGLE, 
                                             EACH_CELL, or EACH_ZONE.

                                             - BEST_SINGLE: For all cells on the input destination data, the \
                                             least-cost path is derived from the cell with the minimum of \
                                             the least-cost paths to source cells. This is the default.

                                             - EACH_CELL: For each cell with valid values on the input \
                                             destination data, at least-cost path is determined and saved \
                                             on the output raster. With this option, each cell of the input \
                                             destination data is treated separately, and a least-cost path \
                                             is determined for each from cell.

                                             - EACH_ZONE: For each zone on the input destination data, \
                                             a least-cost path is determined and saved on the output raster. \
                                             With this option, the least-cost path for each zone begins at \
                                             the cell with the lowest cost distance weighting in the zone.
    ------------------------------------     --------------------------------------------------------------------
    destination_field                        Optional. If not provided, a feature layer is created by the method 
                                             and used as the output.
                                             You can pass in an existing feature layer Item from your GIS to use 
                                             that instead.
                                             Alternatively, you can pass in the name of the output feature layer  that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    output_polyline_name                     Optional. If not provided, a feature layer is created by the method 
                                             and used as the output.
                                             You can pass in an existing feature layer Item from your GIS to use 
                                             that instead.
                                             Alternatively, you can pass in the name of the output feature layer  that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. The GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and 
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
        output_raster : Imagery layer item
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.cost_path_as_polyline(
        input_destination_raster_or_features=input_destination_data,
        input_cost_distance_raster=input_cost_distance_raster,
        input_cost_backlink_raster=input_cost_backlink_raster,
        output_polyline_name=output_polyline_name,
        path_type=path_type,
        destination_field=destination_field,
        context=context,
        future=future,
        **kwargs,
    )


def define_nodata(
    input_raster,
    nodata: dict[str, Any],
    query_filter: Optional[str] = None,
    num_of_bands: Optional[int] = None,
    composite_value: bool = False,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function specifies one or more values to be represented as NoData.
    Function available in ArcGIS Image Server 10.8 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required :class:`~arcgis.raster.ImageryLayer` object. Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    nodata                                   Required dictionary.
                                             The value must be specified in dict form and can have keys - noDataValues, includedRanges

                                             Example:

                                                 | {"noDataValues": [0]}
                                                 | {"noDataValues": [0, 255, 0]}
                                                 | {"includedRanges": [0, 255]}
                                                 | {"includedRanges": [0, 255, 1, 255, 4, 250]}
    ------------------------------------     --------------------------------------------------------------------
    query_filter                             Optional str. An SQL statement to select specific raster in the image collection.
                                             Only the selected rasters will have their NoData values changed.

                                             Example:

                                                "OBJECTID > 3"
    ------------------------------------     --------------------------------------------------------------------
    num_of_bands                             Optional int. The number of bands in the input raster.

                                             Example:

                                                3
    ------------------------------------     --------------------------------------------------------------------
    composite_value                          Optional boolean. Choose whether all bands must be NoData in order
                                             for the pixel to be classified as NoData.

                                             - False : If any of the bands have pixels of NoData,
                                               then the pixel is classified as NoData. This is the default.
                                             - True : All of the bands must have pixels of NoData in
                                               order for the pixel to be classified as NoData.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return: The imagery layer url

    .. code-block:: python

            # Usage Example 1: To set no data values.
            define_nodata_op = define_nodata(input_raster=image_collection,
                                             composite_value=False,
                                             nodata={"noDataValues": [110,105,101]},
                                             num_of_bands=3,
                                             query_filter="OBJECTID < 12",
                                             future=False,
                                             gis=gis
                                            )

    .. code-block:: python

            # Usage Example 2: To set included ranges.
            define_nodata_op = define_nodata(input_raster=image_collection,
                                             composite_value=True,
                                             nodata={"includedRanges": [150, 200, 0, 200, 50, 200]},
                                             num_of_bands=3,
                                             query_filter="OBJECTID > 7",
                                             future=True,
                                             gis=gis
                                            )

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.define_nodata(
        input_raster=input_raster,
        nodata=nodata,
        query_filter=query_filter,
        num_of_bands=num_of_bands,
        composite_value=composite_value,
        future=future,
        **kwargs,
    )


def optimal_path_as_line(
    input_destination_data,
    input_distance_accumulation_raster,
    input_back_direction_raster,
    destination_field: Optional[str] = None,
    path_type: str = "EACH_ZONE",
    output_feature_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    create_network_paths: str = "DESTINATIONS_TO_SOURCES",
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Calculates the optimal path from a source to a destination as a feature.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_destination_data                   Required :class:`~arcgis.raster.ImageryLayer` or :class:`~arcgis.features.FeatureLayer` object. Portal Item can be passed.
                                             A dataset that identifies locations from which the optimal path is
                                             determined to the least costly source.

                                             If the input is a raster, it must consist of cells that have valid values
                                             for the destinations, and the remaining cells must be assigned NoData.
                                             Zero is a valid value.
    ------------------------------------     --------------------------------------------------------------------
    input_distance_accumulation_raster       Required :class:`~arcgis.raster.ImageryLayer` object. Portal Item can be passed.
                                             The distance accumulation raster is used
                                             to determine the optimal path from the sources to the destinations.

                                             The distance accumulation raster is usually created with the
                                             :meth:`~arcgis.raster.functions.gbl.distance_accumulation` or
                                             :meth:`~arcgis.raster.functions.gbl.distance_allocation`. Each cell
                                             in the distance accumulation raster represents the minimum
                                             accumulative cost distance over a surface from each cell to a set of source cells.
    ------------------------------------     --------------------------------------------------------------------
    input_back_direction_raster              Required :class:`~arcgis.raster.ImageryLayer` object. Portal Item can be passed.
                                             The back direction raster contains calculated directions in degrees.
                                             The direction identifies the next cell along the optimal path back to
                                             the least accumulative cost source while avoiding barriers.

                                             The range of values is from 0 degrees to 360 degrees, with 0
                                             reserved for the source cells. Due east (right) is 90, and the
                                             values increase clockwise (180 is south, 270 is west, and 360 is north)
    ------------------------------------     --------------------------------------------------------------------
    destination_field                        Optional string. The field to be used to obtain values for the destination locations.
    ------------------------------------     --------------------------------------------------------------------
    path_type                                Optional string. A keyword defining the manner in which the values and zones on the input destination
                                             data will be interpreted in the cost path calculations.

                                             - EACH_ZONE - For each zone on the input destination data, a least-cost path is determined and saved on the output raster. With this option, the least-cost path for each zone begins at the cell with the lowest cost distance weighting in the zone. This is the default.

                                             - BEST_SINGLE - For all cells on the input destination data, the least-cost path is derived from the cell with the minimum of the least-cost paths to source cells.

                                             - EACH_CELL - For each cell with valid values on the input destination data, a least-cost path is determined and saved on the output raster. With this option, each cell of the input destination data is treated separately, and a least-cost path is determined for each from cell.
    ------------------------------------     --------------------------------------------------------------------
    output_feature_name                      Optional. If not provided, a feature layer is created by the method
                                             and used as the output.

                                             You can pass in an existing feature layer Item from your GIS to use
                                             that instead.

                                             Alternatively, you can pass in the name of the output feature layer  that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dict. Context contains additional settings that affect task execution.
    ------------------------------------     --------------------------------------------------------------------
    create_network_paths                     Optional string or boolean. Specifies whether complete, and possibly
                                             overlapping, paths from the destinations to the sources are calculated
                                             or if  nonoverlapping network paths are created.

                                             - DESTINATIONS_TO_SOURCES (False): Complete paths from the destinations to the sources are calculated, which can be overlapping.  This is default.

                                             - NETWORK_PATHS (True): Nonoverlapping network paths are calculated.

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return: Output Feature Layer Item

    .. code-block:: python

        # Usage Example 1: To calculate the optimal path from a source to a destination.

        destination_data = gis.content.search("my_destination_data")[0].layers[0]
        accumulation_raster = gis.content.search("my_accumulation_raster")[0].layers[0]
        back_direction_raster = gis.content.search("my_back_direction_raster")[0].layers[0]

        optimal_path_op = optimal_path_as_line(input_destination_data=destination_data,
                                               input_distance_accumulation_raster=accumulation_raster,
                                               input_back_direction_raster=back_direction_raster,
                                               output_feature_name="optimal_path_feature",
                                               gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.optimal_path_as_line(
        input_destination_raster_or_features=input_destination_data,
        input_distance_accumulation_raster=input_distance_accumulation_raster,
        input_back_direction_raster=input_back_direction_raster,
        output_polyline_name=output_feature_name,
        destination_field=destination_field,
        path_type=path_type,
        context=context,
        create_network_paths=create_network_paths,
        future=future,
        **kwargs,
    )


def optimal_region_connections(
    input_region_data,
    input_barrier_data=None,
    input_cost_raster=None,
    distance_method: str = "PLANAR",
    connections_within_regions: str = "GENERATE_CONNECTIONS",
    output_optimal_lines_name: Optional[str] = None,
    output_neighbor_connections_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Calculates the optimal connectivity network between two or more input regions.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_region_data                        Required :class:`~arcgis.raster.ImageryLayer` or :class:`~arcgis.features.FeatureLayer` object. Portal Item can be passed.
                                             The input regions to be connected by the optimal network.

                                             If the region input is a raster, the regions are defined by groups
                                             of contiguous (adjacent) cells of the same value. Each region must
                                             be uniquely numbered. The cells that are not part of any region must
                                             be NoData. The raster type must be integer, and the values can be
                                             either positive or negative.

                                             If the region input is a feature dataset, it can be polygons,
                                             lines, or points. Polygon feature regions cannot be composed
                                             of multipart polygons.
    ------------------------------------     --------------------------------------------------------------------
    input_barrier_data                       Required :class:`~arcgis.raster.ImageryLayer` or :class:`~arcgis.features.FeatureLayer` object. Portal Item can be passed.
                                             The dataset that defines the barriers.

                                             The barriers can be defined by an integer or a floating-point raster,
                                             or by a feature layer.
    ------------------------------------     --------------------------------------------------------------------
    input_cost_raster                        Required :class:`~arcgis.raster.ImageryLayer` object. Portal Item can be passed. A raster
                                             defining the impedance or cost to move planimetrically through each
                                             cell.

                                             The value at each cell location represents the cost-per-unit
                                             distance for moving through the cell. Each cell location value
                                             is multiplied by the cell resolution while also compensating
                                             for diagonal movement to obtain the total cost of passing through the cell.

                                             The values of the cost raster can be integer or floating point,
                                             but they cannot be negative or zero (you cannot have a negative or zero cost).
    ------------------------------------     --------------------------------------------------------------------
    distance_method                          Optional String. Specifies whether to calculate the distance using a
                                             planar (flat earth) or a geodesic (ellipsoid) method.

                                             - PLANAR - The distance calculation will be performed on a projected flat plane using a 2D Cartesian coordinate system. This is the default.

                                             - GEODESIC - The distance calculation will be performed on the ellipsoid. Therefore, regardless of input or output projection, the results do not change.
    ------------------------------------     --------------------------------------------------------------------
    connections_within_regions               Optional string. Default - GENERATE_CONNECTIONS

                                             Possible options: GENERATE_CONNECTIONS, NO_CONNECTIONS
    ------------------------------------     --------------------------------------------------------------------
    output_optimal_lines_name                Optional. If not provided, a feature layer is created by the method
                                             and used as the output.
                                             You can pass in an existing feature layer Item from your GIS to use
                                             that instead.
                                             Alternatively, you can pass in the name of the output feature layer
                                             that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists

                                             This is the output polyline feature class of the optimal network of
                                             paths necessary to connect each of the input regions.

                                             Each path (or line) is uniquely numbered, and additional fields in the
                                             attribute table store specific information about the path.
                                             Those fields are the following:

                                             - PATHID - Unique identifier for the path

                                             - PATHCOST - Total accumulative distance or cost for the path

                                             - REGION1 - The first region the path connects

                                             - REGION2 - The other region the path connects

                                             This information provides insight into the paths within the network.

                                             Since each path is represented by a unique line, there will be
                                             multiple lines in locations where paths travel the same route.
    ------------------------------------     --------------------------------------------------------------------
    output_neighbor_connections_name         Optional. If not provided, a feature layer is created by the method
                                             and used as the output.
                                             You can pass in an existing feature layer Item from your GIS to use
                                             that instead.
                                             Alternatively, you can pass in the name of the output feature layer
                                             that should be created by this method to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists

                                             This is the output polyline feature class identifying all paths from
                                             each region to each of its closest or cost neighbors.

                                             Each path (or line) is uniquely numbered, and additional fields in the
                                             attribute table store specific information about the path.
                                             Those fields are the following:

                                             - PATHID - Unique identifier for the path

                                             - PATHCOST - Total accumulative distance or cost for the path

                                             - REGION1 - The first region the path connects

                                             - REGION2 - The other region the path connects

                                             This information provides insight into the paths within the
                                             network and is particularly useful when deciding which paths
                                             should be removed if necessary.

                                             Since each path is represented by a unique line, there will be
                                             multiple lines in locations where paths travel the same route.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`gis.content.create_folder <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return: Returns the following as a named tuple - output_optimum_network_features, output_neighbor_network_features

    .. code-block:: python

        # Usage Example 1: To calculate the optimal connections between regions.

        region_data = gis.content.search("my_region_data")[0].layers[0]
        barrier_data = gis.content.search("my_barrier_data")[0].layers[0]
        cost_raster = gis.content.search("my_cost_raster")[0].layers[0]

        optimal_region_op = optimal_region_connections(input_region_data=region_data,
                                                       input_barrier_data=barrier_data,
                                                       input_cost_raster=cost_raster,
                                                       output_optimal_lines_name="optimal_lines_feature",
                                                       output_neighbor_connections_name="optimal_region_feature",
                                                       gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.optimal_region_connections(
        input_region_raster_or_features=input_region_data,
        input_barrier_raster_or_features=input_barrier_data,
        input_cost_raster=input_cost_raster,
        distance_method=distance_method,
        connections_within_regions=connections_within_regions,
        output_optimal_lines_name=output_optimal_lines_name,
        output_neighbor_connections_name=output_neighbor_connections_name,
        context=context,
        future=future,
        **kwargs,
    )


def _distance_accumulation(
    input_source_raster_or_features,
    input_barrier_raster_or_features=None,
    input_surface_raster=None,
    input_cost_raster=None,
    input_vertical_raster=None,
    vertical_factor="BINARY 1 -30 30",
    input_horizontal_raster=None,
    horizontal_factor="BINARY 1 45",
    source_initial_accumulation=None,
    source_maximum_accumulation=None,
    source_cost_multiplier=None,
    source_direction=None,
    distance_method="PLANAR",
    output_distance_accumulation_raster_name=None,
    output_back_direction_raster_name=None,
    output_source_direction_raster_name=None,
    output_source_location_raster_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.distance_accumulation(
        input_source_raster_or_features=input_source_raster_or_features,
        input_barrier_raster_or_features=input_barrier_raster_or_features,
        input_surface_raster=input_surface_raster,
        input_cost_raster=input_cost_raster,
        input_vertical_raster=input_vertical_raster,
        vertical_factor=vertical_factor,
        input_horizontal_raster=input_horizontal_raster,
        horizontal_factor=horizontal_factor,
        source_initial_accumulation=source_initial_accumulation,
        source_maximum_accumulation=source_maximum_accumulation,
        source_cost_multiplier=source_cost_multiplier,
        source_direction=source_direction,
        distance_method=distance_method,
        output_distance_accumulation_raster_name=output_distance_accumulation_raster_name,
        output_back_direction_raster_name=output_back_direction_raster_name,
        output_source_direction_raster_name=output_source_direction_raster_name,
        output_source_location_raster_name=output_source_location_raster_name,
        context=context,
        future=future,
        **kwargs,
    )


def _distance_allocation(
    input_source_raster_or_features,
    input_barrier_raster_or_features=None,
    input_surface_raster=None,
    input_cost_raster=None,
    input_vertical_raster=None,
    vertical_factor="BINARY 1 -30 30",
    input_horizontal_raster=None,
    horizontal_factor="BINARY 1 45",
    source_initial_accumulation=None,
    source_maximum_accumulation=None,
    source_cost_multiplier=None,
    source_direction=None,
    distance_method="PLANAR",
    output_distance_allocation_raster_name=None,
    output_distance_accumulation_raster_name=None,
    output_back_direction_raster_name=None,
    output_source_direction_raster_name=None,
    output_source_location_raster_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.distance_allocation(
        input_source_raster_or_features=input_source_raster_or_features,
        input_barrier_raster_or_features=input_barrier_raster_or_features,
        input_surface_raster=input_surface_raster,
        input_cost_raster=input_cost_raster,
        input_vertical_raster=input_vertical_raster,
        vertical_factor=vertical_factor,
        input_horizontal_raster=input_horizontal_raster,
        horizontal_factor=horizontal_factor,
        source_initial_accumulation=source_initial_accumulation,
        source_maximum_accumulation=source_maximum_accumulation,
        source_cost_multiplier=source_cost_multiplier,
        source_direction=source_direction,
        distance_method=distance_method,
        output_distance_allocation_raster_name=output_distance_allocation_raster_name,
        output_distance_accumulation_raster_name=output_distance_accumulation_raster_name,
        output_back_direction_raster_name=output_back_direction_raster_name,
        output_source_direction_raster_name=output_source_direction_raster_name,
        output_source_location_raster_name=output_source_location_raster_name,
        context=context,
        future=future,
        **kwargs,
    )


def analyze_changes_using_ccdc(
    input_multidimensional_raster=None,
    bands_for_detecting_change: list[int] = [],
    bands_for_temporal_masking: list[int] = [],
    chi_squared_threshold: float = 0.99,
    min_anomaly_observations: int = 6,
    update_frequency: int = 1,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function evaluates changes in pixel values over time using the CCDC algorithm,
    and generates a multidimensional raster containing the model results.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
                                             The input multidimensional raster must have at least 12 slices, spanning at least 1 year.
    ------------------------------------     --------------------------------------------------------------------
    bands_for_detecting_change               Optional List. The band IDs to use for change detection.
                                             If no band IDs are provided, all the bands from the input raster dataset will be used.
                                             Each element in the list should be within the range 1 to n where n is the
                                             number of bands of the input raster.

                                             Example:

                                                  [1,2,3,4,6]
    ------------------------------------     --------------------------------------------------------------------
    bands_for_temporal_masking               Optional List. The band IDs of the green band and the SWIR band, to be used to
                                             mask for cloud, cloud shadow and snow. If band IDs are not provided, no
                                             masking will occur.
                                             Each element in the list should be within the range 1 to n where n is
                                             the number of bands of the input raster.

                                             Example:

                                                [1,2]
    ------------------------------------     --------------------------------------------------------------------
    chi_squared_threshold                    Optional Float. The chi-square change probability threshold. If an
                                             observation has a calculated change probability that is above this
                                             threshold, it is flagged as an anomaly, which is a potential change
                                             event. The default value is 0.99.

                                             Example:

                                                0.99
    ------------------------------------     --------------------------------------------------------------------
    min_anomaly_observations                 Optional Integer. The minimum number of consecutive anomaly observations
                                             that must occur before an event is considered a change. A pixel must
                                             be flagged as an anomaly for the specified number of consecutive
                                             time slices before it is considered a true change. The default value is 6.
    ------------------------------------     --------------------------------------------------------------------
    update_frequency                         Optional Float. The value that represents the update frequency.
                                             The default value is 1.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}


                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}


                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}


                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}


                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: Imagery layer item

    .. code-block:: python

            # Usage Example 1: This example performs continuous change detection where only one band is used in the change detection
            # and the chi-squared probability threshold is 0.90.
            analyze_changes_using_ccdc_op = analyze_changes_using_ccdc(input_multidimensional_raster=input_multidimensional_raster,
                                                                       bands_for_detecting_change=[0],
                                                                       bands_for_temporal_masking=[],
                                                                       chi_squared_threshold=0.99,
                                                                       min_anomaly_observations=6,
                                                                       update_frequency=1,
                                                                       future=False,
                                                                       gis=gis
                                                                      )

    .. code-block:: python

            # Usage Example 2: This example performs continuous change detection where bands 3 and 7 (indexed at 2 and 6)
            # are used as snow, cloud, and cloud shadow mask.
            analyze_changes_using_ccdc_op = analyze_changes_using_ccdc(input_multidimensional_raster=input_multidimensional_raster,
                                                                       bands_for_detecting_change=[0,1,2,3,4,5,6],
                                                                       bands_for_temporal_masking=[2,6],
                                                                       chi_squared_threshold=0.99,
                                                                       min_anomaly_observations=3,
                                                                       update_frequency=1,
                                                                       future=False,
                                                                       gis=gis
                                                                      )
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.analyze_changes_using_ccdc(
        input_multidimensional_raster=input_multidimensional_raster,
        bands_for_detecting_change=bands_for_detecting_change,
        bands_for_temporal_masking=bands_for_temporal_masking,
        chi_squared_threshold=chi_squared_threshold,
        min_anomaly_observations=min_anomaly_observations,
        update_frequency=update_frequency,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


def detect_change_using_change_analysis_raster(
    input_change_analysis_raster=None,
    change_type: str = "TIME_OF_LATEST_CHANGE",
    max_number_of_changes: int = 1,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    segment_date: str = "BEGINNING_OF_SEGMENT",
    change_direction: str = "ALL",
    filter_by_year: bool = False,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    filter_by_duration: bool = False,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    filter_by_magnitude: bool = False,
    min_magnitude: Optional[float] = None,
    max_magnitude: Optional[float] = None,
    filter_by_start_value: Optional[bool] = None,
    min_start_value: Optional[float] = None,
    max_start_value: Optional[float] = None,
    filter_by_end_value: Optional[bool] = None,
    min_end_value: Optional[float] = None,
    max_end_value: Optional[float] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function generates a raster containing pixel change information using the
    output change analysis raster from the :meth:`~arcgis.raster.analytics.analyze_changes_using_ccdc`
    or :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_change_analysis_raster             Required :class:`~arcgis.raster.ImageryLayer` object. The raster generated from the :meth:`~arcgis.raster.analytics.analyze_changes_using_ccdc` or :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr`
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    change_type                              Optional String. Specifies the change information to calculate.

                                             - TIME_OF_LATEST_CHANGE - Each pixel will contain the date of the most recent change for that pixel in the time series. This is the default.
                                             - TIME_OF_EARLIEST_CHANGE - Each pixel will contain the date of the earliest change for that pixel in the time series.
                                             - TIME_OF_LARGEST_CHANGE - Each pixel will contain the date of the most significant change for that pixel in the time series.
                                             - NUM_OF_CHANGES - Each pixel will contain the total number of times the pixel changed in the time series.
                                             - TIME_OF_LONGEST_CHANGE - Each pixel will contain the date of change at the end of the longest transition segment in the time series. Option available in ArcGIS Image Server 10.9 and higher.
                                             - TIME_OF_SHORTEST_CHANGE - Each pixel will contain the date of change at the end of the shortest transition segment in the time series. Option available in ArcGIS Image Server 10.9 and higher.
                                             - TIME_OF_FASTEST_CHANGE - Each pixel will contain the date of change at the end of the transition that occurred most quickly. Option available in ArcGIS Image Server 10.9 and higher.
                                             - TIME_OF_SLOWEST_CHANGE - Each pixel will contain the date of change at the end of the transition that occurred most slowly. Option available in ArcGIS Image Server 10.9 and higher.

                                             Example:

                                                "TIME_OF_LATEST_CHANGE"
    ------------------------------------     --------------------------------------------------------------------
    max_number_of_changes                    Optional Integer. The maximum number of changes per pixel that will
                                             be calculated. This number corresponds to the number of bands in the output raster.
                                             The default is 1, meaning only one change date will be calculated,
                                             and the output raster will contain only one band.

                                             This parameter is not available when the change_type parameter is set to NUM_OF_CHANGES.

                                             Example:

                                                3
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    segment_date                             Optional string. Specifies whether to extract the date at the beginning
                                             of a change segment, or the end.

                                             This parameter is available only when the input change analysis raster is
                                             the output from the :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.

                                             - BEGINNING_OF_SEGMENT - Extract the date at the beginning of a change segment. This is the default.
                                             - END_OF_SEGMENT - Extract the date at the end of a change segment.

                                             Example:

                                                "END_OF_SEGMENT"

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    change_direction                         Optional string. The direction of change to be included in the analysis.
                                             For example, choose Increasing to only extract date of change information for
                                             periods where the change is in the positive or increasing direction.

                                             This parameter is available only when the input change analysis raster
                                             is the output from the :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.

                                             - ALL - All change directions will be included in the output. This is the default.
                                             - INCREASE - Only change in the positive or increasing direction will be included in the output.
                                             - DECREASE - Only change in the negative or decreasing direction will be included in the output.

                                             Example:

                                                "DECREASE"

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_year                           Optional boolean. Specifies whether to filter by a range of years.

                                             - True - Filter results such that only changes that occurred within a specific range of years is included in the output.
                                             - False - Do not filter results by year. This is the default.

                                             Example:

                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_year                                 Optional int. The earliest year to use to filter results. This parameter
                                             is required if the filter_by_year parameter is set to True.

                                             Example:

                                                2000

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_year                                 Optional int. The latest year to use to filter results. This parameter
                                             is required if the filter_by_year parameter is set to True.

                                             Example:

                                                2005

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_duration                       Optional boolean. Specifies whether to filter by the change duration.
                                             This parameter is available only when the input change analysis raster
                                             is the output from the :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.

                                             - True - Filter results by duration such that only the changes that lasted a given amount of time will be included in the output.
                                             - False - Do not filter results by duration. This is the default.

                                             Example:

                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_duration                             Optional float. The minimum number of consecutive years to include in
                                             the results. This parameter is required if the filter_by_duration parameter
                                             is set to True

                                             Example:

                                                2

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_duration                             Optional float. The maximum number of consecutive years to include
                                             in the results. This parameter is required if the filter_by_duration
                                             parameter is set to True

                                             Example:

                                                4

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_magnitude                      Optional boolean. Specifies whether to filter by change magnitude.

                                             - True - Filter results by magnitude such that only the changes of a given magnitude will be included in the output.
                                             - False - Do not filter results by magnitude. This is the default.

                                             Example:

                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_magnitude                            Optional float. The minimum magnitude to include in the results.
                                             This parameter is required if the filter_by_magnitude
                                             parameter is set to True.

                                             Example:

                                                0.25

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_magnitude                            Optional float. The maximum magnitude to include in the results.
                                             This parameter is required if the filter_by_magnitude parameter is set
                                             to True.

                                             Example:

                                                3

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_start_value                    Optional boolean. Specifies whether to filter by start value. This
                                             parameter is available only when the input change analysis raster
                                             is the output from the :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.

                                             - True - Filter results by start value so that only the change that starts with value defined by a range.
                                             - False - Do not filter by start value. This is the default.

                                             Example:

                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_start_value                          Optional float. The minimum value that defines the range of start value.
                                             This parameter is required if the filter_by_start_value parameter is set
                                             to True.

                                             Example:

                                                0.75

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_start_value                          Optional float. The maximum value that defines the range of start value.
                                             This parameter is required if the filter_by_start_value parameter is
                                             set to True.

                                             Example:

                                                0.9

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    filter_by_end_value                      Optional boolean. Specifies whether to filter by end value. This parameter
                                             is available only when the input change analysis raster is the output
                                             from the :meth:`~arcgis.raster.analytics.analyze_changes_using_landtrendr` function.

                                             - True - Filter results by end value so that only the change that ends with value defined by a range.
                                             - False - Do not filter results by end value. This is the default.

                                             Example:

                                                True

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    min_end_value                            Optional float. The minimum value that defines the range of end value.
                                             This parameter is required if the filter_by_end_value parameter is set
                                             to True.

                                             Example:

                                                -0.12

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    max_end_value                            Optional float. The maximum value that defines the range of end value.
                                             This parameter is required if the filter_by_end_value parameter is set
                                             to True.

                                             Example:

                                                0.35

                                             Parameter available in ArcGIS Image Server 10.9 and higher.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`gis.content.create_folder <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: Imagery layer item

    .. code-block:: python

        # Usage Example 1: This example returns the most recent date at which pixels changed in the input time series.

        detect_change_op = detect_change_using_change_analysis_raster(input_change_analysis_raster=input_change_analysis_raster,
                                                                      change_type="TIME_OF_LATEST_CHANGE",
                                                                      max_number_of_changes=1,
                                                                      gis=gis)

    .. code-block:: python

        # Usage Example 2: This example returns the total number of times the pixels changed in the input time series.

        detect_change_op = detect_change_using_change_analysis_raster(input_change_analysis_raster=input_change_analysis_raster,
                                                                      change_type="NUM_OF_CHANGES",
                                                                      gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.detect_change_using_change_analysis_raster(
        input_change_analysis_raster=input_change_analysis_raster,
        change_type=change_type,
        max_number_of_changes=max_number_of_changes,
        segment_date=segment_date,
        change_direction=change_direction,
        filter_by_year=filter_by_year,
        min_year=min_year,
        max_year=max_year,
        filter_by_duration=filter_by_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        filter_by_magnitude=filter_by_magnitude,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        filter_by_start_value=filter_by_start_value,
        min_start_value=min_start_value,
        max_start_value=max_start_value,
        filter_by_end_value=filter_by_end_value,
        min_end_value=min_end_value,
        max_end_value=max_end_value,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


def manage_multidimensional_raster(
    target_multidimensional_raster,
    manage_mode: str = "APPEND_SLICES",
    variables: Optional[list] = None,
    input_multidimensional_rasters: Optional[list] = None,
    dimension_name: Optional[str] = None,
    dimension_value: Optional[str] = None,
    dimension_description: Optional[str] = None,
    dimension_unit: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function edits a multidimensional raster by adding or deleting variables or dimensions.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    target_multidimensional_raster           Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    manage_mode                              Optional string. Specifies the type of modification that will be performed
                                             on the target raster.

                                             - ADD_DIMENSION - Add a new dimension to the multidimensional raster information.

                                             - APPEND_SLICES - Add slices from another multidimensional raster. Slices are added to the end of the slices for a dimension. This is the default.

                                             - APPEND_VARIABLES - Add one or more variable from another multidimensional raster.

                                             - REPLACE_SLICES - Replace existing slices from another multidimensional raster, at specific dimension values.

                                             - DELETE_VARIABLES - Delete one or more variables from the multidimensional raster.

                                             - REMOVE_DIMENSION - Convert a single slice multidimensional raster into a dimensionless raster.
    ------------------------------------     --------------------------------------------------------------------
    variables                                Optional List. The variable or variables that will be modified in the
                                             target multidimensional raster. This is required if the operation
                                             being performed is a modification of an existing variable.

                                             If no variable is specified, the first variable in the target
                                             multidimensional raster will be modified.
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_rasters           Optional list of input multidimensional raster. This is required
                                             when manage_mode is set to APPEND_SLICES, REPLACE_SLICES, or APPEND_VARIABLES.
    ------------------------------------     --------------------------------------------------------------------
    dimension_name                           Optional string. The name of the dimension to be added to the dataset.
                                             This is required if manage_mode is set to ADD_DIMENSION.
    ------------------------------------     --------------------------------------------------------------------
    dimension_value                          Optional string. The value of the dimension to be added.
                                             This is required if manage_mode is set to ADD_DIMENSION.
    ------------------------------------     --------------------------------------------------------------------
    dimension_description                    Optional string. The description of the dimension to be added.
                                             This is required if manage_mode is set to ADD_DIMENSION.
    ------------------------------------     --------------------------------------------------------------------
    dimension_unit                           Optional string. The unit of the dimension to be modified.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` object. the GIS on which this tool runs. If not specified,
                                             the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return: output_raster : Imagery Layer URL

    .. code-block:: python

        # Usage Example 1: This example appends variables to a multidimensional raster dataset.

        target_mdim_raster = gis.content.search("my_target_mdim_raster")[0].layers[0]
        input_mdim_raster = gis.content.search("my_input_mdim_raster")[0].layers[0]
        variables_to_add = ["variable_1", "variable_2"]

        manage_mdim_op = manage_multidimensional_raster(target_multidimensional_raster=target_mdim_raster,
                                                        manage_mode="APPEND_VARIABLES",
                                                        variables=variables_to_add,
                                                        input_multidimensional_rasters=[input_mdim_raster],
                                                        gis=gis)

        # Usage Example 2: This example adds the StdZ dimension to a multidimensional raster dataset.

        target_mdim_raster = gis.content.search("my_target_mdim_raster")[0].layers[0]

        manage_mdim_op = manage_multidimensional_raster(target_multidimensional_raster=target_mdim_raster,
                                                        manage_mode="ADD_DIMENSION",
                                                        variables="my_variable",
                                                        dimension_name="StdZ",
                                                        dimension_value="0",
                                                        dimension_description="Depth",
                                                        dimension_unit="m",
                                                        gis=gis)

    """

    # task = "ManageMultidimensionalRaster"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.manage_multidimensional_raster(
        target_multidimensional_raster=target_multidimensional_raster,
        manage_mode=manage_mode,
        variables=variables,
        input_multidimensional_rasters=input_multidimensional_rasters,
        dimension_name=dimension_name,
        dimension_value=dimension_value,
        dimension_description=dimension_description,
        dimension_unit=dimension_unit,
        future=future,
        **kwargs,
    )


def sample(
    input_rasters,
    input_location_data,
    resampling_type: str = "NEAREST",
    unique_id_field: Optional[int] = None,
    acquisition_definition: Optional[dict[str, str]] = None,
    statistics_type: str = "MEAN",
    percentile_value: Optional[int] = None,
    buffer_distance: Optional[int] = None,
    layout: str = "ROW_WISE",
    generate_feature_class: bool = False,
    process_as_multidimensional: Optional[bool] = None,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function creates a table that shows the values of cells from a raster, 
    or set of rasters, for defined locations. The locations are defined by raster cells, 
    polygon features, polyline features, or by a set of points.
    The input rasters can be two-dimensional or multidimensional. 
    The structure of the output table changes when the input rasters are multidimensional.
    Function available in ArcGIS Image Server 10.8.1 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_rasters                            Required list of :class:`~arcgis.raster.ImageryLayer` object. List of portal items can be passed.
    ------------------------------------     --------------------------------------------------------------------
    input_location_data                      Required :class:`~arcgis.raster.ImageryLayer` or :class:`~arcgis.features.FeatureLayer`
                                             object. Data identifying positions at which you want a sample taken.
                                             Polyline and polygon feature services are supported when
                                             processAsMultidimensional is set to True in the context. 
    ------------------------------------     --------------------------------------------------------------------
    resampling_type                          Optional str. Resampling algorithm used when sampling a raster.

                                             - NEAREST: Nearest neighbor assignment. This is the default.
                                             - BILINEAR: Bilinear interpolation
                                             - CUBIC: Cubic convolution

                                             Examples:
                                                "NEAREST"
    ------------------------------------     --------------------------------------------------------------------
    unique_id_field                          Optional int. A field containing a different value for every 
                                             location or feature in the input location raster or point features.

                                             Example:
                                                "FID"
    ------------------------------------     --------------------------------------------------------------------
    acquisition_definition                   Optional dictionary. Specify the time, depth or other acquisition \
                                             data associated with the location features.
                                             
                                             Only the following combinations are supported:

                                             - Dimension + Start field or value
                                             - Dimension + Start field or value + End field or value
                                             - Dimension + Start field or value + Relative value or days before + Relative value or days after
                                              
                                             Relative value or days before and Relative value or days after only support non-negative values.
                                             
                                             Statistics will be calculated for variables within this dimension range. 
                                             
                                             Syntax: a list of dictionary objects.

                                                | [{"dimension":  "Dimension",
                                                | "startFieldOrVal": "Start field or value", 
                                                | "endFieldOrVal": "End field or value", 
                                                | "relValOrDaysBefore": "Relative value or days before", 
                                                | "relValOrDaysAfter": "Relative value or days after"}]

                                             Example:

                                                | [{"dimension":  "Dimension",
                                                | "startFieldOrVal": "1999-01-01T00:00:00", 
                                                | "endFieldOrVal": "2019-01-01T00:00:00"}]
    ------------------------------------     --------------------------------------------------------------------
    statistics_type                          Optional string.
                                             The type of statistic to be calculated.

                                             - MINIMUM - Finds the minimum within the specified range.
                                             - MAXIMUM - Finds the maximum within the specified range.
                                             - MEDIAN - Finds the median within the specified range.
                                             - MEAN - Calculates the average for the specified range. This is the default.
                                             - SUM - Calculates the sum of the variables within the specified range.
                                             - MAJORITY - Finds the value that occurs most frequently.
                                             - MINORITY - Finds the value that occurs least frequently.
                                             - STD - Calculates the standard deviation.
                                             - PERCENTILE - Calculates a defined percentile within the specified range.
    ------------------------------------     --------------------------------------------------------------------
    percentile_value                         Optional int. The percentile to calculate when the  
                                             statistics_type parameter is set to PERCENTILE.
                                             This value can range from 0 to 100. The default is 90. 
    ------------------------------------     --------------------------------------------------------------------
    buffer_distance                          Optional int. The specified distance around the location data 
                                             features. The buffer distance is specified in the linear unit 
                                             of the location feature's spatial reference. If the feature 
                                             uses a geographic reference, the unit will be in degrees.
                                             Statistics will be calculated within this buffer area. 
    ------------------------------------     --------------------------------------------------------------------
    layout                                   Optional string. Specifies whether sampled values appear in rows or 
                                             columns in the output table.

                                             - ROW_WISE - Sampled values appear in separate rows in the output table. This is the default.
                                             - COLUMN_WISE - Sampled values appear in separate columns in the output table. This option is only valid when the input multidimensional raster contains one variable and one dimension, and each slice is a single-band raster.
    ------------------------------------     --------------------------------------------------------------------
    generate_feature_class                   Optional bool, Boolean value to determine if this function generates 
                                             a feature layer with sampled values or only a table with sampled values. 
                                             By default, it is False.
    ------------------------------------     --------------------------------------------------------------------
    process_as_multidimensional              Optional bool, Process as multidimensional if set to True, 
                                             if the input is multidimensional raster.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional string. Name of the output feature item or table item to be created.
                                             If not provided, a random name is generated by the method and used as 
                                             the output name. 
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return: :class:`~arcgis.features.FeatureLayer` or :class:`~arcgis.features.Table` object

    .. code-block:: python

        # Usage Example 1: Create a table that shows values of cells from the raster(s) for defined locations.

        raster_1 = gis.content.search("my_raster_1")[0].layers[0]
        raster_2 = gis.content.search("my_raster_2")[0].layers[0]
        rasters = [raster1, raster2]
        location_data = gis.content.search("my_location_data")[0].layers[0]

        sample_op = sample(input_rasters=rasters,
                           input_location_data=location_data,
                           output_name="sample_op_data",
                           gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    if context is None:
        context = {}
    if process_as_multidimensional is not None:
        context.update({"processAsMultidimensional": process_as_multidimensional})

    return gis._tools.rasteranalysis.sample(
        in_rasters=input_rasters,
        in_location_data=input_location_data,
        output_name=output_name,
        resampling_type=resampling_type,
        unique_id_field=unique_id_field,
        acquisition_definition=acquisition_definition,
        statistics_type=statistics_type,
        percentile_value=percentile_value,
        buffer_distance=buffer_distance,
        layout=layout,
        generate_feature_class=generate_feature_class,
        context=context,
        future=future,
        **kwargs,
    )


def merge_multidimensional_rasters(
    input_multidimensional_rasters,
    resolve_overlap_method: str = "FIRST",
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function merges several multidimensional rasters spatially, or across variables and dimensions into one.
    Function available in ArcGIS Image Server 10.9 and higher (not available in ArcGIS Online).

    ====================================     =============================================================================================================================================
    **Parameter**                             **Description**
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    input_multidimensional_rasters           Required list of :class:`~arcgis.raster.ImageryLayer` object. List of input multidimensional rasters to be combined.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    resolve_overlap_method                   Optional string. Specifies the method used to handle overlapping pixels when merging rasters in the combined datasets.

                                             - FIRST - The pixel value in the overlapping areas will be the value from the first raster in the list of input rasters. This is the default.

                                             - LAST - The pixel value in the overlapping areas will be the value from the last raster in the list of input rasters.

                                             - MIN - The pixel value in the overlapping areas will be the minimum value of the overlapping pixels.

                                             - MAX - The pixel value in the overlapping areas will be the maximum value of the overlapping pixels.

                                             - MEAN - The pixel value in the overlapping areas will be the average of the overlapping pixels.

                                             - SUM - The pixel value in the overlapping areas will be the total sum of the overlapping pixels.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}


                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}


                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                               Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}


                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                               Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}


                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}


                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: Bilinear, Nearest, Cubic.

                                               Example:

                                                    {'resamplingMethod': "Nearest"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    gis                                      Optional GIS. The :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     =============================================================================================================================================

    :return:
        The output imagery layer item

    .. code-block:: python

        # Usage Example 1: Merge two multidimensional rasters with different variables..

        merge_mdim_rasters_op = merge_multidimensional_rasters(input_multidimensional_rasters=[input_multidimensional_raster_var1, input_multidimensional_raster_var2],
                                                               resolve_overlap_method="FIRST",
                                                               gis=gis)


    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError(
            "merge_multidimensional_rasters() is not supported on ArcGIS Online"
        )

    return gis._tools.rasteranalysis.merge_multidimensional_rasters(
        input_multidimensional_rasters=input_multidimensional_rasters,
        resolve_overlap_method=resolve_overlap_method,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


def analyze_changes_using_landtrendr(
    input_multidimensional_raster,
    processing_band: Optional[str] = None,
    snapping_date: str = "06-30",
    max_num_segments: int = 5,
    vertex_count_overshoot: int = 2,
    spike_threshold: float = 0.9,
    recovery_threshold: float = 0.25,
    prevent_one_year_recovery: bool = True,
    increasing_recovery_trend: bool = True,
    min_num_observations: int = 6,
    best_model_proportion: float = 1.25,
    pvalue_threshold: float = 0.01,
    output_other_bands: bool = False,
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function evaluates changes in pixel values over time using the Landsat-based detection of trends
    in disturbance and recovery (LandTrendr) method and generates a change analysis raster containing the model results.
    Function available in ArcGIS Image Server 10.9 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required :class:`~arcgis.raster.ImageryLayer` object. The input multidimensional raster.
                                             Portal Item can be passed.
    ------------------------------------     --------------------------------------------------------------------
    processing_band                          Optional string. The band to use for segmenting the pixel value
                                             trajectories over time. Choose the band that will best capture the
                                             changes in the feature you want to observe.

                                             If no band value is specified and the input is multiband imagery,
                                             the first band in the multiband image will be used.

                                             Example:

                                                  "Band_1"
    ------------------------------------     --------------------------------------------------------------------
    snapping_date                            Optional string. The date used to select a slice for each year in the
                                             input multidimensional dataset. The slice with the date closest to
                                             the snapping date will be selected. This parameter is required if
                                             the input dataset contains sub-yearly data.

                                             The default is "06-30" (or June 30), approximately midway through a calendar year.

                                             Example:

                                                "06-30"
    ------------------------------------     --------------------------------------------------------------------
    max_num_segments                         Optional int. The maximum number of segments to be fitted to the
                                             time series for each pixel. The default is 5.

                                             Example:

                                                5
    ------------------------------------     --------------------------------------------------------------------
    vertex_count_overshoot                   Optional int. The number of additional vertices beyond
                                             max_num_segments + 1 that can be used to fit the model during
                                             the initial stage of identifying vertices. Later in the modeling
                                             process, the number of additional vertices will be reduced to
                                             max_num_segments + 1. The default is 2.

                                             Example:

                                                2
    ------------------------------------     --------------------------------------------------------------------
    spike_threshold                          Optional float. The threshold to use for dampening spikes or anomalies
                                             in the pixel value trajectory. The value must range between 0 and 1,
                                             where 1 means no dampening. The default is 0.9.

                                             Example:

                                                0.9
    ------------------------------------     --------------------------------------------------------------------
    recovery_threshold                       Optional float. The recovery threshold value, in years. If a segment
                                             has a recovery rate that is faster than 1/recovery threshold, the segment
                                             is discarded and not included in the time series model. The value must
                                             range between 0 and 1. The default is 0.25.

                                             Example:

                                                0.25
    ------------------------------------     --------------------------------------------------------------------
    prevent_one_year_recovery                Optional boolean. Specifies whether segments that exhibit a one year
                                             recovery will be excluded.

                                             - True - Segments that exhibit a one year recovery will be excluded. This is the default.

                                             - False - Segments that exhibit a one year recovery will not be excluded.

                                             Example:

                                                True
    ------------------------------------     --------------------------------------------------------------------
    increasing_recovery_trend                Optional boolean. Specifies whether the recovery has an increasing (positive) trend.

                                             - True - The recovery has an increasing trend. This is the default.

                                             - False - The recovery has a decreasing trend.

                                             Example:

                                                True
    ------------------------------------     --------------------------------------------------------------------
    min_num_observations                     Optional int. The minimum number of valid observations required to
                                             perform fitting. The number of years in the input multidimensional
                                             dataset must be equal to or greater than this value. The default is 6.

                                             Example:

                                                6
    ------------------------------------     --------------------------------------------------------------------
    best_model_proportion                    Optional float. The best model proportion value. During the model
                                             selection process, the tool will calculate the p-value for each
                                             model and select a model that has the most vertices while
                                             maintaining the smallest (most significant) p-value based on this
                                             proportion value. A value of 1 means the model has the lowest
                                             p-value but may not have a high number of vertices.
                                             The default is 1.25.

                                             Example:

                                                1.25
    ------------------------------------     --------------------------------------------------------------------
    pvalue_threshold                         Optional float. The p-value threshold for a model to be selected.
                                             After the vertices are detected in the initial stage of the model
                                             fitting, the tool will fit each segment and calculate the p-value
                                             to determine the significance of the model. On the next iteration,
                                             the model will decrease the number of segments by one and
                                             recalculate the p-value. This will continue and, if the p-value
                                             is smaller than the value specified in this parameter, the model
                                             will be selected and the tool will stop searching for a better model.
                                             If no such model is selected, the tool will select a model with a
                                             p-value smaller than the lowest p-value × best model proportion value.
                                             The default is 0.01.

                                             Example:

                                                0.01
    ------------------------------------     --------------------------------------------------------------------
    output_other_bands                       Optional boolean. Specifies whether other bands will be included in the
                                             segmentation process.

                                             - True - Other bands will be included. The segmentation and vertices information from the initial segmentation band specified in the processing_band parameter will also be fitted to the remaining bands in the multiband images. The model results will include the segmentation band first, then the remaining bands.

                                             - False - Other bands will not be included. This is the default.

                                             Example:

                                                2
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                                Example:

                                                    {"outSR": {spatial reference}}

                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                                Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                                Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                                Example:

                                                Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                Syntax example with a specified percentage of total
                                                processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             On AGOL, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter on enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return: Imagery layer item

    .. code-block:: python

            # Usage Example 1:
            analyze_changes_using_landtrendr_op = analyze_changes_using_landtrendr(input_multidimensional_raster=input_multidimensional_raster,
                                                                               processing_band="Band_1")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.analyze_changes_using_landtrendr(
        input_multidimensional_raster=input_multidimensional_raster,
        processing_band=processing_band,
        snapping_date=snapping_date,
        max_num_segments=max_num_segments,
        vertex_count_overshoot=vertex_count_overshoot,
        spike_threshold=spike_threshold,
        recovery_threshold=recovery_threshold,
        prevent_one_year_recovery=prevent_one_year_recovery,
        increasing_recovery_trend=increasing_recovery_trend,
        min_num_observations=min_num_observations,
        best_model_proportion=best_model_proportion,
        pvalue_threshold=pvalue_threshold,
        output_other_bands=output_other_bands,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


# def transfer_files(input_files,
#                   output_datastore=None,
#                   tf_filter=None,
#                   return_first_file=False,
#                   context=None,
#                   *,
#                   gis=None,
#                   future=False,
#                   **kwargs):

#    """
#    Transfers files between datastores.
#    The tool does not support file transfers to or from geodatabases.

#    ====================================     ====================================================================
#    **Parameter**                             **Description**
#    ------------------------------------     --------------------------------------------------------------------
#    input_files                              Required str or list, input files or folders that will be copied to the output folder.
#                                             It can be a string specifying the datastore path.

#                                             Example:
#                                             "/rasterStores/testcloudstore/datafolder"
#                                             "/fileShares/testfolder/raster.tif"
#                                             "/cloudStores/testcloud/raster.tif"
#    ------------------------------------     --------------------------------------------------------------------
#    output_datastore                         Optional str, datastore path of the output folder where the files will be copied.
#                                             If not specified, the data is copied to the rasterstore.

#                                             Example:
#                                             "/rasterStores/s3cloudstore/datafolder"
#    ------------------------------------     --------------------------------------------------------------------
#    tf_filter                                Optional str, to specify the filter while transfering files.

#                                             Example:
#                                             "*.tif"
#    ------------------------------------     --------------------------------------------------------------------
#    return_first_file                        Optional bool. Available in ArcGIS Image Server 10.8.1 and higher.
#    ------------------------------------     --------------------------------------------------------------------
#    context                                  Context contains additional settings that affect task execution.

#                                             context parameter overwrites values set through arcgis.env parameter

#                                             This function has the following settings:

#                                              - Parallel Processing Factor (parallelProcessingFactor): controls
#                                                Raster Processing (CPU) service instances.

#                                                Example:
#                                                    Syntax example with a specified number of processing instances:

#                                                    {"parallelProcessingFactor": "2"}

#                                                    Syntax example with a specified percentage of total
#                                                    processing instances:

#                                                    {"parallelProcessingFactor": "60%"}
#    ------------------------------------     --------------------------------------------------------------------
#    gis                                      Optional GIS object. If not specified, the currently active connection
#                                             is used.
#    ------------------------------------     --------------------------------------------------------------------
#    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and
#                                             results will be returned asynchronously.
#    ====================================     ====================================================================

#    :return: Output location

#    .. code-block:: python

#        # Usage Example 1: This example transfers raster.tif from a cloudstore location to a fileshare location.

#        transfer_file = transfer_files(input_files="/cloudStores/testcloud/raster.tif",
#                                       output_datastore="/fileShares/testfolder",
#                                       tf_filter=None,
#                                       gis=gis)

#    """

#    gis = _arcgis.env.active_gis if gis is None else gis


#    return gis._tools.rasteranalysis.transfer_files(input_files=input_files,
#                                                    output_datastore=output_datastore,
#                                                    tf_filter=tf_filter,
#                                                    return_first_file=return_first_file,
#                                                    context=context,
#                                                    future=future,
#                                                    **kwargs)
def zonal_statistics_as_table(
    input_zone_raster_or_features,
    input_value_raster,
    zone_field: str,
    ignore_nodata: bool = True,
    statistic_type: bool = "ALL",
    percentile_values: list[int] = [90],
    process_as_multidimensional: bool = False,
    percentile_interpolation_type: str = "AUTO_DETECT",
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    circular_calculation: bool = False,
    circular_wrap_value: float = 360,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Calculates  the values of a raster within the zones of another dataset and reports the results to a table.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_zone_raster_or_features            Required. The input that defines the zones. Both raster and feature 
                                             can be used for the zone input.
    ------------------------------------     --------------------------------------------------------------------
    input_value_raster                       Required raster. Raster that contains the values on which to summarize a statistic.
    ------------------------------------     --------------------------------------------------------------------
    zone_field                               Required parameter.  The field that defines each zone. It can be an 
                                             integer or a string field of the zone dataset.
    ------------------------------------     --------------------------------------------------------------------
    ignore_nodata                            Optional boolean. Denotes whether NoData values in the value input 
                                             will influence the results of the zone that they fall within.
                                             
                                             - True - Within any particular zone, only cells that have a value in \
                                             the input value raster will be used in determining the output value \
                                             for that zone. NoData cells in the value raster will be ignored in \
                                             the statistic calculation. This is the default.
                                             
                                             - False - Within any particular zone, if any NoData cells exist in the \
                                             value raster, it is deemed that there is insufficient information to \
                                             perform statistical calculations for all the cells in that zone; \
                                             therefore, the entire zone will receive the NoData value on the output raster.
    ------------------------------------     --------------------------------------------------------------------
    statistic_type                           Optional string.  Choose the statistic to calculate.The available options 
                                             when the value raster is integer are ALL, MEAN, MAJORITY, MAXIMUM, MEDIAN, 
                                             MINIMUM, MINORITY, PERCENTILE, RANGE, STD, SUM, VARIETY,  
                                             MIN_MAX, MEAN_STD, and  MIN_MAX_MEAN.

                                             If the value raster is float, the options are ALL, MEAN, MAXIMUM, MINIMUM, 
                                             RANGE, STD, and SUM.

                                             - ALL - All of the statistics will be calculated. \
                                             This is the default.

                                             - MEAN - Calculates the average of all cells in the raster layer to be summarized that \
                                             belong to the same zone as the output cell.

                                             - MAJORITY - Determines the value that occurs most often of all cells in the raster \
                                             layer to be summarized that belong to the same zone as the output cell.

                                             - MAXIMUM - Determines the largest value of all cells in the raster layer \
                                             to be summarized that belong to the same zone as the output cell.

                                             - MEDIAN - Determines the median value of all cells in the raster layer \
                                             to be summarized that belong to the same zone as the output cell.

                                             - MINIMUM - Determines the smallest value of all cells in the raster \
                                             layer to be summarized that belong to the same zone as the output cell.

                                             - MINORITY - Determines the value that occurs least often of all cells in \
                                             the raster layer to be summarized that belong to the same zone as the \
                                             output cell.

                                             - PERCENTILE - Calculates a percentile of all cells in the value raster \
                                             that belong to the same zone as the output cell. The 90th percentile is calculated by default. \
                                             You can specify other values (from 0 to 100) using the Percentile Values parameter.

                                             - RANGE - Calculates the difference between the largest and smallest value of all \
                                             cells in the raster layer to be summarized that belong to the same zone \
                                             as the output cell.

                                             - STD - Calculates the standard deviation of all cells in \
                                             the raster layer to be summarized that belong to the same zone as the output cell.

                                             - SUM - Calculates the total value of all cells in the raster layer to be \
                                             summarized that belong to the same zone as the output cell.

                                             - VARIETY- Calculates the number of unique values for all cells in the raster \
                                             layer to be summarized that belong to the same zone as the output cell.

                                             - MIN_MAX - Both the minimum and maximum statistics are calculated.

                                             - MEAN_STD -  Both the mean and standard deviation statistics \
                                             are calculated.

                                             - MIN_MAX_MEAN - The minimum, maximum and mean statistics are calculated.
    ------------------------------------     --------------------------------------------------------------------
    percentile_values                        Optional list of double values.
                                             The percentile to calculate. The default is 90, for the 90th percentile.
                                             The values can range from 0 to 100. The 0th percentile is essentially 
                                             equivalent to the Minimum statistic, and the 100th Percentile is equivalent to 
                                             Maximum. A value of 50 will produce essentially the same result as the Median statistic.
                                             This option is only available if the Statistics Type parameter is set to PERCENTILE or ALL.
    ------------------------------------     --------------------------------------------------------------------
    process_as_multidimensional              Optional bool, Determines how the input rasters will be processed if they 
                                             are multidimensional.

                                             - False - Statistics will be calculated from the current slice of a \
                                             multidimensional image service. This is the default.
                                             - True - Statistics will be calculated for all dimensions (such as time or depth) \
                                             of a multidimensional image service.
    ------------------------------------     --------------------------------------------------------------------
    percentile_interpolation_type            Optional str. Specifies the method of interpolation to be used when 
                                             the specified percentile value lies between two input cell values.

                                             - AUTO_DETECT - If the input value raster has integer pixel type, the NEAREST method is used. If the input value raster has floating point pixel type, then the LINEAR method is used. This is the default.
                                             - NEAREST - Nearest value to the desired percentile.
                                             - LINEAR - Weighted average of two surrounding values from the desired percentile.
    ------------------------------------     --------------------------------------------------------------------
    circular_calculation                     Optional bool. Denotes whether the statistics calculations will be arithmetic or circular.

                                             - False - Calculates arithmetic statistics. This is the default.
                                             - True - Calculates circular statistics that are appropriate for cyclic quantities, such as compass direction in degrees, daytimes, and fractional parts of real numbers.

                                             Parameter available in ArcGIS Image Server 11 and higher.
    ------------------------------------     --------------------------------------------------------------------
    circular_wrap_value                      Optional float. The possible highest value (upper bound) in the cyclic data. 
                                             It is a positive number, and the default is 360. This value also represents the same quantity 
                                             as the possible lowest value (lower bound).
                                             This parameter is honored only if the circular_calculation parameter is set to True.

                                             | Parameter available in ArcGIS Image Server 11 and higher.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional string. Name of the output feature item or table item to be created.
                                             If not provided, a random name is generated by the method and used as 
                                             the output name. 
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` object. If not specified, the currently active connection
                                             is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and 
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

   						                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
        :class:`~arcgis.features.FeatureLayer`

    .. code-block:: python

        # Usage Example 1: Calculate the values of a raster within defined zones.

        zone_data = gis.content.search("my_zone_data")[0].layers[0]
        value_raster = gis.content.search("my_value_raster")[0].layers[0]

        zonal_stats_table = zonal_statistics_as_table(input_zone_raster_or_features=zone_data,
                                                      input_value_raster=value_raster,
                                                      zone_field="my_zone_field",
                                                      output_name="my_zonal_stats_table",
                                                      gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis

    return gis._tools.rasteranalysis.zonal_statistics_as_table(
        input_zone_raster_or_features=input_zone_raster_or_features,
        input_value_raster=input_value_raster,
        output_name=output_name,
        zone_field=zone_field,
        ignore_nodata=ignore_nodata,
        statistic_type=statistic_type,
        percentile_values=percentile_values,
        process_as_multidimensional=process_as_multidimensional,
        percentile_interpolation_type=percentile_interpolation_type,
        circular_calculation=circular_calculation,
        circular_wrap_value=circular_wrap_value,
        context=context,
        future=future,
        **kwargs,
    )


def compute_change_raster(
    input_from_raster,
    input_to_raster,
    compute_change_method: str = "DIFFERENCE",
    from_classes: Optional[Union[str, list[str]]] = None,
    to_classes: Optional[Union[str, list[str]]] = None,
    filter_method: str = "CHANGED_PIXELS_ONLY",
    transition_class_colors: str = "AVERAGE",
    output_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    from_class_name_field_name: Optional[str] = None,
    to_class_name_field_name: Optional[str] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Function calculates the absolute, relative, or categorical difference between two raster datasets.
    Function available in ArcGIS Image Server 10.9 and higher.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_from_raster                        Required :class:`~arcgis.raster.ImageryLayer` object. The initial or earlier raster to be analyzed.
    ------------------------------------     --------------------------------------------------------------------
    input_to_raster                          Required :class:`~arcgis.raster.ImageryLayer` object. The final or later raster to be analyzed.
                                             This is the raster that will be compared to the initial raster.
    ------------------------------------     --------------------------------------------------------------------
    compute_change_method                    Optional String. Specifies the type of calculation to perform between
                                             the two rasters.

                                             - DIFFERENCE - The mathematical difference, or subtraction, between the pixel values in the rasters will be calculated. This is the default.
                                             - RELATIVE_DIFFERENCE - The difference in pixel values, accounting for the quantities of the values being compared, will be calculated.
                                             - CATEGORICAL_DIFFERENCE  - The difference between two categorical or thematic rasters will be calculated in which the output contains class transitions that occurred between the two rasters.
                                             - SPECTRAL_EUCLIDEAN_DISTANCE - The Euclidean distance between two multiband rasters,
                                                                             where each pixel is treated as a vector. Larger values
                                                                             indicate more change between the images.
                                             - SPECTRAL_ANGLE_DIFFERENCE - The spectral angle between two multiband rasters, where
                                                                           each pixel is treated as a vector. Larger angles indicate
                                                                           more change between the images.
                                             - BAND_WITH_MOST_CHANGE - The band that accounts for the most change in each pixel between
                                                                       two multiband rasters.
                                             Example:

                                                "DIFFERENCE"
    ------------------------------------     --------------------------------------------------------------------
    from_classes                             Optional String or List. Class names from the input_from_raster parameter
                                             to be included in the computation. If no classes are provided, all classes
                                             will be included.

                                             This parameter is honoured when the compute_change_method parameter is
                                             set to CATEGORICAL_DIFFERENCE.

                                             Example:

                                                ["Water", "Developed"]
    ------------------------------------     --------------------------------------------------------------------
    to_classes                               Optional String or List. Class names from the input_to_raster parameter
                                             to be included in the computation. If no classes are provided, all classes
                                             will be included.

                                             This parameter is honoured when the compute_change_method parameter is
                                             set to CATEGORICAL_DIFFERENCE.

                                             Example:

                                                ["Water", "Developed"]
    ------------------------------------     --------------------------------------------------------------------
    filter_method                            Optional String. Specifies the pixels to be categorized in the output raster.

                                             This parameter is honoured when the compute_change_method parameter is
                                             set to CATEGORICAL_DIFFERENCE.

                                             - ALL - All pixels will be categorized in the output. This is the default.
                                             - CHANGED_PIXELS_ONLY - Only the pixels that changed categories will be categorized in the output. All pixels that did not change categories will be grouped in a class called Other.
                                             - UNCHANGED_PIXELS_ONLY  - Only the pixels that did not change categories will be categorized in the output. All pixels that changed categories will be groups in a class called Other.

                                             Example:

                                                "ALL"
    ------------------------------------     --------------------------------------------------------------------
    transition_class_colors                  Optional String. Specifies the pixels to be categorized in the output raster.

                                             This parameter is honoured when the compute_change_method parameter is
                                             set to CATEGORICAL_DIFFERENCE.

                                             - AVERAGE - Use an average of the colors of the from class and to class for the output classes. This is the default.
                                             - FROM_COLOR - Use colors of the from classes for the output.
                                             - TO_COLOR  - Use the colors of the to classes for the output.

                                             Example:

                                                "AVERAGE"
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}


                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                                Example:

                                                    {"outSR": {spatial reference}}


                                             - Snap Raster (snapRaster): The output raster will have its
                                               cells aligned with the specified snap raster.

                                                Example:

                                                    {'snapRaster': {'url': '<image_service_url>'}}


                                             - Cell Size (cellSize): The output raster will have the resolution
                                               specified by cell size.

                                                Example:

                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}


                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                                Example:

                                                Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                Syntax example with a specified percentage of total
                                                processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    from_class_name_field_name               Optional string. A field that stores class names in the input_from_raster.
                                             The function automatically searches for CLASSNAME field or CLASS_NAME field to use.
                                             Use this parameter if the input does not contain these standard field names
                                             Example: "CLASSES"
    ------------------------------------     --------------------------------------------------------------------
    to_class_name_field_name                 Optional string. A field that stores class names in the input_to_raster.
                                             The function automatically searches for CLASSNAME field or CLASS_NAME field to use.
                                             Use this parameter if the input does not contain these standard field names
                                             Example: "CLASSES"
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.

                                             To create Dynamic Imagery Layer as output on ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
        The output imagery layer item

    .. code-block:: python

        # Usage Example 1:

        compute_change_raster_op = compute_change_raster(input_from_raster=input_from_raster,
                                                         input_to_raster=input_to_raster,
                                                         from_classes="Class1",
                                                         to_classes="Class2",
                                                         filter_method="ALL",
                                                         gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.compute_change_raster(
        input_from_raster=input_from_raster,
        input_to_raster=input_to_raster,
        compute_change_method=compute_change_method,
        from_classes=from_classes,
        to_classes=to_classes,
        filter_method=filter_method,
        transition_class_colors=transition_class_colors,
        from_classname_field=from_class_name_field_name,
        to_classname_field=to_class_name_field_name,
        output_name=output_name,
        context=context,
        future=future,
        **kwargs,
    )


def summarize_categorical_raster(
    input_categorical_raster,
    dimension=None,
    area_of_interest=None,
    area_of_interest_id_field=None,
    output_summary_table_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Generates a table containing the pixel count for each class, in each slice of an input categorical raster.
    Function available in ArcGIS Image Server 10.9.1 and higher (not available in ArcGIS Online).

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_from_raster                        Required :class:`~arcgis.raster.ImageryLayer` object. The multidimensional, categorical raster to be summarized.
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Optional String. The name of the dimension to use for the summary. If
                                             there is more than one dimension and no value is specified, all slices
                                             will be summarized using all combinations of dimension values.
    ------------------------------------     --------------------------------------------------------------------
    area_of_interest                         Optional :class:`~arcgis.features.FeatureLayer` object. The polygon feature layer containing the area
                                             or areas of interest to use when calculating the pixel count per category.
                                             If no area of interest is specified, the entire raster will be
                                             included in the analysis.
    ------------------------------------     --------------------------------------------------------------------
    area_of_interest_id_field                Optional String/Integer. The field in the polygon feature layer that
                                             defines each area of interest.
    ------------------------------------     --------------------------------------------------------------------
    output_summary_table_name                Optional String. Name of the output feature item or table item to be created
                                             If not provided, a random name is generated by the method and used as
                                             the output name.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}


                                             - Output Spatial Reference (outSR): The output raster will be
                                               projected into the output spatial reference.

                                               Example:

                                                    {"outSR": {spatial reference}}


                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified percentage of total
                                               processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                                        | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     ====================================================================

    :return:
        Feature Layer

    .. code-block:: python

        # Usage Example 1:

        my_categorical_raster = gis.content.search("categorical_raster", item_type="Imagery Layer")[0].layers[0]
        my_aoi = gis.content.search("area_of_interest_polygon", item_type="Feature Layer")[0].layers[0]

        summarize_categorical_raster_op = summarize_categorical_raster(input_categorical_raster=my_categorical_raster,
                                                                       dimension="StdTime",
                                                                       area_of_interest=my_aoi,
                                                                       gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError(
            "summarize_categorical_raster() is not supported on ArcGIS Online"
        )

    return gis._tools.rasteranalysis.summarize_categorical_raster(
        input_categorical_raster=input_categorical_raster,
        dimension=dimension,
        area_of_interest=area_of_interest,
        area_of_interest_id_field=area_of_interest_id_field,
        output_summary_table_name=output_summary_table_name,
        context=context,
        future=future,
        **kwargs,
    )


def train_random_trees_regression_model(
    input_rasters,
    input_target_data,
    target_value_field=None,
    target_dimension_field=None,
    raster_dimension=None,
    max_number_of_trees=50,
    max_tree_depth=30,
    max_number_of_samples=100000,
    average_points_per_cell="KEEP_ALL_POINTS",
    output_scatter_plots_name=None,
    output_sample_features_name=None,
    percent_samples_for_testing=10,
    output_importance_table_name=None,
    context=None,
    output_ecd_item_name=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Models the relationship between explanatory variables (independent variables) and a target dataset (dependent variable).
    Function available in ArcGIS Image Server 10.9.1 and higher (not available in ArcGIS Online).

    ====================================     =============================================================================================================================================
    **Parameter**                             **Description**
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    input_rasters                            Required :class:`~arcgis.raster.ImageryLayer` object. The single-band, multidimensional, or
                                             multiband rasters, or mosaic datasets, containing explanatory variables.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    input_target_data                        Required :class:`~arcgis.features.FeatureLayer` or :class:`~arcgis.raster.ImageryLayer` object. The raster or feature layer
                                             containing the target variable (dependant variable) data.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    target_value_field                       Optional String. The field name of the information to model in the target
                                             feature layer or raster.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    target_dimension_field                   Optional String. A date field or numeric field in the input feature
                                             layer that defines the dimension values.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    raster_dimension                         Optional String. The dimension name of the input multidimensional raster
                                             (explanatory variables) that links to the dimension in the target data.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    max_number_of_trees                      Optional Integer. The maximum number of trees in the forest. Increasing
                                             the number of trees will lead to higher accuracy rates, although this
                                             improvement will level off. The number of trees increases the processing time linearly.
                                             The default is 50.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    max_tree_depth                           Optional Integer. The maximum depth of each tree in the forest. Depth
                                             determines the number of rules each tree can create, resulting in a decision.
                                             Trees will not grow any deeper than this setting.
                                             The default is 30.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    max_number_of_samples                    Optional Integer. The maximum number of samples that will be used for
                                             the regression analysis. A value that is less than or equal to 0 means
                                             that the system will use all the samples from the input target raster
                                             or feature layer to train the regression model.
                                             The default value is 100,000.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    average_points_per_cell                  Optional String. Specifies whether the average will be calculated when
                                             multiple training points fall into one cell. This parameter is applicable
                                             only when the input target is a feature layer.
                                             Options include:

                                             - KEEP_ALL_POINTS — All points will be used when multiple training points fall into a single cell. \
                                                                  This is the default.
                                             - AVERAGE_POINTS_PER_CELL —The average value of the training points within a cell will be calculated.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_scatter_plots_name                Optional String. The name for the output scatterplots includes scatterplots
                                             of training data, test data, and location test data.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_sample_features_name              Optional String. Name of the output feature class containing
                                             target values and predicted values for training points, test points, and location test points.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    percent_samples_for_testing              Optional Float. Defines the percentage of test points used for error checking.
                                             The tool checks for three types of errors: errors on training points,
                                             errors on test points, and errors on test location points.
                                             The default is 10.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_importance_table_name             Optional String. Name of the output feature item or table item to be created.
                                             This name is to create a table containing information describing the
                                             importance of each explanatory variable used in the model. A larger
                                             number indicates the corresponding variable is more correlated to the
                                             predicted variable and will contribute more in prediction. Values range
                                             between 0 and 1, and the sum of all the values equals 1.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Cell size (cellSize) - Set the output raster cell size, or resolution

                                             - Extent (extent): A bounding box that defines the analysis area.

                                               Example:

                                                    | {"extent": {"xmin": -122.68,
                                                    | "ymin": 45.53,
                                                    | "xmax": -122.45,
                                                    | "ymax": 45.6,
                                                    | "spatialReference": {"wkid": 4326}}}

                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                               Example:

                                                    {"parallelProcessingFactor": "2"}

                                               Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    output_ecd_item_name                     Optional String or Item. The output ECD Portal item to be created.
                                             If an Item is provided, the given item will be overwritten with the ECD instead.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    gis                                      Optional GIS. The :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     ---------------------------------------------------------------------------------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the :meth:`create_folder() <arcgis.gis.ContentManager.create_folder>` can also be passed in as input.

                                             Example:

                                                | {'username': 'user1',
                                                | 'id': '6a3b77c187514ef7873ba73338cf1af8',
                                                | 'title': 'trial'}
    ====================================     =============================================================================================================================================

    :return: Named Tuple

    .. code-block:: python

        # Usage Example 1:

        my_raster_1 = gis.content.search("raster_1", item_type="Imagery Layer")[0].layers[0]
        my_raster_2 = gis.content.search("raster_2", item_type="Imagery Layer")[0].layers[0]
        input_rasters = [my_raster_1, my_raster_2]

        input_target_data = gis.content.search("my_target_data")[0].layers[0]

        train_random_trees_regression_model_op = train_random_trees_regression_model(input_rasters=input_rasters,
                                                                                     input_target_data=input_target_data,
                                                                                     gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError(
            "train_random_trees_regression_model() is not supported on ArcGIS Online"
        )

    return gis._tools.rasteranalysis.train_random_trees_regression_model(
        input_rasters=input_rasters,
        input_target_data=input_target_data,
        target_value_field=target_value_field,
        target_dimension_field=target_dimension_field,
        raster_dimension=raster_dimension,
        max_number_of_trees=max_number_of_trees,
        max_tree_depth=max_tree_depth,
        max_number_of_samples=max_number_of_samples,
        average_points_per_cell=average_points_per_cell,
        percent_samples_for_testing=percent_samples_for_testing,
        output_scatter_plots_name=output_scatter_plots_name,
        output_sample_features_name=output_sample_features_name,
        output_importance_table_name=output_importance_table_name,
        context=context,
        future=future,
        output_ecd_item_name=output_ecd_item_name,
        **kwargs,
    )


def export_to_tile_package(
    input_data, output_name=None, *, gis=None, future=False, **kwargs
):
    """

    Exports a Tiled ImageryLayer portal item to tile package.

    .. note::
        Currently supported only on ArcGIS online.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    input_data          Required Tiled ImageryLayer portal :class:`~arcgis.gis.Item` to be exported as tile package.
    ---------------     --------------------------------------------------------------------
    output_name         Optional string. Name of the Tile Package to be created.
                        If not provided, a Tile Package is created by the method and used as the output.
    ---------------     --------------------------------------------------------------------
    gis                 Optional :class:`~arcgis.gis.GIS`. The GIS on which this tool runs. If not specified, the active GIS is used.
    ---------------     --------------------------------------------------------------------
    future              Optional Boolean. If True, the result will be a GPJob object and
                        results will be returned asynchronously.
    ===============     ====================================================================

    :return: The exported tile package item

    .. code-block:: python

        # Usage Example

        exported_item = export_to_tile_package(input_data = item, output_name = "exported_tile_package")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.export_to_tile_package(
        input_imagery_layer=input_data,
        output_tile_package=output_name,
        future=future,
        **kwargs,
    )


def derive_continuous_flow(
    input_surface_raster: ImageryLayer,
    input_depressions_data: Optional[FeatureLayer] = None,
    input_weight_raster: Optional[ImageryLayer] = None,
    flow_direction_type: Optional[str] = "D8",
    force_flow: Optional[str] = "NORMAL",
    output_flow_accumulation_raster_name: Optional[str] = None,
    output_flow_direction_raster_name: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Generates a raster of accumulated flow into each cell from an input surface raster with no prior sink or depression filling required.
    
    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_surface_raster                     Required :class:`~arcgis.raster.ImageryLayer` object. The input elevation surface.
    ------------------------------------     --------------------------------------------------------------------
    input_depressions_data                   Optional :class:`~arcgis.features.FeatureLayer`/ :class:`~arcgis.raster.ImageryLayer`.\
                                             A dataset that defines real depressions. The depressions can\
                                             be defined either through a raster or a feature layer.\
                                            
                                             If input is a raster, the depression cells must take a valid value, including\
                                             zero, and the areas that are not depressions must be NoData.
    ------------------------------------     --------------------------------------------------------------------
    input_weight_raster                      Optional :class:`~arcgis.raster.ImageryLayer`. A raster that defines the fraction of flow that contributes\
                                             to flow accumulation at each cell. The weight is only applied to flow accumulation.
                                             
                                             If no weight raster is specified, a default weight of 1 will be applied to each cell. 
    ------------------------------------     --------------------------------------------------------------------
    flow_direction_type                      Optional string. Specifies the flow direction type to use.
    
                                             Choice list: ['D8', 'MFD'] 
    
                                                - D8 is for the D8 flow direction type. This is the default. 
                                                - MFD is for the Multi Flow Direction type.
    ------------------------------------     --------------------------------------------------------------------
    force_flow                               Optional string. Specifies if edge cells will always flow outward or follow normal flow rules.
    
                                             Choice list: ['NORMAL', 'FORCE']. The default value is 'NORMAL'.
    ------------------------------------     --------------------------------------------------------------------
    output_flow_accumulation_raster_name     Optional string. If not provided, an Image Service is created by the method and\
                                             used as the output raster.
                                             
                                             The output raster representing flow accumulation (number of upstream cells\
                                             draining to each cell). The output raster is of floating-point type. You can pass\
                                             in an existing Image Service Item from your GIS to use that instead. 
                                            
                                             Alternatively, you can pass in the name of the output Image Service that should be\
                                             created by this method to be used as the output for the tool. A RuntimeError is\
                                             raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    output_flow_direction_raster_name        Optional string. Name of the flow_direction_raster. This parameter determines\
                                             whether flow_direction_raster should be generated or not. Set this parameter,\
                                             in order to generate the flow_direction_raster.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                              - Extent (extent): A bounding box that defines the analysis area.

                                                Example:
                                                    {"extent: {"xmin": -122.68, 
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                              - Output Spatial Reference (outSR): The output raster will be
                                                projected into the output spatial reference.

                                                Example:
                                                    {"outSR": {spatial reference}}

                                              - Snap Raster (snapRaster): The output raster will have its
                                                cells aligned with the specified snap raster.

                                                Example:
                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                              - Cell Size (cellSize): The output raster will have the resolution
                                                specified by cell size.

                                                Example:
                                                    {'cellSize': 11} or {'cellSize': {'url': <image_service_url>}}  or {'cellSize': 'MaxOfIn'}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================     

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.derive_continuous_flow(
        input_surface_raster=input_surface_raster,
        output_flow_accumulation_raster_name=output_flow_accumulation_raster_name,
        input_depressions_data=input_depressions_data,
        input_weight_raster=input_weight_raster,
        output_flow_direction_raster_name=output_flow_direction_raster_name,
        flow_direction_type=flow_direction_type,
        force_flow=force_flow,
        context=context,
        future=future,
        **kwargs,
    )


def mosaic_image(
    input_rasters: list[ImageryLayer],
    target_raster: ImageryLayer,
    mosaic_operator: str = "LAST",
    mosaic_colormap_mode: str = "FIRST",
    no_data_value: Optional[float] = None,
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
):
    """
    Merges multiple existing raster datasets into an existing raster dataset. 
    Function available in ArcGIS Image Server 10.9 and higher.
    Supported only for Cloud Raster format based Imagery Layers.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_rasters                             Required list of :class:`~arcgis.raster.ImageryLayer` objects.
                                              Single or multiple rasters which will be mosaicked to the target raster. 
    ------------------------------------     --------------------------------------------------------------------
    target_raster                            Required :class:`~arcgis.raster.ImageryLayer` object. The raster to which the input rasters will be added.
                                             This must be an existing raster dataset (cloud raster format (CRF) based image service).
    ------------------------------------     --------------------------------------------------------------------
    mosaic_operator                          Optional String. Specifies the method that will be used to mosaic overlapping areas.

                                             - FIRST - The output cell value of the overlapping areas will be the value from the first raster\
                                                       dataset mosaicked into that location.
                                             - LAST - The output cell value of the overlapping areas will be the value from the last raster dataset\
                                                      mosaicked into that location. This is the default.
                                             - BLEND  - The output cell value of the overlapping areas will be a horizontally weighted calculation of\
                                                        the values of the cells in the overlapping area.
                                             - MEAN - The output cell value of the overlapping areas will be the average value of the overlapping cells.
                                             - MINIMUM - The output cell value of the overlapping areas will be the minimum value of the overlapping cells.
                                             - MAXIMUM - The output cell value of the overlapping areas will be the maximum value of the overlapping cells.
                                             - SUM - The output cell value of the overlapping areas will be the total sum of the overlapping cells.
                                             
                                             Example:

                                                "LAST"
    ------------------------------------     --------------------------------------------------------------------
    mosaic_colormap_mode                     Optional String. Specifies the method that will be used to choose which color map from the
                                             input rasters will be applied to the mosaic output.

                                             - FIRST - The color map from the first raster dataset in the list will be applied to the\
                                                       output raster mosaic. This is the default.
                                             - LAST - The color map from the last raster dataset in the list will be applied to the output raster mosaic.
                                             - MATCH  - All the color maps will be considered when mosaicking. If all possible values are\
                                                        already used (for the bit depth), the tool will match the value with the closest available color.
                                             - REJECT  - Only the raster datasets that do not have a color map associated with them will be mosaicked.
                                             
                                             Example:

                                                "FIRST"
    ------------------------------------     --------------------------------------------------------------------
    no_data_value                            Optional Float or Integer. All the pixels with the specified value will be set to NoData
                                             in the output raster dataset (target_raster).

                                             Example:

                                                21
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                             - Resampling Method (resamplingMethod): The output raster will be
                                               resampled to method specified.
                                               The supported values are: BILINEAR, NEAREST, CUBIC.

                                                Example:

                                                   {'resamplingMethod': "NEAREST"}


                                             - Parallel Processing Factor (parallelProcessingFactor): controls
                                               Raster Processing (CPU) service instances.

                                                Example:

                                                Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                Syntax example with a specified percentage of total
                                                processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The :class:`~arcgis.gis.GIS` on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
        The output imagery layer item

    .. code-block:: python

        # Usage Example:

        mosaiced_target_op = mosaic_image(input_rasters=[raster_1, raster_2],
                                          target_raster=raster_3,
                                          mosaic_operator="FIRST",
                                          mosaic_colormap_mode="FIRST",
                                          gis=gis)
    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.mosaic_image(
        input_rasters=input_rasters,
        target_raster=target_raster,
        mosaic_operator=mosaic_operator,
        mosaic_colormap_mode=mosaic_colormap_mode,
        no_data_value=no_data_value,
        context=context,
        gis=gis,
        future=future,
    )


def multidimensional_principal_components(
    input_multidimensional_raster,
    mode="DIMENSION_REDUCTION",
    dimension=None,
    output_principal_components_name: Optional[str] = None,
    output_loadings_name: Optional[str] = None,
    output_eigen_values_table_name: Optional[str] = None,
    variable: Optional[str] = None,
    number_of_principal_components: Optional[str] = "95%",
    context: Optional[dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    Transforms multidimensional rasters into their principal components, loadings, and eigenvalues. It transforms
    the data into a reduced number of components that account for the variance of the data, so that spatial and
    temporal patterns can be readily identified.

    .. note::
           Function available in ArcGIS Image Server 11.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_multidimensional_raster            Required ImageryLayer object. The input multidimensional raster.
    ------------------------------------     --------------------------------------------------------------------
    mode                                     Required String. Specifies the method that will be used to
                                             perform principal component analysis.

                                             - DIMENSION_REDUCTION : The input time series data will be treated as a set of images. Principal components that extract prevalent patterns over time will be computed. This is the default.

                                             - SPATIAL_REDUCTION : The input time series data will be treated as a set of pixels. Principal components that extract prevalent patterns and locations over time will be computed as a set of one-dimensional arrays stored in a table.
    ------------------------------------     --------------------------------------------------------------------
    dimension                                Required String. The dimension name used to process the principal components.
    ------------------------------------     --------------------------------------------------------------------
    output_principal_components_name         | Optional String. If not provided, a service is created by the method and used as the output.

                                             | When the mode parameter is specified as DIMENSION_REDUCTION, the output will be a multiband raster with the components as bands. The first band is the first principal component with the largest eigenvalue, the second band has the principal component with the second largest eigenvalue, and so on.

                                             | When the mode parameter is specified as SPATIAL_REDUCTION, the output is a table
                                             | containing a set of time series data representing the principal components.

                                             | You can pass in an existing Item from your GIS to use that instead.
                                             | Alternatively, you can pass in the name of the output that should be created by this method to be used as the output for the tool.
                                             | A RuntimeError is raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    output_loadings_name                     | Optional String. If not provided, a service is created by the method and used as the output.

                                             | When the mode parameter is specified as DIMENSION_REDUCTION, the output will be a table containing the weights that each input raster contributed to the principal components. These weights define the correlations of the input data and the output principal components.

                                             | When the mode parameter is specified as SPATIAL_REDUCTION, the output is a raster where pixel values are the weights contributing to the principal components. Pixels with larger values are more corelated to the principal components. This output may have a larger cell size than the input raster because a random reprojection is applied to reduce the computation complexity.

                                             | You can pass in an existing Item from your GIS to use that instead.
                                             | Alternatively, you can pass in the name of the output that should be created by this method to be used as the output for the tool.
                                             | A RuntimeError is raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    output_eigen_values_table_name           Optional String. The name for the output eigen values table.

                                             You can pass in an existing Item from your GIS to use
                                             that instead.

                                             Alternatively, you can pass in the name of the output
                                             that should be created by this method to be used as the output for
                                             the tool.
                                             A RuntimeError is raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    variable                                 Optional String. The variable of the input multidimensional raster
                                             used in computation. If the input raster is multidimensional and no
                                             variable is specified, only the first variable will be analyzed, by default.
    ------------------------------------     --------------------------------------------------------------------
    number_of_principal_components           Optional String. The number of principal components to compute, usually
                                             fewer than the number of input rasters. This parameter also takes the
                                             form of percentage (%). For example, 90% means the number of components
                                             that can explain 90% of variance in the data will be computed.
                                             The default is "95%".
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter

                                             This function has the following settings:

                                              - Extent (extent): A bounding box that defines the analysis area.

                                                Example:
                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                              - Output Spatial Reference (outSR): The output raster will be
                                                projected into the output spatial reference.

                                                Example:
                                                    {"outSR": {spatial reference}}

                                              - Snap Raster (snapRaster): The output raster will have its
                                                cells aligned with the specified snap raster.

                                                Example:
                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                              - Parallel Processing Factor (parallelProcessingFactor): controls
                                                Raster Processing (CPU) service instances.

                                                Example:
                                                    Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                    Syntax example with a specified percentage of total
                                                    processing instances:

                                                    {"parallelProcessingFactor": "60%"}

                                              - Resampling Method (resamplingMethod): The output raster will be
                                                resampled to method specified.
                                                The supported values are: BILINEAR, NEAREST, CUBIC.

                                                Example:
                                                    {'resamplingMethod': "NEAREST"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The GIS on which this tool runs. If not specified, the
                                             active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ====================================     ====================================================================

    :return: Named Tuple

    .. code-block:: python

        # Usage Example

        input_mdim_raster = gis.content.search("my_multidimensional_raster", item_type="Imagery Layer")[0].layers[0]
        mdim_pc_op = multidimensional_principal_components(input_multidimensional_raster=input_mdim_raster,
                                                           mode="DIMENSION_REDUCTION",
                                                           output_principal_components_name="op_principal_components",
                                                           output_loadings_name="op_loadings")

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.multidimensional_principal_components(
        input_multidimensional_raster=input_multidimensional_raster,
        mode=mode,
        dimension=dimension,
        output_principal_components_name=output_principal_components_name,
        output_loadings_name=output_loadings_name,
        output_eigen_values_table_name=output_eigen_values_table_name,
        variable=variable,
        number_of_principal_components=number_of_principal_components,
        context=context,
        future=future,
        **kwargs,
    )


def predict_using_regression_model(
    input_rasters,
    input_regression_definition,
    output_predicted_raster_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Predicts data values using the regression model which is output from the TrainRandomTreesRegressionModel tool.
    Function available in ArcGIS Image Server 10.9.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_rasters                            Required ImageryLayer object. The single-band, multidimensional, or
                                             multiband rasters, or mosaic datasets, containing explanatory variables.
    ------------------------------------     --------------------------------------------------------------------
    input_regression_definition              Required String or ECD Item. The JSON string or ECD portal item
                                             that contains the attribute information, statistics or other information
                                             from the regression model. This JSON is the output of the Train Random
                                             Trees Regression model tool.
    ------------------------------------     --------------------------------------------------------------------
    output_predicted_raster_name             Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                                context parameter overwrites values set through arcgis.env parameter

                                                This function has the following settings:

                                                - Cell size (cellSize) - Set the output raster cell size, or resolution

                                                - Output Spatial Reference (outSR): The output raster will be
                                                  projected into the output spatial reference.

                                                Example:
                                                    {"outSR": {spatial reference}}

                                                - Snap Raster (snapRaster): The output raster will have its
                                                  cells aligned with the specified snap raster.

                                                Example:
                                                    {'snapRaster': {'url': '<image_service_url>'}}

                                                - Extent (extent): A bounding box that defines the analysis area.

                                                Example:
                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                                - Cell Alignment (cellAlignment): Adjusts the cell alignent of the output to match
                                                  that of the specified processing extent.
                                                  The supported values are: "Default", "Align with input", "Align with Processing Extent"

                                                - Parallel Processing Factor (parallelProcessingFactor): controls
                                                Raster Processing (CPU) service instances.

                                                Example:
                                                    Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                    Syntax example with a specified percentage of total
                                                    processing instances:

                                                    {"parallelProcessingFactor": "60%"}

                                                - Resampling Method (resamplingMethod): The output raster will be
                                                  resampled to method specified.
                                                  The supported values are: BILINEAR, NEAREST, CUBIC.

                                                Example:
                                                    {'resamplingMethod': "NEAREST"}
    ====================================     ====================================================================

    :return: The imagery layer item

    .. code-block:: python

        # Usage Example

        my_raster_1 = gis.content.search("raster_1", item_type="Imagery Layer")[0].layers[0]
        my_raster_2 = gis.content.search("raster_2", item_type="Imagery Layer")[0].layers[0]
        input_rasters = [my_raster_1, my_raster_2]

        # Use an ECD portal item
        input_regression_definition = gis.content.search("regression_definiton_ecd")[0]

        predict_using_regression_model_op = predict_using_regression_model(input_rasters=input_rasters,
                                                                           input_regression_definition=input_regression_definition,
                                                                           gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.predict_using_regression_model(
        input_rasters=input_rasters,
        input_regression_definition=input_regression_definition,
        output_predicted_raster_name=output_predicted_raster_name,
        context=context,
        future=future,
        **kwargs,
    )
