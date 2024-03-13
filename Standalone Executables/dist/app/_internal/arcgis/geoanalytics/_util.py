import os
import json
import string
import random

import arcgis
from arcgis.gis import Layer, Item
from arcgis.features import FeatureCollection
from arcgis.geoprocessing._job import GPJob


def _prevent_bds_item(item: Item) -> object:
    """checks if the input is a valid input for the GeoAnalytics Tool"""
    if isinstance(item, Item):
        raise ValueError(f"The {item.title} is an Item. Please pass the layer instead.")
    return item


def _id_generator(size: int = 6, chars: str = None, prefix: str = None) -> str:
    """generates a random id of a given length"""
    if chars is None:
        chars = string.ascii_uppercase + string.digits

    if prefix:
        return str(prefix) + "".join(random.choice(chars) for _ in range(size))
    return "".join(random.choice(chars) for _ in range(size))


def _feature_input(self, input_layer):
    input_layer_url = ""
    if isinstance(input_layer, arcgis.gis.Item):
        if input_layer.type.lower() == "feature service":
            input_param = {"url": input_layer.layers[0].url}
        elif input_layer.type.lower() == "feature collection":
            fcdict = input_layer.get_data()
            fc = FeatureCollection(fcdict["layers"][0])
            input_param = fc.layer
        else:
            raise TypeError("item type must be feature service or feature collection")

    elif isinstance(input_layer, arcgis.features.FeatureLayerCollection):
        input_layer_url = input_layer.layers[0].url
        input_param = {"url": input_layer_url}

    elif isinstance(input_layer, FeatureCollection):
        input_param = input_layer.properties

    elif isinstance(input_layer, Layer):
        input_layer_url = input_layer.url
        input_param = {"url": input_layer_url}

    elif isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        input_layer_url = input_layer
        input_param = {"url": input_layer_url}

    else:
        raise Exception(
            "Invalid format of input layer. The following formats "
            "are supported: URL string, feature service item, feat"
            "ure service instance or dictionary."
        )

    return input_param


def _set_context(params):
    out_sr = arcgis.env.out_spatial_reference
    process_sr = arcgis.env.process_spatial_reference
    out_extent = arcgis.env.analysis_extent
    output_datastore = arcgis.env.output_datastore
    default_aggregation_styles = arcgis.env.default_aggregation_styles

    context = {}
    set_context = False

    if default_aggregation_styles is not None and isinstance(
        default_aggregation_styles, bool
    ):
        context["defaultAggregationStyles"] = default_aggregation_styles
        set_context = True
    if out_sr is not None:
        context["outSR"] = {"wkid": int(out_sr)}
        set_context = True
    if out_extent is not None:
        context["extent"] = out_extent
        set_context = True
    if process_sr is not None:
        if isinstance(process_sr, str) and str(process_sr).isdigit() == False:
            context["processSR"] = {"wkt": process_sr}
            set_context = True
        else:
            context["processSR"] = {"wkid": int(process_sr)}
            set_context = True
    if output_datastore is not None:
        context["dataStore"] = output_datastore
        set_context = True

    if set_context:
        params["context"] = json.dumps(context)


def _create_output_service(
    gis,
    output_name,
    output_service_name="Analysis feature service",
    task="GeoAnalytics",
    output_datastore=None,
):
    ok = gis.content.is_service_name_available(output_name, "Feature Service")
    if not ok:
        raise RuntimeError(
            "A feature service by this name already exists: " + output_name
        )
    if output_datastore is None:
        if arcgis.env.output_datastore is not None:
            output_datastore = arcgis.env.output_datastore
        else:
            if gis.properties.isPortal:
                output_datastore = "spatiotemporal"
            else:
                output_datastore = "relational"
    if str(output_datastore).lower().find("/bigdatafileshares/") > -1:
        return None
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
        "options": {"dataSourceType": output_datastore},
    }

    output_service = gis.content.create_service(
        output_name, create_params=createParameters, service_type="featureService"
    )
    description = "Feature service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": output_service_name,
    }
    output_service.update(item_properties)
    return output_service


class GAJob(object):
    """
    Represents a Single GeoAnalytics Job.  The `GAJob` class allows for the asynchronous operation
    of any geoprocessing task.  To request a GAJob task, the code must be called with `future=True`
    or else the operation will occur synchronously.  This class is not intended for users to call
    directly.


    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    gpjob             Required GPJob. The geoprocessing job.
    ----------------  ---------------------------------------------------------------
    return_service    Optional Item. The service to return to the user.
    ----------------  ---------------------------------------------------------------
    add_messages      Optional Boolean. At v1.8.2 a user can request the processing information to be appended to the item.
    ================  ===============================================================

    """

    _gpjob = None
    _return_service = None
    _add_messages = None

    # ----------------------------------------------------------------------
    def __init__(self, gpjob, return_service=None, add_messages=False):
        """
        initializer
        """
        assert isinstance(gpjob, GPJob)
        self._gpjob = gpjob
        self._return_service = return_service
        self._add_messages = add_messages

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<%s GA Job: %s>" % (self.task, self._gpjob._jobid)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s GA Job: %s>" % (self.task, self._gpjob._jobid)

    # ----------------------------------------------------------------------
    @property
    def task(self):
        """
        Returns the task name.

        :return: string
        """
        return self._gpjob.task

    # ----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        return self._gpjob.ellapse_time

    # ----------------------------------------------------------------------
    @property
    def messages(self):
        """
        Returns the GP messages

        :return: List
        """
        return self._gpjob.messages

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        Returns the GP status

        :return: String
        """
        return self._gpjob.status

    # ----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :return: Boolean
        """
        cancel = self._gpjob.cancel()
        if self._return_service:
            self._return_service.delete()
        return cancel

    # ----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :return: Boolean
        """
        return self._gpjob.cancelled()

    # ----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :return: Boolean
        """
        return self._gpjob.running()

    # ----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :return: Boolean
        """
        return self._gpjob.done()

    # ----------------------------------------------------------------------
    def process_info(self):
        """
        Returns the Processing Information for a GeoAnalytics job.

        :return: List or None if process_info does not exist.

        """
        processing_info = None
        if self.result():
            res = self.result()
            if hasattr(res, "_asdict") and "process_info" in res._asdict().keys():
                return getattr(res, "process_info")
            else:
                url = f"{self._gpjob._url}/jobs/{self._gpjob._jobid}"
                params = {"f": "json"}
                res = self._gpjob._gis._con.get(url, params)
                if "results" in res and "processInfo" in res["results"]:
                    url = f"{self._gpjob._url}/jobs/{self._gpjob._jobid}/{res['results']['processInfo']['paramUrl']}"
                    return self._gpjob._gis._con.get(url, params)["value"]
        return None

    # ----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :return: object
        """
        try:
            res = self._gpjob.result()
            if self._return_service:
                return self._return_service
            else:
                return res
        except Exception as e:
            if self._return_service:
                self._return_service.delete()
            raise e
