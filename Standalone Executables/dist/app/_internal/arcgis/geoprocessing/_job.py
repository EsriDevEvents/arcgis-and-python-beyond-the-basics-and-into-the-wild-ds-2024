import os
import datetime
from concurrent.futures import Future
import logging
import json

_log = logging.getLogger(__name__)
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData


class GPJob(object):
    """
    Represents a Single Geoprocessing Job.  The `GPJob` class allows for the asynchronous operation
    of any geoprocessing task.  To request a GPJob task, the code must be called with `future=True`
    or else the operation will occur synchronously.  This class is not intended for users to call
    directly.


    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    future            Required `Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object.  The async object created by
                      the geoprocessing (GP) task.
    ----------------  ---------------------------------------------------------------
    gptool            Required Layer. The Geoprocessing Service
    ----------------  ---------------------------------------------------------------
    jobid             Required String. The unique ID of the GP Job.
    ----------------  ---------------------------------------------------------------
    task_url          Required String. The URL to the :class:`~arcgis.geoprocessing.GPTask`.
    ----------------  ---------------------------------------------------------------
    gis               Required :class:`~arcgis.gis.GIS` . The GIS connection object
    ----------------  ---------------------------------------------------------------
    notify            Optional Boolean.  When set to True, a message will inform the
                      user that the geoprocessing task has completed. The default is
                      False.
    ================  ===============================================================

    """

    _cancelled = None
    _future = None
    _jobid = None
    _url = None
    _gis = None
    _task_name = None
    _is_fa = False
    _is_ra = False
    _is_ortho = False
    _start_time = None
    _end_time = None
    _item_properties = None
    _return_item = None

    # ----------------------------------------------------------------------
    def __init__(self, future, gptool, jobid, task_url, gis, notify=False):
        """
        initializer
        """
        assert isinstance(future, Future)
        self._future = future
        self._start_time = datetime.datetime.now()
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)
        self._gptool = gptool
        self._jobid = jobid
        self._url = task_url
        self._gis = gis
        self._cancelled = False

    # ----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Get the Ellapse Time for the Job
        """
        if self._end_time:
            return self._end_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time

    # ----------------------------------------------------------------------
    def _set_end_time(self, future):
        """sets the finish time"""
        self._end_time = datetime.datetime.now()

    # ----------------------------------------------------------------------
    def _notify(self, future):
        """prints finished method"""
        jobid = str(self).replace("<", "").replace(">", "")
        try:
            res = future.result()
            infomsg = "{jobid} finished successfully.".format(jobid=jobid)
            _log.info(infomsg)
            print(infomsg)
        except Exception as e:
            msg = str(e)
            msg = "{jobid} failed: {msg}".format(jobid=jobid, msg=msg)
            _log.info(msg)
            print(msg)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<%s GP Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s GP Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    @property
    def task(self):
        """Get the task name.

        :return: String

        """
        if self._task_name is None:
            self._task_name = os.path.basename(self._url)
        return self._task_name

    # ----------------------------------------------------------------------
    @property
    def messages(self):
        """
        Get the service's messages

        :return: List
        """
        url = self._url + "/jobs/%s" % self._jobid
        params = {"f": "json", "returnMessages": True}
        if hasattr(self._gis, "_con"):
            res = self._gis._con.post(url, params)
        else:
            res = self._gis.post(url, params)
        if "messages" in res:
            return res["messages"]
        return []

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        Get the GP status

        :return: String
        """
        url = self._url + "/jobs/%s" % self._jobid
        params = {"f": "json", "returnMessages": True}

        if hasattr(self._gis, "_con"):
            res = self._gis._con.post(url, params)
        else:
            res = self._gis.post(url, params)
        if "jobStatus" in res:
            return res["jobStatus"]
        return res

    # ----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :return: Boolean
        """
        if self.done():
            return False
        if self.cancelled():
            return False
        try:
            url = self._url + "/jobs/%s/cancel" % self._jobid
            params = {"f": "json"}
            if hasattr(self._gis, "_con"):
                res = self._gis._con.post(url, params)
            else:
                res = self._gis.post(url, params)
            if "jobStatus" in res:
                self._future.cancel()
                self._future.set_result({"jobStatus": "esriJobCancelled"})
                self._cancelled = True
                return self._cancelled
            self._future.set_result({"jobStatus": "esriJobCancelled"})
            self._future.cancel()

            return res
        except:
            self._future.cancel()
        return True

    # ----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :return: Boolean
        """

        return self._cancelled

    # ----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :return: Boolean
        """
        return self._future.running()

    # ----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :return: Boolean
        """
        return self._future.done()

    # ----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :return: object
        """
        if self.cancelled():
            return None
        if self._is_fa:
            return self._process_fa(self._future.result())
        elif self._is_ra:
            return self._process_ra(self._future.result())
        elif self._is_ortho:
            return self._process_ortho(self._future.result())
        return self._future.result()

    def _process_ortho(self, result):
        """handles the ortho imagery response"""
        import arcgis

        if hasattr(result, "_fields"):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if isinstance(value, dict) and "featureSet" in value:
                    r[key] = arcgis.features.FeatureCollection(value)
                elif (
                    isinstance(value, dict)
                    and "url" in value
                    and value["url"].lower().find("imageserver")
                ):
                    return value["url"]
                elif (
                    isinstance(value, dict)
                    and "url" in value
                    and value["url"].lower().find("featureserver")
                ):
                    return arcgis.features.FeatureLayerCollection(
                        url=value["url"], gis=self._gis
                    )
                elif (
                    isinstance(value, dict)
                    and "itemId" in value
                    and len(value["itemId"]) > 0
                ):
                    if not value["itemId"] in iids:
                        r[key] = arcgis.gis.Item(self._gis, value["itemId"])
                        if self._item_properties:
                            _item_properties = {
                                "properties": {
                                    "jobUrl": self._url + "/jobs/" + self._jobid,
                                    "jobType": "GPServer",
                                    "jobId": self._jobid,
                                    "jobStatus": "completed",
                                }
                            }
                            r[key].update(item_properties=_item_properties)
                        iids.append(value["itemId"])
                elif len(str(value)) > 0 and value:
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]

            return r
        else:
            value = result
            if self.task == "AlterProcessingStates":
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    processing_states = value.replace("'", '"')
                    processing_states = json.loads(processing_states.replace('u"', '"'))
                    return processing_states

            if isinstance(value, DataFile) and self.task != "GenerateReport":
                return self._gis._con.post(value.to_dict()["url"], {})
            if isinstance(value, (RasterData, LinearUnit)):
                return value
            elif isinstance(value, str) and value.lower().find("imageserver") > -1:
                return value
            elif isinstance(value, (dict, tuple, list)) == False:
                return value
            elif "itemId" in value and len(value["itemId"]) > 0:
                itemid = value["itemId"]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + "/jobs/" + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed",
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif isinstance(value, dict) and "items" in value:
                itemid = list(value["items"].keys())[0]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + "/jobs/" + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed",
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif self.task == "QueryCameraInfo":
                import pandas as pd

                columns = value["schema"]
                data = value["content"]
                return pd.DataFrame(data, columns=columns)
            elif (
                isinstance(value, dict)
                and "url" in value
                and value["url"].lower().find("imageserver")
            ):
                return value["url"]
            elif (
                isinstance(value, dict)
                and "url" in value
                and value["url"].lower().find("featureserver")
            ):
                return arcgis.features.FeatureLayerCollection(
                    url=value["url"], gis=self._gis
                )
            elif isinstance(value, dict) and "featureSet" in value:
                return arcgis.features.FeatureCollection(value)
            return value

    def _process_ra(self, result):
        import arcgis

        if isinstance(result, arcgis.features.FeatureLayer):
            if self._item_properties:
                _item_properties = {
                    "properties": {
                        "jobUrl": self._url + "/jobs/" + self._jobid,
                        "jobType": "GPServer",
                        "jobId": self._jobid,
                        "jobStatus": "completed",
                    }
                }
                self._return_item.update(item_properties=_item_properties)
            return self._return_item
        if hasattr(result, "_fields"):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if isinstance(value, dict) and "featureSet" in value:
                    r[key] = arcgis.features.FeatureCollection(value)
                elif (
                    isinstance(value, dict)
                    and "itemId" in value
                    and len(value["itemId"]) > 0
                ):
                    if not value["itemId"] in iids:
                        r[key] = arcgis.gis.Item(self._gis, value["itemId"])
                        if self._item_properties:
                            _item_properties = {
                                "properties": {
                                    "jobUrl": self._url + "/jobs/" + self._jobid,
                                    "jobType": "GPServer",
                                    "jobId": self._jobid,
                                    "jobStatus": "completed",
                                }
                            }
                            r[key].update(item_properties=_item_properties)
                        iids.append(value["itemId"])
                elif len(str(value)) > 0 and value:
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]
            if (
                self.task == "CalculateDistance"
                or self.task == "DetermineOptimumTravelCostNetwork"
                or self.task == "FlowDirection"
                or self.task == "CalculateTravelCost"
            ):
                m = {}
                if isinstance(r, dict):
                    for key, value in r.items():
                        m[key[0 : key.rindex("_") + 1] + "service"] = r[key]
                    r = m

            if self.task == "InterpolatePoints":
                if "process_info" in r.keys():
                    process_info = r["process_info"]
                    html_final = "<b>The following table contains cross validation statistics:</b><br></br><table style='width: 250px;margin-left: 2.5em;'><tbody>"
                    for row in process_info:
                        temp_dict = json.loads(row)
                        if isinstance(temp_dict["message"], list):
                            html_final += (
                                "<tr><td>"
                                + temp_dict["message"][0]
                                + "</td><td style='float:right'>"
                                + temp_dict["params"][
                                    temp_dict["message"][1].split("${")[1].split("}")[0]
                                ]
                                + "</td></tr>"
                            )

                    html_final += "</tbody></table><br></br>"
                    from IPython.display import HTML

                    process_info_html = HTML(html_final)
                    r["process_info"] = process_info_html
                    r["output_raster"].update(
                        item_properties={"description": html_final}
                    )

            return_value_names = []
            for key, value in r.items():
                return_value_names.append(key)
            num_returns = len(r)
            if num_returns == 1:
                return r[return_value_names[0]]

            else:
                ret_names = []
                for return_value in return_value_names:
                    ret_names.append(return_value)
                import collections

                NamedTuple = collections.namedtuple("FunctionOutput", ret_names)
                function_output = NamedTuple(**r)
                return function_output

        elif isinstance(result, arcgis.raster.ImageryLayer):
            return result
        else:
            value = result
            if (
                isinstance(value, dict)
                and "itemId" in value
                and len(value["itemId"]) > 0
            ):
                itemid = value["itemId"]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + "/jobs/" + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed",
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif isinstance(value, dict) and "url" in value:
                return value["url"]
            elif isinstance(value, dict) and "contentList" in value:
                if value == "":
                    return None
                elif isinstance(value["contentList"], str):
                    return json.loads(value["contentList"])
                return value["contentList"]
            elif isinstance(value, dict) and "modelInfo" in value:
                try:
                    dict_output = json.loads(value["modelInfo"])
                    return dict_output
                except:
                    return value
            elif isinstance(value, dict) and "result" in value:
                return value["result"]
            elif isinstance(value, dict) and "items" in value:
                itemid = list(value["items"].keys())[0]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + "/jobs/" + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed",
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
            elif isinstance(value, dict) and "featureSet" in value:
                return arcgis.features.FeatureCollection(value)
            elif isinstance(value, list) and value is not None:
                output_model_list = []
                from arcgis.learn import Model

                for element in value:
                    if isinstance(element, dict):
                        if "id" in element.keys():
                            item = arcgis.gis.Item(gis=self._gis, itemid=element["id"])
                            output_model_list.append(Model(item))
                return output_model_list
            elif isinstance(value, dict) and "id" in value:
                itemid = value["id"]
                item = arcgis.gis.Item(gis=self._gis, itemid=itemid)
                if self._item_properties:
                    _item_properties = {
                        "properties": {
                            "jobUrl": self._url + "/jobs/" + self._jobid,
                            "jobType": "GPServer",
                            "jobId": self._jobid,
                            "jobStatus": "completed",
                        }
                    }
                    item.update(item_properties=_item_properties)
                return item
        return result

    def _process_fa(self, result):
        import arcgis

        HAS_ITEM = False
        if hasattr(result, "_fields"):
            r = {}
            iids = []
            for key in result._fields:
                value = getattr(result, key)
                if (
                    self.task
                    in [
                        "AggregatePoints",
                        "ConnectOriginsToDestinations",
                        "SummarizeNearby",
                        "InterpolatePoints",
                    ]
                    and isinstance(value, dict)
                    and "featureSet" in value
                ):
                    r[key] = arcgis.features.FeatureCollection(value)
                elif isinstance(value, dict) and "featureSet" in value:
                    HAS_ITEM = True
                    r[key] = arcgis.features.FeatureCollection(value)
                elif (
                    isinstance(value, dict)
                    and "itemId" in value
                    and len(value["itemId"]) > 0
                ):
                    if not value["itemId"] in iids:
                        HAS_ITEM = True
                        r[key] = arcgis.gis.Item(self._gis, value["itemId"])
                        iids.append(value["itemId"])
                elif len(str(value)) > 0 and value:
                    r[key] = value
                elif HAS_ITEM == False and (
                    self.task
                    in [
                        "AggregatePoints",
                        "CreateWatersheds",
                        "PlanRoutes",
                        "ConnectOriginsToDestinations",
                        "SummarizeNearby",
                        "InterpolatePoints",
                    ]
                    or self.task == "ConnectOriginsToDestinations"
                ):
                    r[key] = value
            if len(r) == 1:
                return r[list(r.keys())[0]]
            return r
        else:
            value = result
            if "itemId" in value and len(value["itemId"]) > 0:
                itemid = value["itemId"]
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif self.task.lower() == "createroutelayers":
                return [
                    arcgis.gis.Item(gis=self._gis, itemid=itemid)
                    for itemid in result["items"]
                ]
            elif (
                isinstance(value, dict)
                and "items" in value
                and len(set(value["items"].keys())) == 1
            ):
                itemid = list(value["items"].keys())[0]
                return arcgis.gis.Item(gis=self._gis, itemid=itemid)
            elif (
                isinstance(value, dict)
                and "items" in value
                and len(set(value["items"].keys())) > 1
            ):
                return [
                    arcgis.gis.Item(gis=self._gis, itemid=itemid)
                    for itemid in result["items"]
                ]
            elif isinstance(value, dict) and "featureSet" in value:
                return arcgis.features.FeatureCollection(value)
            return value
        return result


class RAJob(GPJob):
    """
    Represents a Single Raster Geoprocessing Job.  The `RAJob` class allows for the asynchronous operation
    of any geoprocessing task.  To request a GPJob task, the code must be called with `future=True`
    or else the operation will occur synchronously.  This class is not intended for users to call
    directly.


    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    gpjob
    ----------------  ---------------------------------------------------------------
    item
    ================  ===============================================================

    """

    _item = None
    _gpjob = None

    # ----------------------------------------------------------------------
    def __init__(self, gpjob: GPJob, item: "Item" = None):
        """
        initializer
        """
        self._gpjob = gpjob
        self._item = item

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<%s Raster Analysis Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s Raster Analysis Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    @property
    def task(self):
        """Returns the task name.
        :return: string
        """
        return self._gpjob.task

    # ----------------------------------------------------------------------
    @property
    def messages(self):
        """
        Returns the service's messages

        :return: List
        """
        return self._gpjob.messages

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the GP status

        :return: String
        """
        return self._gpjob.status

    # ----------------------------------------------------------------------
    @property
    def elapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        return self._gpjob.ellapse_time

    # ----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :return: object
        """
        try:
            return self._gpjob.result()
        except Exception as e:
            from arcgis.gis import Item

            if isinstance(self._item, Item):
                self._item.delete()
            elif isinstance(self._item, (tuple, list)):
                [i.delete() for i in self._item if isinstance(i, Item)]
            raise e

    # ----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :return: boolean
        """
        res = self._gpjob.cancel()
        if self.cancelled():
            from arcgis.gis import Item

            if isinstance(self._item, Item):
                self._item.delete()
            elif isinstance(self._item, (tuple, list)):
                [i.delete() for i in self._item if isinstance(i, Item)]
        return res

    # ----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :return: boolean
        """
        return self._gpjob.cancelled()

    # ----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :return: boolean
        """
        return self._gpjob.running()

    # ----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :return: boolean
        """
        return self._gpjob.done()


class OMJob(GPJob):
    """
    Represents a Single Raster orthomapping Job.  The `OMJob` class allows for the asynchronous operation
    of any geoprocessing task.  To request a GPJob task, the code must be called with `future=True`
    or else the operation will occur synchronously.  This class is not intended for users to call
    directly.


    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    gpjob             Represents the GP Job Object
    ----------------  ---------------------------------------------------------------
    item              Item object that needs to be updated by the OMJob
    ================  ===============================================================

    """

    _item = None
    _gpjob = None
    _flight_details = None

    # ----------------------------------------------------------------------
    def __init__(self, gpjob: GPJob, item: "Item" = None):
        """
        initializer
        """
        self._gpjob = gpjob
        self._item = item

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<%s Orthomapping Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s Orthomapping Job: %s>" % (self.task, self._jobid)

    # ----------------------------------------------------------------------
    @property
    def task(self):
        """Returns the task name.
        :return: string
        """
        return self._gpjob.task

    # ----------------------------------------------------------------------
    @property
    def messages(self):
        """
        Returns the service's messages

        :return: List
        """
        return self._gpjob.messages

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the GP status

        :return: String
        """
        return self._gpjob.status

    # ----------------------------------------------------------------------
    @property
    def elapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        return self._gpjob.ellapse_time

    # ----------------------------------------------------------------------
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :return: object
        """
        try:
            op = self._gpjob.result()
            self._update_flight_info()
            return op
        except Exception as e:
            from arcgis.gis import Item

            if isinstance(self._item, Item):
                self._item.delete()
            elif isinstance(self._item, (tuple, list)):
                [i.delete() for i in self._item if isinstance(i, Item)]
            raise e

    # ----------------------------------------------------------------------
    def cancel(self):
        """
        Attempt to cancel the call. If the call is currently being executed
        or finished running and cannot be cancelled then the method will
        return False, otherwise the call will be cancelled and the method
        will return True.

        :return: boolean
        """
        res = self._gpjob.cancel()
        if self.cancelled():
            from arcgis.gis import Item

            if isinstance(self._item, Item):
                self._item.delete()
            elif isinstance(self._item, (tuple, list)):
                [i.delete() for i in self._item if isinstance(i, Item)]
        return res

    # ----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :return: boolean
        """
        return self._gpjob.cancelled()

    # ----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :return: boolean
        """
        return self._gpjob.running()

    # ----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :return: boolean
        """
        return self._gpjob.done()

    def _update_flight_info(self):
        flight_json_details = self._flight_details
        update_flight_json = False
        if isinstance(flight_json_details, dict):
            # project_item = flight_json_details.get("project_item", None)
            item_name = flight_json_details.get("item_name", None)
            mission = flight_json_details.get("mission", None)
            update_flight_json = flight_json_details.get("update_flight_json", None)
            processing_states = flight_json_details.get("processing_states", None)
            adjust_settings = flight_json_details.get("adjust_settings", None)

        if update_flight_json:
            import json

            job_messages = self.messages
            rm = mission._project_item.resources
            mission_json = mission._mission_json
            resource = mission._resource_info
            resource_name = resource["resource"]

            start_time = (
                self._gpjob._start_time.isoformat(timespec="milliseconds") + "Z"
            )
            end_time = self._gpjob._end_time.isoformat(timespec="milliseconds") + "Z"

            mission_json["jobs"].update(
                {
                    item_name: {
                        "messages": job_messages,
                        "checked": True,
                        "progress": 100,
                        "success": True,
                        "startTime": start_time,
                        "completionTime": end_time,
                    }
                }
            )
            if item_name == "reset":
                keys = [
                    "adjustment",
                    "matchControlPoint",
                    "colorCorrection",
                    "computeControlPoints",
                    "seamline",
                    "appendControlPoints",
                    "report",
                    "queryControlPoints",
                    "ortho",
                    "dsm",
                    "dtm",
                ]
                for key in keys:
                    if key in mission_json["jobs"].keys():
                        if key != "adjustment":
                            mission_json["jobs"].update({key: {"checked": False}})
                        else:
                            mission_json["jobs"].update(
                                {key: {"checked": False, "mode": "Quick"}}
                            )

                item_keys = ["ortho", "dsm", "dtm"]
                for key in item_keys:
                    if key in mission_json["items"].keys():
                        mission_json["items"].update({key: {}})

            if processing_states is not None:
                mission_json["processingSettings"].update(
                    {item_name: processing_states}
                )
            if adjust_settings is not None:
                mode = adjust_settings.pop("mode", None)
                mission_json["jobs"][item_name].update({"mode": mode})
                mission_json["adjustSettings"].update(adjust_settings)

            properties = json.loads(resource["properties"])

            if self._item:
                item = ""
                url = ""
                item_props = json.loads(self._item)
                if "serviceProperties" in item_props.keys():
                    if "serviceUrl" in item_props["serviceProperties"].keys():
                        url = item_props["serviceProperties"]["serviceUrl"]
                    if "itemProperties" in item_props.keys():
                        if "itemId" in item_props["itemProperties"].keys():
                            itemid = item_props["itemProperties"]["itemId"]
                elif "itemId" in item_props.keys():
                    itemid = item_props["itemId"]
                    portal_item = mission._gis.content.get(itemid)
                    url = portal_item.url
                elif "url" in item_props.keys():
                    url = item_props["url"]

                mission_json["items"].update(
                    {item_name: {"itemId": itemid, "url": url}}
                )

                properties = json.loads(resource["properties"])
                properties_items = properties["items"]
                products = []
                for dict_item in properties_items:
                    products.append(dict_item["product"])
                if item_name not in products:
                    properties_items.append(
                        {"product": item_name, "id": itemid, "created": True}
                    )
                else:
                    index = products.index(item_name)
                    properties_items[index] = {
                        "product": item_name,
                        "id": itemid,
                        "created": True,
                    }
                properties.update({"items": properties_items})

            import tempfile, uuid, os

            fname = resource_name.split("/")[1]
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, fname)
            with open(temp_file, "w") as writer:
                json.dump(mission_json, writer)
            del writer

            try:
                rm.update(
                    file=temp_file,
                    text=mission_json,
                    folder_name="flights",
                    file_name=fname,
                    properties=properties,
                )
            except:
                raise RuntimeError("Error updating the mission resource")
