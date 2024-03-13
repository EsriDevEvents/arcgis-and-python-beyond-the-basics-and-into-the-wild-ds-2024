from __future__ import print_function
import datetime

import inspect
import logging
import sys
import json
import time
import datetime
import collections
import concurrent.futures

import arcgis
from arcgis.gis import GIS
from arcgis.gis._impl._con import Connection
from arcgis.features import FeatureSet, FeatureCollection, Table
from arcgis.auth.tools import LazyLoader

mapping = LazyLoader("arcgis.mapping")
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing._tool import _camelCase_to_underscore
from arcgis._impl.common._utils import _date_handler
from arcgis.geoprocessing._job import GPJob

_log = logging.getLogger(__name__)


def _layer_input(input_layer):
    # Will be used exclusively by RA tools
    input_param = input_layer

    url = ""
    if isinstance(input_layer, arcgis.gis.Item):
        if input_layer.type == "Image Collection":
            input_param = {"itemId": input_layer.itemid}
        else:
            if "layers" in input_layer:
                input_param = input_layer.layers[0]._lyr_dict
            else:
                raise TypeError("No layers in input layer Item")

    elif isinstance(input_layer, arcgis.features.FeatureLayerCollection):
        input_param = input_layer.layers[0]._lyr_dict

    elif isinstance(input_layer, arcgis.features.FeatureCollection):
        input_param = input_layer.properties

    elif isinstance(input_layer, arcgis.gis.Layer):
        input_param = input_layer._lyr_dict
        from arcgis.raster import ImageryLayer
        import json

        if isinstance(input_layer, ImageryLayer):
            if "options" in input_layer._lyr_json:
                if isinstance(
                    input_layer._lyr_json["options"], str
                ):  # sometimes the rendering info is a string
                    # load json
                    layer_options = json.loads(input_layer._lyr_json["options"])
                else:
                    layer_options = input_layer._lyr_json["options"]

                if "imageServiceParameters" in layer_options:
                    # get renderingRule and mosaicRule
                    input_param.update(layer_options["imageServiceParameters"])

    elif isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        if "http:" in input_layer or "https:" in input_layer:
            input_param = {"url": input_layer}
        else:
            input_param = {"uri": input_layer}

    else:
        raise Exception(
            "Invalid format of input layer. url string, layer Item, layer instance or dict supported"
        )

    if "url" in input_param:
        url = input_param["url"]
        if "/RasterRendering/" in url:
            url = input_layer._uri
            input_param = {"uri": url}
            return input_param
    if "ImageServer" in url or "MapServer" in url:
        if "serviceToken" in input_param:
            url = url + "?token=" + input_param["serviceToken"]
            input_param.update({"url": url})

    return input_param


def _layer_input_gp(input_layer):
    input_param = input_layer

    input_layer_url = ""
    if isinstance(input_layer, arcgis.gis.Item):
        if "layers" in input_layer:
            input_param = input_layer.layers[0]._lyr_dict
        else:
            raise TypeError("No layers in input layer Item")

    elif isinstance(input_layer, arcgis.features.FeatureLayerCollection):
        input_param = input_layer.layers[0]._lyr_dict

    elif isinstance(input_layer, arcgis.features.FeatureCollection):
        input_param = input_layer.properties

    elif isinstance(input_layer, arcgis.features.FeatureSet):
        input_param = input_layer.to_dict()

    elif isinstance(input_layer, arcgis.gis.Layer):
        input_param = input_layer._lyr_dict

    elif isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        if "http:" in input_layer or "https:" in input_layer:
            input_param = {"url": input_layer}
        else:
            input_param = {"uri": input_layer}

    else:
        raise Exception(
            "Invalid format of input layer. url string, layer Item, layer instance or dict supported"
        )

    return input_param


def _feature_input(input_layer):
    input_param = input_layer

    input_layer_url = ""
    if isinstance(input_layer, arcgis.gis.Item):
        if input_layer.type.lower() == "feature service":
            input_param = input_layer.layers[0]._lyr_dict
        elif input_layer.type.lower() == "big data file share":
            input_param = input_layer.layers[0]._lyr_dict
        elif input_layer.type.lower() == "feature collection":
            fcdict = input_layer.get_data()
            fc = FeatureCollection(fcdict["layers"][0])
            input_param = fc.layer
        else:
            raise TypeError("item type must be feature service or feature collection")

    elif isinstance(input_layer, arcgis.features.FeatureLayerCollection):
        input_param = input_layer.layers[0]._lyr_dict

    elif isinstance(input_layer, arcgis.features.FeatureCollection):
        input_param = input_layer.properties

    elif isinstance(input_layer, arcgis.gis.Layer):
        input_param = input_layer._lyr_dict

    elif isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        input_param = {"url": input_layer}

    else:
        raise Exception(
            "Invalid format of input layer. url string, feature service Item, feature service instance or dict supported"
        )

    return input_param


def _analysis_job(gptool, task, params):
    """Submits an Analysis job and returns the job URL for monitoring the job
    status in addition to the json response data for the submitted job."""

    # Unpack the Analysis job parameters as a dictionary and add token and
    # formatting parameters to the dictionary. The dictionary is used in the
    # HTTP POST request. Headers are also added as a dictionary to be included
    # with the POST.
    #
    # print("Submitting analysis job...")

    task_url = "{}/{}".format(gptool.url, task)
    submit_url = "{}/submitJob".format(task_url)

    params["f"] = "json"
    try:
        resp = gptool._con.post(submit_url, params, token=gptool._token)
    except RuntimeError:
        resp = gptool._con.post(submit_url, params)
    # print(resp)
    return task_url, resp, resp["jobId"]


def _analysis_job_status(gptool, task_url, job_info):
    """Tracks the status of the submitted Analysis job."""

    if "jobId" in job_info:
        # Get the id of the Analysis job to track the status.
        #
        try:
            job_id = job_info.get("jobId")
            job_url = "{}/jobs/{}".format(task_url, job_id)
            params = {"f": "json"}
            try:
                job_response = gptool._con.get(job_url, params, token=gptool._token)
            except Exception as e:
                job_response = gptool._con.get(job_url, params)

            # Query and report the Analysis job status.
            #
            num_messages = 0
            if "jobStatus" in job_response:
                while not job_response.get("jobStatus") == "esriJobSucceeded":
                    time.sleep(1)
                    try:
                        job_response = gptool._con.get(
                            job_url, params, token=gptool._token
                        )
                    except Exception as e:
                        job_response = gptool._con.get(job_url, params)

                    # print(job_response)
                    messages = (
                        job_response["messages"] if "messages" in job_response else []
                    )
                    num = len(messages)
                    if num > num_messages:
                        for index in range(num_messages, num):
                            msg = messages[index]
                            if arcgis.env.verbose:
                                print(msg["description"])
                            if msg["type"] == "esriJobMessageTypeInformative":
                                _log.info(msg["description"])
                            elif msg["type"] == "esriJobMessageTypeWarning":
                                _log.warning(msg["description"])
                            elif msg["type"] == "esriJobMessageTypeError":
                                _log.error(msg["description"])
                                # print(msg['description'], file=sys.stderr)
                            else:
                                _log.warning(msg["description"])
                        num_messages = num

                    if job_response.get("jobStatus") == "esriJobFailed":
                        raise Exception("Job failed.")
                    elif job_response.get("jobStatus") == "esriJobCancelled":
                        raise Exception("Job cancelled.")
                    elif job_response.get("jobStatus") == "esriJobTimedOut":
                        raise Exception("Job timed out.")

                if "results" in job_response:
                    return job_response
                else:
                    retry_counter = 0
                    while retry_counter < 5:
                        time.sleep(retry_counter + 1)
                        try:
                            job_response = gptool._con.get(
                                job_url, params, token=gptool._token
                            )
                        except Exception as e:
                            job_response = gptool._con.get(job_url, params)
                        if "results" in job_response:
                            return job_response
                        retry_counter += 1
                if "results" in job_response:
                    return job_response
                else:
                    raise Exception("No job results.")

            else:
                raise Exception("No job results.")
        except KeyboardInterrupt:
            cancel_url = "%s/jobs/%s/cancel" % (task_url, job_info["jobId"])
            params = {"f": "json"}
            job_info = gptool._con.get(path=cancel_url, params=params)
            job_info = _analysis_job_status(gptool, task_url, job_info)
    else:
        raise Exception("No job url.")


def _analysis_job_results(gptool, task_url, job_info, job_id=None):
    """Use the job result json to get information about the feature service
    created from the Analysis job."""

    # Get the paramUrl to get information about the Analysis job results.
    #
    if job_id is None:
        job_id = job_info.get("jobId")

    if "results" in job_info:
        results = job_info.get("results")
        result_values = {}
        for key in list(results.keys()):
            param_value = results[key]
            if "paramUrl" in param_value:
                param_url = param_value.get("paramUrl")
                result_url = "{}/jobs/{}/{}".format(task_url, job_id, param_url)

                params = {"f": "json"}
                _set_env_params(params, {})
                try:
                    param_result = gptool._con.get(
                        result_url, params, token=gptool._token
                    )
                except:
                    param_result = gptool._con.get(result_url, params)
                if isinstance(param_result, list):
                    result_values[key] = [value.get("value") for value in param_result]
                else:
                    job_value = param_result.get("value")
                    result_values[key] = job_value
        return result_values
    else:
        raise Exception("Unable to get analysis job results.")


def _future_op(
    gptool,
    task_url,
    job_info,
    job_id,
    param_db,
    return_values,
    return_messages,
):
    job_info = _analysis_job_status(gptool, task_url, job_info)
    resp = _analysis_job_results(gptool, task_url, job_info, job_id)

    # ---------------------async-out---------------------#
    output_dict = {}
    for retParamName in resp.keys():
        output_val = resp[retParamName]
        try:
            ret_param_name, ret_val = _get_output_value(
                gptool, output_val, param_db, retParamName
            )
            output_dict[ret_param_name] = ret_val
        except KeyError:
            pass  # cannot handle unexpected output as return tuple will change

    # tools with output map service - add another output:
    # result_layer = '' #***self.properties.resultMapServerName
    if gptool.properties.resultMapServerName != "":
        job_id = job_info.get("jobId")
        result_layer_url = (
            gptool._url.replace("/GPServer", "/MapServer") + "/jobs/" + job_id
        )

        output_dict["result_layer"] = mapping.MapImageLayer(
            result_layer_url, gptool._gis
        )

    num_returns = len(resp)
    if return_messages:
        return (
            _return_output(num_returns, output_dict, return_values),
            job_info,
        )

    return _return_output(num_returns, output_dict, return_values)


def _execute_gp_tool(
    gis,
    task_name,
    params,
    param_db,
    return_values,
    use_async,
    url,
    webtool=False,
    add_token=True,
    return_messages=False,
    future=False,
):
    if gis is None:
        gis = arcgis.env.active_gis
    elif (  # Checks if the GIS is not a GIS class but has the _con property
        isinstance(gis, GIS) == False
        and hasattr(gis, "_con")
        and isinstance(gis._con, Connection)
    ):
        gis = gis._con

    if isinstance(gis, Connection):
        log = logging.getLogger()
        log.warning("Using Connection object over GIS object")
        ngis = GIS(set_active=False)
        ngis._con = gis
        gis = ngis

    gp_params = {"f": "json"}

    # ---------------------in---------------------#
    for param_name, param_value in params.items():
        # print(param_name + " = " + str(param_value))
        if param_name in param_db:
            py_type, gp_param_name = param_db[param_name]
            if param_value is None:
                param_value = ""
            gp_params[gp_param_name] = param_value
            if py_type == FeatureSet:
                if webtool:
                    if isinstance(param_value, (tuple, list)):
                        gp_params[gp_param_name] = [
                            _layer_input_gp(p) for p in param_value
                        ]
                    else:
                        gp_params[gp_param_name] = _layer_input_gp(param_value)

                else:
                    try:
                        from arcgis.features.geo._accessor import (
                            _is_geoenabled,
                        )
                    except:

                        def _is_geoenabled(o):
                            return False

                    if type(param_value) == FeatureSet:
                        gp_params[gp_param_name] = param_value.to_dict()
                    elif _is_geoenabled(param_value):
                        gp_params[gp_param_name] = json.loads(
                            json.dumps(
                                param_value.spatial.__feature_set__,
                                default=_date_handler,
                            )
                        )
                    elif isinstance(param_value, arcgis.gis.Layer):
                        gp_params[gp_param_name] = _layer_input_gp(param_value)
                    elif type(param_value) == str:
                        try:
                            klass = py_type
                            gp_params[gp_param_name] = klass.from_str(param_value)

                        except:
                            pass

            elif py_type in [LinearUnit, DataFile, RasterData]:
                if type(param_value) in [LinearUnit, DataFile, RasterData]:
                    gp_params[gp_param_name] = param_value.to_dict()

                elif type(param_value) == str:
                    try:
                        klass = py_type
                        gp_params[gp_param_name] = klass.from_str(param_value)

                    except:
                        pass

                elif isinstance(param_value, arcgis.gis.Layer):
                    gp_params[gp_param_name] = param_value._lyr_dict

            elif py_type == datetime.datetime:
                gp_params[gp_param_name] = _date_handler(param_value)
    # --------------------------------------------#

    _set_env_params(gp_params, params)

    # for param_name, param_value in gp_params.items():
    #     print(param_name + " = " + str(param_value))

    gptool = arcgis.gis._GISResource(url, gis)

    if use_async:
        task_url = "{}/{}".format(url, task_name)
        submit_url = "{}/submitJob".format(task_url)
        if add_token and submit_url.lower().find("arcgis.com") == -1:
            try:
                job_info = gptool._con.post(submit_url, gp_params, token=gptool._token)
            except:
                job_info = gptool._con.post(submit_url, gp_params)
        else:
            job_info = gptool._con.post(submit_url, gp_params)
        job_id = job_info["jobId"]
        if future:
            executor = concurrent.futures.ThreadPoolExecutor(1)
            future = executor.submit(
                _future_op,
                *(
                    gptool,
                    task_url,
                    job_info,
                    job_id,
                    param_db,
                    return_values,
                    return_messages,
                ),
            )
            executor.shutdown(False)
            gpjob = GPJob(
                future=future,
                gptool=gptool,
                jobid=job_id,
                task_url=task_url,
                gis=gptool._gis,
                notify=arcgis.env.verbose,
            )
            return gpjob
        job_info = _analysis_job_status(gptool, task_url, job_info)
        resp = _analysis_job_results(gptool, task_url, job_info, job_id)

        # ---------------------async-out---------------------#
        output_dict = {}
        for retParamName in resp.keys():
            output_val = resp[retParamName]
            try:
                ret_param_name, ret_val = _get_output_value(
                    gptool, output_val, param_db, retParamName
                )
                output_dict[ret_param_name] = ret_val
            except KeyError:
                pass  # cannot handle unexpected output as return tuple will change

        # tools with output map service - add another output:
        # result_layer = '' #***self.properties.resultMapServerName
        if gptool.properties.resultMapServerName != "":
            job_id = job_info.get("jobId")
            result_layer_url = (
                url.replace("/GPServer", "/MapServer") + "/jobs/" + job_id
            )

            output_dict["result_layer"] = mapping.MapImageLayer(
                result_layer_url, gptool._gis
            )

        num_returns = len(resp)
        if return_messages:
            return (
                _return_output(num_returns, output_dict, return_values),
                job_info,
            )

        return _return_output(num_returns, output_dict, return_values)

    else:  # synchronous
        exec_url = url + "/" + task_name + "/execute"
        if add_token:
            try:
                resp = gptool._con.post(exec_url, gp_params, token=gptool._token)
            except:
                resp = gptool._con.post(exec_url, gp_params)
        else:
            resp = gptool._con.post(exec_url, gp_params)

        output_dict = {}

        for result in resp["results"]:
            retParamName = result["paramName"]

            output_val = result["value"]
            try:
                ret_param_name, ret_val = _get_output_value(
                    gptool, output_val, param_db, retParamName
                )
                output_dict[ret_param_name] = ret_val
            except KeyError:
                pass  # cannot handle unexpected output as return tuple will change

        num_returns = len(resp["results"])
        if return_messages:
            return (
                _return_output(num_returns, output_dict, return_values),
                job_info,
            )
        return _return_output(num_returns, output_dict, return_values)


def _set_env_params(gp_params, params):
    # copy environment variables if set
    if "env:outSR" not in params and arcgis.env.out_spatial_reference is not None:
        gp_params["env:outSR"] = arcgis.env.out_spatial_reference
    if (
        "env:processSR" not in params
        and arcgis.env.process_spatial_reference is not None
    ):
        gp_params["env:processSR"] = arcgis.env.process_spatial_reference
    if "returnZ" not in params and arcgis.env.return_z is not False:
        gp_params["returnZ"] = True
    if "returnM" not in params and arcgis.env.return_m is not False:
        gp_params["returnM"] = True


def _return_output(num_returns, output_dict, return_values):
    if num_returns == 1:
        return output_dict[return_values[0]["name"]]
    else:
        ret_names = []
        for return_value in return_values:
            ret_names.append(return_value["name"])
        # ret_names = output_dict.keys() # CANT USE - the order matters
        NamedTuple = collections.namedtuple("ToolOutput", ret_names)
        tool_output = NamedTuple(**output_dict)
        return tool_output


def _get_output_value(gptool, output_val, param_db, retParamName):
    ret_param_name = _camelCase_to_underscore(retParamName)

    ret_type, _ = param_db[ret_param_name]

    ret_val = output_val
    if output_val is not None:
        if ret_type in [FeatureSet, LinearUnit, DataFile, RasterData, Table]:
            jsondict = output_val
            if (
                "mapImage" in jsondict
            ):  # http://resources.esri.com/help/9.3/arcgisserver/apis/rest/gpresult.html#mapimage
                ret_val = jsondict
            elif ret_type == Table and "url" in jsondict:
                ret_val = arcgis.features.Table(jsondict["url"], gptool._gis)
            elif ret_type == FeatureSet and "url" in jsondict:
                ret_val = arcgis.features.FeatureLayer(jsondict["url"], gptool._gis)
            elif len(jsondict) == 0 or jsondict == {}:
                ret_val = None
            else:
                result = ret_type.from_dict(jsondict)
                result._con = gptool._con
                result._token = gptool._token
                ret_val = result
        else:
            ret_val = output_val
    return ret_param_name, ret_val
