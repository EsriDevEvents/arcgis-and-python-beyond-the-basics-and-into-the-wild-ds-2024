"""

Contains an internal GP method used to calculate Credit Usage for Tool
This is a private method and could change without warning. Do not use.

"""
import json
import arcgis as _arcgis
from arcgis.geoprocessing import DataFile, LinearUnit, RasterData
from arcgis.geoprocessing._support import _execute_gp_tool


def _estimate_credits(task, parameters, gis=None):
    """
    Estimates the number of credits a spatial analysis operation will take.

    =======================     ====================================================================
    **Parameter**                **Description**
    -----------------------     --------------------------------------------------------------------
    task                        Required String. The name of the analysis tool.
    -----------------------     --------------------------------------------------------------------
    parameters                  Required String.  The input parameters for the tool.
    -----------------------     --------------------------------------------------------------------
    gis                         Optional GIS.  The enterprise connection object.
    =======================     ====================================================================

    :return: Float value indicating the maximum cost

    """

    if gis is None and _arcgis.env.active_gis:
        gis = _arcgis.env.active_gis
    elif gis is None and _arcgis.env.active_gis is None:
        raise Exception("A GIS must be provided and/or set as active.")
    if gis.version >= [6, 4] and gis._portal.is_arcgisonline:
        url = gis.properties["helperServices"]["creditEstimation"]["url"]
        gptask = "EstimateCredits"
        url = "{base}/{gptask}/execute".format(base=url, gptask=gptask)
        params = {
            "f": "json",
            "taskName": task,
            "taskParameters": json.dumps(parameters),
        }
        kwargs = locals()
        param_db = {
            "task": (str, "taskName"),
            "parameters": (str, "taskParameters"),
            "credit_estimate": (str, "creditEstimate"),
        }
        return_values = [
            {"name": "credit_estimate", "display_name": "creditEstimate", "type": str},
        ]
        res = _execute_gp_tool(
            gis,
            gptask,
            kwargs,
            param_db,
            return_values,
            False,
            url,
            webtool=True,
            add_token=False,
        )
        if "cost" in res:
            return float(res["cost"])
        elif "maximumCost" in res:
            return float(res["maximumCost"])
        return res
    return
