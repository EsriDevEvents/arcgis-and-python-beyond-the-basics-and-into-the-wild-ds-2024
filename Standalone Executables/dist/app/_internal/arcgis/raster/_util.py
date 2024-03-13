import json as _json
from arcgis.raster._layer import ImageryLayer as _ImageryLayer

# from arcgis.raster._layer import Raster as _Raster
from arcgis.features import FeatureLayer as _FeatureLayer
import arcgis as _arcgis
import string as _string
import random as _random
from arcgis._impl.common._utils import _date_handler
import datetime
from arcgis.geometry import Geometry as _Geometry
import numbers
import time
import os
from urllib.parse import urljoin, quote, unquote, urlparse
import sys


import logging as _logging

_LOGGER = _logging.getLogger(__name__)

try:
    import numpy as _np
    import requests as _requests
except:
    pass


def _get_layer_info(input_layer):
    input_param = input_layer

    url = ""
    from arcgis.raster import Raster as _Raster
    from arcgis.gis import Item as _Item
    from arcgis.gis import Layer as _Layer

    if isinstance(input_layer, _Raster):
        if hasattr(input_layer, "_engine_obj"):
            input_layer = input_layer._engine_obj
    if isinstance(input_layer, _Item):
        if input_layer.type == "Image Collection":
            input_param = {"itemId": input_layer.itemid}
        else:
            if "layers" in input_layer:
                input_param = input_layer.layers[0]._lyr_dict
                try:
                    if isinstance(input_param, dict) and "url" in input_param.keys():
                        url = input_param["url"]
                        if "token" not in url:
                            from arcgis.raster.functions.utility import (
                                _generate_layer_token,
                            )

                            token = _generate_layer_token(input_layer, url)
                            if token is not None:
                                if input_layer.type == "Feature Service":
                                    input_param.update({"serviceToken": token})
                                else:
                                    url = input_param["url"] + "?token=" + token
                            input_param.update({"url": url})
                except:
                    pass
            else:
                raise TypeError("No layers in input layer Item")

    elif isinstance(input_layer, _Layer):
        input_param = input_layer._lyr_dict
        from arcgis.raster import ImageryLayer
        import json

        if isinstance(input_layer, _ImageryLayer) or isinstance(input_layer, _Raster):
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

            try:
                if isinstance(input_param, dict) and "url" in input_param.keys():
                    url = input_param["url"]
                    if "token" not in url:
                        from arcgis.raster.functions.utility import (
                            _generate_layer_token,
                        )

                        token = _generate_layer_token(input_layer, url)
                        if token is not None:
                            url = input_param["url"] + "?token=" + token
                        input_param.update({"url": url})
                        if "serviceToken" in input_param.keys():
                            del input_param["serviceToken"]
            except:
                pass

        elif isinstance(input_layer, _FeatureLayer):
            input_param = input_layer._lyr_dict
            try:
                if isinstance(input_param, dict) and "url" in input_param.keys():
                    url = input_param["url"]
                    if "serviceToken" not in input_param:
                        from arcgis.raster.functions.utility import (
                            _generate_layer_token,
                        )

                        token = _generate_layer_token(input_layer, url)
                        if token is not None:
                            input_param.update({"serviceToken": token})
                        input_param.update({"url": url})
            except:
                pass

    elif isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        if "http:" in input_layer or "https:" in input_layer:
            input_param = {"url": input_layer}
        else:
            input_param = {"uri": input_layer}

    else:
        raise Exception("Invalid format for env parameter")

    if "ImageServer" in url or "MapServer" in url:
        if "serviceToken" in input_param:
            url = url + "?token=" + input_param["serviceToken"]
            input_param.update({"url": url})

    return input_param


def _set_context(params, function_context=None):
    out_sr = _arcgis.env.out_spatial_reference
    process_sr = _arcgis.env.process_spatial_reference
    out_extent = _arcgis.env.analysis_extent
    mask = _arcgis.env.mask
    snap_raster = _arcgis.env.snap_raster
    cell_size = _arcgis.env.cell_size
    parallel_processing_factor = _arcgis.env.parallel_processing_factor

    context = {}

    if out_sr is not None:
        context["outSR"] = {"wkid": int(out_sr)}

    if out_extent is not None:
        context["extent"] = out_extent

    if process_sr is not None:
        context["processSR"] = {"wkid": int(process_sr)}

    if mask is not None:
        context["mask"] = _get_layer_info(mask)

    if cell_size is not None:
        if isinstance(cell_size, _ImageryLayer):
            context["cellSize"] = {"url": cell_size._url}
        elif isinstance(cell_size, str):
            if "http:" in cell_size or "https:" in cell_size:
                context["cellSize"] = {"url": cell_size}
            else:
                context["cellSize"] = cell_size
        else:
            context["cellSize"] = cell_size

    if snap_raster is not None:
        context["snapRaster"] = _get_layer_info(snap_raster)

    if parallel_processing_factor is not None:
        context["parallelProcessingFactor"] = parallel_processing_factor

    if function_context is not None:
        if context is not None:
            context.update({k: function_context[k] for k in function_context.keys()})

        else:
            context = function_context

    if context:
        params["context"] = _json.dumps(context)


def _id_generator(size=6, chars=_string.ascii_uppercase + _string.digits):
    return "".join(_random.choice(chars) for _ in range(size))


def _set_time_param(time):
    time_val = time
    if time is not None:
        if type(time) is list:
            if isinstance(time[0], datetime.datetime):
                if time[0].tzname() is None or time[0].tzname() != "UTC":
                    time[0] = time[0].astimezone(datetime.timezone.utc)
            if isinstance(time[1], datetime.datetime):
                if time[1].tzname() is None or time[1].tzname() != "UTC":
                    time[1] = time[1].astimezone(datetime.timezone.utc)
            starttime = _date_handler(time[0])
            endtime = _date_handler(time[1])
            if starttime is None:
                starttime = "null"
            if endtime is None:
                endtime = "null"
            time_val = "%s,%s" % (starttime, endtime)
        else:
            time_val = _date_handler(time)

    return time_val


def _to_datetime(dt):
    import datetime

    try:
        if dt < 0:
            return datetime.datetime(1970, 1, 1) + datetime.timedelta(
                seconds=(dt / 1000)
            )
        else:
            return datetime.datetime.fromtimestamp(
                dt / 1000, tz=datetime.timezone.utc
            ).replace(tzinfo=None)
    except:
        return dt


def _datetime2ole(date):
    # date = datetime.strptime(date, '%d-%b-%Y')
    import datetime

    OLE_TIME_ZERO = datetime.datetime(1899, 12, 30)
    delta = date - OLE_TIME_ZERO
    return float(delta.days) + (float(delta.seconds) / 86400)


def _ole2datetime(oledt):
    import datetime

    OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)
    try:
        return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))
    except:
        return datetime.datetime.fromtimestamp(
            oledt / 1000, tz=datetime.timezone.utc
        ).replace(tzinfo=None)


def _iso_to_datetime(timestamp):
    format_string = "%Y-%m-%dT%H:%M:%S%z"
    try:
        colon = timestamp[-3]
        colonless_timestamp = timestamp
        if colon == ":":
            colonless_timestamp = timestamp[:-3] + timestamp[-2:]
        dt_ob = datetime.datetime.strptime(colonless_timestamp, format_string)
        return dt_ob.replace(tzinfo=None)
    except:
        try:
            format_string = "%Y-%m-%dT%H:%M:%S"
            dt_ob = datetime.datetime.strptime(timestamp, format_string)
            return dt_ob
        except:
            return timestamp


def _check_if_iso_format(timestamp):
    format_string = "%Y-%m-%dT%H:%M:%S%z"
    try:
        colon = timestamp[-3]
        colonless_timestamp = timestamp
        if colon == ":":
            colonless_timestamp = timestamp[:-3] + timestamp[-2:]
        dt_ob = datetime.datetime.strptime(colonless_timestamp, format_string)
        return True
    except:
        try:
            format_string = "%Y-%m-%dT%H:%M:%S"
            dt_ob = datetime.datetime.strptime(timestamp, format_string)
            return dt_ob
        except:
            return False


def _time_filter(time_extent, ele):
    if time_extent is not None:
        if isinstance(time_extent, datetime.datetime):
            if ele < time_extent:
                return True
            else:
                return False
        elif isinstance(time_extent, list):
            if isinstance(time_extent[0], datetime.datetime) and isinstance(
                time_extent[1], datetime.datetime
            ):
                if time_extent[0] < ele and ele < time_extent[1]:
                    return True
                else:
                    return False

        else:
            return True
    else:
        return True


def _linear_regression(sample_size, date_list, x, y):
    ncoefficient = 2
    if sample_size < ncoefficient:
        _LOGGER.warning(
            "Trend line cannot be drawn. Insufficient points to plot Linear Trend Line"
        )
        return [], []

    AA = _np.empty([sample_size, ncoefficient], dtype=float, order="C")
    BB = _np.empty([sample_size, 1], dtype=float, order="C")
    XX = _np.empty([ncoefficient, 1], dtype=float, order="C")
    for i in range(sample_size):
        n = 0
        AA[i][n] = date_list[i]
        AA[i][n + 1] = 1
        BB[i] = y[i]

    x1 = _np.linalg.lstsq(AA, BB, rcond=None)[0]

    YY = []
    for i in range(sample_size):
        y_temp = x1[0][0] * date_list[i] + x1[1][0]
        YY.append(y_temp)
    return x, YY


def _harmonic_regression(sample_size, date_list, x, y, trend_order):
    PI2_Year = 3.14159265 * 2 / 365.25

    ncoefficient = 2 * (trend_order + 1)
    if sample_size < ncoefficient:
        _LOGGER.warning(
            "Trend line cannot be drawn. Insufficient points to plot Harmonic Trend Line for trend order "
            + str(trend_order)
            + ". Please try specifying a lower trend order."
        )
        return [], []

    AA = _np.empty([sample_size, ncoefficient], dtype=float, order="C")
    BB = _np.empty([sample_size, 1], dtype=float, order="C")
    XX = _np.empty([ncoefficient, 1], dtype=float, order="C")

    for i in range(sample_size):
        n = 0
        AA[i][n] = date_list[i]
        AA[i][n + 1] = 1

        for j in range(1, trend_order + 1):
            AA[i][n + 2 * j] = _np.sin(PI2_Year * j * date_list[i])
            AA[i][n + 2 * j + 1] = _np.cos(PI2_Year * j * date_list[i])

        BB[i] = y[i]

    x1 = _np.linalg.lstsq(AA, BB, rcond=None)[0]
    YY = []
    for i in range(sample_size):
        y_temp = x1[0][0] * date_list[i] + x1[1][0]
        for q in range(2, len(x1), 2):
            y_temp = y_temp + x1[q][0] * _np.sin(
                2 * 3.14159265358979323846 * (q / 2) * date_list[i] / 365.25
            )
            y_temp = y_temp + x1[q + 1][0] * _np.cos(
                2 * 3.14159265358979323846 * (q / 2) * date_list[i] / 365.25
            )
        YY.append(y_temp)
    return x, YY


def _epoch_to_iso(dt):
    import datetime

    try:
        if dt < 0:
            return (
                datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
                + datetime.timedelta(seconds=(dt / 1000))
            ).isoformat()
        else:
            return datetime.datetime.fromtimestamp(
                dt / 1000, tz=datetime.timezone.utc
            ).isoformat()
    except:
        return dt


def _datetime2ole(date):
    # date = datetime.strptime(date, '%d-%b-%Y')
    import datetime

    OLE_TIME_ZERO = datetime.datetime(1899, 12, 30)
    delta = date - OLE_TIME_ZERO
    return float(delta.days) + (float(delta.seconds) / 86400)


def _ole2datetime(oledt):
    import datetime

    OLE_TIME_ZERO = datetime.datetime(1899, 12, 30, 0, 0, 0)
    try:
        return OLE_TIME_ZERO + datetime.timedelta(days=float(oledt))
    except:
        return datetime.datetime.fromtimestamp(
            oledt / 1000, tz=datetime.timezone.utc
        ).replace(tzinfo=None)


def _iso_to_datetime(timestamp):
    format_string = "%Y-%m-%dT%H:%M:%S%z"
    try:
        colon = timestamp[-3]
        colonless_timestamp = timestamp
        if colon == ":":
            colonless_timestamp = timestamp[:-3] + timestamp[-2:]
        dt_ob = datetime.datetime.strptime(colonless_timestamp, format_string)
        return dt_ob.replace(tzinfo=None)
    except:
        try:
            format_string = "%Y-%m-%dT%H:%M:%S.%f%z"
            dt_ob = datetime.datetime.strptime(colonless_timestamp, format_string)
            return dt_ob.replace(tzinfo=None)
        except:
            try:
                format_string = "%Y-%m-%dT%H:%M:%S"
                dt_ob = datetime.datetime.strptime(timestamp, format_string)
                return dt_ob
            except:
                return timestamp


def _check_if_iso_format(timestamp):
    format_string = "%Y-%m-%dT%H:%M:%S%z"
    try:
        colon = timestamp[-3]
        colonless_timestamp = timestamp
        if colon == ":":
            colonless_timestamp = timestamp[:-3] + timestamp[-2:]
        dt_ob = datetime.datetime.strptime(colonless_timestamp, format_string)
        return True
    except:
        try:
            format_string = "%Y-%m-%dT%H:%M:%S"
            dt_ob = datetime.datetime.strptime(timestamp, format_string)
            return dt_ob
        except:
            return False


def _local_function_template(
    operation_number=None,
    percentile_value=None,
    percentile_interpolation_type=None,
):
    template_dict = {
        "name": "max_rft",
        "description": "A raster function template.",
        "function": {
            "pixelType": "UNKNOWN",
            "name": "Cell Statistics",
            "description": "Calculates a per-cell statistic from multiple rasters.  The available statistics are Majority, Maximum, Mean, Median, Minimum, Minority, Range, Standard Deviation, Sum, and Variety.",
            "type": "LocalFunction",
            "_object_id": 1,
        },
        "arguments": {
            "Rasters": {
                "name": "Rasters",
                "value": {
                    "elements": [],
                    "type": "ArgumentArray",
                    "_object_id": 2,
                },
                "aliases": ["__IsRasterArray__"],
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 3,
            },
            "Operation": {
                "name": "Operation",
                "value": "",
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 4,
            },
            "CellsizeType": {
                "name": "CellsizeType",
                "value": 2,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 5,
            },
            "ExtentType": {
                "name": "ExtentType",
                "value": 1,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 6,
            },
            "ProcessAsMultiband": {
                "name": "ProcessAsMultiband",
                "value": True,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 7,
            },
            "MatchVariable": {
                "name": "MatchVariable",
                "value": True,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 8,
            },
            "UnionDimension": {
                "name": "UnionDimension",
                "value": False,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 9,
            },
            "PercentileValue": {
                "name": "PercentileValue",
                "value": 90,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 5,
            },
            "PercentileInterpolationType": {
                "name": "PercentileInterpolationType",
                "value": 1,
                "isDataset": False,
                "isPublic": False,
                "type": "RasterFunctionVariable",
                "_object_id": 6,
            },
            "type": "LocalFunctionArguments",
            "_object_id": 10,
        },
        "functionType": 0,
        "thumbnail": "",
    }
    if operation_number is not None:
        template_dict["arguments"]["Operation"]["value"] = operation_number
    if percentile_value is not None:
        template_dict["arguments"]["PercentileValue"]["value"] = percentile_value
    if percentile_interpolation_type is not None:
        template_dict["arguments"]["PercentileInterpolationType"][
            "value"
        ] = percentile_interpolation_type
    return template_dict


# def _percentile_function_template(
#    ignore_nodata=False, percentile=90, percentile_interpolation_type=False
# ):
#    template_dict = {
#        "name": "Raster Function Template",
#        "description": "A raster function template.",
#        "function": {
#            "pixelType": "UNKNOWN",
#            "name": "Percentile Function",
#            "description": "Compute percentile value across the input rasters.",
#            "type": "PercentileFunction",
#            "_object_id": 1,
#        },
#        "arguments": {
#            "Rasters": {
#                "name": "Rasters",
#                "isDataset": False,
#                "isPublic": False,
#                "type": "RasterFunctionVariable",
#                "_object_id": 2,
#            },
#            "IgnoreNoData": True,
#            "Percentile": 90,
#            "InterpolatePercentile": False,
#            "type": "PercentileFunctionArguments",
#            "_object_id": 3,
#        },
#        "functionType": 0,
#        "type": "RasterFunctionTemplate",
#        "_object_id": 4,
#    }

#    if ignore_nodata is not None:
#        template_dict["arguments"]["IgnoreNoData"] = ignore_nodata

#    if percentile is not None:
#        template_dict["arguments"]["Percentile"] = percentile

#    if percentile_interpolation_type is not None:
#        template_dict["arguments"][
#            "InterpolatePercentile"
#        ] = percentile_interpolation_type

#    return template_dict


def _get_geometry(data):
    if data is None:
        return None

    if isinstance(data, _Geometry):
        return data
    elif isinstance(data, _arcgis.raster.Raster):
        return _Geometry(data.extent)
    elif isinstance(data, _ImageryLayer):
        return _Geometry(data.extent)
    elif isinstance(data, _FeatureLayer):
        return _get_geometry_from_feature_layer(data)
    else:
        return data


def _get_geometry_from_feature_layer(data):
    geo = None
    layer_fset = data.query()
    try:
        for ele in layer_fset.features:
            geo = geo.union(_Geometry(ele.geometry)) if geo else _Geometry(ele.geometry)
    except:
        _LOGGER.warning(
            "Failure while constructing the union of the individual feature geometries"
        )
    return geo


def build_query_string(field_name, operator, field_values):
    operator_map = {
        "equals": "=",
        "less_than": "<",
        "greater_than": ">",
        "not_equals": "<>",
        "not_less_than": ">=",
        "not_greater_than": "<=",
    }

    if operator in operator_map:
        if isinstance(field_values, numbers.Number):
            return field_name + " " + operator_map[operator] + " " + str(field_values)
        elif isinstance(field_values, str):
            return field_name + " " + operator_map[operator] + " '" + field_values + "'"
        else:
            raise TypeError("field_value must be numeric or string")

    elif operator in [
        "starts_with",
        "ends_with",
        "not_starts_with",
        "not_ends_with",
        "contains",
        "not_contains",
    ]:
        if not isinstance(field_values, str):
            raise TypeError("field_value must be string")
        if operator == "starts_with":
            return field_name + " LIKE " + "'" + field_values + "%'"
        elif operator == "ends_with":
            return field_name + " LIKE" + "'%" + field_values + "'"
        elif operator == "not_starts_with":
            return field_name + " NOT LIKE " + "'" + field_values + "%'"
        elif operator == "not_ends_with":
            return field_name + " NOT LIKE " + "'%" + field_values + "'"
        elif operator == "contains":
            return field_name + " LIKE " + "'%" + field_values + "%'"
        elif operator == "not_contains":
            return field_name + " NOT LIKE " + "'%" + field_values + "%'"
    elif operator == "in":
        if not isinstance(field_values, list):
            raise TypeError('field_values must be type list for operator "in"')
        values = "("
        for item in field_values:
            if not (isinstance(item, numbers.Number) or isinstance(item, str)):
                raise TypeError("item in field_values must be numeric or string")
            if values == "(":
                values += "'" + item + "'" if isinstance(item, str) else str(item)
            else:
                values += (
                    ",'" + item + "'" if isinstance(item, str) else "," + str(item)
                )
        values += ")"
        return field_name + " IN " + values
    elif operator == "not_in":
        values = "("
        for item in field_values:
            if not (isinstance(item, numbers.Number) or isinstance(item, str)):
                raise TypeError("item in field_values must be numeric or string")
            if values == "(":
                values += "'" + item + "'" if isinstance(item, str) else str(item)
            else:
                values += (
                    ",'" + item + "'" if isinstance(item, str) else "," + str(item)
                )
        values += ")"
        return field_name + " NOT IN " + values
    else:
        raise ValueError("invalid operator value")


def _generate_direct_access_url(gis=None, expiration=None):
    """helper fn to get the direct access url for azure storage"""
    gis = _arcgis.env.active_gis if gis is None else gis
    url = "%s/sharing/rest/content/users/%s/generateDirectAccessUrl" % (
        gis._portal.url,
        gis.users.me.username,
    )
    params = {"f": "json", "storeType": "rasterStore"}
    if expiration is not None:
        params.update({"expiration": expiration})
    else:
        params.update({"expiration": 1440})
    res = gis._portal.con.post(url, params)
    if isinstance(res, dict):
        if "url" in res.keys():
            return res["url"]
        else:
            raise RuntimeError("Couldn't generate direct access url")
    else:
        raise RuntimeError("Couldn't generate direct access url")


def _print_on_same_line(msg):
    """helper method for printing text on same line"""
    last_msg_length = (
        len(_print_on_same_line.last_msg)
        if hasattr(_print_on_same_line, "last_msg")
        else 0
    )
    print(" " * last_msg_length, end="\r")
    print(msg, end="\r")
    sys.stdout.flush()
    _print_on_same_line.last_msg = msg


def _print_progress_bar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    unit="items",
    fill="█",
):
    """method for displaying a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)

    _print_on_same_line(
        f"\r{prefix} |{bar}| {percent}% complete ({iteration}/{total} {unit})\t{suffix}"
    )
    if iteration == total:
        print("\n")


def _ra_upload_allowed_extensions():
    """
    returns a list of valid file upload extensions supported for creating
    hosted imagery layers from raster datasets on Enterprise/AGOL
    """
    return (
        "1b,5gud,a11,a12,a13,a14,a15,a16,a17,a18,a19,a1a,a1b,a1c,a1d,a1e,"
        "a1f,a1g,a1h,a1j,a21,a22,a23,a24,a25,a26,a27,a28,a29,a2a,a2b,a2c,"
        "a2d,a2e,a2f,a2g,a2h,a2j,a31,a32,a33,a34,a35,a36,a37,a38,a39,a3a,"
        "a3b,a3c,a3d,a3e,a3f,a3g,a3h,a3j,a41,a42,a43,a44,a45,a46,a47,a48,"
        "a49,a4a,a4b,a4c,a4d,a4e,a4f,a4g,a4h,a4j,adf,ads,afr,asc,at1,at2,"
        "at3,at4,at5,at6,at7,at8,at9,ata,atb,atc,atd,ate,atf,atg,ath,atj,"
        "att,aux,avg,bag,bil,bin,bip,blw,blx,bmp,bpw,bqw,bsq,bt,bundle,"
        "bundlx,c11,c12,c13,c14,c15,c16,c17,c18,c19,c1a,c1b,c1c,c1d,c1e,"
        "c1f,c1g,c1h,c1j,c21,c22,c23,c24,c25,c26,c27,c28,c29,c2a,c2b,c2c,"
        "c2d,c2e,c2f,c2g,c2h,c2j,c41,c42,c43,c44,c45,c46,c47,c48,c49,c4a,"
        "c4b,c4c,c4d,c4e,c4f,c4g,c4h,c4j,c51,c52,c53,c54,c55,c56,c57,c58,"
        "c59,c5a,c5b,c5c,c5d,c5e,c5f,c5g,c5h,c5j,c61,c62,c63,c64,c65,c66,"
        "c67,c68,c69,c6a,c6b,c6c,c6d,c6e,c6f,c6g,c6h,c6j,c71,c72,c73,c74,"
        "c75,c76,c77,c78,c79,c7a,c7b,c7c,c7d,c7e,c7f,c7g,c7h,c7j,c81,c82,"
        "c83,c84,c85,c86,c87,c88,c89,c8a,c8b,c8c,c8d,c8e,c8f,c8g,c8h,c8j,"
        "c91,c92,c93,c94,c95,c96,c97,c98,c99,c9a,c9b,c9c,c9d,c9e,c9f,c9g,"
        "c9h,c9j,ca1,ca2,ca3,ca4,ca5,ca6,ca7,ca8,ca9,caa,cab,cac,cad,cae,"
        "caf,cag,cah,caj,cb1,cb2,cb3,cb4,cb5,cb6,cb7,cb8,cb9,cba,cbb,cbc,"
        "cbd,cbe,cbf,cbg,cbh,cbj,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,cca,"
        "ccb,ccc,ccd,cce,ccf,ccg,cch,ccj,cd1,cd2,cd3,cd4,cd5,cd6,cd7,cd8,"
        "cd9,cda,cdb,cdc,cdd,cde,cdf,cdg,cdh,cdi,cdj,ce1,ce2,ce3,ce4,ce5,"
        "ce6,ce7,ce8,ce9,cea,ceb,cec,ced,cee,cef,ceg,ceh,cej,cf1,cf2,cf3,"
        "cf4,cf5,cf6,cf7,cf8,cf9,cfa,cfb,cfc,cfd,cfe,cff,cfg,cfh,cfj,cg1,"
        "cg2,cg3,cg4,cg5,cg6,cg7,cg8,cg9,cga,cgb,cgc,cgd,cge,cgf,cgg,cgh,"
        "cgj,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,cha,chb,chc,chd,che,chf,"
        "chg,chh,chj,cit,cj1,cj2,cj3,cj4,cj5,cj6,cj7,cj8,cj9,cja,cjb,cjc,"
        "cjd,cje,cjf,cjg,cjh,cjj,ck1,ck2,ck3,ck4,ck5,ck6,ck7,ck8,ck9,cka,"
        "ckb,ckc,ckd,cke,ckf,ckg,ckh,ckj,cl1,cl2,cl3,cl4,cl5,cl6,cl7,cl8,"
        "cl9,cla,clb,clc,cld,cle,clf,clg,clh,clj,clr,cm1,cm2,cm3,cm4,cm5,"
        "cm6,cm7,cm8,cm9,cma,cmb,cmc,cmd,cme,cmf,cmg,cmh,cmj,cn1,cn2,cn3,"
        "cn4,cn5,cn6,cn7,cn8,cn9,cna,cnb,cnc,cnd,cne,cnf,cng,cnh,cnj,co1,"
        "co2,co3,co4,co5,co6,co7,co8,co9,coa,cob,coc,cod,coe,cof,cog,coh,"
        "coj,cos,cot,cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9,cpa,cpb,cpc,cpd,"
        "cpe,cpf,cpg,cph,cpj,cq1,cq2,cq3,cq4,cq5,cq6,cq7,cq8,cq9,cqa,cqb,"
        "cqc,cqd,cqe,cqf,cqg,cqh,cqj,cr1,cr2,cr3,cr4,cr5,cr6,cr7,cr8,cr9,"
        "cra,crb,crc,crd,cre,crf,crg,crh,crj,cs1,cs2,cs3,cs4,cs5,cs6,cs7,"
        "cs8,cs9,csa,csb,csc,csd,cse,csf,csg,csh,csj,ct1,ct2,ct3,ct4,ct5,"
        "ct6,ct7,ct8,ct9,cta,ctb,ctc,ctd,cte,ctf,ctg,cth,ctj,cub,dat,dbf,"
        "ddf,dem,dim,dt0,dt1,dt2,elas,eph,ers,f11,f12,f13,f14,f15,f16,f17,"
        "f18,f19,f1a,f1b,f1c,f1d,f1e,f1f,f1g,f1h,f1j,f21,f22,f23,f24,f25,"
        "f26,f27,f28,f29,f2a,f2b,f2c,f2d,f2e,f2f,f2g,f2h,f2j,f31,f32,f33,"
        "f34,f35,f36,f37,f38,f39,f3a,f3b,f3c,f3d,f3e,f3f,f3g,f3h,f3j,f41,"
        "f42,f43,f44,f45,f46,f47,f48,f49,f4a,f4b,f4c,f4d,f4e,f4f,f4g,f4h,"
        "f4j,f51,f52,f53,f54,f55,f56,f57,f58,f59,f5a,f5b,f5c,f5d,f5e,f5f,"
        "f5g,f5h,f5j,fit,flt,fst,gc,geo,gff,gif,gis,gn1,gn2,gn3,gn4,gn7,gn9,"
        "gna,gnb,gnc,gnd,gng,gnj,gr2,grb,grb2,grc,grd,grib,grib2,gtx,gxf,h1,"
        "h4,h5,ha1,ha1,ha2,ha3,ha4,ha5,ha6,ha7,ha8,ha9,haa,hab,hac,had,hae,"
        "haf,hag,hah,haj,hdf,hdf4,hdf5,hdr,he4,he5,hf2,hgt,hr1,hr2,hr3,hr4,"
        "hr5,hr6,hr7,hr8,i1,i11,i12,i13,i14,i15,i16,i17,i18,i19,i1a,i1b,i1c,"
        "i1d,i1e,i1f,i1g,i1h,i1j,i2,i21,i22,i23,i24,i25,i26,i27,i28,i29,i2a,"
        "i2b,i2c,i2d,i2e,i2f,i2g,i2h,i2j,i3,i31,i32,i33,i34,i35,i36,i37,i38,"
        "i39,i3a,i3b,i3c,i3d,i3e,i3f,i3g,i3h,i3j,i4,i41,i42,i43,i44,i45,i46,"
        "i47,i48,i49,i4a,i4b,i4c,i4d,i4e,i4f,i4g,i4h,i4j,i5,i51,i52,i53,i54,"
        "i55,i56,i57,i58,i59,i5a,i5b,i5c,i5d,i5e,i5f,i5g,i5h,i5j,i6,i7,i8,i9,"
        "idx,ige,imd,img,iv1,iv2,iv3,iv4,iv5,iv6,iv7,iv8,iv9,iva,ivb,ivc,ivd,"
        "ive,ivf,ivg,ivh,ivj,j2c,j2k,ja1,ja2,ja3,ja4,ja5,ja6,ja7,ja8,ja9,jaa,"
        "jab,jac,jad,jae,jaf,jag,jah,jaj,jg1,jg2,jg3,jg4,jg5,jg6,jg7,jg8,jg9,"
        "jga,jgb,jgc,jgd,jge,jgf,jgg,jgh,jgj,jgw,jn1,jn2,jn3,jn4,jn5,jn6,jn7,"
        "jn8,jn9,jna,jnb,jnc,jnd,jne,jnf,jng,jnh,jnj,jo1,jo2,jo3,jo4,jo5,jo6,"
        "jo7,jo8,jo9,joa,job,joc,jod,joe,jof,jog,joh,joj,jp2,jpc,jpg,jpw,jpx,"
        "jr1,jr2,jr3,jr4,jr5,jr6,jr7,jr8,jr9,jra,jrb,jrc,jrd,jre,jrf,jrg,jrh,"
        "jrj,json,kap,l11,l12,l13,l14,l15,l16,l17,l18,l19,l1a,l1b,l1c,l1d,l1e,"
        "l1f,l1g,l1h,l1j,l21,l22,l23,l24,l25,l26,l27,l28,l29,l2a,l2b,l2c,l2d,"
        "l2e,l2f,l2g,l2h,l2j,l31,l32,l33,l34,l35,l36,l37,l38,l39,l3a,l3b,l3c,"
        "l3d,l3e,l3f,l3g,l3h,l3j,l41,l42,l43,l44,l45,l46,l47,l48,l49,l4a,l4b,"
        "l4c,l4d,l4e,l4f,l4g,l4h,l4j,l51,l52,l53,l54,l55,l56,l57,l58,l59,l5a,"
        "l5b,l5c,l5d,l5e,l5f,l5g,l5h,l5j,lan,las,lbl,lf1,lf2,lf3,lf4,lf5,lf6,"
        "lf7,lf8,lf9,lfa,lfb,lfc,lfd,lfe,lff,lfg,lfh,lfj,lgg,ln1,ln2,ln3,ln4,"
        "ln5,ln6,ln7,ln8,ln9,lna,lnb,lnc,lnd,lne,lnf,lng,lnh,lnj,lrc,m11,m12,"
        "m13,m14,m15,m16,m17,m18,m19,m1a,m1b,m1c,m1d,m1e,m1f,m1g,m1h,m1j,m21,"
        "m22,m23,m24,m25,m26,m27,m28,m29,m2a,m2b,m2c,m2d,m2e,m2f,m2g,m2h,m2j,"
        "map,max,memory,met,mi1,mi2,mi3,mi4,mi5,mi6,mi7,mi8,mi9,mia,mib,mic,"
        "mid,mie,mif,mig,mih,mij,min,mm1,mm2,mm3,mm4,mm5,mm6,mm7,mm8,mm9,mma,"
        "mmb,mmc,mmd,mme,mmf,mmg,mmh,mmj,mpl,mpr,mrf,mtl,n1,nc,nc4,nes,nsf,ntf,"
        "oa1,oa2,oa3,oa4,oa5,oa6,oa7,oa8,oa9,oaa,oab,oac,oad,oae,oaf,oag,oah,"
        "oaj,oh1,oh2,oh3,oh4,oh5,oh6,oh7,oh8,oh9,oha,ohb,ohc,ohd,ohe,ohf,ohg,"
        "ohh,ohj,on1,on2,on3,on4,on5,on6,on7,on8,on9,ona,onb,onc,ond,one,onf,"
        "ong,onh,onj,ovr,ow1,ow2,ow3,ow4,ow5,ow6,ow7,ow8,ow9,owa,owb,owc,owd,"
        "owe,owf,owg,owh,owj,paux,pbm,pgm,pgw,pix,png,ppm,prj,pro,properties,"
        "psi,pvl,r0,raw,rpb,rpc,rrd,rst,rv1,rv2,rv3,rv4,rv5,rv6,rv7,rv8,rv9,"
        "rva,rvb,rvc,rvd,rve,rvf,rvg,rvh,rvj,sdat,sdw,sid,sta,stk,sv,tc1,tc2,"
        "tc3,tc4,tc5,tc6,tc7,tc8,tc9,tca,tcb,tcc,tcd,tce,tcf,tcg,tch,tcj,ter,"
        "tf1,tf2,tf3,tf4,tf5,tf6,tf7,tf8,tf9,tfa,tfb,tfc,tfd,tfe,tff,tfg,tfh,"
        "tfj,tfrd,tfw,tif,tiff,til,tl1,tl2,tl3,tl4,tl5,tl6,tl7,tl8,tl9,tla,tlb,"
        "tlc,tld,tle,tlf,tlg,tlh,tlj,tn1,tn2,tn3,tn4,tn5,tn6,tn7,tn8,tn9,tna,"
        "tnb,tnc,tnd,tne,tnf,tng,tnh,tnj,toc,tp1,tp2,tp3,tp4,tp5,tp6,tp7,tp8,"
        "tp9,tpa,tpb,tpc,tpd,tpe,tpf,tpg,tph,tpj,tq1,tq2,tq3,tq4,tq5,tq6,tq7,"
        "tq8,tq9,tqa,tqb,tqc,tqd,tqe,tqf,tqg,tqh,tqj,tr1,tr2,tr3,tr4,tr5,tr6,"
        "tr7,tr8,tr9,tra,trb,trc,trd,tre,trf,trg,trh,trj,trl,tt1,tt2,tt3,tt4,"
        "tt5,tt6,tt7,tt8,tt9,tta,ttb,ttc,ttd,tte,ttf,ttg,tth,ttj,txt,ul1,ul2,"
        "ul3,ul4,ul5,ul6,ul7,ul8,ul9,ula,ulb,ulc,uld,ule,ulf,ulg,ulh,ulj,vh1,"
        "vh2,vh3,vh4,vh5,vh6,vh7,vh8,vh9,vha,vhb,vhc,vhd,vhe,vhf,vhg,vhh,vhj,"
        "view,vn1,vn2,vn3,vn4,vn5,vn6,vn7,vn8,vn9,vna,vnb,vnc,vnd,vne,vnf,vng,"
        "vnh,vnj,vrt,vt1,vt2,vt3,vt4,vt5,vt6,vt7,vt8,vt9,vta,vtb,vtc,vtd,vte,"
        "vtf,vtg,vth,vtj,wo,xml,xpm,xyz,gdb,pjg,pzp,ppg".split(",")
    )


def _is_primary_file(file):
    """Returns the file path if it is a primary file, otherwise return None."""

    # CRF folder check
    if file.endswith(".bundle") and "/_alllayers" in file:
        return file[: file.rfind("/_alllayers")]

    # Common raster dataset formats check
    if file.lower().endswith(
        (
            ".tiff",
            ".tif",
            ".mrf",
            ".img",
            ".jp2",
            ".jpx",
            ".j2k",
            ".sid",
            ".ntf",
            ".nsf",
            ".hdf",
            ".hdf4",
            ".hdf5",
            ".h4",
            ".h5",
            ".he4",
            ".he5",
            ".grib",
            ".grb",
            ".grib2",
            ".grb2",
            ".bin",
            ".dat",
            ".nc",
            ".nc4",
        )
    ):
        return file
    return


class _ImageryUploaderAGOL:
    """helper class for concurrently uploading multiple files to user's rasterstore on AGOL"""

    def __init__(
        self,
        file_list,
        container,
        auto_renew,
        upload_properties,
        task,
        raster_type,
        gis,
    ):
        from azure.storage.blob import ContainerClient
        from azure.core.exceptions import (
            ClientAuthenticationError,
            ServiceResponseError,
            ServiceRequestError,
        )

        self.ContainerClient = ContainerClient
        (
            self.ClientAuthenticationError,
            self.ServiceResponseError,
            self.ServiceRequestError,
        ) = (
            ClientAuthenticationError,
            ServiceResponseError,
            ServiceRequestError,
        )

        self.file_list = file_list
        self.container = container
        self.auto_renew = auto_renew
        self.task = task
        self.raster_type = raster_type
        self.gis = gis
        self.all_files = []
        self.mosaic_data_info = []
        self.primary_files = []
        self.single_primary_file = (
            False
            if self.raster_type != "Raster Dataset"
            or (
                len(self.file_list) > 1
                and any(item["is_dir"] for item in self.file_list)
            )
            else True
        )
        for i, d in enumerate(file_list):
            self.all_files.extend([(f, i) for f in d["files_list"]])
        self.url_list = []
        use_defaults = False
        if upload_properties is None or not isinstance(upload_properties, dict):
            upload_properties_lower = {
                "maxuploadconcurrency": 6,
                "maxworkerthreads": None,
                "displayprogress": False,
            }
            use_defaults = True
        else:
            upload_properties_lower = {
                k.lower(): v for k, v in upload_properties.items()
            }

        if "maxuploadconcurrency" in upload_properties_lower and (
            isinstance(upload_properties_lower["maxuploadconcurrency"], int)
            or upload_properties_lower["maxuploadconcurrency"] is None
        ):
            self.max_upload_concurrency = upload_properties_lower[
                "maxuploadconcurrency"
            ]
        else:
            self.max_upload_concurrency = 6

        if "maxworkerthreads" in upload_properties_lower and (
            isinstance(upload_properties_lower["maxworkerthreads"], int)
            or upload_properties_lower["maxworkerthreads"] is None
        ):
            self.max_worker_threads = upload_properties_lower["maxworkerthreads"]
        else:
            self.max_worker_threads = None

        display_flag = _arcgis.env.verbose
        if "displayprogress" in upload_properties_lower and isinstance(
            upload_properties_lower["displayprogress"], bool
        ):
            self.display_progress = upload_properties_lower["displayprogress"]
            if not self.display_progress and not use_defaults:
                display_flag = False
        else:
            self.display_progress = False

        if display_flag:
            self.display_progress = True

    def upload_file(self, file_item):
        """method to upload single file"""

        file_name, i = file_item
        prefix = self.file_list[i]["prefix"]
        current_time_str = prefix[:-1][8:]
        is_dir = self.file_list[i]["is_dir"]
        data_for_md = self.file_list[i]["data_for_md"]
        if is_dir:
            folder_path = self.file_list[i]["file_name"]
            root = os.path.dirname(file_name)
            basename_len = self.file_list[i]["basename_len"]
            blobname = prefix + (root + "/" + os.path.basename(file_name))[
                basename_len + 1 :
            ].replace(os.sep, "/")
        else:
            blobname = prefix + os.path.basename(file_name).replace(os.sep, "/")

        while True:
            try:
                blob = self.container.get_blob_client(blobname)

                with open(file_name, "rb") as data:
                    blob.upload_blob(
                        data,
                        blob_type="BlockBlob",
                        max_concurrency=self.max_upload_concurrency,
                    )

                url = blob.url.split("?", 1)[0]
                url_suffix = (
                    "arcgis.com" if "arcgis.com" in url else "blob.core.windows.net"
                )

                if is_dir:
                    if self.file_list[i]["single_image"]:
                        self.url_list.append(url)
                    else:
                        if url != "":
                            if data_for_md:
                                source = folder_path
                                if self.task == "CreateImageCollection":
                                    target = os.path.basename(source)
                                else:
                                    folder_match = quote(
                                        prefix + os.path.basename(source)
                                    )
                                    folder_url = url[
                                        0 : url.find(folder_match) + len(folder_match)
                                    ]
                                    target = unquote(
                                        folder_url.replace(
                                            folder_url[
                                                0 : folder_url.find(url_suffix)
                                                + len(url_suffix)
                                            ],
                                            "/vsiaz",
                                        )
                                    )

                                data_path = {
                                    "source": source,
                                    "target": target,
                                }
                                if data_path not in self.mosaic_data_info:
                                    self.mosaic_data_info.append(data_path)

                            if self.single_primary_file:
                                primary_file = _is_primary_file(url)
                                if primary_file:
                                    if primary_file not in self.primary_files:
                                        self.primary_files.append(primary_file)
                            url = url[
                                0 : url.find(current_time_str) + len(current_time_str)
                            ]
                            if url not in self.url_list:
                                self.url_list.append(url)
                else:
                    if data_for_md:
                        if self.task == "CreateImageCollection":
                            data_path = os.path.dirname(file_name)
                        else:
                            source = os.path.dirname(file_name)
                            target = os.path.dirname(
                                url.replace(
                                    url[0 : url.find(url_suffix) + len(url_suffix)],
                                    "/vsiaz",
                                )
                            )
                            data_path = {"source": source, "target": target}

                        if data_path not in self.mosaic_data_info:
                            self.mosaic_data_info.append(data_path)

                    if self.single_primary_file:
                        primary_file = _is_primary_file(url)
                        if primary_file:
                            if primary_file not in self.primary_files:
                                self.primary_files.append(primary_file)

                    if (
                        url not in self.url_list
                        and os.path.dirname(url) not in self.url_list
                    ):
                        self.url_list.append(url)

                break
            except (
                self.ClientAuthenticationError,
                self.ServiceResponseError,
                self.ServiceRequestError,
            ) as err:
                if self.auto_renew:
                    sas_url = _generate_direct_access_url(self.gis)
                    self.container = self.ContainerClient.from_container_url(sas_url)
                    continue
                else:
                    raise
            except Exception as err:
                raise err

        return file_name

    def upload_all_files(self):
        """method to upload multiple files concurrently"""
        result, mosaic_data_info = self.run(self.all_files)
        return result, mosaic_data_info

    def run(self, all_files):
        """helper method for creating a thread pool for uploading multiple files"""
        import concurrent.futures as _cf

        if self.display_progress:
            l = len(all_files)
            start = time.time()
            _print_progress_bar(0, l, prefix=" ", suffix=" ", length=30, unit="files")

        with _cf.ThreadPoolExecutor(self.max_worker_threads) as executor:
            futures = [executor.submit(self.upload_file, arg) for arg in all_files]

            if self.display_progress:
                current = 0
                for future in _cf.as_completed(futures):
                    res = future.result()
                    current += 1
                    end = time.time()
                    elapsed_time = str(datetime.timedelta(seconds=end - start))
                    time_comp = ":" if current == l else ">"
                    _print_progress_bar(
                        current,
                        l,
                        prefix=f"{os.path.basename(res)} uploaded",
                        suffix=f"[time elapsed{time_comp} {elapsed_time}]",
                        length=30,
                        unit="files",
                    )

        if len(self.primary_files) == 1:
            self.url_list[:] = self.primary_files
        return self.url_list, self.mosaic_data_info


def _upload_imagery_agol(
    files,
    gis=None,
    direct_access_url=None,
    auto_renew=True,
    upload_properties=None,
    single_image=False,
    raster_type=None,
    task=None,
):
    """uploads imagery to user's rasterstore on AGOL and returns the list of urls"""

    try:
        from azure.storage.blob import ContainerClient
        from azure.core.exceptions import (
            ClientAuthenticationError,
            ServiceResponseError,
            ServiceRequestError,
        )
    except:
        _LOGGER.warning(
            "Install Azure library packages for Python."
            + "(Azure SDK for Python - azure-storage-blob: 12.1<= version <=12.17)"
            + "\n(https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package)"
        )
    gis = _arcgis.env.active_gis if gis is None else gis
    if direct_access_url is None:
        sas_url = _generate_direct_access_url(gis)
    else:
        sas_url = direct_access_url
    container = ContainerClient.from_container_url(sas_url)
    if not isinstance(files, list):
        files = [files]

    file_list = []
    time_list = []
    allowed_extensions = _ra_upload_allowed_extensions()

    is_data_for_md = False
    if (isinstance(raster_type, str)) and raster_type == "mosaic_dataset":
        is_data_for_md = True
        current_time = int(time.time())

    all_files = True
    for file in files:
        to_upload = True
        file_dict = {}
        if not is_data_for_md:
            current_time = int(time.time())
            while current_time in time_list:
                current_time += 1
            time_list.append(current_time)
        file_dict["prefix"] = "_images/" + str(current_time) + "/"
        file_dict["file_name"] = file
        file_dict["single_image"] = single_image
        if is_data_for_md:
            file_dict["data_for_md"] = True
        else:
            file_dict["data_for_md"] = False
        if os.path.exists(file):
            if os.path.isdir(file):
                all_files = False
                file_dict["is_dir"] = True
                file_dict["basename_len"] = len(os.path.dirname(file))
                if not ".gdb" in file:
                    file_dict["files_list"] = [
                        os.path.join(root, f)
                        for root, d_names, f_names in os.walk(file)
                        for f in f_names
                        if os.path.splitext(f)[1][1:].lower() in allowed_extensions
                    ]
                else:
                    file_dict["files_list"] = [
                        os.path.join(root, f)
                        for root, d_names, f_names in os.walk(file)
                        for f in f_names
                    ]

                if len(file_dict["files_list"]) == 0:
                    to_upload = False
            else:
                file_dict["is_dir"] = False
                if os.path.splitext(file)[1][1:].lower() in allowed_extensions:
                    file_dict["files_list"] = [file]
                else:
                    to_upload = False

            if to_upload:
                file_list.append(file_dict)
    if len(file_list) == 0:
        raise RuntimeError("No supported files to upload")
    if all_files:
        for file in file_list:
            file["prefix"] = file_list[0]["prefix"]

    uploader = _ImageryUploaderAGOL(
        file_list,
        container,
        auto_renew,
        upload_properties,
        task,
        raster_type,
        gis,
    )
    mosaic_data_info = []
    url_list, mosaic_data_info = uploader.upload_all_files()
    if is_data_for_md:
        return url_list, mosaic_data_info
    return url_list


def _upload_imagery_enterprise(files, raster_type_name=None, gis=None):
    """uploads a file to the image layer to enterprise and returns the item id"""

    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    url = "%s/uploads/upload" % ra_url
    params = {"f": "json"}

    if not isinstance(files, list):
        files = [files]

    item_ids_list = []
    res = {}

    append_path = False
    for file in files:
        item_id_dict = {}
        if os.path.exists(file):
            if os.path.isdir(file):
                if file.endswith(".crf") or raster_type_name != "Raster Dataset":
                    append_path = True
                elif not file.endswith(".crf") or raster_type_name == "Raster Dataset":
                    for dir_ele in [x[0] for x in os.walk(file)]:
                        if dir_ele.endswith(".crf"):
                            append_path = True  # case when parent of the crf folder is specified and raster type is specified as Raster Dataset, we need to append path
                folder = os.path.basename(file)
                basename_len = len(os.path.dirname(file))
                for root, d_names, f_names in os.walk(file):
                    for f in f_names:
                        fp = os.path.join(root, f)
                        path = ("/" + root + "/" + f)[basename_len + 1 :].replace(
                            os.sep, "/"
                        )
                        item_id = None
                        try:
                            item_id = _upload(path=fp, gis=gis)
                        except Exception as e:
                            if "(Error Code: 403)" in str(e):
                                pass
                            else:
                                _LOGGER.warning("file: " + str(fp) + " " + str(e))

                        if item_id is not None:
                            if append_path:
                                item_id_dict = {
                                    "itemId": item_id,
                                    "path": path,
                                }
                                item_ids_list.append(item_id_dict)
                                item_id_dict = {}
                            else:
                                item_ids_list.append(item_id)

            else:
                files_param = {"file": file}
                item_id = None
                try:
                    item_id = _upload(path=file, gis=gis)
                except Exception as e:
                    _LOGGER.warning("file: " + str(file) + " " + str(e))
                if item_id is not None:
                    item_ids_list.append(item_id)

    return item_ids_list

    # ----------------------------------------------------------------------


def _upload(path, description=None, gis=None):
    """
    The ``upload`` method uploads a new item to the server.

    .. note::
        Once the operation is completed successfully, item id of the uploaded item is returned.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    path                Optional string. Filepath of the file to upload.
    ---------------     --------------------------------------------------------------------
    description         Optional string. Descriptive text for the uploaded item.
    ===============     ====================================================================

    :return: Item id of uploaded item

    """
    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    if (os.path.getsize(path)) < 1000000000:
        url = ra_url + "/uploads/upload"
        params = {
            "f": "json",
            "filename": os.path.basename(path),
            "overwrite": True,
        }
        files = {}
        files["file"] = path
        if description:
            params["description"] = description
        res = gis._con.post(path=url, postdata=params, files=files, timeout=None)
        if "error" in res:
            raise Exception(res)
        else:
            return res["item"]["itemID"]
    else:
        file_path = path
        item_id = _register_upload(file_path, gis=gis)
        _upload_by_parts(item_id, file_path, gis=gis)
        return _commit_upload(item_id, gis=gis)


# ----------------------------------------------------------------------
def _register_upload(file_path, gis=None):
    """returns the itemid for the upload by parts logic"""
    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    r_url = "%s/uploads/register" % ra_url
    params = {"f": "json", "itemName": os.path.basename(file_path)}
    reg_res = gis._con.post(r_url, params, timeout=None)
    if "item" in reg_res and "itemID" in reg_res["item"]:
        return reg_res["item"]["itemID"]
    return None


# ----------------------------------------------------------------------
def _upload_by_parts(item_id, file_path, gis=None):
    """loads a file for attachmens by parts"""
    import mmap, tempfile

    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    b_url = "%s/uploads/%s" % (ra_url, item_id)
    upload_part_url = "%s/uploadPart" % b_url
    params = {"f": "json"}
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = 100000000
        steps = int(os.fstat(f.fileno()).st_size / size)
        if os.fstat(f.fileno()).st_size % size > 0:
            steps += 1
        for i in range(steps):
            files = {}
            tempFile = os.path.join(tempfile.gettempdir(), "split.part%s" % i)
            if os.path.isfile(tempFile):
                os.remove(tempFile)
            with open(tempFile, "wb") as writer:
                writer.write(mm.read(size))
                writer.flush()
                writer.close()
            del writer
            files["file"] = tempFile
            params["partId"] = i + 1
            res = gis._con.post(
                upload_part_url, postdata=params, files=files, timeout=None
            )
            if "error" in res:
                raise Exception(res)
            os.remove(tempFile)
            del files
        del mm
    return True


# ----------------------------------------------------------------------
def _commit_upload(item_id, gis=None):
    """commits an upload by parts upload"""

    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    b_url = "%s/uploads/%s" % (ra_url, item_id)
    commit_part_url = "%s/commit" % b_url
    params = {"f": "json", "parts": _uploaded_parts(itemid=item_id, gis=gis)}
    res = gis._con.post(commit_part_url, params, timeout=None)
    if "error" in res:
        raise Exception(res)
    else:
        return res["item"]["itemID"]


# ----------------------------------------------------------------------
def _uploaded_parts(itemid, gis=None):
    """
    returns the parts uploaded for a given item

    ==================   ==============================================
    Arguments           Description
    ------------------   ----------------------------------------------
    itemid               required string. Id of the uploaded by parts item.
    ==================   ==============================================

    """
    ra_url = gis.properties.helperServices["rasterAnalytics"]["url"]
    url = ra_url + "/uploads/%s/parts" % itemid
    params = {"f": "json"}
    res = gis._con.get(url, params)
    return ",".join(res["parts"])


def _get_extent(extdict=None):
    """
    This method is used to convert the JSON presentation of extent (with spatial reference)
    to arcpy.Extent object, so that it can be set to the GP environment.
    :param context: context parameter contains output spatial reference info
    :return geometry object and geometry coordinate
    """
    try:
        import arcpy
    except:
        return None, None
    outext = arcpy.Extent
    extsr = ""
    try:
        if extdict is None:
            return outext, extsr
        # Note: creating geometry directly from envelope JSON gave me a _passthrough
        # which does not provide a extent object.
        if (
            "xmin" in extdict
            and "xmax" in extdict
            and "ymin" in extdict
            and "ymax" in extdict
        ):
            xmin = extdict["xmin"]
            ymin = extdict["ymin"]
            xmax = extdict["xmax"]
            ymax = extdict["ymax"]
            extjson = {
                "rings": [
                    [
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                    ]
                ]
            }
            if "spatialReference" in extdict:
                srdict = extdict["spatialReference"]
                extjson.update({"spatialReference": srdict})
                extsr = srdict

            polygon = arcpy.AsShape(extjson, True)
            outext = polygon.extent
        return outext, extsr
    except:
        return outext, extsr


def _get_stac_api_search_items(
    api_search_endpoint, query, request_method, request_params, get_all_items
):
    """
    This method is used to retrieve all the STAC Items from a search query.
    :param api_search_endpoint: URL of the STAC API (/search) endpoint. The STAC API where
                                the search needs to be performed.
    :param query: The GET/POST request query dictionary that can be used to query a
                  STAC API's search endpoint.
    :param request_method: The HTTP request method used with the STAC API for
                           making the search ("GET" or "POST").
    :param request params: requests.get()/post() method parameters used for
                           the STAC API search request (specified in dictionary format).
    :param get_all_items: Boolean speciying whether to return all the items (retrieving
                          them from all the pages) or just the items from the first
                          page of matches.
    :return list (of STAC Item dictionaries)
    """

    all_items = []
    more_items = True

    while more_items:
        if request_method.upper() == "GET":
            data = _requests.get(api_search_endpoint, params=query, **request_params)
        else:
            data = _requests.post(api_search_endpoint, json=query, **request_params)

        if data.status_code != 200 or data.headers.get("content-type") not in [
            "application/json",
            "application/geo+json",
            "application/json;charset=utf-8",
            "application/geo+json; charset=utf-8",
        ]:
            raise RuntimeError(
                f"Invalid Response: Please verify that the specified query is correct-\n{data.text}"
            )

        json_data = data.json()
        if "type" not in json_data or json_data["type"] != "FeatureCollection":
            raise RuntimeError(
                f"Invalid JSON Response from the STAC API: Please verify that the specified query is correct-\n{json_data}"
            )

        json_data = data.json()
        items = json_data["features"]
        if not get_all_items:
            return items
        all_items.extend(items)
        next_request = next(
            (link for link in json_data["links"] if link["rel"] == "next"), None
        )
        if next_request:
            query = (
                urlparse(next_request["href"]).query
                if request_method.upper() == "GET"
                else dict(query, **next_request["body"])
            )
        else:
            more_items = False

    return all_items


def _get_stac_metadata_file(item):
    """
    This method is used to retrieve the metadata file of a valid STAC item.
    :param item: input STAC Item (JSON dictionary)
    :return string (URL of the STAC Item metadata file)
    """

    planetary_computer_map = {
        **dict.fromkeys(
            [
                "3dep-seamless",
                "3dep-lidar-dsm",
                "cop-dem-glo-30",
                "cop-dem-glo-90",
                "3dep-lidar-hag",
                "3dep-lidar-intensity",
                "3dep-lidar-pointsourceid",
                "noaa-c-cap",
                "3dep-lidar-returns",
                "3dep-lidar-dtm-native",
                "3dep-lidar-classification",
                "3dep-lidar-dtm",
                "gap",
                "alos-dem",
                "io-lulc",
                "drcog-lulc",
                "chesapeake-lc-7",
                "chesapeake-lc-13",
                "chesapeake-lu",
                "io-lulc-9-class",
                "io-biodiversity",
                "ecmwf-forecast",
            ],
            "data",
        ),
        **dict.fromkeys(
            [
                "sentinel-1-rtc",
                "hgb",
                "gnatsgo-rasters",
                "mobi",
                "chloris-biomass",
                "jrc-gsw",
                "hrea",
                "noaa-nclimgrid-monthly",
                "usda-cdl",
                "esa-cci-lc",
                "noaa-climate-normals-gridded",
                "noaa-cdr-sea-surface-temperature-whoi",
                "noaa-cdr-ocean-heat-content",
                "esa-worldcover",
            ],
            "All COGs",
        ),
        **dict.fromkeys(
            [
                "daymet-annual-pr",
                "daymet-daily-hi",
                "gridmet",
                "daymet-annual-na",
                "daymet-monthly-na",
                "daymet-annual-hi",
                "daymet-monthly-hi",
                "daymet-monthly-pr",
                "terraclimate",
                "daymet-daily-pr",
                "daymet-daily-na",
            ],
            "zarr-https",
        ),
        **dict.fromkeys(
            [
                "sentinel-1-grd",
                "sentinel-3-olci-wfr-l2-netcdf",
                "sentinel-3-synergy-v10-l2-netcdf",
                "sentinel-3-olci-lfr-l2-netcdf",
                "sentinel-3-slstr-lst-l2-netcdf",
                "sentinel-3-slstr-wst-l2-netcdf",
                "sentinel-3-synergy-syn-l2-netcdf",
                "sentinel-3-synergy-vgp-l2-netcdf",
                "sentinel-3-synergy-vg1-l2-netcdf",
            ],
            "safe-manifest",
        ),
        **dict.fromkeys(
            [
                "esa-cci-lc-netcdf",
                "noaa-climate-normals-netcdf",
                "noaa-cdr-sea-surface-temperature-whoi-netcdf",
                "noaa-cdr-ocean-heat-content-netcdf",
            ],
            "netcdf",
        ),
        **dict.fromkeys(
            [
                "noaa-mrms-qpe-24h-pass2",
                "noaa-mrms-qpe-1h-pass1",
                "noaa-mrms-qpe-1h-pass2",
            ],
            "cog",
        ),
        **dict.fromkeys(["landsat-c2-l2", "landsat-c2-l1"], "mtl.txt"),
        **dict.fromkeys(["sentinel-2-l2a"], "product-metadata"),
        **dict.fromkeys(["mtbs"], "burn-severity"),
        **dict.fromkeys(["alos-fnf-mosaic"], "C"),
        **dict.fromkeys(["nrcan-landcover"], "landcover"),
        **dict.fromkeys(["nasadem"], "elevation"),
        **dict.fromkeys(["naip"], "image"),
    }

    earth_search_map = {
        **dict.fromkeys(["sentinel-s2-l2a-cogs", "sentinel-2-l2a"], 1),
        **dict.fromkeys(
            ["sentinel-s2-l2a", "sentinel-s2-l1c", "sentinel-2-l1c"],
            ("visual", "productInfo.json"),
        ),
        **dict.fromkeys(["naip"], "image"),
        **dict.fromkeys(["landsat-c2-l2"], "mtl.txt"),
        **dict.fromkeys(["sentinel-1-grd"], "safe-manifest"),
        **dict.fromkeys(["cop-dem-glo-30", "cop-dem-glo-90"], "data"),
    }

    sentinel_hub_map = {
        **dict.fromkeys(["sentinel-2"], ("data", "productInfo.json")),
        **dict.fromkeys(["sentinel-1"], ("s3", "manifest.safe")),
    }

    product_file_map = {
        "planetarycomputer.microsoft.com/api/stac": planetary_computer_map,
        "earth-search.aws.element84.com": earth_search_map,
        "services.sentinel-hub.com/api": sentinel_hub_map,
    }

    stacs = list(product_file_map.keys())

    self_link = next(
        (link["href"] for link in item["links"] if link["rel"] == "self"), None
    )

    if self_link is None:
        return

    item_stac = next((stac for stac in stacs if stac in self_link), None)

    if item_stac is None:
        return

    collection_id = (
        item["collection"]
        if "collection" in item
        else (
            item["properties"]["constellation"] if item_stac == stacs[2] else item["id"]
        )
    )

    target = product_file_map[item_stac].get(collection_id)

    href = None
    if isinstance(target, str):
        href = (
            [
                cog["href"]
                for cog in item["assets"].values()
                if cog["href"].endswith((".tif", ".tiff"))
            ]
            if target == "All COGs"
            else item["assets"][target]["href"]
        )
    elif isinstance(target, int):
        href = item["links"][target]["href"]
    elif isinstance(target, tuple):
        directory = os.path.dirname(item["assets"][target[0]]["href"])
        if collection_id == "sentinel-s2-l2a":
            directory = os.path.dirname(directory)
        href = f"{directory}/{target[1]}"

    href = (
        rf"/vsis3{href[4:]}"
        if href is not None and isinstance(href, str) and href.startswith("s3")
        else href
    )

    if collection_id.startswith(("sentinel-2", "sentinel-s2", "landsat")):
        href = rf"{href}\Multiband"

    return href


def _get_stac_links(stac_json, rel):
    """
    This method is used to retrieve all the links matching the specified relation type from a STAC Item or Catalog.
    :param stac_json: input STAC Item or Catalog (JSON dictionary).
    :param rel: relationship type used to filter  the links.
    :return list (of URLs matching the rel filter)
    """
    if "links" not in stac_json:
        raise RuntimeError(f"Invalid STAC Item/Catalog-\n{stac_json}")
    links = stac_json["links"]
    rel_links = [l for l in links if l["rel"] == rel]
    link_hrefs = [l["href"] for l in rel_links]

    all_links = []
    for l in link_hrefs:
        if l.startswith("http"):
            link = l
        else:
            link = urljoin(os.path.dirname(stac_json["links"][0]["href"]) + "/", l)
        all_links.append(link)
    return all_links


def _get_all_stac_catalog_items(stac_json, request_params={}):
    """
    This method is used to get all items from a STAC catalog and all its subcatalogs. Will traverse any subcatalogs recursively.
    :param stac_json: input Static STAC (Catalog - JSON dictionary)
    :param request_params: requests.get() method parameters used for the STAC Item and Catalog requests (passed through the RasterCollection.from_stac_catalog() method call).
    :return generator (of all items retrived in the Catalog)
    """
    for item_link in _get_stac_links(stac_json, "item"):
        item_res = _requests.get(item_link, **request_params)
        if item_res.status_code != 200 or item_res.headers.get("content-type") not in [
            "application/json",
            "application/geo+json",
            "application/json;charset=utf-8",
        ]:
            raise RuntimeError(f"Invalid STAC Item-\n{item_res.text}")
        item_json = item_res.json()
        yield item_json

    children = _get_stac_links(stac_json, "child")
    for child in children:
        child_res = _requests.get(child, **request_params)
        if child_res.status_code != 200 or child_res.headers.get(
            "content-type"
        ) not in [
            "application/json",
            "application/geo+json",
            "application/json;charset=utf-8",
        ]:
            raise RuntimeError(f"Invalid STAC Catalog-\n{child_res.text}")
        child_json = child_res.json()
        yield from _get_all_stac_catalog_items(child_json, request_params)
