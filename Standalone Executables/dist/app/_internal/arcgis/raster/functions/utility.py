from .._layer import ImageryLayer, Raster, RasterCollection, _ArcpyRasterCollection
from arcgis.gis import Item
import numbers
from arcgis.features.layer import FeatureLayer
from arcgis.auth import (
    EsriUserTokenAuth,
    EsriOAuth2Auth,
    EsriNotebookAuth,
    EsriAPIKeyAuth,
    ArcGISProAuth,
    EsriBuiltInAuth,
    EsriGenTokenAuth,
)
import requests

_aggregating_functions = [
    "max",
    "min",
    "med",
    "mean",
    "majority",
    "sum",
    "std",
    "variety",
    "geometric_median",
    "minority",
    "merge_rasters",
]


def _raster_input(raster, raster2=None):
    layer = None

    # if input is a rastercollection, get the list of rasters and use that as the input

    if isinstance(raster, RasterCollection) or isinstance(raster2, RasterCollection):
        import inspect

        fn_name_l2 = inspect.stack()[2][3]
        fn_name_l1 = inspect.stack()[1][3]
        if (
            fn_name_l2 != "to_multidimensional_raster"
            and fn_name_l1 != "_simple_collection"
        ):
            if (
                not fn_name_l2 in _aggregating_functions
                and not fn_name_l1 in _aggregating_functions
            ):
                raise RuntimeError(
                    "RasterCollection object cannot be specified as input to non aggregating functions"
                )
            if isinstance(raster, RasterCollection):
                if (
                    hasattr(raster, "_ras_coll_engine")
                ) and raster._ras_coll_engine != _ArcpyRasterCollection:
                    raster = raster._ras_coll_engine_obj._rasters_list
                else:
                    if hasattr(raster, "_ras_coll_engine_obj"):
                        return (
                            raster._ras_coll_engine_obj,
                            raster._ras_coll_engine_obj,
                            raster._ras_coll_engine_obj,
                        )
                    else:
                        return raster, raster, raster

            if (
                isinstance(raster2, RasterCollection)
            ) and raster2._ras_coll_engine != _ArcpyRasterCollection:
                if (
                    hasattr(raster2, "_ras_coll_engine")
                ) and raster2._ras_coll_engine != _ArcpyRasterCollection:
                    raster2 = raster2._ras_coll_engine_obj._rasters_list
                else:
                    if hasattr(raster2, "_ras_coll_engine_obj"):
                        return (
                            raster2._ras_coll_engine_obj,
                            raster2._ras_coll_engine_obj,
                            raster2._ras_coll_engine_obj,
                        )
                    else:
                        return raster2, raster2, raster2

    if isinstance(raster, Raster):
        if hasattr(raster, "_engine_obj"):
            raster = raster._engine_obj
    if isinstance(raster2, Raster):
        if hasattr(raster2, "_engine_obj"):
            raster2 = raster2._engine_obj
    if isinstance(raster, list):
        for index, ele in enumerate(raster):
            if isinstance(ele, Raster):
                if hasattr(ele, "_engine_obj"):
                    raster[index] = ele._engine_obj
    if isinstance(raster2, list):
        for index, ele in enumerate(raster2):
            if isinstance(ele, Raster):
                if hasattr(ele, "_engine_obj"):
                    raster2[index] = ele._engine_obj

    if raster2 is not None:
        if isinstance(raster2, (ImageryLayer, Raster)) and isinstance(
            raster, (ImageryLayer, Raster)
        ):
            if raster2._rendering_rule_from_item:
                raster2._fn = None
                raster2._fnra = None
            layer = raster2
            raster_ra = _get_raster_ra(raster2)
            if raster._datastore_raster and raster2._datastore_raster:
                if raster2._fn is not None:
                    if raster._uri == raster2._uri:
                        raster2 = raster2._fn
                    else:
                        raster2 = _replace_raster_url(raster2._fn, raster2._uri)
                else:
                    if raster2._uri == raster._uri:
                        oids = raster2.filtered_rasters()
                        if oids is None:
                            raster2 = "$$"
                        elif len(oids) == 1:
                            raster2 = "$" + str(oids[0])
                        else:
                            raster2 = ["$" + str(x) for x in oids]
                    else:
                        raster2 = raster2._uri
            else:
                if raster2._fn is not None:
                    if raster._url == raster2._url:
                        raster2 = raster2._fn
                    else:
                        if raster2._datastore_raster is False:
                            try:
                                url = raster2._url
                                if (
                                    (hasattr(raster2, "_lazy_token"))
                                    and raster2._lazy_token is None
                                ) or not hasattr(raster2, "_lazy_token"):
                                    raster2._lazy_token = _generate_layer_token(
                                        raster2, url
                                    )
                                if isinstance(raster2._lazy_token, str):
                                    url = url + "?token=" + raster2._lazy_token
                                raster2 = _replace_raster_url(raster2._fn, url)
                            except:
                                if "url" in raster2._lyr_dict:
                                    url = raster2._lyr_dict["url"]
                                    if "serviceToken" in raster2._lyr_dict:
                                        url = (
                                            url
                                            + "?token="
                                            + raster2._lyr_dict["serviceToken"]
                                        )
                                    raster2 = _replace_raster_url(raster2._fn, url)
                                else:
                                    raster2 = _replace_raster_url(
                                        raster2._fn, raster2._url
                                    )
                        else:
                            raster2 = _replace_raster_url(raster2._fn, raster2._uri)
                else:
                    if raster2._url == raster._url:
                        oids = raster2.filtered_rasters()
                        if oids is None:
                            raster2 = "$$"
                        elif len(oids) == 1:
                            raster2 = "$" + str(oids[0])
                        else:
                            raster2 = ["$" + str(x) for x in oids]
                    else:
                        if raster2._datastore_raster is False:
                            try:
                                url = raster2._url
                                if (
                                    (hasattr(raster2, "_lazy_token"))
                                    and raster2._lazy_token is None
                                ) or not hasattr(raster2, "_lazy_token"):
                                    raster2._lazy_token = _generate_layer_token(
                                        raster2, url
                                    )

                                if isinstance(raster2._lazy_token, str):
                                    url = url + "?token=" + raster2._lazy_token
                                raster2 = url
                            except:
                                if "url" in raster2._lyr_dict:
                                    url = raster2._lyr_dict["url"]
                                    if "serviceToken" in raster2._lyr_dict:
                                        url = (
                                            url
                                            + "?token="
                                            + raster2._lyr_dict["serviceToken"]
                                        )
                                    raster2 = url
                                else:
                                    raster2 = raster2._url
                        else:
                            raster2 = raster2._uri
        elif isinstance(raster2, (ImageryLayer, Raster)) and not isinstance(
            raster, (ImageryLayer, Raster)
        ):
            layer = raster2
            raster_ra = _get_raster_ra(raster2)
            raster2 = _get_raster(raster2)

        elif isinstance(raster2, list):
            mix_and_match = False  # mixing rasters from two image services
            # try:
            #     r0 = raster2[0]
            #     r1 = raster2[1]
            #     if r0._fn is None and r1._fn is None and r0._url != r1._url:
            #         mix_and_match = True
            # except:
            #     pass

            for r in raster2:  # layer is first non numeric raster in list
                if not isinstance(r, numbers.Number):
                    layer = r
                    break

            for r in raster2:
                if r._datastore_raster:
                    if not isinstance(r, numbers.Number):
                        if r._uri != layer._uri:
                            mix_and_match = True
                else:
                    if not isinstance(r, numbers.Number):
                        if r._url != layer._url:
                            mix_and_match = True

            raster_ra = [_get_raster_ra(r) for r in raster2]
            if mix_and_match:
                raster2 = [_get_raster_url(r, layer) for r in raster2]
            else:
                raster2 = [_get_raster(r) for r in raster2]
        else:  # secondinput maybe scalar for arithmetic functions, or a chained raster fn
            layer = None
            # raster = raster
            raster_ra = raster2
        return layer, raster2, raster_ra

    if isinstance(raster, (ImageryLayer, Raster)):
        if raster._rendering_rule_from_item:
            raster._fn = None
            raster._fnra = None
        layer = raster
        raster_ra = _get_raster_ra(raster)
        raster = _get_raster(raster)

    elif isinstance(raster, list):
        mix_and_match = False  # mixing rasters from two image services
        # try:
        #     r0 = raster[0]
        #     r1 = raster[1]
        #     if r0._fn is None and r1._fn is None and r0._url != r1._url:
        #         mix_and_match = True
        # except:
        #     pass

        for r in raster:
            if isinstance(r, (ImageryLayer, Raster)):
                if r._datastore_raster:
                    layer = r
        if layer is None:
            for r in raster:  # layer is first non numeric raster in list
                if not isinstance(r, numbers.Number):
                    layer = r
                    break

        for r in raster:
            if not isinstance(r, numbers.Number):
                if r._datastore_raster:
                    if r._uri != layer._uri:
                        mix_and_match = True
                else:
                    if r._url != layer._url:
                        mix_and_match = True

        raster_ra = [_get_raster_ra(r) for r in raster]
        if mix_and_match:
            raster = [_get_raster_url(r, layer) for r in raster]
        else:
            raster = [_get_raster(r) for r in raster]
    else:  # maybe scalar for arithmetic functions, or a chained raster fn
        layer = None
        # raster = raster
        raster_ra = raster

    return layer, raster, raster_ra


def _get_raster(raster):
    if isinstance(raster, (ImageryLayer, Raster)):
        if raster._fn is not None:
            raster = raster._fn
        else:
            oids = raster.filtered_rasters()
            if oids is None and raster.mosaic_rule is not None:
                raster = _get_raster_ra(raster)
            elif oids is None:
                raster = "$$"
            elif len(oids) == 1:
                raster = "$" + str(oids[0])
            else:
                raster = ["$" + str(x) for x in oids]
    return raster


def _replace_raster_url(obj, url=None):
    # replace all "Raster" : '$$' with url
    if isinstance(obj, dict):
        value = {k: _replace_raster_url(v, url) for k, v in obj.items()}
    elif isinstance(obj, list):
        value = [_replace_raster_url(elem, url) for elem in obj]
    else:
        value = obj

    if value == "$$":
        return url
    elif isinstance(value, str) and len(value) > 0 and value[0] == "$":
        return url + "/" + value.replace("$", "")
    else:
        return value


def _get_raster_url(raster, layer):
    if isinstance(raster, (ImageryLayer, Raster)):
        if raster._fn is not None:
            if raster._datastore_raster and layer._datastore_raster:
                if raster._uri == layer._uri or isinstance(raster._uri, bytes):
                    raster = raster._fn
                else:
                    raster = _replace_raster_url(raster._fn, raster._uri)

            else:
                if raster._url == layer._url:
                    raster = raster._fn
                else:
                    try:
                        url = raster._url
                        if (
                            (hasattr(raster, "_lazy_token"))
                            and raster._lazy_token is None
                        ) or not hasattr(raster, "_lazy_token"):
                            raster._lazy_token = _generate_layer_token(raster, url)
                        if isinstance(raster._lazy_token, str):
                            url = url + "?token=" + raster._lazy_token
                        raster = _replace_raster_url(raster._fn, url)
                    except:
                        if "url" in raster._lyr_dict:
                            url = raster._lyr_dict["url"]
                            if "serviceToken" in raster._lyr_dict:
                                url = url + "?token=" + raster._lyr_dict["serviceToken"]
                            raster = _replace_raster_url(raster._fn, url)
                        else:
                            raster = _replace_raster_url(raster._fn, raster._url)

        else:
            if raster._datastore_raster and layer._datastore_raster:
                if raster._uri == layer._uri:
                    raster = "$$"
                else:
                    raster = raster._uri
            else:
                if raster._url == layer._url:
                    raster = "$$"
                else:
                    try:
                        url = raster._url
                        if (
                            (hasattr(raster, "_lazy_token"))
                            and raster._lazy_token is None
                        ) or not hasattr(raster, "_lazy_token"):
                            raster._lazy_token = _generate_layer_token(raster, url)
                        if isinstance(raster._lazy_token, str):
                            url = url + "?token=" + raster._lazy_token
                        raster = url
                    except:
                        if "url" in raster._lyr_dict:
                            url = raster._lyr_dict["url"]
                            if "serviceToken" in raster._lyr_dict:
                                url = url + "?token=" + raster._lyr_dict["serviceToken"]
                            raster = url
                        else:
                            raster = raster._url

            # oids = raster.filtered_rasters()
            # if oids is None:
            #     raster = '$$'
            # elif len(oids) == 1:
            #     raster = '$' + str(oids[0])
            # else:
            #     raster = ['$' + str(x) for x in oids]
    return raster


def _get_raster_ra(raster):
    if isinstance(raster, (ImageryLayer, Raster)):
        try:
            url = raster._url
            if (
                (hasattr(raster, "_lazy_token")) and raster._lazy_token is None
            ) or not hasattr(raster, "_lazy_token"):
                raster._lazy_token = _generate_layer_token(raster, url)
            if isinstance(raster._lazy_token, str):
                url = url + "?token=" + raster._lazy_token
        except:
            if "url" in raster._lyr_dict:
                url = raster._lyr_dict["url"]
            if "serviceToken" in raster._lyr_dict:
                url = url + "?token=" + raster._lyr_dict["serviceToken"]
        if raster._fnra is not None:
            raster_ra = raster._fnra
        else:
            raster_ra = {}
            if raster._mosaic_rule is not None:
                if raster._datastore_raster:
                    raster_ra["uri"] = raster._uri
                else:
                    raster_ra["url"] = url
                raster_ra["mosaicRule"] = raster._mosaic_rule
            else:
                if raster._datastore_raster:
                    raster_ra = raster._uri

                else:
                    raster_ra = url

            # if raster._mosaic_rule is not None:
            #    raster_ra['mosaicRule'] = raster._mosaic_rule
    elif isinstance(raster, Item):
        raise RuntimeError(
            "Item not supported as input. Use ImageryLayer - e.g. item.layers[0]"
        )
        # raster_ra = {
        #    'itemId': raster.itemid
        # }
    else:
        raster_ra = raster

    return raster_ra


def _raster_input_rft(raster, raster2=None):
    if isinstance(raster, (ImageryLayer, Raster)) or isinstance(raster, FeatureLayer):
        raster_ra = _get_raster_ra_rft(raster)

    elif isinstance(raster, list):
        raster_ra = [_get_raster_ra_rft(r) for r in raster]

    else:  # maybe scalar for arithmetic functions, or a chained raster fn
        raster_ra = raster

    return raster_ra


def _get_raster_ra_rft(raster):
    if isinstance(raster, Raster):
        if hasattr(raster, "_engine_obj"):
            raster = raster._engine_obj
    if isinstance(raster, (ImageryLayer, Raster)):
        if raster._rendering_rule_from_item:
            raster._fn = None
            raster._fnra = None
        try:
            url = raster._url
            if (
                (hasattr(raster, "_lazy_token")) and raster._lazy_token is None
            ) or not hasattr(raster, "_lazy_token"):
                raster._lazy_token = _generate_layer_token(raster, url)
            if isinstance(raster._lazy_token, str):
                url = url + "?token=" + raster._lazy_token
        except:
            if "url" in raster._lyr_dict:
                url = raster._lyr_dict["url"]
            if "serviceToken" in raster._lyr_dict:
                url = url + "?token=" + raster._lyr_dict["serviceToken"]
        if raster._fnra is not None:
            raster_ra = raster._fnra
        else:
            raster_ra = {}
            if raster._mosaic_rule is not None:
                if raster._datastore_raster:
                    raster_ra["uri"] = raster._uri
                else:
                    raster_ra["url"] = url
                raster_ra["mosaicRule"] = raster._mosaic_rule
            else:
                if raster._datastore_raster:
                    raster_ra = raster._uri
                else:
                    raster_ra = url

    elif isinstance(raster, FeatureLayer):
        raster_ra = raster._url
        # if raster._mosaic_rule is not None:
        #    raster_ra['mosaicRule'] = raster._mosaic_rule
    elif isinstance(raster, Item):
        raise RuntimeError(
            "Item not supported as input. Use ImageryLayer - e.g. item.layers[0]"
        )
    else:
        raster_ra = raster

    return raster_ra


def _input_rft(input_layer):
    if isinstance(input_layer, dict):
        input_param = input_layer

    elif isinstance(input_layer, str):
        if (
            "/fileShares/" in input_layer
            or "/rasterStores/" in input_layer
            or "/cloudStores/" in input_layer
            or ("http" not in input_layer and "https" not in input_layer)
        ):
            input_param = {"uri": input_layer}
        else:
            input_param = {"url": input_layer}

    elif isinstance(input_layer, list):
        input_param = []
        for il in input_layer:
            if isinstance(il, str):
                if (
                    "/fileShares/" in il
                    or "/rasterStores/" in il
                    or "/cloudStores/" in il
                ):
                    input_param.append({"uri": il})
                else:
                    input_param.append({"url": il})
            elif isinstance(il, dict):
                input_param.append(il)
    else:
        input_param = input_layer
    return input_param


def _find_object_ref(rft_dict, record, instance):
    for k, v in rft_dict.items():
        if isinstance(v, dict):
            if "_object_id" in v:
                record[v["_object_id"]] = v
            if "isPublic" in v:
                if v["isPublic"] is True:
                    instance._is_public_flag = True
            _find_object_ref(v, record, instance)

        elif isinstance(v, list):
            for ele in v:
                if isinstance(ele, dict):
                    if "_object_id" in ele:
                        record[ele["_object_id"]] = ele
                    if "isPublic" in ele:
                        if ele["isPublic"] is True:
                            instance._is_public_flag = True
                    _find_object_ref(ele, record, instance)
    return _replace_object_id(rft_dict, record)


def _replace_object_id(rft_dict, record):
    if isinstance(rft_dict, dict):
        if "_object_ref_id" in rft_dict.keys():
            ref_value = record[rft_dict["_object_ref_id"]]
            rft_dict = ref_value
            return _replace_object_id(rft_dict, record)
        else:
            for k, v in rft_dict.items():
                if isinstance(v, dict):
                    rft_dict[k] = _replace_object_id(v, record)

                elif isinstance(v, list):
                    for n, ele in enumerate(v):
                        if isinstance(ele, dict):
                            v[n] = _replace_object_id(ele, record)
    return rft_dict


def _python_variable_name(var):
    var = "".join(e for e in var if e.isalnum() or e == "_")
    return var


def _pixel_type_string_to_long(pixel_type):
    pixel_type = pixel_type.upper()
    if pixel_type == "U1":
        return 0
    elif pixel_type == "U2":
        return 1
    elif pixel_type == "U4":
        return 2
    elif pixel_type == "U8":
        return 3
    elif pixel_type == "S8":
        return 4
    elif pixel_type == "U16":
        return 5
    elif pixel_type == "S16":
        return 6
    elif pixel_type == "U32":
        return 7
    elif pixel_type == "S32":
        return 8
    elif pixel_type == "F32":
        return 9
    elif pixel_type == "F64":
        return 10
    elif pixel_type == "C16":
        return 13
    elif pixel_type == "C32":
        return 14
    elif pixel_type == "C64":
        return 11
    elif pixel_type == "C128":
        return 12
    elif pixel_type == "D8":
        return 15
    elif pixel_type == "D10":
        return 16
    elif pixel_type == "D12":
        return 17
    elif pixel_type == "D16":
        return 18
    elif pixel_type == "D20":
        return 19
    elif pixel_type == "D24":
        return 20
    elif pixel_type == "D32":
        return 21
    elif pixel_type == "D36":
        return 22
    else:
        return -1


def _generate_layer_token(layer, url):
    token = layer._gis._con._create_token(url)
    if token:
        if isinstance(
            layer._gis._con._session.auth,
            (
                EsriUserTokenAuth,
                EsriOAuth2Auth,
                EsriNotebookAuth,
                EsriAPIKeyAuth,
                ArcGISProAuth,
                EsriBuiltInAuth,
                EsriGenTokenAuth,
            ),
        ):
            temp_r = requests.get(
                url, {"f": "json", "token": token}, verify=layer._gis._verify_cert
            )
            if temp_r.status_code != 200:
                return None
            elif (
                temp_r.status_code == 200
                and temp_r.text.lower().find("invalid token") > -1
            ):
                return None
            else:
                return token
        else:
            return token
    else:
        return None


def _set_multidimensional_rules(function_chain=None, function_chain_ra=None):
    from ... import env

    match_variables = env.match_variables
    union_dimension = env.union_dimension

    if (match_variables is not None) and isinstance(match_variables, bool):
        if (
            function_chain is not None
        ) and "MatchVariable" not in function_chain.keys():
            function_chain["rasterFunctionArguments"]["MatchVariable"] = match_variables
        if (
            function_chain_ra is not None
        ) and "MatchVariable" not in function_chain_ra.keys():
            function_chain_ra["rasterFunctionArguments"][
                "MatchVariable"
            ] = match_variables
    if (union_dimension is not None) and isinstance(union_dimension, bool):
        if (
            function_chain is not None
        ) and "UnionDimension" not in function_chain.keys():
            function_chain["rasterFunctionArguments"][
                "UnionDimension"
            ] = union_dimension
        if (
            function_chain_ra is not None
        ) and "UnionDimension" not in function_chain.keys():
            function_chain_ra["rasterFunctionArguments"][
                "UnionDimension"
            ] = union_dimension


def _get_dimension_names(lyr):
    dim_list = []
    if isinstance(lyr, Raster):
        if hasattr(lyr, "_engine_obj"):
            lyr = lyr._engine_obj
    md_info = lyr.multidimensional_info
    if md_info:
        for ele in md_info["multidimensionalInfo"]["variables"]:
            for ele_dim in ele["dimensions"]:
                dim_list.append(ele_dim["name"])
    return dim_list
