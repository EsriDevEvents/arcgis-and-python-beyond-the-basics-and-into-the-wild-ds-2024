import os
import sys
import logging as _logging
import urllib.parse
import concurrent.futures
from types import MethodType
from functools import lru_cache
from arcgis.auth.tools import LazyLoader
from arcgis.gis import GIS
from arcgis import env

arcgis = LazyLoader("arcgis")
_mixins = LazyLoader("arcgis._impl.common._mixins")

from arcgis.geoprocessing._support import _execute_gp_tool
from arcgis.geoprocessing._tool import (
    _camelCase_to_underscore,
    _generate_param,
    _inspect_tool,
)

_log = _logging.getLogger(__name__)


###########################################################################
def _input_string_params(spec, name_type, name_param, num_spaces=20):
    """creates the input strings for the lambda"""
    src_code = ""
    optional_code = ""
    param_inputs = ""
    default_db = {}
    if len(spec) > 0:
        param_name, param_dval = spec[0]
        param_type = name_type[param_name]

        # src_code += _generate_param(name_param, param_dval, param_name, param_type)
        # param_inputs += f"{param_name}={param_name},"

        for param_name_dval in spec[:]:  # [ (param_name, param_dval) ]
            param_name, param_dval = param_name_dval
            param_type = name_type[param_name]
            if param_dval:
                optional_code += f"{param_name}=None,"
                default_db[param_name] = param_dval
            else:
                src_code += f"{param_name}"
            # src_code += _generate_param(name_param, param_dval, param_name, param_type)
            src_code += ","
            if src_code.find(",,") > -1:
                src_code = src_code.replace(",,", ",")
            param_inputs += f"{param_name}={param_name},"
        # src_code += ","
    src_code += ""
    if src_code.endswith(",") and optional_code:
        src_code += optional_code
    elif src_code.endswith(",") == False and optional_code:
        src_code += ","
        src_code += optional_code
    return src_code, param_inputs, default_db


###########################################################################
def _build_lambda(self, input_strings, param_inputs):
    """builds the lambda"""
    if input_strings.startswith(","):
        input_strings = input_strings[1:]

    return eval(
        f"lambda self, {input_strings}: self._run_tool({param_inputs})".replace(
            "\n", ""
        )
        .replace('''"""''', "'")
        .replace(":str", "")
    )


###########################################################################
class GPInfo:
    """
    Provides access to additional information about the GP Service.
    """

    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    @property
    def item_info(self) -> dict:
        """

        :return: the service's item information

        """
        url = f"{self._url}/iteminfo"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    @property
    def metadata(self) -> str:
        """

        :return: the service's metadata

        """
        url = f"{self._url}/metadata"

        return self._gis._con.get(url, {}, try_json=False)

    @property
    def thumbnail(self) -> str:
        """

        :return: the service's thumbnail

        """
        url = f"{self._url}/thumbnail"
        return self._gis._con.get(url, {}, try_json=False)


###########################################################################
class GPTask:
    """
    The GP Task resource represents a single task in a geoprocessing
    service published using ArcGIS Server. It provides basic information
    about the task including its name and display name. It also provides
    detailed information about the various input and output parameters
    exposed by the task.
    """

    _gis = None
    _url = None
    _parent = None
    _properties = None
    _choice_list = None
    _default_db = None

    def __init__(
        self,
        url: str,
        parent: "GPService",
        gis: GIS = None,
    ):
        self._parent = parent
        self._url = url
        if gis is None and env.active_gis is None:
            gis = GIS()
        elif gis is None and env.active_gis:
            gis = env.active_gis
        self._gis = gis
        name = self.properties["name"]
        uses_map_as_result = self._parent.properties.resultMapServerName != ""

        (
            helpstring,
            name_name,
            name_type,
            return_values,
            spec,
            name_param,
            choice_list_db_param,
        ) = _inspect_tool(self.properties, uses_map_as_result)
        param_db = {}
        for param_name_dval in spec:  # [ (param_name, param_dval) ]
            param_name, param_dval = param_name_dval
            param_type = name_type[param_name]
            gp_param_name = name_name[param_name]
            param_db[param_name] = (param_type.__name__, gp_param_name)
        return_values2 = []
        for retval in return_values:
            param_db[retval["name"]] = (retval["type"].__name__, retval["display_name"])
            return_values2.append(
                {
                    "name": retval["name"],
                    "display_name": retval["display_name"],
                    "type": retval["type"].__name__,
                }
            )
        self._return_values = return_values2
        input_string, param_string, self._default_db = _input_string_params(
            spec, name_type, name_param
        )
        self._param_db = param_db
        l = _build_lambda(
            self=self, input_strings=input_string, param_inputs=param_string
        )
        l.__doc__ = helpstring

        setattr(self, _camelCase_to_underscore(name), MethodType(l, self))

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _run_tool(self, **kwargs):
        """runs the tool"""
        future = True
        param_db = self._param_db
        return_values = self._return_values
        run_async = (
            self._parent.properties["executionType"] != "esriExecutionTypeSynchronous"
        )
        for key, value in kwargs.items():
            if key in self._default_db and value is None:
                kwargs[key] = self._default_db[key]

        return _execute_gp_tool(
            self._gis,
            os.path.basename(self._url),
            kwargs,
            param_db,
            return_values,
            run_async,
            os.path.dirname(self._url),
            future=future,
        )

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def properties(self) -> dict:
        """
        Returns the Service's Properties

        :return: Dictionary
        """
        if self._properties is None:
            params = {"f": "json"}
            self._properties = _mixins.PropertyMap(
                self._gis._con.get(self._url, params)
            )
        return self._properties

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def help_url(self) -> str:
        """
        Returns the URL to the documentation for the GP Task.

        :return: string
        """
        return self.properties.get("helpUrl", "")

    # ----------------------------------------------------------------------
    def default_value(self, name: str) -> object:
        """
        Gets a parameter's default value.


        :return: object

        """
        name = _camelCase_to_underscore(name=name)
        for param in self.properties["parameters"]:
            if _camelCase_to_underscore(param["name"]) == name:
                return param["defaultValue"]
        return None

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def name(self) -> str:
        """
        Name of the geoprocessing tasks

        :return: string

        """
        return _camelCase_to_underscore(self.properties["name"])

    # ----------------------------------------------------------------------
    @property
    @lru_cache(maxsize=10)
    def choice_list(self) -> dict:
        """
        Returns a Map of Parameters with Choice Lists.

        :return: Dictionary

        """
        if self._choice_list is None:
            self._choice_list = {}

            for param in self.properties["parameters"]:
                name = _camelCase_to_underscore(name=param["name"])
                cl = param.get("choiceList", None)
                if cl:
                    self._choice_list[name] = cl
        return self._choice_list


###########################################################################
class GPService:
    """
    A geoprocessing service can contain one or more tools that use input
    data from a client application, process it, and return output in the
    form of features, maps, reports, files, or services. These tools are
    first authored and run in ArcGIS Pro or ArcGIS Desktop, typically as
    a custom model or script tools, before being shared to an ArcGIS
    Server.
    """

    _gis = None
    _url = None
    _info = None
    _tasks = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS = None):
        self._url = url
        if gis is None and env.active_gis is None:
            gis = GIS()
        elif gis is None and env.active_gis:
            gis = env.active_gis
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> dict:
        """
        Returns the Service's Properties

        :return: Dictionary
        """
        if self._properties is None:
            params = {"f": "json"}
            self._properties = _mixins.PropertyMap(
                self._gis._con.get(self._url, params)
            )
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def tasks(self) -> list:
        """returns the :class:`GP Tasks <arcgis.geoprocessing.GPTask>`"""
        if self._tasks is None:
            self._tasks = [
                GPTask(
                    url=self._url + urllib.parse.quote(f"/{task}"),
                    gis=self._gis,
                    parent=self,
                )
                for task in self.properties["tasks"]
            ]
        return self._tasks

    # ----------------------------------------------------------------------
    def refresh(self):
        """
        Reloads the Service Information
        """
        self._tasks = None
        self._properties = None

    # ----------------------------------------------------------------------
    @property
    def info(self) -> GPInfo:
        """
        :return:
            :class:`~arcgis.geoprocessing.GPInfo`
        """
        if self._info is None:
            url = f"{self._url}/info"
            self._info = GPInfo(url, self._gis)
        return self._info
