import os, csv
from typing import Optional
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
import datetime as _dt


########################################################################
class LogManager(object):
    """
    Logs are the records written by the various components of Notebook server.
    You can query the logs and change various log settings. An object of this
    class can be created using :attr:`~arcgis.gis.nb.NotebookServer.logs` property of the
    :class:`~arcgis.gis.nb.NotebookServer` class
    """

    _url = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< LogManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< LogManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def clean(self):
        """
        Deletes all the log files on all server machines in the site. This is an irreversible
        operation.

        This operation forces the server to clean the logs, which has the effect of freeing
        up disk space. However, it is not required that you invoke this operation because
        the server periodically purges old logs.

        :return: Boolean

        """
        params = {
            "f": "json",
        }
        url = "{}/clean".format(self._url)
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def settings(self):
        """
        Get/set the current log settings.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        value                  Required dict. A dictionary with the key/values pairs to modify settings.
        ==================     ====================================================================

        :return: PropertyMap
        """
        params = {"f": "json"}
        url = self._url + "/settings"
        try:
            res = self._con.get(url, params)
            return PropertyMap(res)
        except:
            return ""

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value: dict):
        """
        See main ``settings`` property docstring.
        """
        params = {"f": "json"}
        current = dict(self.settings)
        allowed_keys = list(current.keys())
        for k in allowed_keys:
            if k in value:
                params[k] = value[k]
            else:
                params[k] = current[k]
        url = self._url + "/settings/edit"
        self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        since_server_start: bool = False,
        level: str = "WARNING",
        services: str = "*",
        machines: str = "*",
        server: str = "*",
        codes: Optional[str] = None,
        process_IDs: Optional[str] = None,
        export: bool = False,
        export_type: str = "CSV",  # CSV or TAB
        out_path: Optional[str] = None,
    ):
        """
        The query operation on the logs resource provides a way to
        aggregate, filter, and page through logs across the entire site.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        start_time             Optional string/integer. The most recent time to query.  Default is now.
                               Time can be specified in milliseconds since UNIX epoch, or as an
                               ArcGIS Server timestamp.

                               Example for string:
                               "startTime": "2011-08-01T15:17:20"

                               Example for integer:
                               "startTime": 1312237040123
        ------------------     --------------------------------------------------------------------
        end_time               Optional string. The oldest time to include in the result set. You
                               can use this to limit the query to the last n minutes or hours as
                               needed. Default is the beginning of all logging.
        ------------------     --------------------------------------------------------------------
        since_server_start     Optional string. Gets only the records written since the server
                               started (True).  The default is False.
        ------------------     --------------------------------------------------------------------
        level                  Optional string. Gets only the records with a log level at or more
                               severe than the level declared here. Can be one of (in severity
                               order): DEBUG, VERBOSE, FINE, INFO, WARNING, SEVERE. The
                               default is WARNING.
        ------------------     --------------------------------------------------------------------
        services               Optional string. Query records related to a specific service.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        machines               Optional string. Query records related to a specific machine.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        server                 Optional string. Query records related to a specific server.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        codes                  Optional string. Gets only the records with the specified code.
                               The default is all.  See http://server.arcgis.com/en/server/latest/administer/windows/log-codes-overview.htm
        ------------------     --------------------------------------------------------------------
        process_IDs            Optional string. Query by the machine process ID that logged the event.
        ------------------     --------------------------------------------------------------------
        export                 Optional string. Boolean indicating whether to export the query
                               results.  The default is False (don't export).
        ------------------     --------------------------------------------------------------------
        export_type            Optional string. The export file type. CSV or TAB are the choices,
                               CSV is the default.
        ------------------     --------------------------------------------------------------------
        out_path               Optional string. The path to download the log file to.
        ==================     ====================================================================

        :return:
           A JSON of the log items that match the query. If export option is set to True, the
           output log file path is returned.


        """

        if codes is None:
            codes = []
        if process_IDs is None:
            process_IDs = []
        allowed_levels = ("SEVERE", "WARNING", "INFO", "FINE", "VERBOSE", "DEBUG")
        qFilter = {"services": "*", "machines": "*", "server": "*"}
        if len(process_IDs) > 0:
            qFilter["processIds"] = process_IDs
        if len(codes) > 0:
            qFilter["codes"] = codes
        params = {
            "f": "json",
            "sinceServerStart": since_server_start,
            "pageSize": 10000,
        }
        url = "{url}/query".format(url=self._url)
        if start_time is not None and isinstance(start_time, _dt.datetime):
            params["startTime"] = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        if end_time is not None and isinstance(end_time, _dt.datetime):
            params["endTime"] = end_time.strftime("%Y-%m-%dT%H:%M:%S")
        if level.upper() in allowed_levels:
            params["level"] = level
        if server != "*":
            qFilter["server"] = server.split(",")
        if services != "*":
            qFilter["services"] = services.split(",")
        if machines != "*":
            qFilter["machines"] = machines.split(",")
        params["filter"] = qFilter
        if export is True and out_path is not None:
            messages = self._con.get(url, params)
            with open(out_path, mode="wb") as f:
                hasKeys = False
                if export_type == "TAB":
                    csvwriter = csv.writer(f, delimiter="\t")
                else:
                    csvwriter = csv.writer(f)
                for message in messages["logMessages"]:
                    if hasKeys == False:
                        csvwriter.writerow(message.keys())
                        hasKeys = True
                    csvwriter.writerow(message.values())
                    del message
            del messages
            return out_path
        else:
            return self._con.get(url, params)
