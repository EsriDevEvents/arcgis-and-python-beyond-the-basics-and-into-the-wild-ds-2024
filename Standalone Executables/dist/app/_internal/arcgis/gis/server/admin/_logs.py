from __future__ import annotations
from __future__ import absolute_import
from __future__ import print_function
import csv
from datetime import datetime
from .._common import BaseServer
from arcgis.gis import GIS
from typing import Optional


########################################################################
class LogManager(BaseServer):
    """
    Helper class for the management of logs by administrators.

    Logs are the transaction records written by the various components
    of ArcGIS Server.  You can query the logs, change various log settings,
    and check error messages for helping to determine the nature of an issue.

    """

    _url = None
    _con = None
    _json_dict = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object..
        ==================     ====================================================================

        """
        connection = gis
        super(LogManager, self).__init__(gis=gis, url=url)
        self._url = url
        self._con = connection
        if initialize:
            self._init(connection)

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def count_error_reports(self, machine: str = "*") -> dict:
        """
        This operation counts the number of error reports (crash reports) that have been generated
        on each machine.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        machine                Optional string. The name of the machine on which to count the
                               reports. The default will return the count for all machines in a site.
        ==================     ====================================================================

        :return:
           A dictionary with machine name and report count.

        """
        params = {"f": "json", "machine": machine}
        url = self._url + "/countErrorReports"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def clean(self) -> bool:
        """
        Deletes all the log files on all server machines in the site. This is an irreversible
        operation.

        This operation forces the server to clean the logs, which has the effect of freeing
        up disk space. However, it is not required that you invoke this operation because
        the server periodically purges old logs.

        :return:
           A boolean indicating success (True) or failure (False).

        """
        params = {
            "f": "json",
        }
        url = "{}/clean".format(self._url)
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        """Gets the current log settings."""
        params = {"f": "json"}
        url = self._url + "/settings"
        try:
            return self._con.post(path=url, postdata=params)["settings"]
        except:
            return ""

    # ----------------------------------------------------------------------
    def edit(
        self,
        level: str = "WARNING",
        log_dir: Optional[str] = None,
        max_age: int = 90,
        max_report_count: int = 10,
    ) -> dict:
        """
        Provides log editing capabilities for the entire site.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        level                  Optional string. The log level.  Can be one of (in severity order):
                               OFF, DEBUG, VERBOSE, FINE, INFO, WARNING, SEVERE. The default is WARNING.
        ------------------     --------------------------------------------------------------------
        log_dir                Optional string. The file path to the root of the log directory.
        ------------------     --------------------------------------------------------------------
        max_age                Optional integer. The number of days that a server should save a
                               log file. The default is 90.
        ------------------     --------------------------------------------------------------------
        ax_report_count        Optional integer. The maximum number of error report files per
                               machine. The default is 10.
        ==================     ====================================================================


        :return:
           A JSON with the edited settings.

        """
        url = self._url + "/settings/edit"
        allowed_levels = (
            "OFF",
            "SEVERE",
            "WARNING",
            "INFO",
            "FINE",
            "VERBOSE",
            "DEBUG",
        )
        current_settings = self.settings
        current_settings["f"] = "json"

        if level.upper() in allowed_levels:
            current_settings["logLevel"] = level.upper()
        if log_dir is not None:
            current_settings["logDir"] = log_dir
        if max_age is not None and isinstance(max_age, int):
            current_settings["maxLogFileAge"] = max_age
        if (
            max_report_count is not None
            and isinstance(max_report_count, int)
            and max_report_count > 0
        ):
            current_settings["maxErrorReportsCount"] = max_report_count
        return self._con.post(path=url, postdata=current_settings)

    # ----------------------------------------------------------------------
    def query(
        self,
        start_time: int | datetime | None = None,
        end_time: int | datetime | None = None,
        since_server_start: bool = False,
        level: str = "WARNING",
        services: str = "*",
        machines: str = "*",
        server: str = "*",
        codes: str | None = None,
        process_IDs: str | None = None,
        export: bool = False,
        export_type: str = "CSV",
        out_path: str | None = None,
        max_records_return: int = 5000,
    ):
        """
        The query operation on the logs resource provides a way to
        aggregate, filter, and page through logs across the entire site.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        start_time             Optional Integer or datetime. The most recent time to query.  Default is now.
                               Time can be specified in milliseconds since UNIX epoch.


                               Example for integer:
                               start_time = 1312237040123

        ------------------     --------------------------------------------------------------------
        end_time               Optional String or datetime. The oldest time to include in the result set. You
                               can use this to limit the query to the last n minutes or hours as
                               needed.

                               If ```sinceLastStart``` is true, the default is all logs since the
                               server was started.
        ------------------     --------------------------------------------------------------------
        since_server_start     Optional Bool. Gets only the records written since the server
                               started (True).  The default is False.
        ------------------     --------------------------------------------------------------------
        level                  Optional String. Gets only the records with a log level at or more
                               severe than the level declared here. Can be one of (in severity
                               order): DEBUG, VERBOSE, FINE, INFO, WARNING, SEVERE. The
                               default is WARNING.
        ------------------     --------------------------------------------------------------------
        services               Optional String. Query records related to a specific service.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        machines               Optional String. Query records related to a specific machine.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        server                 Optional String. Query records related to a specific server.
                               The default is all.
        ------------------     --------------------------------------------------------------------
        codes                  Optional String. Gets only the records with the specified code.
                               The default is all.  See http://server.arcgis.com/en/server/latest/administer/windows/log-codes-overview.htm
        ------------------     --------------------------------------------------------------------
        process_IDs            Optional String. Query by the machine process ID that logged the event.
        ------------------     --------------------------------------------------------------------
        export                 Optional String. Boolean indicating whether to export the query
                               results.  The default is False (don't export).
        ------------------     --------------------------------------------------------------------
        export_type            Optional String. The export file type. CSV or TAB are the choices,
                               CSV is the default.
        ------------------     --------------------------------------------------------------------
        out_path               Optional String. The path to download the log file to.
        ------------------     --------------------------------------------------------------------
        max_records_return     Optional Int. The maximum amount of records to return. Default is 5000
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
        }
        # set page size
        params["pageSize"] = max_records_return if max_records_return < 5000 else 5000
        # if greater than 5000 then will enter hasMore loop else, correct number returned
        max_records_return -= 5000
        url = "{url}/query".format(url=self._url)
        if start_time is not None and isinstance(start_time, datetime):
            params["startTime"] = int(start_time.timestamp() * 1000)
        elif start_time:
            params["startTime"] = start_time
        else:
            params["startTime"] = int(datetime.now().timestamp() * 1000)
        if end_time is not None and isinstance(end_time, datetime):
            params["endTime"] = int(end_time.timestamp() * 1000)
        elif end_time and isinstance(end_time, int):
            params["endTime"] = end_time
        if level.upper() in allowed_levels:
            params["level"] = level
        if server != "*":
            qFilter["server"] = server.split(",")
        if services != "*":
            qFilter["services"] = services.split(",")
        if machines != "*":
            qFilter["machines"] = machines.split(",")
        params["filter"] = qFilter

        logs = self._con.post(path=url, postdata=params)
        # determine if more logs are available to query
        has_more = logs["hasMore"]

        # If the hasMore member of the response object is true,
        # pass the end time as the startTime parameter
        # for the next request to get the next set of records
        loop = 0
        new_logs = {}
        while max_records_return > 1 and has_more:
            if has_more:
                # get new start time from logs endTime in first loop then from new_logs endTime after
                params["startTime"] = (
                    logs["endTime"] if loop == 0 else new_logs["endTime"]
                )
                loop = 1
                # page size is 5000 or less
                params["pageSize"] = (
                    max_records_return if max_records_return < 5000 else 5000
                )
                # decides if more records needed or not
                max_records_return -= 5000
                # new logs to query
                new_logs = self._con.post(path=url, postdata=params)
                has_more = new_logs["hasMore"]
                # append new log messages to logs to return
                for log_message in new_logs["logMessages"]:
                    logs["logMessages"].append(log_message)
                else:
                    break
        # if export true then no values returned, file written to
        if export is True and out_path is not None:
            with open(file=out_path, mode="w") as f:
                hasKeys = False
                if export_type == "TAB":
                    csvwriter = csv.writer(f, delimiter="\t")
                else:
                    csvwriter = csv.writer(f)
                for message in logs["logMessages"]:
                    if hasKeys == False:
                        csvwriter.writerow(list(message.keys()))
                        hasKeys = True
                    csvwriter.writerow(list(message.values()))
                    del message
            del logs
            return out_path
        else:
            return logs
