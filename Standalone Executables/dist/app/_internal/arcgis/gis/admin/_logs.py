"""
Allows access to the Portal Logs
"""
from datetime import datetime, timezone
from typing import Optional, Union
from .. import GIS
from ._base import BasePortalAdmin


########################################################################
class Logs(BasePortalAdmin):
    """
    Logs are records written by various components of the portal. You can
    query the logs, clean the logs, and edit log settings.

    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    gis               required GIS, portal connection object
    ----------------  ---------------------------------------------------------------
    url               required string, web address of the log resource
    ================  ===============================================================

    """

    _gis = None
    _url = None
    _con = None
    _portal = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        if isinstance(gis, GIS):
            self._url = url
            self._gis = gis
            self._portal = gis._portal
            self._con = gis._con
        else:
            raise ValueError("gis object must be of type GIS")

    # ----------------------------------------------------------------------
    def clean(self):
        """
        Deletes all the log files on the machine hosting Portal for ArcGIS.
        This operation allows you to free up disk space. The logs cannot be
        recovered after executing this operation.

        .. code-block:: python

            USAGE: Clean logs from your Portal Admin API

            from arcgis.gis import GIS
            gis = GIS("https://yourportal.com/portal", "portaladmin", "password")
            logs = gis.admin.logs
            resp = logs.clean()
            print(resp)

            # Output
            True

        :return:
            Boolean True or False depicting success

        """
        url = "%s/clean" % self._url
        params = {"f": "json"}
        res = self._con.post(path=url, postdata=params)
        if isinstance(res, dict) and "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    @property
    def settings(self):
        """
        Get/Set the current log settings for the portal.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        value             required dictionary, the dictionary of the log settings
        ================  ===============================================================

        .. code-block:: python

            USAGE: Print out the Log Settings

            from arcgis.gis import GIS
            gis = GIS("https://yourportal.com/portal", "portaladmin", "password")
            logs = gis.admin.logs
            logsettings = logs.settings
            for key, value in dict(logsettings).items():
                print("{} : {}".format(key, value))

            # Output
            logDir : C:\\arcgisportal\\logs
            logLevel : INFO
            maxErrorReportsCount : 10
            maxLogFileAge : 90
            usageMeteringEnabled : False

        :return:
            Dictionary of key/value pairs of log settings

        """
        url = "%s/settings" % self._url
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value: dict):
        """
        See main ``settings`` property docstring.
        """
        url = "%s/settings/edit" % self._url
        params = {"f": "json"}
        if isinstance(value, dict):
            for k, v in value.items():
                params[k] = v
        else:
            raise ValueError("Value must be a dictionary")
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def query(
        self,
        start_time: Union[datetime, float],
        end_time: Optional[Union[datetime, float]] = None,
        level: str = "WARNING",
        query_filter: Union[str, dict] = "*",
        page_size: int = 1000,
    ):
        """
        The query operation allows you to aggregate, filter, and page
        through logs written by the portal.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        start_time        required datetime/float. The most recent time to query.

                          Local date corresponding to the POSIX timestamp, such as is
                          returned by time.time(). This may raise OverflowError, if the
                          timestamp is out of the range of values supported by the
                          platform. It's common for this to be restricted to years from
                          1970 through 2038.
                          Time can be specified as a portal timestamp (format in
                          "%Y-%m-%dT%H:%M:%S") or in seconds since UNIX epoch. For
                          :Examples:
                          Datetime Object: datetime.datetime.now()
                          Timestamp: "2015-08-01T15:17:20,123"
                          Seconds: 1312237040.123/time.time()
                          Default: datetime.datetime.now()
        ----------------  ---------------------------------------------------------------
        end_time          optional datetime/float, The oldest time to include in the
                          result set. You can use this to limit the query to the last
                          number of minutes, hours, days, months, and years as needed.

                          Local date corresponding to the POSIX timestamp, such as is
                          returned by time.time(). This may raise OverflowError, if the
                          timestamp is out of the range of values supported by the
                          platform. It's common for this to be restricted to years from
                          1970 through 2038.

                          Datetime Object: datetime.datetime.now()
                          Timestamp: "2015-08-01T15:17:20,123"
                          Seconds: 1312237040.123/time.time()
                          Default: datetime.datetime.now()
        ----------------  ---------------------------------------------------------------
        level             optional string, Can be one of [OFF, SEVERE, WARNING, INFO,
                          FINE, VERBOSE, DEBUG]. Returns only records with a log level at
                          or more severe than the level specified.
                          Default: WARNING
        ----------------  ---------------------------------------------------------------
        query_filter      optional dict, Filtering is allowed by any combination of
                          codes, users, and source components. The filter accepts a comma
                          delimited list of filter definitions. If any definition is
                          omitted, it defaults to all ("*").
                          :Example:

                          {"codes":[204000-205999,212015,219114], "users":["admin","jcho"],
                          "source": ["PORTAL ADMIN"]}

                          The source of logged events are generated from the sharing,
                          administrative, and portal components of the software.
                          For example:
                           - Events related to publishing and users are categorized under
                             SHARING.
                           - Events related to security and indexing are categorized under
                             PORTAL ADMIN.
                           - Events related to installing the software are categorized
                             under PORTAL.
        ----------------  ---------------------------------------------------------------
        page_size         optional integer, the number of log records to return. The
                          default is 1000
        ================  ===============================================================

        :return:
           dictionary of messages
        """
        from datetime import datetime

        url = "%s/query" % self._url
        if isinstance(start_time, datetime):
            start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(start_time, str):
            try:
                datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            except:
                raise Exception(
                    "Invalid start_time string, must be in the format YYYY-MM-DDTHH:MM:SS"
                )
        elif isinstance(start_time, tuple(list(int) + [float])):
            start_time = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        if end_time is None:
            end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(end_time, datetime):
            end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(end_time, str):
            try:
                datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
            except:
                raise Exception(
                    "Invalid end_time string, must be in the format YYYY-MM-DDTHH:MM:SS"
                )
        elif isinstance(end_time, tuple(list(int) + [float])):
            end_time = datetime.fromtimestamp(end_time, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        if query_filter == "*":
            query_filter = {"codes": [], "users": [], "source": "*"}
        params = {
            "startTime": start_time,
            "endTime": end_time,
            "level": level,
            "f": "json",
            "filterType": "json",
            "pageSize": page_size,
        }
        if query_filter:
            params["filter"] = query_filter
        return self._con.get(path=url, params=params)
