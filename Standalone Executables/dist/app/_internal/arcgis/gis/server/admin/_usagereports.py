"""
This resource is a collection of all the usage reports created within
your site. The Create Usage Report operation lets you define a new
usage report.
"""
from __future__ import absolute_import
from __future__ import print_function
import json
from .._common import BaseServer
from urllib.parse import quote
from arcgis.gis import GIS
from typing import Optional


########################################################################
class ReportManager(BaseServer):
    """
    A utility class for managing usage reports for ArcGIS Server.

    """

    _con = None
    _json_dict = None
    _url = None
    _json = None
    _metrics = None
    _reports = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(ReportManager, self).__init__(url=url, gis=gis)
        if url.lower().endswith("/usagereports"):
            self._url = url
        else:
            self._url = url + "/usagereports"
        self._con = gis
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def list(self) -> list:
        """
        Retrieves a list of reports on the server.

        :return:
            A list of reports found.

        """
        if self.properties is None:
            self._init()
        self._reports = []
        if isinstance(self.properties["metrics"], list):
            for r in self.properties["metrics"]:
                url = f"{self._url}/{quote(str(r['reportname']))}"
                self._reports.append(Report(url=url, gis=self._con))
                del url
        return self._reports

    # ----------------------------------------------------------------------
    @property
    def settings(self) -> dict:
        """
        Gets the current usage reports settings. The usage reports
        settings are applied to the entire site. When usage
        reports are enabled, service usage statistics are collected and
        persisted to a statistics database. When usage reports are
        disabled, the statistics are not collected. The interval
        parameter defines the duration (in minutes) during which the usage
        statistics are sampled or aggregated (in-memory) before being
        written out to the statistics database. Database entries are
        deleted after the interval specified in the max_history parameter (
        in days), unless the max_history parameter is 0, for which the
        statistics are persisted forever.
        """
        params = {"f": "json"}
        url = self._url + "/settings"
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def edit(
        self,
        interval: str,
        enabled: bool = True,
        max_history: int = 0,
    ) -> dict:
        """
        Edits the usage reports settings that are applied to the entire site.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        interval               Required string. Defines the duration (in minutes) for which the
                               usage statistics are aggregated or sampled, in-memory, before being
                               written out to the statistics database.
        ------------------     --------------------------------------------------------------------
        enabled                Optional string. When usage reports are enabled, service
                               usage statistics are collected and persisted to a statistics
                               database. When usage reports are disabled, the statistics are not
                               collected.  The default is True (enabled).
        ------------------     --------------------------------------------------------------------
        max_history            Optional integer. The number of days after which usage statistics
                               are deleted from the statistics database. If the max_history
                               parameter is set to 0 (the default value), the statistics are
                               persisted forever.
        ==================     ====================================================================


        :return:
            A JSON message indicating success.

        """
        params = {
            "f": "json",
            "maxHistory": max_history,
            "enabled": enabled,
            "samplingInterval": interval,
        }
        url = self._url + "/settings/edit"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def create(
        self,
        reportname: str,
        queries: list,
        metadata: Optional[str] = None,
        since: str = "LAST_DAY",
        from_value: Optional[int] = None,
        to_value: Optional[int] = None,
        aggregation_interval: Optional[str] = None,
    ) -> dict:
        """
        Creates a new usage report. A usage report is created by submitting
        a JSON representation of the usage report to this operation.
        See `CreateUsageReport <https://developers.arcgis.com/rest/enterprise-administration/server/createusagereport.htm>`_
        for details on the REST request bundled by this method.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        reportname               Required string. The unique name of the report.
        --------------------     --------------------------------------------------------------------
        queries                  Required list of Python dictionaries for which to generate the
                                 report. Each dictionary has two keys: ``resourceURIs`` and
                                 ``metrics``

                                 .. code-block:: python

                                     # Usage Example

                                     [{"resourceURIs": ["/services/Folder_name/",
                                                        "Forest_loss.FeatureServer"],
                                       "metrics": ["RequestCount,RequestsFailed"]}]

                                 Each key's corresponding value is a list of strings specifying
                                 a resource for which to gather metrics, or the metrics to
                                 gather, respectively.

                                 - ``resourceURIs`` --
                                     Comma-separated list that specifies the services or folders
                                     for which to gather metrics, formatted as below:

                                     - ``services/`` -
                                       Entire Site
                                     - ``services/Folder/`` -
                                       Folder within a Site. Reports metrics aggregated across all
                                       services within that folder and any sub-folders.
                                     - ``services/Folder/ServiceName.ServiceType`` -
                                       Service in a specified folder.
                                        - services/Folder_name/Map_bv_999.MapServer
                                     - ``service`` -
                                       If in the root folder
                                        - Map_bv_999.MapServer

                                 - ``metrics`` --
                                     Comma-separated string of specific measures to gather.

                                     - ``RequestCount`` —
                                       the number of requests received
                                     - ``RequestsFailed`` —
                                       the number of requests that failed
                                     - ``RequestsTimedOut`` —
                                       the number of requests that timed out
                                     - ``RequestMaxResponseTime`` —
                                       the maximum response time
                                     - ``RequestAvgResponseTime`` —
                                       the average response time
                                     - ``ServiceActiveInstances`` —
                                       the maximum number of active (running) service instances sampled at 1 minute
                                       intervals for a specified service
                                     - ``ServiceRunningInstancesMax`` — the maximum number of active (running) service
                                       instances, sampled at one-minute intervals for a specified service. If you
                                       include this metric, it must be the only metric included in the report.
        --------------------     --------------------------------------------------------------------
        metadata                 Optional string. Any JSON object representing presentation tier
                                 data for the usage report, such as report title, colors,
                                 line-styles, etc. Also used to denote visibility in ArcGIS Server
                                 Manager for reports created with the Administrator Directory. To
                                 make any report created in the Administrator Directory visible to
                                 Manager, include *"managerReport":true* in the metadata JSON object.
                                 When this value is not set (default), reports are not visible in
                                 Manager. This behavior can be extended to any client that wants to
                                 interact with the Administrator Directory. Any user-created value
                                 will need to be processed by the client.
        --------------------     --------------------------------------------------------------------
        since                    Optional string. The time duration of the report. The supported
                                 values are: LAST_DAY, LAST_WEEK, LAST_MONTH, LAST_YEAR, CUSTOM

                                 - ``LAST_DAY`` represents a time range spanning the previous 24 hours.
                                   This is the default value.
                                 - ``LAST_WEEK`` represents a time range spanning the previous 7 days.
                                 - ``LAST_MONTH`` represents a time range spanning the previous 30 days.
                                 - ``LAST_YEAR`` represents a time range spanning the previous 365 days.
                                 - ``CUSTOM`` represents a time range that is specified using the from
                                   and to parameters.
        --------------------     --------------------------------------------------------------------
        from_value               Optional integer. Only valid when ``since`` is CUSTOM. The timestamp
                                 in milliseconds (since January 1, 1970, 00:00:00 GMT, the Unix epoch)
                                 for the beginning period of the report.

                                 .. code-block:: python

                                    # usage Example:

                                    import datetime as dt

                                    >>> sept1_2020 = int(dt.datetime(2020, 9, 1).timestamp()) * 1000
                                        sept1_2020

                                        1598943600000
        --------------------     --------------------------------------------------------------------
        to_value                 Optional integer. Only valid when ``since`` is CUSTOM. The timestamp
                                 in milliseconds (since January 1, 1970, 00:00:00 GMT, the Unix epoch)
                                 for the ending period of the report.

                                 .. code-block:: python

                                    # usage Example:

                                    import datetime as dt

                                    now = int(dt.datetime.now().timestamp()) * 1000
        --------------------     --------------------------------------------------------------------
        aggregation_interval     Optional string. The aggregation interval in minutes. Server metrics
                                 are aggregated and returned for time slices aggregated using the
                                 specified aggregation interval. The time range for the report,
                                 specified using the *since* parameter (and *from_value* and
                                 *to_value* when since is CUSTOM) is split into multiple slices, each
                                 covering an aggregation interval. Server metrics are then aggregated
                                 for each time slice and returned as data points in the report data.
                                 When the aggregation_interval is not specified, the following defaults
                                 are used:

                                   - ``LAST_DAY``: 30 minutes
                                   - ``LAST_WEEK``: 4 hours
                                   - ``LAST_MONTH``: 24 hours
                                   - ``LAST_YEAR``: 1 week
                                   - ``CUSTOM``: 30 minutes up to 1 day, 4 hours up to 1 week, 1
                                   day up to 30 days, and 1 week for longer periods.

                                 If the interval specified in Usage Reports Settings is more than
                                 the aggregationInterval, the interval is used instead.
        ====================     ====================================================================


        :return:
            A :class:`~arcgis.gis.server.Report` object.


        .. code-block:: python

            USAGE EXAMPLE:

            import datetime as dt
            from arcgis.gis import GIS

            >>> gis = GIS(profile="your_ent_profile", verify_cert=False)

            >>> gis_servers = gis.admin.servers.list()

            >>> gis_server = gis_servers[1]

            >>> now = int(dt.datetime.now().timestamp()) * 1000
            >>> sept1_2020 = int(dt.datetime(2020, 9, 1).timestamp()) * 1000

            >>> query_obj = [{"resourceURIs": ["services/Map_bv_999.MapServer"],
                              "metrics": ["RequestCount"]}]

            >>> r = gis_server.usage.create(reportname="SampleReport",
                                            queries=query_obj,
                                            metadata="This could be any String or JSON Object.",
                                            since="CUSTOM",
                                            from_value=sept1_2020,
                                            to_value=now)
            >>> r

                <Report at https://server_url:6443/arcgis/admin/usagereports/SampleReport>
        """
        url = self._url + "/add"
        temp = False
        params = {
            "reportname": reportname,
            "since": since,
        }
        if not metadata:
            params["metadata"] = {
                "temp": temp,
                "title": reportname,
                "managerReport": False,
            }
        else:
            params["metadata"] = metadata
        if isinstance(queries, dict):
            params["queries"] = [queries]
        elif isinstance(queries, list):
            params["queries"] = queries
        if aggregation_interval:
            params["aggregationInterval"] = aggregation_interval
        if since.lower() == "custom":
            params["to"] = to_value
            params["from"] = from_value
        p = {"f": "json", "usagereport": params}
        res = self._con.post(path=url, postdata=p)
        #  Refresh the metrics object
        self._init()
        for report in self.list():
            if str(report.reportname).lower() == reportname.lower():
                return report
        return res

    # ----------------------------------------------------------------------
    def quick_report(
        self,
        since: str = "LAST_WEEK",
        queries: str = "services/",
        metrics: str = "RequestsFailed",
    ) -> dict:
        """
        Generates an on the fly usage report for a service, services, or folder.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        since                    Optional string. The time duration of the report. The supported
                                 values are: LAST_DAY, LAST_WEEK, LAST_MONTH, or LAST_YEAR.

                                 - ``LAST_DAY`` represents a time range spanning the previous 24 hours.
                                   This is the default value.
                                 - ``LAST_WEEK`` represents a time range spanning the previous 7 days.
                                 - ``LAST_MONTH`` represents a time range spanning the previous 30 days.
                                 - ``LAST_YEAR`` represents a time range spanning the previous 365 days.
        --------------------     --------------------------------------------------------------------
        queries                  Required string. A string of resourceURIs for which to generate the report.
                                 Specified as a comma-separated sting of services or folders for which to
                                 gather metrics.

                                    - ``services/`` -- Entire Site
                                    - ``services/Folder/`` -- Folder within a Site. Reports metrics
                                      aggregated across all services within that Folder and Sub-Folders.
                                    - ``services/Folder/ServiceName.ServiceType`` -- Service in a
                                      specified folder, for example:
                                         - services/Folder_Name/Map_bv_999.MapServer
                                         - services/Fodler_Name/ServiceName.ServiceType
                                    - ``root folder`` -- Service in the root folder
                                         - Map_bv_999.MapServer.

                                 .. code-block:: python

                                     queries="services/Hydroligic_Data/Lake_algae.FeatureServer,services/Mountains"
        --------------------     --------------------------------------------------------------------
        metrics                  Optional string. Comma separated list of metrics to be reported.

                                 Supported metrics are:

                                    - RequestCount -- the number of requests received
                                    - RequestsFailed -- the number of requests that failed
                                    - RequestsTimedOut -- the number of requests that timed out
                                    - RequestMaxResponseTime -- the maximum response time
                                    - RequestAvgResponseTime -- the average response time
                                    - ServiceActiveInstances -- the maximum number of active
                                      (running) service instances sampled at 1 minute intervals,
                                      for a specified service

                                 .. code-block:: python

                                     metrics="RequestCount,RequestsFailed"
        ====================     ====================================================================

        :return:
            A Python dictionary of data on a successful query.

        .. code-block:: python

           # Usage Example:

           >>> gis = GIS(profile="my_own_portal", verify_cert=False)

           >>> gis_servers = gis.admin.servers.list()

           >>> srv = gis_servers[0]

           >>> query_string = "services/Forests/Forests_degraded_2000.MapServer,services/Lakes/Lakes_drought_levels.MapServer"
           >>> qk_report = srv.usage.quick_report(since = "LAST_MONTH",
                                                  queries = query_string,
                                                  metrics = "RequestCount,RequestsFailed")

           >>> qk_report

               {'report': {'reportname': '1fa828eb31664485ae5c25c76c86e28d',
                           'metadata': '{"temp":true,"title":"1fa828eb31664485ae5c25c76c86e28d","managerReport":false}',
                           'time-slices': [1598914800000, 1599001200000, 1599087600000, ... 1601420400000],
                           'report-data': [[{'resourceURI': 'services/Forests/Forests_degraded_2000.MapServer',
                                             'metric-type': 'RequestCount', 'data': [None, 17, 928, ... 20]},
                                            {'resourceURI': 'services/Forests/Forests_degraded_2000.MapServer',
                                             'metric-type': 'RequestsFailed', 'data': [None, 225, None, ... 0]},
                                            {'resourceURI': 'services/Lakes/Lakes_drought_levels.MapServer',
                                             'metric-type': 'RequestCount', 'data': [0, 0, 7, ... 71]},
                                            {'resourceURI': 'services/Lakes/Lakes_drought_levels.MapServer',
                                             'metric-type': 'RequestsFailed', 'data': [None, None, 1 ... , 0]}]]}}
        """
        from uuid import uuid4

        queries = {"resourceURIs": queries.split(","), "metrics": metrics.split(",")}
        reportname = uuid4().hex
        metadata = {
            "temp": True,
            "title": reportname,
            "managerReport": False,
        }
        res = self.create(
            reportname=reportname, queries=queries, since=since, metadata=metadata
        )
        if isinstance(res, Report):
            data = res.query()
            res.delete()
            return data
        return res


########################################################################
class Report(BaseServer):
    """
    **(This class should not be created by a user)**

    A utility class representing a single usage report returned by ArcGIS Server.

    A Usage Report is used to obtain ArcGIS Server usage data for specified
    resources during a given time period. It specifies the parameters for
    obtaining server usage data, time range (parameters since, from_value, to_value),
    aggregation interval, and queries (which specify the metrics to be
    gathered for a collection of server resources, such as folders and
    services).
    """

    _con = None
    _url = None
    _json = None
    _reportname = None
    _since = None
    _from = None
    _to = None
    _aggregationInterval = None
    _queries = None
    _metadata = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False):
        """
        Constructor

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        url                    Required string. The machine URL.
        ------------------     --------------------------------------------------------------------
        gis                    Optional string. The GIS or Server object.
        ------------------     --------------------------------------------------------------------
        initialize             Optional string. Denotes whether to load the machine properties at
                               creation (True). Default is False.
        ==================     ====================================================================

        """
        super(Report, self).__init__(url=url, gis=gis)
        self._con = gis
        self._url = url
        if initialize:
            self._init()

    # ----------------------------------------------------------------------
    def edit(self) -> dict:
        """
        Edits the usage report. To edit a usage report, submit
        the complete JSON representation of the usage report which
        includes updates to the usage report properties. The name of the
        report cannot be changed when editing the usage report.

        Values are changed in the class, to edit a property like
        metrics, pass in a new value.

        :return:
            A JSON indicating success.

        """

        usagereport_dict = {
            "reportname": self._reportname,
            "queries": self._queries,
            "since": self._since,
            "metadata": self._metadata,
            "to": self._to,
            "from": self._from,
            "aggregationInterval": self._aggregationInterval,
        }
        params = {"f": "json", "usagereport": json.dumps(usagereport_dict)}
        url = self._url + "/edit"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def delete(self) -> dict:
        """
        Deletes this usage report.

        :return:
            A JSON indicating success.
        """
        url = self._url + "/delete"
        params = {
            "f": "json",
        }
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def query(self, query_filter: Optional[str] = None) -> dict:
        """
        Retrieves server usage data for this report. This operation
        aggregates and filters server usage statistics for the entire
        ArcGIS Server site. The report data is aggregated in a time slice,
        which is obtained by dividing up the time duration by the default
        (or specified) aggregationInterval parameter in the report. Each
        time slice is represented by a timestamp, which represents the
        ending period of that time slice.

        In the JSON response, the queried data is returned for each metric-
        resource URI combination in a query. In the report-data section,
        the queried data is represented as an array of numerical values. A
        response of null indicates that data is not available or requests
        were not logged for that metric in the corresponding time-slice.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        query_filter           Optional string. The report data can be filtered by the machine
                               where the data is generated. The filter accepts a comma-separated
                               list of machine names; * represents all machines.
        ==================     ====================================================================


        :return:
            A JSON containing the server usage data.


        .. code-block:: python

            USAGE EXAMPLE 1: Filters for the specified machines

            {"machines": ["WIN-85VQ4T2LR5N", "WIN-239486728937"]}

        .. code-block:: python

            USAGE EXAMPLE 2: No filtering, all machines are accepted

            {"machines": "*"}

        """
        if query_filter is None:
            query_filter = {"machines": "*"}
        params = {"f": "json", "filter": query_filter, "filterType": "json"}
        url = self._url + "/data"
        return self._con.get(path=url, params=params)
