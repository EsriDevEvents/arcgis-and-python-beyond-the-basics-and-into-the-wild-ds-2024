"""
Provides functions to gather usage statistics for Portal/ArcGIS Online
"""
import os
import time
import datetime
from typing import Optional
from .._impl._con import Connection
from ..._impl.common._mixins import PropertyMap
from ..._impl.common._utils import local_time_to_online, timestamp_to_datetime
from ...gis import GIS
from ._base import BasePortalAdmin


########################################################################


class AGOLUsageReports(BasePortalAdmin):
    """
    Simple Usage Reports from ArcGIS Online

    .. note::
        Usage reports can contain users outside your organization.

    """

    _json_dict = {}
    _json = None
    _con = None
    _portal = None
    _gis = None
    _url = None

    # ----------------------------------------------------------------------

    def _init(self, connection=None):
        """loads the properties into the class"""
        self._json_dict = {}
        self._properties = PropertyMap(self._json_dict)

    # ----------------------------------------------------------------------
    def generate_report(
        self,
        focus: str = "org",
        report_type: str = "users",
        title: Optional[str] = None,
        duration: Optional[str] = None,
        start_time: Optional[datetime.datetime] = None,
        notify: bool = False,
        future: bool = True,
    ):
        """
        Generates the reports of the overall usage of the organizations.
        Reports define organization usage metrics for either a weekly or
        monthly time frame.


        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        focus               Optional String. The report type. Currently, only
                            the organization (`org`) report type is supported.
        ---------------     ----------------------------------------------------
        report_type         Required String. The type of report to generate.

                            Values:
                                - 'content'
                                - 'users'
                                - 'activity'
                                - 'credits'
                                - 'serviceUsages'
                                - 'itemUsages'
        ---------------     ----------------------------------------------------
        title               *deprecated* Optional String.  The output report item's title.
        ---------------     ----------------------------------------------------
        duration            Optional String. Specifies the time duration for the
                            reports. This parameter is required when `report_type`
                            is set to `credits`, `activity`, `serviceUsages`, or `itemUsages`.

                            .. note::
                                The `daily` value is only available when `report_type` is
                                set to `activity`.
                                The `yearly` value is only available when `report_type`
                                is set to `itemUsages`.

                            Values:
                                - 'daily'
                                - 'weekly'
                                - 'monthly'
                                - 'quarterly'
                                - 'yearly'
        ---------------     ----------------------------------------------------
        start_time          Optional datetime.datetime. The start time of the
                            time duration. The time format is Unix time with millisecond
                            precision. If `duration = 'weekly'`, the start_time
                            value must be a time on Sunday or Monday GMT.
                            If `duration = 'monthly`, the start_time value must
                            be on the first day of the month.
        ---------------     ----------------------------------------------------
        notify              Optional Boolean. The Job will print a message upon
                            task completion.
        ---------------     ----------------------------------------------------
        future              Optional Boolean. Returns an asynchronous Job when
                            `True`, when `False`, returns an :class:`~arcgis.gis.Item`.
        ===============     ====================================================


        :return: Async Job Object or :class:`~arcgis.gis.Item`

        """
        url = f"{self._gis._portal.resturl}community/users/{self._gis.users.me.username}/report"
        params = {"f": "json", "reportType": focus, "reportSubType": report_type}

        # Perform Checks
        if duration and duration.lower() not in [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
        ]:
            raise ValueError("Invalid `duration` value %s" % duration)
        if duration:
            duration = duration.lower()
            if duration == "daily" and report_type != "activity":
                raise ValueError(
                    "Duration set to 'daily' can only be used with report type 'activity'."
                )
            elif duration == "yearly" and report_type != "itemUsages":
                raise ValueError(
                    "Duration set to 'yearly' can only be used with report type 'itemUsages'."
                )
            params["timeDuration"] = duration
        if (
            report_type in ["credits", "activity", "serviceUsages", "itemUsages"]
            and duration is None
        ):
            raise ValueError(
                "For the report type specified, a duration must also be specified."
            )

        # Assign parameters
        if not start_time is None and isinstance(start_time, datetime.datetime):
            params["startTime"] = local_time_to_online(start_time)
        elif not start_time is None and isinstance(start_time, int):
            params["startTime"] = start_time
        count = 0
        resp = self._con.post(url, params)
        if "itemId" in resp and future:
            from arcgis._impl._async.jobs import ItemStatusJob

            item = self._gis.content.get(resp["itemId"])
            while item is None:
                time.sleep(1)
                item = self._gis.content.get(resp["itemId"])
                if count == 15:
                    raise Exception(
                        "The report cannot be generated, please resubmit the operation."
                    )
                count += 1
            item = self._gis.content.get(resp["itemId"])
            isj = ItemStatusJob(
                item=item, task_name="Generate Report", notify=notify, gis=self._gis
            )
            if future:
                return isj
            return isj.result()
        return resp

    # ----------------------------------------------------------------------
    def credit(
        self,
        start_time: Optional[datetime.datetime] = None,
        time_frame: str = "week",
        export: bool = False,
    ):
        """
        Creates a Panda's dataframe or CSV file reporting on credit consumption
        within an ArcGIS Online organization.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        start_time          optional datetime, the time to step back from.  If
                            None, the current time is used.
        ---------------     ----------------------------------------------------
        time_frame          optional string, is the timeframe report to create.
                            Allowed values: today, week (default), 14days, 30days,
                            60days, 90days, 6months, year
        ---------------     ----------------------------------------------------
        export              optional boolean, if `True`, a csv is generated from
                            the request. If `False`, a Panda's dataframe is
                            returned. Default is `False`
        ===============     ====================================================

        :return:
             string path to csv file or Panda's Dataframe (default) that
             records the total number of credits consumed per:

             * `hour` if ``time_frame`` is `today`
             * `day` if ``time_frame`` is `week`, `7days`, `14days`, `30days`,
               `60days` or `90days`
             * `week` if ``time_frame`` is `6months`
             * `month` if ``time_frame`` is `year`

        .. code-block:: python

            # Usage example

            >>> usage_reporter = gis.admin.usage_reports

            >>> usage_reporter.credit(start_time= jan2_23,
                                      time_frame= "week")

                date	credits
                ________________________________________
                0	2022-12-26 16:00:00	173.1696
                ...
                6	2023-01-01 16:00:00	177.6483

        """
        out_folder = None
        if start_time is None:
            start_time = datetime.datetime.now()
        if isinstance(start_time, datetime.datetime) == False:
            raise ValueError("start_time and end_time must be datetime objects")
        if time_frame.lower() == "today":
            end_time = start_time - datetime.timedelta(days=1)
            period = "1h"
        elif time_frame.lower() in ["7days", "week"]:
            end_time = start_time - datetime.timedelta(days=7)
            period = "1d"
        elif time_frame.lower() == "14days":
            end_time = start_time - datetime.timedelta(days=14)
            period = "1d"
        elif time_frame.lower() in ["month", "30days"]:
            end_time = start_time - datetime.timedelta(days=30)
            period = "1d"
        elif time_frame.lower() == "60days":
            end_time = start_time - datetime.timedelta(days=60)
            period = "1d"
        elif time_frame.lower() == "90days":
            end_time = start_time - datetime.timedelta(days=90)
            period = "1d"
        elif time_frame.lower() == "6months":
            end_time = start_time - datetime.timedelta(days=180)
            period = "1w"
        elif time_frame.lower() == "year":
            end_time = start_time - datetime.timedelta(days=365)
            period = "1m"
        # Convert to timestamps
        end_time = str(int(local_time_to_online(dt=end_time)))
        start_time = str(int(local_time_to_online(dt=start_time)))
        f = "json"
        if export:
            f = "csv"
        if export and (out_folder is None or os.path.isdir(out_folder) == False):
            import tempfile

            out_folder = tempfile.gettempdir()
        params = {
            "f": f,
            "startTime": end_time,
            "endTime": start_time,
            "vars": "credits,num",
            "groupby": "stype,etype",
            "period": period,
        }
        res = self._con.post(path=self._url, postdata=params)
        if export:
            return res
        elif isinstance(res, (dict, PropertyMap)):
            import pandas as pd

            data = res["data"][0]["credits"]
            for row in data:
                if isinstance(row[0], str):
                    row[0] = int(row[0])
                row[0] = timestamp_to_datetime(timestamp=row[0])

            df = pd.DataFrame.from_records(
                data=data, columns=["date", "credits"], coerce_float=True
            )
            df["credits"] = df["credits"].astype(float)
            return df
        return res

    # ----------------------------------------------------------------------
    def users(
        self, start_time: Optional[datetime.datetime] = None, time_frame: str = "week"
    ):
        """
        Creates a credit usage report for resources of an ArcGIS Online
        organization with results aggregated by specific `username` and user's
        organization id.

        .. note::
            Reports can contain users outside your organization.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        start_time          optional datetime, the time to step back from.  If
                            None, the current time is used.
        ---------------     ----------------------------------------------------
        time_frame          optional string, is the timeframe report to create.
                            Allowed values: today, week, 14days, 30days, 60days,
                            90days, 6months, year
        ===============     ====================================================

        :return:
             dictionary reporting the number of credits consumed by users
             through this organization.

             Results are aggregated by:
               * `hour` if ``time_frame`` is `today`
               * `day` if ``time_frame`` is `week`, `7days`, `14days`, `30days`,
                 `60days` or `90days`
               * `week` if ``time_frame`` is `6months`
               * `month` if ``time_frame`` is `year`

        .. code-block:: python

            # Usage Example:

            >>> from arcgis.gis import GIS
            >>> import datetime as dt

            >>> gis = GIS(profile="my_organizational_profile")
            >>> usage_reporter = gis.admin.usage_reports

            >>> jan2_23 = dt.datetime(2023, 1, 2)
            >>> user_usg = usage_reporter.users(start_time = jan2_23,
                                                time_frame = "week")

            >>> list(user_usg.keys())
            ['startTime', 'endTime', 'period', 'data']

            >>> type(user_usg["data"])
            list

            ### The data key's value will be a list of
            ### dictionaries. Each dictionary will have varying keys.
            ### If the dictonary has no userOrgId key, that indicates
            ### a public user account.

            >>> user_usg['data'][1]
            {'username': '<user_name1>',
             'credits': [['1672099200000', '0.0'],
                         ['1672185600000', '0.0'],
                         ...
                         ['1672617600000', '2.0E-4']]}

           >>> user_usg['data'][2]
           {'username': '<user_name2>',
            'userOrgId': 'JXrNeAy8ce1q2b4l'
            'credits': [['1672099200000', '0.0'],
                        ['1672185600000', '0.0'],
                       ...
                        ['1672617600000', '0.0']]}

        """
        out_folder = None
        if start_time is None:
            start_time = datetime.datetime.now()
        if isinstance(start_time, datetime.datetime) == False:
            raise ValueError("start_time and end_time must be datetime objects")
        end_time = start_time - datetime.timedelta(days=1)
        period = "1h"
        if time_frame.lower() in ["7days", "week"]:
            end_time = start_time - datetime.timedelta(days=7)
            period = "1d"
        elif time_frame.lower() == "14days":
            end_time = start_time - datetime.timedelta(days=14)
            period = "1d"
        elif time_frame.lower() == "30days":
            end_time = start_time - datetime.timedelta(days=30)
            period = "1d"
        elif time_frame.lower() == "60days":
            end_time = start_time - datetime.timedelta(days=60)
            period = "1d"
        elif time_frame.lower() == "90days":
            end_time = start_time - datetime.timedelta(days=90)
            period = "1d"
        elif time_frame.lower() == "6months":
            end_time = start_time - datetime.timedelta(days=180)
            period = "1w"
        elif time_frame.lower() == "year":
            end_time = start_time - datetime.timedelta(days=365)
            period = "1m"
        # Convert to timestamps
        end_time = str(int(local_time_to_online(dt=end_time)))
        start_time = str(int(local_time_to_online(dt=start_time)))
        params = {
            "f": "json",
            "startTime": end_time,
            "endTime": start_time,
            "vars": "credits",
            "groupby": "username,userorgid",
            "period": period,
        }
        res = self._con.post(path=self._url, postdata=params)
        return res

    # ----------------------------------------------------------------------
    def applications(
        self, start_time: Optional[datetime.datetime] = None, time_frame: str = "week"
    ):
        """
        Creates a usage report for all registered application logins for a
        given ArcGIS Online organization.

        .. note::
            Output can contain users outside your organization
            that used organization applications

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        start_time          optional datetime, the time to step back from.  If
                            None, the current time is used.
        ---------------     ----------------------------------------------------
        time_frame          optional string, is the timeframe report to create.
                            Allowed values: today, week, 14days, 30days, 60days,
                            90days, 6months, year
        ===============     ====================================================

        :return:
             dictionary with the number of application logins grouped by
             application and username.

             Results aggregated by:

             - `hour` if ``time_frame`` is `today`
             - `day` if ``time_frame`` is `week`, `7days`, `14days`, `30days`,
               `60days` or `90days`
             - `week` if ``time_frame`` is `6months`
             - `month` if ``time_frame`` is `year`

        .. code-block:: python

            # Usage example:

            >>> import datetime as dt
            >>> from arcgis.gis import GIS

            >>> gis = GIS(profile="my_organizational_profile)
            >>> jan2_23 = dt.datetime(2023, 1, 2)

            >>> usage_reporter = gis.admin.usage_reports

            >>> usage_reporter.applications(start_time= jan2_23,
                                            time_frame="week")

            {'startTime': 1672099200000,
            'endTime': 1672704000000,
            'period': '1d',
            'data': [{'etype': 'svcusg',
                    'stype': 'applogin',
                    'username': <username 1>,
                    'userOrgId': 'JXwx ... Ok2o',
                    'appId': 'arcgisnotebooks',
                    'appOrgId': 'Ab3e ... q0o7i',
                    'num': [['1672099200000', '0'],
                            ...
                            ['1672444800000', '4'],
                            ['1672531200000', '3'],
                            ['1672617600000', '0']]},
             ...
             ...
                    {'etype': 'svcusg',
                     'stype': 'applogin',
                     'username': 'external username2',
                     'userOrgId': 'JLxMbZo4ex3kOa2o',
                     'appId': 'arcgisonline',
                     'appOrgId': 'Ab3e ... q0o7i',
                     'num': [['1672099200000', '0'],
                             ...
                             ['1672444800000', '62'],
                             ['1672531200000', '10'],
                             ['1672617600000', '0']]}]}

        """
        out_folder = None
        if start_time is None:
            start_time = datetime.datetime.now()
        if isinstance(start_time, datetime.datetime) == False:
            raise ValueError("start_time and end_time must be datetime objects")
        end_time = start_time - datetime.timedelta(days=1)
        period = "1h"
        if time_frame.lower() in ["7days", "week"]:
            end_time = start_time - datetime.timedelta(days=7)
            period = "1d"
        elif time_frame.lower() == "14days":
            end_time = start_time - datetime.timedelta(days=14)
            period = "1d"
        elif time_frame.lower() == "30days":
            end_time = start_time - datetime.timedelta(days=30)
            period = "1d"
        elif time_frame.lower() == "60days":
            end_time = start_time - datetime.timedelta(days=60)
            period = "1d"
        elif time_frame.lower() == "90days":
            end_time = start_time - datetime.timedelta(days=90)
            period = "1d"
        elif time_frame.lower() == "6months":
            end_time = start_time - datetime.timedelta(days=180)
            period = "1w"
        elif time_frame.lower() == "year":
            end_time = start_time - datetime.timedelta(days=365)
            period = "1m"
        # Convert to timestamps
        end_time = str(int(local_time_to_online(dt=end_time)))
        start_time = str(int(local_time_to_online(dt=start_time)))
        params = {
            "f": "json",
            "startTime": end_time,
            "endTime": start_time,
            "vars": "num",
            "groupby": "appId",
            "eType": "svcusg",
            "sType": "applogin",
            "period": period,
        }
        res = self._con.post(path=self._url, postdata=params)
        return res

    # ----------------------------------------------------------------------
    def _custom(
        self,
        start_time,
        end_time,
        vars=None,
        period=None,
        groupby=None,
        name=None,
        stype=None,
        etype=None,
        appId=None,
        device_id=None,
        username=None,
        app_org_id=None,
        user_org_id=None,
        host_org_id=None,
    ):
        """
        returns the usage statistics value
        """
        if (
            isinstance(start_time, datetime.datetime) == False
            or isinstance(end_time, datetime.datetime) == False
        ):
            raise ValueError("start_time and end_time must be datetime objects")

        url = self._url

        start_time = str(int(local_time_to_online(dt=start_time)))
        end_time = str(int(local_time_to_online(dt=end_time)))

        params = {
            "f": "json",
            "startTime": end_time,
            "endTime": start_time,
            "vars": vars,
            "period": period,
            "groupby": groupby,
            "name": name,
            "stype": stype,
            "etype": etype,
            "appId": appId,
            "deviceId": device_id,
            "username": username,
            "appOrgId": app_org_id,
            "userOrgId": user_org_id,
            "hostOrgId": host_org_id,
        }

        params = {key: item for key, item in params.items() if item is not None}
        return self._con.post(path=url, postdata=params)
