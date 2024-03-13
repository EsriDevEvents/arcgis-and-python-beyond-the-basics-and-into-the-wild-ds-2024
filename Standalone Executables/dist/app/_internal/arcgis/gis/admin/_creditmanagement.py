########################################################################
from typing import Optional
import datetime


class CreditManager(object):
    """
    Manages an ArcGIS Online organization's credits for users and sites

    **Example Usage**

    .. code-block:: python

        from arcgis.gis import GIS
        gis = GIS(profile='agol_account')
        cm = gis.admin.credits
        cm.allocate("user1", 100)



    """

    _gis = None
    _con = None
    _portal = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Constructor"""
        self._gis = gis
        self._portal = gis._portal
        self._con = self._portal.con

    # ----------------------------------------------------------------------
    @property
    def credits(self):
        """returns the current number of credits on the GIS"""
        try:
            return self._gis.properties.availableCredits
        except:
            return 0

    # ----------------------------------------------------------------------
    @property
    def is_enabled(self):
        """
        :return:
            A boolean that shows whether the organization has credit budgeting enabled.
        """
        if "creditAssignments" in self._gis.properties:
            return self._gis.properties.creditAssignments == "enabled"
        else:
            return False

    # ----------------------------------------------------------------------
    def enable(self):
        """
        enables credit allocation on ArcGIS Online
        """
        return self._gis.update_properties({"creditAssignments": "enabled"})

    # ----------------------------------------------------------------------
    def disable(self):
        """
        disables credit allocation on ArcGIS Online
        """
        return self._gis.update_properties({"creditAssignments": "disabled"})

    # ----------------------------------------------------------------------
    @property
    def default_limit(self):
        """
        Gets/Sets the default credit allocation for ArcGIS Online
        """
        return self._gis.properties.defaultUserCreditAssignment

    # ----------------------------------------------------------------------
    @default_limit.setter
    def default_limit(self, value: float):
        """
        Gets/Sets the default credit allocation for ArcGIS Online
        """
        params = {"defaultUserCreditAssignment": value}
        self._gis.update_properties(params)

    # ----------------------------------------------------------------------
    def allocate(self, username: str, credits: Optional[float] = None):
        """
        Allows organization administrators to allocate credits for
        organizational users in ArcGIS Online

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        username                        Required string.The name of the user to assign credits to.
        ---------------------------     --------------------------------------------------------------------
        credits                         Optional float. The number of credits to assign to a user. If None
                                        is provided, it sets user to unlimited credits.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False

        """
        if hasattr(username, "username"):
            username = getattr(username, "username")
        if not credits is None:
            params = {
                "f": "json",
                "userAssignments": [{"username": username, "credits": credits}],
            }
            path = "portals/self/assignUserCredits"
            res = self._con.post(path, params)
            if "success" in res:
                return res["success"]
            return res
        else:
            return self.deallocate(username=username)

    # ----------------------------------------------------------------------
    def deallocate(self, username: str):
        """
        Allows organization administrators to set credit limit to umlimited for
        organizational users in ArcGIS Online

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        username                        Required string.The name of the user to set to unlimited credits.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False

        """
        if hasattr(username, "username"):
            username = getattr(username, "username")
        params = {"usernames": [username], "f": "json"}
        path = "portals/self/unassignUserCredits"
        res = self._con.post(path, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def credit_usage(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        time_frame: str = "week",
    ):
        """
        returns the total credit consumption for a given time period.

        ===================   ===============================================
        **arguements**        **description**
        -------------------   -----------------------------------------------
        start_time            datetime.datetime object. This is the date to
                              start at.
        -------------------   -----------------------------------------------
        end_time              datetime.datetime object. This is the stop time
                              to look for credit consumption. It needs to be
                              at least 1 day previous than then start_time.
        -------------------   -----------------------------------------------
        time_frame            Optional string. is the timeframe report to create.
                              Allowed values: today, week (default), 14days, 30days,
                              60days, 90days, 6months, year

                              If end_time is specified, this parameter is ignored.
        ===================   ===============================================

        returns: dictionary
        """
        from ..._impl.common._utils import local_time_to_online

        if start_time and end_time:
            if (
                isinstance(start_time, datetime.datetime) == False
                or isinstance(end_time, datetime.datetime) == False
            ):
                raise ValueError("start_time and end_time must be datetime objects")

            # calculate time in days between start and end time to assign correct period
            time_elapsed = start_time - end_time
            time_elapsed = time_elapsed.days
            # Convert to timestamps
            if isinstance(start_time, datetime.datetime):
                start_time = str(int(local_time_to_online(dt=start_time)))
            if isinstance(end_time, datetime.datetime):
                end_time = str(int(local_time_to_online(dt=end_time)))
            if time_elapsed <= 1:
                # one day
                period = "1h"
            elif time_elapsed in range(2, 183):
                # less than 6 months
                period = "1d"
            elif time_elapsed in range(183, 265):
                # between 6 months to 1 year
                period = "1w"
            elif time_elapsed >= 365:
                # 1 year or more
                period = "1m"
        elif end_time is None:
            if start_time is None:
                start_time = datetime.datetime.now()
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
            else:
                raise ValueError("Indicate a valid end_time or time_frame")
            # Convert to timestamps
            end_time = str(int(local_time_to_online(dt=end_time)))
            start_time = str(int(local_time_to_online(dt=start_time)))
        path = "portals/self/usage"
        params = {
            "f": "json",
            "startTime": end_time,
            "endTime": start_time,
            "period": period,
            "groupby": "stype,etype",
            "vars": "credits,num",
        }
        data = self._con.get(path, params)
        res = {}
        for d in data["data"]:
            if d["stype"] in res:
                res[d["stype"]] += sum([float(a[1]) for a in d["credits"]])
            else:
                res[d["stype"]] = sum([float(a[1]) for a in d["credits"]])
        return res
