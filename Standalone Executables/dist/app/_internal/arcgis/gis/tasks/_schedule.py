from __future__ import annotations
import json
import datetime
from arcgis.gis import GIS, User, Item
from arcgis._impl.common._isd import InsensitiveDict
from arcgis._impl.common._utils import local_time_to_online


###########################################################################
class BaseTask(object):
    """
    Base Schedule Class
    """

    _url = None
    _gis = None
    _con = None
    _properties = None

    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        if self._properties is None:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = InsensitiveDict(res)
        return self._properties


###########################################################################
class Run(BaseTask):
    """
    Represents a single run of a scheduled task.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string. The URL to the REST endpoint.
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS. The GIS object.
    ==================     ====================================================================
    """

    _gis = None
    _url = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        super(Run, self)
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"<{self.__class__.__name__} @ {self.properties.runId}>"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Removes the Task from the System.

        :return: Boolean

        """
        url = f"{self._url}/delete"
        params = {"f": "json"}
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def update(self, status: str | None = None, description: str | None = None) -> bool:
        """
        Updates the Run's Status Message and Result Message.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        status                 Optional String. The status of the run.  The allowed values are:
                               `scheduled`, `executing`, `succeeded`, `failed`, or `skipped`.
        ------------------     --------------------------------------------------------------------
        description            Optional String. Updates the descriptive message associated with the
                               current `Run`.
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json"}
        status_values = [
            "scheduled",
            "executing",
            "succeeded",
            "failed",
            "skipped",
        ]
        if status is None and description is None:
            return False
        if status and status.lower() in status_values:
            params["status"] = status.lower()
        elif status and status.lower() not in status_values:
            raise ValueError("Invalid status")
        elif status is None:
            params["status"] = self.properties.status
        if description:
            params["result"] = description
        elif description is None:
            params["result"] = self.properties.result

        url = f"{self._url}/update"
        res = self._gis._con.post(url, params)
        if "success" in res:
            self._properties = None
            return res["success"]
        return res


###########################################################################
class Task(BaseTask):
    """
    Represents a scheduled task that can be modified for a user.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string. The URL to the REST endpoint.
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS. The GIS object.
    ==================     ====================================================================

    """

    _url = None
    _gis = None

    def __init__(self, url: str, gis: GIS):
        super(Task, self)
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"<Task @ {self.properties.id}>"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Removes the Task from the System.

        :return: Boolean

        """
        url = f"{self._url}/delete"
        params = {"f": "json"}
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def enable(self, enabled: bool) -> bool:
        """
        The `enable` method allows administrators to enable or disable the scheduled task..

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        enabled                Required Boolean.  If True, the status of the task is set to active.
                               If False, the task is set active to False.
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json"}
        if enabled == True:
            url = f"{self._url}/enable"
        elif enabled == False:
            url = f"{self._url}/disable"
        else:
            raise ValueError("`enabled` must be a boolean value")
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def start(self) -> bool:
        """
        Starts a task if it is actively running.

        :return: Boolean

        """
        return self.update(is_active=True)

    # ----------------------------------------------------------------------
    def stop(self) -> bool:
        """
        Stops a task if it is actively running.

        :return: Boolean

        """
        return self.update(is_active=False)

    # ----------------------------------------------------------------------
    def update(
        self,
        item: Item | None = None,
        cron: str | None = None,
        task_type: str | None = None,
        occurences: int = 10,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        title: str | None = None,
        parameters: dict | None = None,
        task_url: str | None = None,
        is_active: bool | None = None,
    ) -> bool:
        """
        Updates the current Task

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Optional Item. The item to update the schedule for.
        ------------------     --------------------------------------------------------------------
        cron                   Optional String. The executution time syntax.
        ------------------     --------------------------------------------------------------------
        task_type              Required String. The type of task, either executing a notebook or
                               updating an Insights workbook, that will be executed against the
                               specified item.  For notebook server tasks use ``ExecuteNotebook``,
                               for Insights notebook use: ``UpdateInsightsWorkbook``. Use
                               ``ExecuteSceneCook`` to cook scene tiles. Use ``ExecuteWorkflowManager``
                               to run workflow manager tasks.
                               Values: `ExecuteNotebook`, `UpdateInsightsWorkbook`,
                               `ExecuteSceneCook`, `ExecuteWorkflowManager`, `ExecuteReport`, or
                               `GPService`ns are
                               ``ExecuteNotebook`` or ``UpdateInsightsWorkbook``
        ------------------     --------------------------------------------------------------------
        occurences             Optional Integer. The maximum number of occurrences this task should execute.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The date/time when the task will begin executing.
        ------------------     --------------------------------------------------------------------
        end_date               Optional Datetime.  The date/time when the task will stop executing.
        ------------------     --------------------------------------------------------------------
        title                  Optional String. The name of the task.
        ------------------     --------------------------------------------------------------------
        parameters             Optional Dict.  Additional key/value pairs for execution of notebooks.
        ------------------     --------------------------------------------------------------------
        task_url               Optional String. A response URL with a set of results.
        ------------------     --------------------------------------------------------------------
        is_active              Optional Bool. Determines if the tasks is currently running.
        ==================     ====================================================================

        :return: Boolean or Dict on error.

        """
        SPECIALS = {
            "reboot": "@reboot",
            "hourly": "0 * * * *",
            "daily": "0 0 * * *",
            "weekly": "0 0 * * 0",
            "monthly": "0 0 1 * *",
            "yearly": "0 0 1 1 *",
            "annually": "0 0 1 1 *",
            "midnight": "0 0 * * *",
        }
        params = {
            "title": title or self.properties.title,
            "type": task_type or self.properties.type,
            "taskUrl": task_url or "",
            "parameters": parameters,
            "itemId": None,
            "minute": None,
            "hour": None,
            "dayOfMonth": None,
            "month": None,
            "dayOfWeek": None,
            "maxOccurrences": occurences or self.properties.maxOccurrences,
            "isActive": None,
            "f": "json",
        }
        if is_active is None:
            params.pop("isActive", None)
        else:
            params["isActive"] = json.dumps(is_active)
        if task_url is None:
            params.pop("taskUrl", None)
        if cron is None:
            params["minute"] = self.properties.cronSchedule.minute
            params["hour"] = self.properties.cronSchedule.hour
            params["dayOfMonth"] = self.properties.cronSchedule.dayOfMonth
            params["month"] = self.properties.cronSchedule.month
            params["dayOfWeek"] = self.properties.cronSchedule.dayOfWeek
        elif isinstance(cron, str) and cron in SPECIALS:
            cron = SPECIALS[cron].split(" ")
            params["minute"] = cron[0]
            params["hour"] = cron[1]
            params["dayOfMonth"] = cron[2]
            params["month"] = cron[3]
            params["dayOfWeek"] = cron[4]
        else:
            cron = cron.split(" ")
            params["minute"] = cron[0]
            params["hour"] = cron[1]
            params["dayOfMonth"] = cron[2]
            params["month"] = cron[3]
            params["dayOfWeek"] = cron[4]

        if isinstance(item, Item):
            params["itemId"] = item.itemid
        elif item:
            params["itemId"] = item
        else:
            params["itemId"] = self.properties.itemId

        if start_date:
            params["startDate"] = local_time_to_online(dt=start_date)
        else:
            params.pop("startDate", None)
        if end_date:
            params["endDate"] = local_time_to_online(dt=end_date)
        else:
            params.pop("endDate", None)
        if parameters:
            params["parameters"] = json.dumps(parameters)
        elif "parameters" in self.properties:
            params["parameters"] = self.properties.parameters
        url = f"{self._url}/update"
        res = self._gis._con.post(url, params)
        if "success" in res:
            self._properties = None
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    @property
    def runs(self) -> list:
        """
        Returns the Runs for the Task.  The maximum number of runs returned is 30

        :return: List
        """
        runs = []
        url = f"{self._url}/runs"
        params = {"f": "json"}
        res = self._gis._con.get(url, params)
        for t in res["runs"]:
            run_url = f"{self._url}/runs/{t['runId']}"
            runs.append(Run(url=run_url, gis=self._gis))
            del t
        return runs


###########################################################################
class TaskManager(object):
    """

    Provides the functions to create, update and delete scheduled tasks.

    This operation is for Enterprise configuration 10.8.1+.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    Required string. The URL to the REST endpoint.
    ------------------     --------------------------------------------------------------------
    user                   Required User. The user to perform the Task management on.
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS. The GIS object.
    ==================     ====================================================================


    """

    _tasks = None

    def __init__(self, url: str, user: User, gis: GIS):
        """initializer"""
        self._url = url
        self._user = user
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"<User {self._user.username} Tasks>"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ----------------------------------------------------------------------
    def search(
        self,
        item: Item | None = None,
        active: bool | None = None,
        types: str | None = None,
    ) -> list[Task]:
        """
        This property allows users to search for tasks based on criteria.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        item              Optional Item. The item to query tasks about.
        ----------------  -------------------------------------------------------------------------------
        active            Optional Bool. Queries tasks based on active status.
        ----------------  -------------------------------------------------------------------------------
        types             Optional String. The type of notebook execution for the item.  This can be
                          ``ExecuteNotebook``, or ``UpdateInsightsWorkbook``.
        ================  ===============================================================================

        :return: List of :class:`~arcgis.gis.tasks.Task` objects

        """
        if item is None and active is None and types is None:
            return self.all
        else:
            _tasks = []
            url = f"{self._gis._portal.resturl}community/users/{self._user.username}/tasks"
            params = {
                "num": 100,
                "start": 1,
            }
            if item:
                params["itemId"] = item.itemid
            if not active is None:
                params["active"] = active
            if types:
                params["types"] = types
            res = self._gis._con.get(url, params)
            for t in res["tasks"]:
                url = f"{self._url}/{t['id']}"
                _tasks.append(Task(url=url, gis=self._gis))
            while res["nextStart"] != -1:
                params["start"] = res["nextStart"]
                res = self._gis._con.get(url, params)
                for t in res["tasks"]:
                    url = f"{self._url}/{t['id']}"
                    _tasks.append(Task(url=url, gis=self._gis))
            return _tasks
        return []

    # ----------------------------------------------------------------------
    def create(
        self,
        item: Item,
        cron: str,
        task_type: str,
        occurences: int = 10,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        title: str | None = None,
        parameters: dict | None = None,
    ) -> Task:
        """
        Creates a new scheduled task for a notebook `Item`.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The item to schedule a task on.
        ------------------     --------------------------------------------------------------------
        cron                   Required String. The CRON statement. This should be in the from of:

                               `<minute> <hour> <day of month> <month> <day of week>`

                               Example to run a task weekly, use: `0 0 * * 0`
        ------------------     --------------------------------------------------------------------
        task_type              Required String. The type of task, either executing a notebook or
                               updating an Insights workbook, that will be executed against the
                               specified item.  For notebook server tasks use ``ExecuteNotebook``,
                               for Insights notebook use: ``UpdateInsightsWorkbook``. Use
                               ``ExecuteSceneCook`` to cook scene tiles. Use ``ExecuteWorkflowManager``
                               to run workflow manager tasks.
                               Values: `ExecuteNotebook`, `UpdateInsightsWorkbook`,
                               `ExecuteSceneCook`, `ExecuteWorkflowManager`, `ExecuteReport`, or
                               `GPService`
        ------------------     --------------------------------------------------------------------
        occurences             Optional Integer. The total number of instance that can run at a single time.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The begin date for the task to run.
        ------------------     --------------------------------------------------------------------
        end_date               Optional Datetime. The end date for the task to run.
        ------------------     --------------------------------------------------------------------
        title                  Optional String. The title of the scheduled task.
        ------------------     --------------------------------------------------------------------
        parameters             Optional Dict. Optional collection of Key/Values that will be given
                               to the task.  The dictionary will be added to the task run
                               request. This parameter is required for ``ExecuteSceneCook`` tasks.

                               Example:

                                   | {
                                   |    "service_url": <scene service URL>,
                                   |    "num_of_caching_service_instances": 2, (2 instances are required)
                                   |    "layer": "{<list of scene layers to cook>}", //The default is all layers
                                   |    "update_mode": "PARTIAL_UPDATE_NODES"
                                   | }


        ==================     ====================================================================

        :return:
            :class:`~arcgis.gis.tasks.Task` object

        """
        SPECIALS = {
            "reboot": "@reboot",
            "hourly": "0 * * * *",
            "daily": "0 0 * * *",
            "weekly": "0 0 * * 0",
            "monthly": "0 0 1 * *",
            "yearly": "0 0 1 1 *",
            "annually": "0 0 1 1 *",
            "midnight": "0 0 * * *",
        }
        url = f"{self._url}/createTask"
        params = {
            "f": "json",
            "title": title,
            "type": task_type,
            "parameters": None,
            "itemId": None,
            "startDate": start_date,
            "endDate": end_date,
            "minute": None,
            "hour": None,
            "dayOfMonth": None,
            "month": None,
            "dayOfWeek": None,
            "maxOccurrences": occurences,
        }
        if isinstance(item, Item):
            params["itemId"] = item.itemid
        else:
            params["itemId"] = item
        if title is None:
            params.pop("title", None)
        if start_date:
            params["startDate"] = local_time_to_online(dt=start_date)
        else:
            params.pop("startDate", None)
        if end_date:
            params["endDate"] = local_time_to_online(dt=end_date)
        else:
            params.pop("endDate", None)
        if cron in SPECIALS:
            cron = SPECIALS[cron]
        if cron:
            cron = cron.split(" ")
            while len(cron) < 5:
                cron.append("*")
            params.update(
                {
                    "minute": cron[0],
                    "hour": cron[1],
                    "dayOfMonth": cron[2],
                    "month": cron[3],
                    "dayOfWeek": cron[4],
                }
            )
        if parameters:
            params["parameters"] = json.dumps(parameters)
        res = self._gis._con.post(url, params)

        if "success" in res and res["success"]:
            url = f"{self._url}/{res['taskId']}"
            return Task(url, gis=self._gis)
        return res

    # ----------------------------------------------------------------------
    @property
    def all(self) -> list:
        """
        returns all the current user's tasks

        :return:
            List of :class:`~arcgis.gis.tasks.Task` objects

        """
        if self._tasks is None:
            self._tasks = []
            url = f"{self._gis._portal.resturl}community/users/{self._user.username}/tasks"
            params = {
                "num": 100,
                "start": 1,
            }
            res = self._gis._con.get(url, params)
            for t in res["tasks"]:
                url = f"{self._url}/{t['id']}"
                self._tasks.append(Task(url=url, gis=self._gis))
            while res["nextStart"] != -1:
                params["start"] = res["nextStart"]
                res = self._gis._con.get(url, params)
                for t in res["tasks"]:
                    url = f"{self._url}/{t['id']}"
                    self._tasks.append(Task(url=url, gis=self._gis))
        return self._tasks

    # ----------------------------------------------------------------------
    @property
    def count(self) -> int:
        """
        Returns the number of tasks a user has

        :return: Int
        """
        url = f"{self._gis._portal.resturl}community/users/{self._user.username}/tasks"
        params = {
            "num": 1,
            "start": 1,
        }
        res = self._gis._con.get(url, params)
        return res["total"]
