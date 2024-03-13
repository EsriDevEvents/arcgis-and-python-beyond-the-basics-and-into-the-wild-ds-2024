from __future__ import annotations
import datetime
import json
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS, Item
from arcgis.gis.tasks._schedule import TaskManager, Task
from arcgis._impl.common._utils import local_time_to_online


###########################################################################
class TaskManager(_BaseKube):
    """
    Provides access to the tasks resources defined on the ArcGIS
    Enterprise.
    """

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ---------------------------------------------------------------------
    def task(self, task_id: str):
        """
        This operation returns information on a specific task, such as the task's title, parameters, and schedule.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required String. The specific task to get.
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    def edit_task(
        self,
        task_id: str,
        item: Item,
        cron: str,
        task_type: str,
        occurences: int = 10,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        title: str = None,
        parameters: dict = None,
    ):
        """
        This operation allows you to edit and update the properties of a preexisting task
        (CleanGPJobs, BackupRetentionCleaner at 10.9.1, and CreateBackup at 10.9.1).
        Updates that have been made to a task will go into effect during its next scheduled execution.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required String. The task to edit.
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The item to schedule a task on.
        ------------------     --------------------------------------------------------------------
        cron                   Required String. The CRON statement. This should be in the from of:

                               `<minute> <hour> <day of month> <month> <day of week>`

                               Example to run a task weekly, use: `0 0 * * 0`
        ------------------     --------------------------------------------------------------------
        task_type              Required String. The type of task, either executing a notebook or
                               updating an Insights workbook, that will be executed against the
                               specified item.  For notebook server tasks use: `ExecuteNotebook`,
                               for Insights notebook use: `UpdateInsightsWorkbook`. Use
                               `ExecuteSceneCook` to cook scene tiles. Use `ExecuteWorkflowManager`
                               to run workflow manager tasks.
        ------------------     --------------------------------------------------------------------
        occurences             Optional Integer. The total number of instance that can run at a single time.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The begin date for the task to run.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The end date for the task to run.
        ------------------     --------------------------------------------------------------------
        title                  Optional String. The title of the scheduled task.
        ------------------     --------------------------------------------------------------------
        parameters             Optional Dict. Optional collection of Key/Values that will be given
                               to the task.  The dictionary will be added to the task run
                               request. This parameter is required for `ExecuteSceneCook` tasks.

                               Example

                               ```
                               {
                                   "service_url": <scene service URL>,
                                   "num_of_caching_service_instances": 2, //2 instances are required
                                   "layer": "{<list of scene layers to cook>}", //The default is all layers
                                   "update_mode": "PARTIAL_UPDATE_NODES"
                               }
                               ```

        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/update"
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

    # ---------------------------------------------------------------------
    def delete_task(self, task_id: str):
        """
        This operation deletes a task. Once the task is deleted, all associated runs and
        resources are deleted as well.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to delete
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/delete"
        params = {"f": "json"}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def enable_task(self, task_id: str):
        """
        This operation enables a previously disabled task,
        setting its taskState to active.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to enable
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/enable"
        params = {"f": "json"}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def diable_task(self, task_id: str):
        """
        This operation disables a specific task and suspends any
        upcoming runs scheduled for the task.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to disable
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/disable"
        params = {"f": "json"}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def runs(self, task_id: str):
        """
        This resource returns a list of all runs that have been completed for a
        specific task.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string.
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/runs"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    def run(self, task_id: str, run_id: str):
        """
        This resource returns information on a specific run for a task.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to get the run from.
        ------------------     --------------------------------------------------------------------
        run_id                 Required string. The run to get.
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/runs/{run_id}"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ---------------------------------------------------------------------
    def edit_run(self, task_id: str, run_id: str, status: str, results):
        """
        This operation updates an existing run for a scheduled task.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to edit the run on.
        ------------------     --------------------------------------------------------------------
        run_id                 Required string. The run to edit.
        ------------------     --------------------------------------------------------------------
        status                 Required string. Set the status of the run.

                                Values: "scheduled" | "executing" | "succeeded" | "skipped" | "failed"
                                        | "submitfailed"
        ------------------     --------------------------------------------------------------------
        results                Required string. A result string for this run.
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/runs/{run_id}/update"
        params = {"f": "json", "status": status, "results": results}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def delete_run(self, task_id: str, run_id: str):
        """
        The delete operation removes a specified run for a scheduled task.
        Deleting a run also deletes corresponding resource files associated with the run.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_id                Required string. The task to delete the run on.
        ------------------     --------------------------------------------------------------------
        run_id                 Required string. The run to delete.
        ==================     ====================================================================
        """
        url = f"{self._url}/{task_id}/runs/{run_id}/delete"
        params = {"f": "json"}
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def create_task(
        self,
        item: Item,
        cron: str,
        task_type: str,
        occurences: int = 10,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        title: str = None,
        parameters: dict = None,
    ) -> Task:
        """
        This operation creates scheduled tasks for your deployment that run
        automatically. Once the task has been created, it can be updated using
        the Update operation. In addition, scheduled tasks can be disabled, reenabled,
        and deleted through other operations in the ArcGIS Enterprise Administrator API.

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
                               specified item.  For notebook server tasks use: `ExecuteNotebook`,
                               for Insights notebook use: `UpdateInsightsWorkbook`. Use
                               `ExecuteSceneCook` to cook scene tiles. Use `ExecuteWorkflowManager`
                               to run workflow manager tasks.
        ------------------     --------------------------------------------------------------------
        occurences             Optional Integer. The total number of instance that can run at a single time.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The begin date for the task to run.
        ------------------     --------------------------------------------------------------------
        start_date             Optional Datetime. The end date for the task to run.
        ------------------     --------------------------------------------------------------------
        title                  Optional String. The title of the scheduled task.
        ------------------     --------------------------------------------------------------------
        parameters             Optional Dict. Optional collection of Key/Values that will be given
                               to the task.  The dictionary will be added to the task run
                               request. This parameter is required for `ExecuteSceneCook` tasks.

                               Example

                               ```
                               {
                                   "service_url": <scene service URL>,
                                   "num_of_caching_service_instances": 2, //2 instances are required
                                   "layer": "{<list of scene layers to cook>}", //The default is all layers
                                   "update_mode": "PARTIAL_UPDATE_NODES"
                               }
                               ```

        ==================     ====================================================================

        :return: Task
        """
        tm = TaskManager(self, self._gis)
        return tm.create_task(
            item,
            cron,
            task_type,
            occurences,
            start_date,
            end_date,
            title,
            parameters,
        )
