from __future__ import annotations
from typing import TypeVar
from arcgis.gis import GIS, Item
from arcgis._impl.common._mixins import PropertyMap
import concurrent.futures

__all__ = ["NotebookManager"]

K = TypeVar("K")
V = TypeVar("V")


########################################################################
class NotebookManager(object):
    """
    Provides access to managing a site's notebooks. An object of this
    class can be created using :attr:`~arcgis.gis.nb.NotebookServer.notebooks` property of the
    :class:`~arcgis.gis.nb.NotebookServer` class
    """

    _url = None
    _gis = None
    _properties = None
    _nbs = None
    _snapshot = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, nbs):
        """Constructor"""
        self._url = url
        self._nbs = nbs
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
        return "< NotebookManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< NotebookManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @staticmethod
    def _future_job(
        fn,
        task_name,
        jobid=None,
        task_url=None,
        notify=False,
        gis=None,
        **kwargs,
    ):
        """
        runs the job asynchronously

        :return: Job object
        """
        from arcgis._impl._async.jobs import Job

        tp = concurrent.futures.ThreadPoolExecutor(1)
        try:
            future = tp.submit(fn=fn, **kwargs)
        except:
            future = tp.submit(fn, **kwargs)
        tp.shutdown(False)
        return Job(future, task_name, jobid, task_url, notify, gis=gis)

    # ----------------------------------------------------------------------
    def execute_notebook(
        self,
        item: Item,
        update_portal_item: bool = True,
        parameters: list | None = None,
        save_parameters: bool = False,
        instance_type: str | None = None,
        timeout: float | int = 50,
        future: bool = False,
    ):
        """

        The Execute Notebook operation allows administrators to remotely
        run a notebook in their ArcGIS Notebook Server site. The notebook
        specified in the operation will be run with all cells in order.

        Using this operation, you can schedule the execution of a notebook,
        either once or with a regular occurrence. This allows you to
        automate repeating tasks such as data collection and cleaning,
        content updates, and portal administration. On Linux machines, use
        a cron job to schedule the executeNotebook operation; on Windows
        machines, you can use the Task Scheduler app.

        .. note::
            To run this operation, you must be logged in with an ArcGIS
            Enterprise portal account. You cannot execute notebooks from
            the ArcGIS Notebook Server primary site administrator
            account.

        You can specify parameters to be used in the notebook at execution
        time. If you've specified one or more parameters, they'll be
        inserted into the notebook as a new cell. This cell will be placed
        at the beginning of the notebook, unless you have added the tag
        parameters to a cell.

        ====================    ====================================================================
        **Parameter**            **Description**
        --------------------    --------------------------------------------------------------------
        item                    Required :class:`~arcgis.gis.Item`. Opens an existing portal item.
        --------------------    --------------------------------------------------------------------
        update_portal_item      Optional Boolean. Specifies whether you want to update the
                                notebook's portal item after execution. The default is true. You may
                                want to specify true when the notebook you're executing contains
                                information that needs to be updated, such as a workflow that
                                collects the most recent version of a dataset. It may not be
                                important to update the portal item if the notebook won't store any
                                new information after executing, such as an administrative notebook
                                that emails reminders to inactive users.
        --------------------    --------------------------------------------------------------------
        parameters              Optional List. An optional array of parameters to add to the
                                notebook for this execution. The parameters will be inserted as a
                                new cell directly after the cell you have tagged ``parameters``.
                                Separate parameters with a comma. Use the format "x":1 when
                                defining parameters with numbers, and "y":"text" when defining
                                parameters with text strings.
        --------------------    --------------------------------------------------------------------
        save_parameters         Optional Boolean.  Specifies whether the notebook parameters cell
                                should be saved in the notebook for future use. The default is
                                false.
        --------------------    --------------------------------------------------------------------
        instance_type           Optional String. The instance type.
        --------------------    --------------------------------------------------------------------
        timeout                 Optional Int. The number of minutes to run the instance before timeout.
        --------------------    --------------------------------------------------------------------
        future                  Optional boolean. If True, a Job object will be returned and the process
                                will not wait for the task to complete. The default is False, which means wait for results.
        ====================    ====================================================================

        :return: Dict else If ``future = True``, then the result is
                 a `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object.
                 Call ``result()`` to get the response

        """
        from arcgis.gis import Item

        url = self._url + "/executeNotebook"
        itemid = None
        if isinstance(item, str):
            itemid = item
        elif isinstance(item, Item):
            itemid = item.itemid
        params = {
            "f": "json",
            "itemId": itemid,
            "updatePortalItem": update_portal_item,
            "saveInjectedParameters": save_parameters,
            "instanceTypeName": instance_type,
            "executionTimeoutInMinutes": timeout,
        }
        if parameters:
            params["notebookParameters"] = parameters
        if future:

            def _fn(url, params, nbs):
                import time

                start_job = self._gis._con.post(url, params)
                if "jobUrl" in start_job:
                    resp = self._gis._con.get(start_job["jobUrl"], {"f": "json"})
                else:
                    return start_job
                if "status" in resp and resp["status"].lower() != "success":
                    status = self._gis._con.get(start_job["jobUrl"], {"f": "json"})
                    i = 0
                    while status["status"].lower() != "completed":
                        time.sleep(0.3 * i)
                        if status["status"].lower() == "failed":
                            return status
                        elif (
                            status["status"].lower().find("fail") > -1
                            or status["status"].lower().find("error") > -1
                        ):
                            raise Exception(f"Job Fail {status}")
                        status = self._gis._con.get(start_job["jobUrl"], {"f": "json"})
                        i += 1
                        if i > 20:
                            i = 20
                    return status
                return resp

            return NotebookManager._future_job(
                fn=_fn,
                task_name="Execute Notebook",
                gis=self._gis,
                **{"url": url, "params": params, "nbs": self._nbs},
            )
        res = self._gis._con.post(url, params)
        return res

    # ----------------------------------------------------------------------
    def open_notebook(
        self,
        itemid: str,
        templateid: str | None = None,
        nb_runtimeid: str | None = None,
        template_nb: str | None = None,
        instance_type: str | None = None,
        *,
        future: bool = False,
    ):
        """

        Opens a notebook on the notebook server

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        itemid                  Required String. Opens an existing portal item.
        ------------------      --------------------------------------------------------------------
        templateid              Optional String. The id of the portal notebook template. To get the
                                system templates, look at the sample notebooks group:

                                .. code-block:: python

                                    >>> from arcgis.gis import GIS
                                    >>> gis = GIS()
                                    >>> grp = gis.groups.search("title:(esri sample notebooks) AND
                                    >>>                                 owner:\"esri_notebook\")[0]
                                    >>> grp.content
        ------------------      --------------------------------------------------------------------
        nb_runtimeid            Optional String. The runtime to use to generate a new notebook.
        ------------------      --------------------------------------------------------------------
        template_nb             Optional String. The start up template for the notebook.
        ------------------      --------------------------------------------------------------------
        instance_type           Optional String. The name of the instance type.
        ------------------      --------------------------------------------------------------------
        future                  Optional Bool.
        ==================      ====================================================================

        :return: Dict

        """

        def _fn(url, params, nbs):
            """used to fire off async job"""
            import time

            start_job = self._gis._con.post(url, params)
            status_url = start_job.get("jobUrl", None) or start_job.get(
                "notebookStatusUrl", None
            )
            if status_url:
                resp = self._gis._con.get(status_url, {"f": "json"})
            else:
                return start_job
            if "status" in resp and resp["status"].lower() != "success":
                status = self._gis._con.get(status_url, {"f": "json"})
                i = 0
                while status["status"].lower() != "completed":
                    time.sleep(0.3 * i)
                    if status["status"].lower() == "failed":
                        return status
                    elif (
                        status["status"].lower().find("fail") > -1
                        or status["status"].lower().find("error") > -1
                    ):
                        raise Exception(f"Job Fail {status}")
                    status = self._gis._con.get(status_url, {"f": "json"})
                    i += 1
                    if i > 20:
                        i = 20
                return status

        if hasattr(itemid, "id"):
            itemid = getattr(itemid, "id")
        params = {
            "itemId": itemid,
            "templateId": templateid,
            "notebookRuntimeId": nb_runtimeid,
            "templateNotebook": template_nb,
            "async": True,
            "f": "json",
            "instanceTypeName": instance_type,
        }
        url = self._url + "/openNotebook"

        if future:
            return NotebookManager._future_job(
                fn=_fn,
                task_name="Open Notebook",
                gis=self._gis,
                **{"url": url, "params": params, "nbs": self._nbs},
            )
        else:
            res = self._con.post(url, params)
            status_url = res.get("jobUrl", None) or res.get("notebookStatusUrl", None)
            if status_url:
                import time

                job_url = status_url
                params = {"f": "json"}
                job_res = self._con.get(job_url, params)
                while job_res["status"].upper() != "COMPLETED":
                    job_res = self._con.get(job_url, params)
                    if job_res["status"].lower().find("fail") > -1:
                        return job_res
                    time.sleep(2.5)
                return job_res
            return res
