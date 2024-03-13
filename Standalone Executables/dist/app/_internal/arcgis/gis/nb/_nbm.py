import os
from typing import Optional
from arcgis.gis import GIS, Item
from arcgis._impl.common._mixins import PropertyMap
import concurrent.futures


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
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def list(self):
        """
        Returns a list of notebook instances on the Notebook Server

        :return: List of :class:`~arcgis.gis.nb.Notebook` objects

        """
        return [
            Notebook(url=self._url, item_id=nbs["id"], properties=nbs)
            for nbs in self.properties.notebooks
        ]

    # ----------------------------------------------------------------------
    @property
    def runtimes(self):
        """
        Returns a list of all runtimes

        :return: List
        """
        url = self._url + "/runtimes"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "runtimes" in res:
            return [
                Runtime(url=url + "/{rid}".format(rid=r["id"]), gis=self._gis)
                for r in res["runtimes"]
            ]
        return []

    # ----------------------------------------------------------------------
    @property
    def snapshots(self):
        """
        Provides access to managing Notebook's snapshots

        :return: :class:`~arcgis.gis.nb.SnapshotManager`
        """
        if self._snapshot is None:
            from ._snapshot import SnapshotManager

            url = self._url + "/snapshots"
            self._snapshot = SnapshotManager(url=url, gis=self._gis)
        return self._snapshot

    # ----------------------------------------------------------------------
    def restore_runtime(self):
        """
        This operation restores the two default notebook runtimes in ArcGIS
        Notebook Server - ArcGIS Notebook Python 3 Standard and ArcGIS
        Notebook Python 3 Advanced - to their original settings.
        """
        url = self._url + "/runtimes/restore"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

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
        parameters: Optional[list] = None,
        save_parameters: bool = False,
        future: bool = False,
    ):
        """

        The Execute Notebook operation allows administrators and users with
        the `Create and Edit Notebooks` privilege to remotely
        run a notebook that they own.  The notebook pecified in the operation will be run with all
        cells in order.

        Using this operation, you can schedule the execution of a notebook,
        either once or with a regular occurrence. This allows you to
        automate repeating tasks such as data collection and cleaning,
        content updates, and portal administration. On Linux machines, use
        a cron job to schedule the executeNotebook operation; on Windows
        machines, you can use the Task Scheduler app.

        .. note::
            To run this operation in ArcGIS Enterprise, you must log in with
            an Enterprise account. You cannot execute notebooks using the
            ArcGIS Notebook Server primary site administrator account.

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
        future                  Optional boolean. If True, a Job object will be returned and the process
                                will not wait for the task to complete. The default is False, which means wait for results.
        ====================    ====================================================================

        :return: Dict else If ``future = True``, then the result is
                 a `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object.
                 Call ``result()`` to get the response

        .. code-block:: python

            # Usage Example:

            >>> from arcgis.gis import GIS
            >>> gis = GIS("home")
            >>> nb_server = gis.notebook_server[0]

            >>> notebook_item = gis.content.get('<notebook_item_id>')

            >>> nb_mgr = nb_server.notebooks
            >>> nb_mgr.execute_notebook(notebook_item)

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
        templateid: Optional[str] = None,
        nb_runtimeid: Optional[str] = None,
        template_nb: Optional[str] = None,
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
        future                  Optional Bool. If True, the job will run asynchronously.
        ==================      ====================================================================

        :return: Dict or Job

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

        params = {
            "itemId": itemid,
            "templateId": templateid,
            "notebookRuntimeId": nb_runtimeid,
            "templateNotebook": template_nb,
            "async": True,
            "f": "json",
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
                job_url = status_url
                params = {"f": "json"}
                job_res = self._con.get(job_url, params)
                while job_res["status"] != "COMPLETED":
                    job_res = self._con.get(job_url, params)
                    if job_res["status"].lower().find("fail") > -1:
                        return job_res
                return job_res
            return res

    # ----------------------------------------------------------------------
    def _add_runtime(
        self,
        name,
        image_id,
        version="10.7",
        container_type="docker",
        image_pull_string="",
        max_cpu=1.0,
        max_memory=4.0,
        max_memory_unit="g",
        max_swap_memory="",
        max_swap_unit="g",
        shared_memory=None,
        shared_memory_unit="m",
        docker_runtime="",
        manifest=None,
        **kwargs,
    ):
        """
        **WARNING: private method, this will change in future releases**

        Added a new docker image to the notebook server.
        """
        url = self._url + "/runtimes/register"
        params = {
            "f": "json",
            "name": name,
            "version": version,
            "imageId": image_id,
            "containerType": container_type,
            "imagePullString": image_pull_string,
            "maxCpu": float(max_cpu),
            "maxMemory": float(max_memory),
            "maxMemoryUnit": max_memory_unit,
            "maxSwapMemory": max_swap_memory,
            "maxSwapMemoryUnit": max_swap_unit,
            "sharedMemory": shared_memory,
            "sharedMemoryUnit": shared_memory_unit,
            "dockerRuntime": docker_runtime,
            "f": "json",
        }

        for k, v in kwargs.items():
            params[k] = v
        res = self._con.post(
            url,
            params,
            files={"manifestFile": manifest},
        )
        return res


########################################################################
class Runtime(object):
    """
    Provides information about the properties of a specific notebook runtime in your ArcGIS Notebook Server site
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
        return "< Runtime @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< Runtime @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the current runtime from the ArcGIS Notebook Server

        :return: Boolean

        """
        url = self._url + "/unregister"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def update(
        self,
        name: Optional[str] = None,
        image_id: Optional[str] = None,
        max_cpu: Optional[float] = None,
        max_memory: Optional[float] = None,
        memory_unit: Optional[str] = None,
        max_swap_memory: Optional[str] = None,
        swap_memory_unit: Optional[str] = None,
        shared_memory: Optional[str] = None,
        docker_runtime: Optional[str] = None,
        shared_unit: Optional[str] = None,
        version: Optional[str] = None,
        container_type: Optional[str] = None,
        pull_string: Optional[str] = None,
        require_advanced_priv: Optional[bool] = None,
        manifest: Optional[str] = None,
    ):
        """
        This operation allows you to update the properties of a notebook
        runtime in ArcGIS Notebook Server. These settings will be applied
        to every container to which the runtime is applied.

        You can use this operation to update the resource limits of the
        runtime, such as maximum CPU and maximum memory. You can also use
        it to extend either of the default notebook runtimes, in order to
        make additional Python modules available to your notebook authors,
        or as a step in making ArcGIS Notebook Server able to use graphical
        processing units (GPUs).



        """
        url = self._url + "/update"
        if manifest is None:
            manifest = ""
        if manifest:
            file = {"manifestFile": manifest}

        params = {
            "name": name,
            "version": version,
            "imageId": image_id,
            "containerType": container_type,
            "imagePullString": pull_string,
            "requiresAdvancedPrivileges": require_advanced_priv,
            "maxCpu": max_cpu or float(self.properties.maxCpu),
            "maxMemory": max_memory or float(self.properties.maxMemory),
            "maxMemoryUnit": memory_unit or "g",
            "maxSwapMemory": max_swap_memory or "",
            "maxSwapMemoryUnit": swap_memory_unit or "g",
            "sharedMemory": shared_memory or "",
            "sharedMemoryUnit": shared_unit or "m",
            "dockerRuntime": docker_runtime,
            "f": "json",
        }
        import json

        for k in list(params.keys()):
            if params[k] is None and k in self.properties:
                params[k] = self.properties[k]
            elif params[k] is None:
                params[k] = ""
            if isinstance(params[k], bool):
                params[k] = json.dumps(params[k])
            elif isinstance(params[k], (int, float)):
                params[k] = float(params[k])

        if len(params) == 1:
            return False
        res = self._con.post(
            url,
            params,
            files={"manifestFile": manifest},
        )
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def manifest(self):
        """
        This resource returns a JSON representation of all the Python
        libraries supported in the specified notebook runtime. Notebook
        authors who open notebooks using this runtime are able to import
        any of the libraries in the manifest into their notebooks.

        :return: List of Dictionaries

        """
        url = self._url + "/manifest"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "libraries" in res:
            return res["libraries"]
        return res


###########################################################################
class Notebook(object):
    """
    This represents an individual notebook resource in the notebook server.
    """

    _url = None
    _item_id = None
    _properties = None
    _gis = None

    # ----------------------------------------------------------------------
    def __init__(self, url, item_id, properties=None, gis=None):
        self._url = url + "/%s" % item_id
        self._item_id = item_id
        if properties:
            self._properties = properties
        if gis is None:
            from arcgis.env import active_gis

            gis = active_gis
        self._gis = gis

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
        return "< Notebook @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< Notebook @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def close(self):
        """
        This operation stops a running notebook. You can use it to free up
        space in your notebook container. Idle notebooks are automatically
        cleaned up according to the duration defined by the
        idleNotebookThreshold property. The default value for that property
        is 24 hours.

        :return: Boolean

        """
        params = {"f": "json"}
        url = self._url + "/closeNotebook"
        res = self._gis._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res
