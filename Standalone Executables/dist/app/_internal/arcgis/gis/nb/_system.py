import os
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
from typing import List, Dict, Any, Optional


########################################################################
class ContainerNotebook(object):
    """
    Represents an individual notebook container
    """

    _con = None
    _gis = None
    _url = None
    _properties = None

    def __init__(self, url: str, gis: GIS):
        """initializer"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< ContainerNotebook @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< ContainerNotebook @ {url}>".format(url=self._url)

    @property
    def properties(self) -> PropertyMap:
        """
        The container notebook properties

        :return: PropertyMap
        """
        params = {"f": "json"}
        return PropertyMap(self._con.get(self._url, params))

    def close(self) -> bool:
        """This operation stops a running notebook

        :return: Boolean
        """
        url = f"{self._url}/close"
        params = {"f": "json"}
        return self._con.post(url, params).get("status", "failure") == "success"


########################################################################
class DirectoryManager(object):
    """
    Manages and maintains a collection of all server directories.
    """

    _url = None
    _con = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self) -> PropertyMap:
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< DirectoryManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< DirectoryManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def list(self) -> List[Dict[str, Any]]:
        """
        returns the current registered directories

        :return: List

        """
        self._properties = None
        val = dict(self.properties)
        return val["directories"]

    # ----------------------------------------------------------------------
    def register(self, name: str, path: str, directory_type: str) -> bool:
        """
        This operation registers a new data directory from your local
        machine with the ArcGIS Notebook Server site. Registering a local
        folder as a data directory allows your notebook authors to work with
        files in the folder.

        ==================     ====================================================================
        **Parameter**          **Description**
        ------------------     --------------------------------------------------------------------
        name	               The name of the directory.
        ------------------     --------------------------------------------------------------------
        path	               The full path to the directory on your machine.
        ------------------     --------------------------------------------------------------------
        directory_type	       The type of directory. Values: DATA | WORKSPACE | OUTPUT
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json", "name": name, "path": path, "type": directory_type}
        url = self._url + "/register"
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def unregister(self, directory_id: str) -> bool:
        """
        This operation unregisters an existing directory from the ArcGIS
        Notebook Server site.

        ==================     ====================================================================
        **Parameter**          **Description**
        ------------------     --------------------------------------------------------------------
        directory_id           Required String.  The directory ID to remove.
        ==================     ====================================================================

        :return: Boolean

        """
        params = {"f": "json"}
        url = self._url + "/{uid}/unregister".format(uid=directory_id)
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class WebAdaptor(object):
    """
    This resource provides information about the ArcGIS Web Adaptor
    configured with your ArcGIS Notebook Server site. ArcGIS Web Adaptor
    is a web application that runs in a front-end web server. One of the
    Web Adaptor's primary responsibilities is to forward HTTP requests
    from end users to ArcGIS Notebook Server in a round-robin fashion.
    The Web Adaptor acts a reverse proxy, providing the end users with
    an entry point into the system, hiding the server itself, and
    providing some degree of immunity from back-end failures.

    The front-end web server could authenticate incoming requests against
    your enterprise identity stores and provide specific authentication
    schemes like Integrated Windows Authentication (IWA), HTTP Basic or
    Digest.

    Most importantly, ArcGIS Web Adaptor provides your end users with a
    well-defined entry point into your system without exposing the internal
    details of your server site. ArcGIS Notebook Server will trust requests
    being forwarded by ArcGIS Web Adaptor and will not challenge the user
    for any credentials. However, the authorization of the request (by
    looking up roles and permissions) is still enforced by the server site.

    ArcGIS Notebooks use the WebSocket protocol for communication. You can
    update the maximum size of the file sent using WebSocket by updating your
    site's webSocketMaxHeapSize property.
    """

    _url = None
    _con = None
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
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< WebAdaptor @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< WebAdaptor @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def unregister(self):
        """
        Unregisters a WebAdaptor for the Notebook Server

        :return: Boolean
        """
        url = self._url + "/unregister"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class WebAdaptorManager(object):
    """
    Manages and configures web adaptors for the ArcGIS Notebook Server.
    """

    _url = None
    _con = None
    _gis = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self) -> PropertyMap:
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< WebAdaptorManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< WebAdaptorManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def register(
        self,
        name: str,
        ip: str,
        webadaptor_url: str,
        http_port: int,
        https_port: int,
        description: Optional[str] = "",
    ):
        """
        Registers a new :class:`web adapter <arcgis.gis.nb.WebAdaptor>`.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Required String. The name of the web adapter
        ------------------     --------------------------------------------------------------------
        ip                     Required String. The IP of the web adapter.
        ------------------     --------------------------------------------------------------------
        webadaptor_url         Required String. The URI endpoint of the web adpater.
        ------------------     --------------------------------------------------------------------
        http_port              Required Integer. The port number of the web adapter
        ------------------     --------------------------------------------------------------------
        https_port             Required Integer. The secure port of the web adapter.
        ------------------     --------------------------------------------------------------------
        description            Optional String. The optional web adapter description.
        ==================     ====================================================================

        :return: Boolean

        """
        params = {
            "f": "json",
            "machineName": name,
            "machineIP": ip,
            "webAdaptorURL": webadaptor_url,
            "description": description,
            "httpPort": http_port,
            "httpsPort": https_port,
        }
        url = self._url + "/register"
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def config(self) -> Dict[str, Any]:
        """
        This is a property that allows for the retreival and manipulation of web adaptors.

        You can use this operation to change the Web Adaptor configuration
        and the sharedkey attribute. The sharedkey attribute must be present
        in the request.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        config                 Required dict. The configuration items to be updated for this web
                               adaptor. Always include the web adaptor's sharedkey attribute.
        ==================     ====================================================================

        :return:
            A boolean indicating success (True), else a Python dictionary containing an error message.
        """
        url = self._url + "/config"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @config.setter
    def config(self, config) -> None:
        """
        See main ``config`` property docstring.
        """
        url = self._url + "/config/update"
        params = {"f": "json", "webAdaptorConfig": config}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def list(self) -> List[WebAdaptor]:
        """
        Returns all registered Web Adapters

        :return: List of :class:`~arcgis.gis.nb.WebAdaptor` objects
        """
        url = self._url
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "webAdaptors" in res:
            return [
                WebAdaptor(self._url + "/{wa}".format(wa=wa["id"]), gis=self._gis)
                for wa in res["webAdaptors"]
            ]
        return res


########################################################################
class Container(object):
    """
    This represents a single hosted notebook container.
    """

    _url = None
    _con = None
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
            res = self._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< Container @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< Container @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def sessions(self):
        """
        When an ArcGIS Notebook is opened in a container, a computational
        engine called a kernel is launched; the kernel runs while the notebook
        is active and executes all the work done by the notebook. This
        resource tracks the active kernels running in the specified container
        to provide information about current notebook sessions.


        :return: List of Dict

        ==================     ====================================================================
        **Response**           **Description**
        ------------------     --------------------------------------------------------------------
        ID	               The unique ID string for the container session.
        ------------------     --------------------------------------------------------------------
        path	               The working path to the running ArcGIS Notebook, ending in .ipynb.
        ------------------     --------------------------------------------------------------------
        kernel	               Properties describing the kernel. They are as follows:

                                    + last_activity: The date and time of the most recent action performed by the kernel.
                                    + name: The name of the kernel. At 10.7, this value is python3.
                                    + id: The unique ID string for the kernel.
                                    + execution_state: Whether the kernel is currently executing an action or is idle.
                                    + connections: The number of users currently accessing the notebook.
        ==================     ====================================================================
        """
        url = self._url + "/sessions"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "sessions" in res:
            return res["sessions"]
        return res

    @property
    def notebooks(self) -> List[ContainerNotebook]:
        """
        A list of all notebooks currently open in the container

        :return: List of :class:`~arcgis.gis.nb.ContainerNotebook` objects
        """
        nbs = []
        url = f"{self._url}/notebooks"
        params = {"f": "json"}
        notebooks = self._con.get(url, params).get("notebooks", [])
        for nb in notebooks:
            n_id = nb["id"]
            n_url = f"{url}/{n_id}"
            nbs.append(ContainerNotebook(url=n_url, gis=self._gis))
        return nbs

    def logs(self, count: Optional[int] = 1000) -> List[str]:
        """
        Returns the container logs

        :return: List[str]
        """
        params = {"f": "json", "tail": count}
        url = f"{self._url}/logs"
        return self._con.post(url, params).get("containerLogs", [])

    def terminate(self) -> bool:
        """
        Stops the container

        :return: Boolean
        """
        url = f"{self._url}/terminateContainer"
        params = {"f": "json"}
        return self._con.post(url, params).get("status", "fail") == "success"

    @property
    def statistics(self) -> Dict[str, Any]:
        """
        Returns information about the current container

        :return: Dict[str,Any]

        """
        url = self._url + "/statistics"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    def shutdown(self):
        """
        Terminates the current container

        :return: Boolean
        """
        url = self._url + "/terminateContainer"
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"] == "success"
        return res


########################################################################
class SystemManager(object):
    """
    The System resource is a collection of server-wide resources in your
    ArcGIS Notebook Server site. Within this resource, you can access
    information and perform operations pertaining to licenses, Web
    Adaptors, containers, server properties, directories, Jobs, and the
    configuration store.
    """

    _url = None
    _con = None
    _gis = None
    _dir = None
    _wam = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        """Constructor"""
        self._url = url
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self) -> PropertyMap:
        """loads the properties"""
        try:
            url = self._url + "/properties"
            params = {"f": "json"}
            res = self._con.get(url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "< SystemManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< SystemManager @ {url}>".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> PropertyMap:
        """
        ArcGIS Notebook Server has configuration properties that govern
        some of its intricate behavior. This resource is a container for
        these properties. The properties are available to all server
        objects and extensions through the server environment interface.

        You can use this property to get and/or set the available properties.

        .. code-block:: python

            #Usage Example to set property:
            >>> nbserver = gis.notebook_server[0]

            >>> nbserver.system.properties = {"webSocketSize" : 32}


        See the REST API documention for `Notebook Server System properties <https://developers.arcgis.com/rest/enterprise-administration/notebook/server-properties.htm>`_
        for current complete list of available properties.

        :return: dictionary-like PropertyMap
        """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def recent_statistics(self) -> Dict[str, Any]:
        """
        returns statistics about the current state of the notebook server

        :return: Dictionary
        """
        try:
            url = self._url + "/statistics/mostRecent"
            params = {"f": "json"}
            return self._con.get(url, params)
        except:
            raise Exception(
                (
                    "Recent Statistics is not supported on your cur"
                    "rent version of Notebook Server, please use v10.8.1+"
                )
            )

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, value: dict) -> None:
        """
        See main ``properties`` property docstring
        """
        properties = {
            "dockerConnectionPort": 2375,
            "webSocketSize": 16,
            "maxContainersPerNode": 20,
            "containersStartPort": 30001,
            "containersStopPort": 31000,
            "idleNotebookThreshold": 1440,
            "containerCreatedThreshold": 60,
            "dockerConnectionHost": "localhost",
        }
        props = {}
        url = self._url + "/properties/update"
        params = {"f": "json", "properties": {}}
        current = dict(self.properties)
        for k in current.keys():
            if k in value:
                params["properties"][k] = value[k]
            else:
                params["properties"][k] = current[k]
        res = self._con.post(url, params)
        if not "status" in res:
            raise Exception(res)

    # ----------------------------------------------------------------------
    @property
    def containers(self) -> List[Container]:
        """
        Returns a list of active containers.

        :return: List of :class:`containers <arcgis.gis.nb.Container>`
        """
        container = []
        url = self._url + "/containers"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "containers" in res:
            for c in res["containers"]:
                cid = c["id"]
                curl = self._url + "/containers/{cid}".format(cid=cid)
                container.append(Container(url=curl, gis=self._gis))
        return container

    # ----------------------------------------------------------------------
    @property
    def licenses(self) -> Dict[str, Any]:
        """
        Gets the license resource list.  The licenses resource lists the
        current license level of ArcGIS Notebook Sever and all authorized
        extensions. Contact Esri Customer Service if you have questions
        about license levels or expiration properties.
        """
        url = self._url + "/licenses"
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def web_adaptors(self) -> WebAdaptorManager:
        """
        returns a list of web adapters

        :return: List
        """
        if self._wam is None:
            url = self._url + "/webadaptors"
            self._wam = WebAdaptorManager(url=url, gis=self._gis)
        return self._wam

    # ----------------------------------------------------------------------
    @property
    def jobs(self) -> List[Dict[str, Any]]:
        """
        This resource is a collection of all the administrative jobs
        (asynchronous operations) created within your site. When operations
        that support asynchronous execution are run, ArcGIS Notebook Server
        creates a new job entry that can be queried for its current status
        and messages.

        :return: List

        """
        url = self._url + "/jobs"
        params = {"f": "json"}
        res = self._con.get(url, params)
        if "asyncJobs" in res:
            return res["asyncJobs"]
        return res

    # ----------------------------------------------------------------------

    def delete_all_jobs(self) -> bool:
        """
        Administrators can clean up an open notebook and execute notebook
        jobs on demand. Administrators can view and delete all jobs.
        Non-administrative users with create and edit notebook privileges
        can only view and delete their own jobs.
        Only jobs in completed or failed states will be cleaned up.

        :return: Boolean
        """
        params = {"f": "json"}
        url = self._url + "/jobs/deleteAll"
        resp = self._con.post(url, params)
        return resp["status"] == "success"

    # ----------------------------------------------------------------------
    def list_jobs(self, num: int = 100, details: bool = False) -> List[Dict[str, Any]]:
        """
        This resource is a collection of all the administrative jobs
        (asynchronous operations) created within your site. When operations
        that support asynchronous execution are run, ArcGIS Notebook Server
        creates a new job entry that can be queried for its current status
        and messages. This is used for Notebook Server 10.9+

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        details                Optional Bool.  For 10.9+ Notebook Servers, to get the expanded
                               details of a Job, set the details to `True`. `False` will provide
                               back a shortened job status.
        ------------------     --------------------------------------------------------------------
        num                    Optional Integer.  The number of jobs to return.  The default is 100.
                               This is only valid on 10.9+.
        ==================     ====================================================================

        :return: List

        """
        url = self._url + "/jobs"
        params = {"f": "json", "detail": details, "num": num}
        res = self._con.get(url, params)
        if "asyncJobs" in res:
            return res["asyncJobs"]
        return res

    # ----------------------------------------------------------------------
    @property
    def directories(self) -> DirectoryManager:
        """Provides access to registering directories

        :return: :class:`~arcgis.gis.nb.DirectoryManager`
        """
        if self._dir is None:
            url = self._url + "/directories"
            self._dir = DirectoryManager(url=url, gis=self._gis)
        return self._dir

    # ----------------------------------------------------------------------
    def job_details(self, job_id: str) -> Dict[str, Any]:
        """
        A job represents the asynchronous execution of an operation in
        ArcGIS Notebook Server. You can acquire progress information by
        periodically querying the job.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        job_id                 Required String. The unique identifier of the job.
        ==================     ====================================================================

        :return: Dict

        """
        url = self._url + "/jobs/{jid}".format(jid=job_id)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def config_store(self) -> Dict[str, Any]:
        """
        The configuration store maintains configurations for ArcGIS Notebook
        Server. Typical configurations include all the resources such as
        machines and security rules that are required to power the site. In
        a way, the configuration store is a physical representation of a site.

        Every ArcGIS Notebook Server machine, when it joins the site, is
        provided with a connection to the configuration store and it can
        thereafter participate in the management of the site. You can change
        the store's properties during runtime using the edit operation.

        The Administrator API that runs on every server machine is capable
        of reading and writing to the store. As a result, the store must be
        accessible to every server machine within the site. The default
        implementation is built on top of a file system and stores all the
        configurations in a hierarchy of folders and files.

        :return: Dict

        """
        url = self._url + "/configStore"
        params = {"f": "json"}
        return self._con.get(url, params)
