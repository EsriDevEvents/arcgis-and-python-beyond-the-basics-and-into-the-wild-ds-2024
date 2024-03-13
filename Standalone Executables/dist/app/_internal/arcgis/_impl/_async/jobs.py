import concurrent.futures
import uuid
import datetime
import logging

_log = logging.getLogger()


def _execute(fn, **kwargs):
    """runs a generic function with optional inputs"""
    if len(kwargs) > 0:
        fn(**kwargs)
    else:
        fn()


class Job(object):
    """represents an asynchronous job"""

    _future = None
    _gis = None
    _task_name = None
    _start_time = None
    _end_time = None
    _verbose = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        future,
        task_name,
        jobid=None,
        task_url=None,
        notify=False,
        gis=None,
    ):
        self._start_time = datetime.datetime.now()
        self._task_name = task_name
        self._future = future
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)

        self._end_time = None
        self._url = task_url
        if jobid is None:
            self._jobid = uuid.uuid4().hex
        else:
            self._jobid = jobid

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self._task_name} job {self._jobid}>"

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self._task_name} job {self._jobid}>"

    # ----------------------------------------------------------------------
    def cancelled(self):
        """
        Return True if the call was successfully cancelled.

        :return: boolean
        """
        return self._future.cancelled()

    # ----------------------------------------------------------------------
    def running(self):
        """
        Return True if the call is currently being executed and cannot be cancelled.

        :return: boolean
        """
        return self._future.running()

    # ----------------------------------------------------------------------
    def done(self):
        """
        Return True if the call was successfully cancelled or finished running.

        :return: boolean
        """
        return self._future.done()

    # ----------------------------------------------------------------------
    @property
    def ellapse_time(self):
        """
        Returns the Ellapse Time for the Job
        """
        if self._end_time:
            return self._end_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time

    # ----------------------------------------------------------------------
    def _set_end_time(self, future):
        """sets the finish time"""
        self._end_time = datetime.datetime.now()

    # ----------------------------------------------------------------------
    def _notify(self, future):
        """prints finished method"""
        jobid = self._jobid
        try:
            res = future.result()
            infomsg = "{jobid} finished successfully.".format(jobid=jobid)
            _log.info(infomsg)
            print(infomsg)
        except Exception as e:
            msg = str(e)
            msg = "{jobid} failed: {msg}".format(jobid=jobid, msg=msg)
            _log.info(msg)
            print(msg)

    # ----------------------------------------------------------------------
    def result(self):
        """returns the job result"""
        return self._future.result()


class GeometryJob(Job):
    _future = None
    _gis = None
    _task_name = None
    _start_time = None
    _end_time = None
    _verbose = None
    _wkid = None

    def __init__(
        self,
        future: concurrent.futures.Future,
        task_name: str,
        jobid: str = None,
        task_url: str = None,
        notify: bool = False,
        gis=None,
        out_wkid: int = None,
    ) -> None:
        super(GeometryJob, self)
        self._start_time = datetime.datetime.now()
        self._task_name = task_name
        self._future = future
        self._wkid = out_wkid
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)

        self._end_time = None
        self._url = task_url
        if jobid is None:
            self._jobid = uuid.uuid4().hex
        else:
            self._jobid = jobid

    # ----------------------------------------------------------------------
    def result(self):
        """returns the job result"""
        from arcgis.geometry import Geometry

        if self._task_name == "lengths":
            res = self._future.result()
            return res.get("lengths", [])
        elif self._wkid:  # set the output sr to that wkid integer
            sr = {"spatialReference": {"wkid": self._wkid}}
            res = self._future.result()
            if isinstance(res, (list, tuple)):
                [g.update(sr) for g in res if not "spatialReference" in g]
                return [Geometry(g) for g in res]
            elif isinstance(res, dict) and not "spatialReference" in res:
                res.update(sr)
                return Geometry(res)
            else:
                return res
        else:
            return self._future.result()


class ItemStatusJob(Job):
    """represents an Item"""

    _future = None
    _gis = None
    _task_name = None
    _start_time = None
    _end_time = None
    _verbose = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        item,
        task_name,
        jobid=None,
        job_type=None,
        notify=False,
        gis=None,
    ):
        executor = concurrent.futures.ThreadPoolExecutor(1)
        future = executor.submit(self._status, *(item, jobid, job_type))
        executor.shutdown(False)
        self._start_time = datetime.datetime.now()
        self._task_name = task_name
        self._future = future
        if notify:
            self._future.add_done_callback(self._notify)
        self._future.add_done_callback(self._set_end_time)

        self._end_time = None
        self._url = None
        if jobid is None:
            self._jobid = uuid.uuid4().hex
        else:
            self._jobid = jobid

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self._task_name} job>"

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self._task_name} job>"

    # ----------------------------------------------------------------------
    def _status(self, item, job_id, job_type):
        """Processes the asynchronous job"""
        import time
        from arcgis.gis import Item

        while item.status(job_id=job_id, job_type=job_type).get("status") in [
            "processing"
        ]:
            status = item.status(job_id=job_id, job_type=job_type).get("status")
            if status in ["failed", "completed"]:
                return item.status(job_id=job_id, job_type=job_type)
            else:
                return item
        return item

    # ----------------------------------------------------------------------
