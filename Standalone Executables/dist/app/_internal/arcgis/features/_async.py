import concurrent.futures


def _run_async(fn, **inputs):
    """runs the inputs asynchronously"""
    tp = concurrent.futures.ThreadPoolExecutor(1)
    try:
        future = tp.submit(fn=fn, **inputs)
    except:
        future = tp.submit(fn, **inputs)
    tp.shutdown(False)
    return future


class EditFeatureJob(object):
    """
    Represents a Single Editing Job.  The `EditFeatureJob` class allows for the
    asynchronous operation of the :meth:`~arcgis.features.FeatureLayer.edit_features`
    method. This class is not intended for users to initialize directly, but is
    retuned by :meth:`~arcgis.features.FeatureLayer.edit_features` when `future=True`.


    ================  ===============================================================
    **Parameter**      **Description**
    ----------------  ---------------------------------------------------------------
    future            Future. The future request.
    ----------------  ---------------------------------------------------------------
    connection        The GIS connection object.
    ================  ===============================================================

    """

    _future = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, future, connection):
        """
        initializer
        """
        assert isinstance(future, concurrent.futures.Future)
        self._con = connection
        self._future = future

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<Edit Feature Job>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def task(self):
        """Returns the task name.
        :return: string
        """
        return "Edit Features Job"

    # ----------------------------------------------------------------------
    @property
    def messages(self):
        """
        returns the GP messages

        :return: List
        """
        return self._future.result()

    # ----------------------------------------------------------------------
    @property
    def status(self):
        """
        returns the Job status

        :return: bool - True means running, False means finished
        """
        return self.running()

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
    def result(self):
        """
        Return the value returned by the call. If the call hasn't yet completed
        then this method will wait.

        :return: object
        """
        try:
            res = self._future.result()
            url = res.get("resultUrl", None)
            if url is None:
                return None
            return self._con.get(url)
        except Exception as e:
            raise e
