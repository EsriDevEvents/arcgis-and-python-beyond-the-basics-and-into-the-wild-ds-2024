from arcgis import GIS
from typing import Optional, Dict, Union, List
from ._task import Task
from ._util import _Util


class RealTimeAnalytics(Task):
    """
    the ``RealTimeAnalytics`` class implements Task and provides public facing methods to
    access RealTimeAnalytics API endpoints.
    """

    _id = ""
    _gis = None
    _util = None
    _item = None

    def __init__(self, gis: GIS, util: _Util, item: Optional[Dict] = None):
        self._gis = gis
        self._util = util

        if item:
            self._item = item
            self._id = item["id"]

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "<%s id:%s label:%s>" % (
            type(self).__name__,
            self._id,
            self._item["label"],
        )

    # ----------------------------------------------------------------------
    def start(self) -> Dict:
        """
        Start the Real-Time Analytics for the given ID.

        :return: response of realtime_analytics start

        .. code-block:: python

            # Start real-time analytics

            # Method: <item>.start()

            sample_realtime_task.start()

        """
        return self._util._start("analytics/realtime", self._id)

    # ----------------------------------------------------------------------
    def stop(self) -> Dict:
        """
        Stop the Real-Time Analytics for the given ID.
        Return True if the the Real-Time Analytics was successfully stopped.

        :return: boolean

        .. code-block:: python

            # Stop real-time analytics

            # Method: <item>.stop()

            sample_realtime_task.stop()

        """
        return self._util._stop("analytics/realtime", self._id)

    # ----------------------------------------------------------------------
    @property
    def status(self) -> Dict:
        """
        Get the status of the running Real-Time Analytics for the given ID.

        :return: response of Real-Time Analytics status

        .. code-block:: python

            # Retrieve status of real-time analytics task

            # Property: <item>.status()

            status = sample_realtime_task.status
            status

        """
        return self._util._status("analytics/realtime", self._id)

    # ----------------------------------------------------------------------
    @property
    def metrics(self) -> Dict:
        """
        Get the metrics of the running Real-Time Analytics for the given ID.

        :return: response of Real-Time Analytics metrics

        .. code-block:: python

            # Retrieve metrics of real-time analytics task

            # Property: <item>.metrics()

            metrics = sample_realtime_task.metrics
            metrics

        """
        return self._util._metrics("analytics/realtime/metrics", self._id)

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes an existing Real-Time Analytics task instance.

        :return: A boolean containing True (for success) or False (for failure) a dictionary with details is returned.

        .. code-block:: python

            # Delete a real-time analytics

            # Method: <item>.delete()

            sample_realtime_task.delete

        """
        return self._util._delete("analytics/realtime", self._id)
