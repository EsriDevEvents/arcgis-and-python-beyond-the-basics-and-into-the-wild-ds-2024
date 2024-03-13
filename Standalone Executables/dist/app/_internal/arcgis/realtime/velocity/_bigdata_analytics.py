from arcgis import GIS
from typing import Optional, Dict, Union, List
from ._task import Task
from ._util import _Util


class BigDataAnalytics(Task):
    """
    The ``BigDataAnalytics`` class implements Task and provides public facing methods to
    access BigDataAnalytics API endpoints.
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
        Start the Big Data Analytics for the given ID.

        :return: response of bigdata_analytics start

        .. code-block:: python

            # Start big data analytics

            # Method: <item>.start()

            sample_bigdata_task.start()
        """
        return self._util._start("analytics/bigdata", self._id)

    # ----------------------------------------------------------------------
    def stop(self) -> Dict:
        """
        Stop the Big Data Analytics for the given ID.
        Return True if the Big Data Analytics was successfully stopped.

        :return: boolean

        .. code-block:: python

            # Stop big data analytics

            # Method: <item>.stop()

            sample_bigdata_task.stop()
        """
        return self._util._stop("analytics/bigdata", self._id)

    # ----------------------------------------------------------------------
    @property
    def status(self) -> Dict:
        """
        Get the status of the running Big Data Analytics for the given ID.

        :return: response of Big Data Analytics status

        .. code-block:: python

            # Retrieve status of big data analytics task

            # Property: <item>.status()

            status = sample_bigdata_task.status
            status
        """
        return self._util._status("analytics/bigdata", self._id)

    # ----------------------------------------------------------------------
    @property
    def metrics(self) -> Dict:
        """
        Get the metrics of the running Big Data Analytics for the given ID.

        :return: response of Big Data Analytics metrics

        .. code-block:: python

            # Retrieve metrics of big data analytics task

            # Property: <item>.metrics()

            metrics = sample_bigdata_task.metrics
            metrics
        """
        return self._util._metrics("analytics/bigdata/metrics", self._id)

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes an existing Big Data Analytics instance.

        :return: A boolean containing True (for success) or
         False (for failure) a dictionary with details is returned.

        .. code-block:: python

            # Delete a big data analytics

            # Method: <item>.delete()

            sample_bigdata_task.delete
        """
        return self._util._delete("analytics/bigdata", self._id)
