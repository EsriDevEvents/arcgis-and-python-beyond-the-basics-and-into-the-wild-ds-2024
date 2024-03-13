from arcgis import GIS
from typing import Optional, Dict, Union, List
from ._task import Task
from ._util import _Util


class Feed(Task):
    """
    The ``Feed`` class implements Task and provides public facing methods to access Feed API endpoints.
    """

    _id = ""
    _gis = None
    _util = None
    _item = None

    def __init__(self, gis: GIS, util: _Util, item: Optional[Dict] = None):
        self._gis = gis
        self._util = util
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
        Start the Feed for the given ID.

        :return: response of feed start

        .. code-block:: python

            # start feed

            # Method: <item>.start

            sample_feed.start()

        """
        return self._util._start("feed", self._id)

    # ----------------------------------------------------------------------
    def stop(self) -> Dict:
        """
        Stop the Feed for the given ID.
        Return True if the Feed was successfully stopped.

        :return: boolean

        .. code-block:: python

            # stop feed

            # Method: <item>.stop

            sample_feed.stop()

        """
        return self._util._stop("feed", self._id)

    # ----------------------------------------------------------------------
    @property
    def status(self) -> Dict:
        """
        Get the status of the running Feed for the given ID.
        :return: response of Feed status

        .. code-block:: python

            # Retrieve status of sample_feed

            # Method: <item>.status

            Sample_feed = feeds.get("id")
            status = sample_feed.status
            status
        """
        return self._util._status("feed", self._id)

    # ----------------------------------------------------------------------
    @property
    def metrics(self) -> Dict:
        """
        Get the metrics of the running Feed for the given ID.

        :return: response of feed metrics

        .. code-block:: python

            # Method: <item>.metrics

            # Retrieve metrics of sample_feed

            metrics = sample_feed.metrics
            metrics
        """
        return self._util._metrics("feed/metrics", self._id)

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes an existing Feed instance.

        :return: A boolean containing True (for success) or
         False (for failure) a dictionary with details is returned.

        .. code-block:: python

            # Delete a feed

            # Method: <item>.delete()

            sample_feed.delete()
        """
        return self._util._delete("feed", self._id)
