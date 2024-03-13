from arcgis import GIS
from typing import Optional, Dict, Union, List

from ._realtime_analytics import RealTimeAnalytics
from ._util import _Util
import logging

_LOGGER = logging.getLogger(__name__)


class RealTimeAnalyticsManager:
    """
    Used to manage real-time analytic items.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
      url                    URL of the ArcGIS Velocity organization.
    ------------------     --------------------------------------------------------------------
      gis                    an authenticated :class:`arcigs.gis.GIS` object.
    ==================     ====================================================================

    """

    _gis = None
    _util = None

    def __init__(self, url: str, gis: GIS):
        self._gis = gis

        self._util = _Util(gis, url)

    @property
    def items(self) -> List[RealTimeAnalytics]:
        """
        Get all real-time analytic items.

        :return: returns a collection of all real-time analytics items with id and label.

        .. code-block:: python

            # Get all real-time analytics items

            all_realtime_analytics = realtime_analytics.items
            all_realtime_analytics

        """
        all_realtime_analytics_response = self._util._get_request("analytics/realtime")

        if (
            all_realtime_analytics_response is not None
            and type(all_realtime_analytics_response) is list
        ):
            realtime_analytics_items = [
                RealTimeAnalytics(self._gis, self._util, realtime_item)
                for realtime_item in all_realtime_analytics_response
            ]
            return realtime_analytics_items
        elif all_realtime_analytics_response is None:
            _LOGGER.warning("No Real-time analytic items found for the user.")
            return []
        else:
            raise Exception(
                f"Error retrieving Real-time analytic items. Velocity response: ${all_realtime_analytics_response}"
            )

    def get(self, id) -> RealTimeAnalytics:
        """
        Get real-time analytic item by ID.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
          id                     Unique ID of a real-time analytic.
        ==================     ====================================================================

        :return: endpoint response of real-time analytics for the given id and label

        .. code-block:: python

            # Get real-time analytics by id
            # Method: <item>.get(id)

            sample_realtime_task = realtime_analytics.get("id")

        """
        realtime_analytics_item = self._util._get("analytics/realtime", id)
        return RealTimeAnalytics(self._gis, self._util, realtime_analytics_item)
