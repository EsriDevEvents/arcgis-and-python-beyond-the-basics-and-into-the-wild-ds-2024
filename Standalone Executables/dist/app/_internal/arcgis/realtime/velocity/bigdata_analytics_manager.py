from typing import Optional, Dict, Union, List

from arcgis import GIS

from ._bigdata_analytics import BigDataAnalytics
from ._util import _Util
import logging

_LOGGER = logging.getLogger(__name__)


class BigDataAnalyticsManager:
    """
    Used to manage big data analytic items.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    URL of the ArcGIS Velocity organization.
    ------------------     --------------------------------------------------------------------
    gis                    An authenticated :class:`arcigs.gis.GIS` object.
    ==================     ====================================================================

    """

    _gis = None
    _util = None

    def __init__(self, url: str, gis: GIS):
        self._gis = gis

        self._util = _Util(gis, url)

    @property
    def items(self) -> List[BigDataAnalytics]:
        """
        Get all big data analytic items.

        :return: returns a collection of all configured Big Data Analytics items

        .. code-block:: python

            # Get all big data analytics

            all_bigdata_analytics = bigdata_analytics.items
            all_bigdata_analytics
        """
        all_bigdata_analytics_response = self._util._get_request("analytics/bigdata")
        if (
            all_bigdata_analytics_response is not None
            and type(all_bigdata_analytics_response) is list
        ):
            bigdata_analytics_items = [
                BigDataAnalytics(self._gis, self._util, bigdata_item)
                for bigdata_item in all_bigdata_analytics_response
            ]
            return bigdata_analytics_items
        elif all_bigdata_analytics_response is None:
            _LOGGER.warning("No Big-data Analytic items found for the user.")
            return []
        else:
            raise Exception(
                f"Error retrieving Big-data Analytic items. Velocity response: ${all_bigdata_analytics_response}"
            )

    def get(self, id) -> BigDataAnalytics:
        """
        Get big data analytic items by ID.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Unique ID of a big data analytic task.
        ===============     ====================================================================

        :return: endpoint response of Big Data Analytics for the given id and label

        .. code-block:: python

            # Get big data analytics by id
            # Method: <item>.get(id)

            sample_bigdata_task = bigdata_analytics.get("id")

        """
        bigdata_analytics_item = self._util._get("analytics/bigdata", id)
        return BigDataAnalytics(self._gis, self._util, bigdata_analytics_item)
