from arcgis import GIS

from .bigdata_analytics_manager import BigDataAnalyticsManager
from .feeds_manager import FeedsManager
from .realtime_analytics_manager import RealTimeAnalyticsManager
from ._util import _Util


class Velocity:
    """
    Provides access to real-time analytics, big data analytics, and feeds in ArcGIS Velocity.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    URL of the ArcGIS Velocity organization.
    ------------------     --------------------------------------------------------------------
    gis                    an authenticated :class:`arcigs.gis.GIS` object.
    ==================     ====================================================================

    .. code-block:: python

        # Connect to a Velocity instance:

        gis = GIS(url="url",username="username",password="password",)

        velocity = gis.velocity
        velocity

    .. code-block:: python

        # UsageExample:

        from arcgis.gis import GIS
        gis = GIS(url="url",username="username",password="password",)

        velocity = gis.velocity
        velocity

    """

    _gis = None
    _url = None
    _subinfo = None
    _velocity = None
    # manager instances
    _feeds = None
    _realtime_analytics = None
    _bigdata_analytics = None
    _velocity = None
    _util = None

    def __init__(self, url: str, gis: GIS):
        self._gis = gis
        self._url = url
        # Optional may set gis and _url to Velocity if we need to access these
        # in other classes internally such as: Velocity._gis, Velocity._url
        Velocity._gis = gis
        Velocity._url = url
        # set reference for Util Velocity
        Velocity._util = _Util(gis, url)

    @property
    def feeds(self) -> FeedsManager:
        """
        Provides access to the resource manager for managing configured feeds in ArcGIS Velocity.

        :return: :class:`~arcgis.realtime.velocity.FeedsManager`

        .. code-block:: python

            # Get instance of feeds from `velocity`:

            feeds = velocity.feeds
            feeds

        """
        if self._feeds is None:
            self._feeds = FeedsManager(url=self._url, gis=self._gis)
        return self._feeds

    @property
    def realtime_analytics(self) -> RealTimeAnalyticsManager:
        """
         Provides access to the resource manager for managing configured real-time analytic tasks in ArcGIS Velocity.

        :return: :class:`~arcgis.realtime.velocity.RealTimeAnalyticsManager`

        .. code-block:: python

            # Get instance of realtime_analytics from `velocity`:

            realtime_analytics = velocity.realtime_analytics
            realtime_analytics
        """
        if self._realtime_analytics is None:
            self._realtime_analytics = RealTimeAnalyticsManager(
                url=self._url, gis=self._gis
            )
        return self._realtime_analytics

    @property
    def bigdata_analytics(self) -> BigDataAnalyticsManager:
        """
        Provides access to the resource manager for managing configured big data analytic tasks in ArcGIS Velocity.

        :return: :class:`~arcgis.realtime.velocity.BigDataAnalyticsManager`

        .. code-block:: python

            # Get instance of bigdata_analytics from `velocity`:

            bigdata_analytics = velocity.bigdata_analytics
            bigdata_analytics

        """
        if self._bigdata_analytics is None:
            self._bigdata_analytics = BigDataAnalyticsManager(
                url=self._url, gis=self._gis
            )
        return self._bigdata_analytics
