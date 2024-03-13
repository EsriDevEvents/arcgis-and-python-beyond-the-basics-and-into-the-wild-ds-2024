"""
arcgis.realtime.Velocity provides API functions to automate the ArcGIS Velocity REST API.
ArcGIS Velocity is a real-time and big data processing and analysis capability of ArcGIS Online.
It enables you to ingest, visualize, analyze, store, and act upon data from Internet of Things (IoT) sensors.

"""
from .velocity import Velocity

from .velocity import FeedsManager
from .velocity import RealTimeAnalyticsManager
from .velocity import BigDataAnalyticsManager

from .feeds_manager import Feed
from .realtime_analytics_manager import RealTimeAnalytics
from .bigdata_analytics_manager import BigDataAnalytics
from .http_authentication_type import NoAuth, CertificateAuth, BasicAuth
from ._reserved_fields import _ReservedFields
