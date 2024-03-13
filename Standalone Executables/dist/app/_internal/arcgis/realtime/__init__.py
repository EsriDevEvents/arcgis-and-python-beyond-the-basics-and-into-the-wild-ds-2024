"""
The arcgis.realtime module provides API functions to work with and automate real-time data feeds, continuous processing and analysis of streaming data, and stream layers.

This module contains the following:

    + :class:`~arcgis.realtime.StreamLayer` provides types and functions for receiving real-time data feeds and sensor data streamed from
      the GIS to perform continuous processing and analysis. It includes support for stream layers that allow Python scripts
      to subscribe to the streamed feature data or broadcast updates or alerts.
    + The :class:`~arcgis.realtime.Velocity`  class and the various submodules provide API functions to automate the ArcGIS Velocity REST API.

"""

from .stream_layer import StreamLayer
from .velocity import Velocity
