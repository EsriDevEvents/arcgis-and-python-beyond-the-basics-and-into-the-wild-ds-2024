"""
arcgis.realtime.StreamLayer provides types and functions for receiving real-time data feeds and sensor data streamed from
the GIS to perform continuous processing and analysis on the streaming data. It includes support for stream layers that allow Python scripts
to subscribe to the streamed feature data or to broadcast updates and alerts.

"""
from arcgis.gis import *
from arcgis.features import *

from urllib.parse import urlencode


class StreamLayer(Layer):
    """
    Allows Python scripts to subscribe to the feature data streamed from the GIS, using ArcGIS
    GeoEvent Server or ArcGIS Velocity, or to broadcast updates and alerts. This class can be used to perform continuous processing and
    analysis on streaming data as it is received.
    """

    # autobahn, twisted, pyOpenssl, service_identity
    def __init__(self, url, gis=None):
        super(StreamLayer, self).__init__(url, gis)
        self._streamtoken = self.properties.streamUrls[0].token
        self._streamurl = self.properties.streamUrls[0].urls[0]
        self._out_sr = self.properties.spatialReference.wkid
        self.filter = {}
        self._on_features = None
        self._on_disconnect = None
        self._on_error = None
        try:
            from arcgis.gis.server._service._adminfactory import AdminServiceGen

            self.service = AdminServiceGen(service=self, gis=gis)
        except:
            pass

    @property
    def out_sr(self):
        """Get/Set the spatial reference of the streamed features."""
        return self._out_sr

    @out_sr.setter
    def out_sr(self, value):
        self._out_sr = value

    @property
    def filter(self):
        """
        Get/Set property used for filtering the streamed features so they meet spatial and SQL like criteria,
        and return the specified fields.
        """
        return self._filter

    @filter.setter
    def filter(self, value):
        self._filter = value

    def subscribe(self, on_features, on_open=None, on_disconnect=None, on_error=None):
        """
        Allows Python scripts to subscribe to the feature data streamed from the GIS using ArcGIS
        GeoEvent Server or ArcGIS Velocity. Subscribing to the streamed data can be used to perform continuous processing and
        analysis of real-time data as it is received.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        on_features         callback function that is called every time features are streamed
                            to the client.
        ---------------     --------------------------------------------------------------------
        on_open             callback function called when the connection to the streaming server
                            is created.
        ---------------     --------------------------------------------------------------------
        on_disconnect       callback function called when the connection to the streaming server
                            is closed.
        ---------------     --------------------------------------------------------------------
        on_error            callback function called if the connection recieves an error.
        ===============     ====================================================================

        """
        try:
            import sys
            import ssl
            from twisted.internet import reactor
            from twisted.python import log

            from autobahn.twisted.websocket import (
                WebSocketClientFactory,
                WebSocketClientProtocol,
                connectWS,
            )
        except:
            raise ImportError(
                "Install autobahn, twisted, pyOpenssl, service_identity packages to subscribe"
            )

        url = self._streamurl

        params = {"token": self._streamtoken}
        params.update(self.filter)

        if self.out_sr != self.properties.spatialReference.wkid:
            params["outSR"] = self.out_sr

        url = "{url}/subscribe?{params}".format(url=url, params=urlencode(params))

        class StreamServiceClientProtocol(WebSocketClientProtocol):
            def onOpen(self):
                if on_open is not None:
                    on_open()

            def onMessage(self, payload, isBinary):
                if isBinary:
                    print("Binary message received: {0} bytes".format(len(payload)))
                else:
                    msg = format(payload.decode("utf8"))
                    on_features(msg)

        factory = WebSocketClientFactory(url, headers={"token": self._streamtoken})

        factory.protocol = StreamServiceClientProtocol
        connectWS(factory)

        reactor.run()
