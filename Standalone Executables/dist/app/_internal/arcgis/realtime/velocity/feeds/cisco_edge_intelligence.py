from typing import ClassVar

from arcgis.realtime.velocity.feeds.mqtt import MQTT


class CiscoEdgeIntelligence(MQTT):
    """
    Receive events from a Cisco Edge Intelligence broker. This data class can be used to define the feed configuration
    and to create the feed.

    =====================       ========================================================================================
    **Parameter**                **Description**
    ---------------------       ----------------------------------------------------------------------------------------
    label                       String. Unique label for this feed instance.
    ---------------------       ----------------------------------------------------------------------------------------
    description                 String. Feed description.
    ---------------------       ----------------------------------------------------------------------------------------
    host                        String. Hostname of the Cisco Edge Intelligence broker prefixed with
                                "``tcp://``" for non-SSL or "``ssl://``" for SSL connections.
    ---------------------       ----------------------------------------------------------------------------------------
    port                        String. Port on which the Cisco Edge Intelligence broker is accessible.
    ---------------------       ----------------------------------------------------------------------------------------
    topic                       String. Topic over which event messages stream.
    ---------------------       ----------------------------------------------------------------------------------------
    qos_level                   int. Quality of Service (QoS) level defines the guarantee of delivery for a specific
                                message. In MQTT 3.1.1, a QoS of 0 means a message is delivered at most once, a QoS of 1
                                at least once, and a QoS of 2 exactly once. The default is: 0.
    =====================       ========================================================================================

    =====================       ========================================================================================
    **Optional Argument**       **Description**
    =====================       ========================================================================================
    username                    Username for basic authentication.
    ---------------------       ----------------------------------------------------------------------------------------
    password                    Password for basic authentication.
    ---------------------       ----------------------------------------------------------------------------------------
    client_id                   The client ID ArcGIS Velocity will use to connect to the Cisco Edge
                                Intelligence broker.
    ---------------------       ----------------------------------------------------------------------------------------
    data_format                 [:class:`~arcgis.realtime.velocity.input.EsriJsonFormat`, :class:`~arcgis.realtime.velocity.input.GeoJsonFormat`, :class:`~arcgis.realtime.velocity.input.DelimitedFormat`, :class:`~arcgis.realtime.velocity.input.JsonFormat`, :class:`~arcgis.realtime.velocity.input.XMLFormat`].
                                An instance that contains the data format configuration for this feed. Configure only
                                allowed formats. If this is not set right during initialization, a format will be
                                auto-detected and set from a sample of the incoming data. This sample will be fetched
                                from the configuration provided so far in the init.
    ---------------------       ----------------------------------------------------------------------------------------
    track_id_field              String. Name of the field from the incoming data that should be set as
                                track ID.
    ---------------------       ----------------------------------------------------------------------------------------
    geometry                    [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                                that will be used to create geometry objects from the incoming data.
    ---------------------       ----------------------------------------------------------------------------------------
    time                        [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                will be used to create time information from the incoming data.
    =====================       ========================================================================================

    :return: A data class with cisco edge intelligence feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import CiscoEdgeIntelligence
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

        cisco_edge_config = CiscoEdgeIntelligence(
            label="feed_name",
            description="feed_description",
            host="cisco_host",
            port="cisco_port",
            topic="cisco_topic",
            qos_level=0,
            username="cisco_username",
            password="cisco_password",
            client_id="cisco_client_id",
            data_format=None
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        cisco_edge_feed = feeds.create(cisco_edge_config)
        cisco_edge_feed.start()
        feeds.items

    """

    # FeedTemplate properties
    _name: ClassVar[str] = "kinetic"
