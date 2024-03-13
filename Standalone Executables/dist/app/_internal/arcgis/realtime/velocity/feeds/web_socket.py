from typing import Union, Optional, ClassVar
from dataclasses import dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds._feed_template import _FeedTemplate
from arcgis.realtime.velocity.feeds.geometry import (
    _HasGeometry,
    SingleFieldGeometry,
    XYZGeometry,
)
from arcgis.realtime.velocity.feeds.time import _HasTime, TimeInstant, TimeInterval
from arcgis.realtime.velocity.input.format import (
    EsriJsonFormat,
    GeoJsonFormat,
    JsonFormat,
    DelimitedFormat,
    XMLFormat,
    _format_from_config,
)


@dataclass
class WebSocket(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Receives events from a web socket. This data class can be used to define the feed configuration and use it
    to create the feed.

    ==================          ====================================================================
    **Parameter**                **Description**
    ------------------          --------------------------------------------------------------------
    label                       String. Unique label for the feed instance.
    ------------------          --------------------------------------------------------------------
    description                 String. Feed description.
    ------------------          --------------------------------------------------------------------
    url                         String. WebSocket URL over which messages are received.
    ==================          ====================================================================

    =====================       ===============================================================================
    **Optional Argument**       **Description**
    =====================       ===============================================================================
    data_format                 [:class:`~arcgis.realtime.velocity.input.EsriJsonFormat`, :class:`~arcgis.realtime.velocity.input.GeoJsonFormat`, :class:`~arcgis.realtime.velocity.input.DelimitedFormat`, :class:`~arcgis.realtime.velocity.input.JsonFormat`, :class:`~arcgis.realtime.velocity.input.XMLFormat`].
                                An instance that contains the data format
                                configuration for this feed. Configure only allowed formats.
                                If this is not set right during initialization, a format will be
                                auto-detected and set from a sample of the incoming data. This sample
                                will be fetched from the configuration provided so far in the init.
    ---------------------       -------------------------------------------------------------------------------
    track_id_field              String. Name of the field from the incoming data that should be set as
                                track ID.
    ---------------------       -------------------------------------------------------------------------------
    geometry                    [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                                that will be used to create geometry objects from the incoming data.
    ---------------------       -------------------------------------------------------------------------------
    time                        [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                will be used to create time information from the incoming data.
    =====================       ===============================================================================

    :return: A data class with Web Socket feed configuration configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import WebSocket
        from arcgis.realtime.velocity.input.format import DelimitedFormat
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

        web_socket = WebSocket(
            label="feed_name",
            description="feed_description",
            url = "http://feed_url.com"
        )

        # create web socket feed

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        web_socket_feed = feeds.create(web_socket)
        web_socket_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Web Socket specific properties
    url: str

    # user can define these properties even after initialization
    data_format: Optional[
        Union[DelimitedFormat, EsriJsonFormat, GeoJsonFormat, JsonFormat, XMLFormat]
    ] = None
    # FeedTemplate properties
    track_id_field: Optional[str] = None
    # HasGeometry properties
    geometry: Optional[Union[XYZGeometry, SingleFieldGeometry]] = None
    # HasTime properties
    time: Optional[Union[TimeInstant, TimeInterval]] = None

    # FeedTemplate properties
    _name: ClassVar[str] = "websocket"

    def __post_init__(self):
        if Velocity is None:
            return
        self._util = Velocity._util

        # validation of fields
        if self._util.is_valid(self.label) == False:
            raise ValueError(
                "Label should only contain alpha numeric, _ and space only"
            )

        # generate dictionary of this feed object's properties that will be used to query test-connection and
        # sample-messages Rest endpoint
        feed_properties = self._generate_feed_properties()

        test_connection = self._util.test_connection(
            input_type="feed", payload=feed_properties
        )
        # Test connection to make sure feed can fetch schema
        if test_connection is True:
            # test connection succeeded. Now try getting sample messages
            sample_payload = {
                "properties": {
                    "maxSamplesToCollect": 5,
                    "timeoutInMillis": 5000,
                }
            }
            self._dict_deep_merge(sample_payload, feed_properties)
            # Sample messages to fetch schema/fields
            sample_messages = self._util.sample_messages(
                input_type="feed", payload=sample_payload
            )

            if sample_messages["featureSchema"] is not None:
                # sample messages succeeded. Get Feature Schema from it
                self._set_fields(sample_messages["featureSchema"])

            # if Format was not specified by user, use the auto-detected format from the sample messages response as this feed object's format.
            if self.data_format is None:
                self.data_format = _format_from_config(sample_messages)

            print(
                "Feature Schema retrieved from the Feed:",
                sample_messages["featureSchema"],
            )

            # initiate actions for each of the following properties if it was set at init
            if self.track_id_field is not None:
                self.set_track_id(self.track_id_field)
            if self.geometry is not None:
                self.set_geometry_config(self.geometry)
            if self.time is not None:
                self.set_time_config(self.time)

    def _build(self) -> dict:
        feed_configuration = {
            "id": "",
            "label": self.label,
            "description": self.description,
            "feed": {**self._generate_schema_transformation()},
            "properties": {"executable": True},
        }

        feed_properties = self._generate_feed_properties()
        self._dict_deep_merge(feed_configuration["feed"], feed_properties)
        print(feed_configuration)
        return feed_configuration

    def _generate_feed_properties(self) -> dict:
        feed_properties = {
            "name": self._name,
            "properties": {f"{self._name}.url": self.url},
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
