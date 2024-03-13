from typing import Union, Optional, ClassVar
from dataclasses import field, dataclass

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
class MQTT(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Receive events from an MQTT broker. This data class can be used to define the feed configuration and to
    create the feed.

    ==================          ========================================================================================
    **Parameter**                **Description**
    ------------------          ----------------------------------------------------------------------------------------
    label                       String. Unique label for this feed instance.
    ------------------          ----------------------------------------------------------------------------------------
    description                 String. Feed description.
    ------------------          ----------------------------------------------------------------------------------------
    host                        String. Hostname of the of the broker prefixed with "``tcp://``" for non-SSL or
                                "``ssl://``" for SSL connections.
    ------------------          ----------------------------------------------------------------------------------------
    port                        int. Port on which the MQTT broker is accessible.
    ------------------          ----------------------------------------------------------------------------------------
    topic                       String. Topic over which event messages stream.
    ------------------          ----------------------------------------------------------------------------------------
    qos_level                   int. Quality of Service (QoS) level defines the guarantee of delivery for a specific
                                message. In MQTT 3.1.1, a QoS of 0 means a message is delivered at most once, a QoS of 1
                                at least once, and a QoS of 2 exactly once. The default is: 0.
    ==================          ========================================================================================

    =====================       ========================================================================================
    **Optional Argument**       **Description**
    =====================       ========================================================================================
    username                    String. Username for basic authentication.
    ---------------------       ----------------------------------------------------------------------------------------
    password                    String. Password for basic authentication.
    ---------------------       ----------------------------------------------------------------------------------------
    client_id                   String. Client ID ArcGIS Velocity will use to connect to the MQTT broker.
    ---------------------       ----------------------------------------------------------------------------------------
    data_format                 [:class:`~arcgis.realtime.velocity.input.EsriJsonFormat`, :class:`~arcgis.realtime.velocity.input.GeoJsonFormat`, :class:`~arcgis.realtime.velocity.input.DelimitedFormat`, :class:`~arcgis.realtime.velocity.input.JsonFormat`, :class:`~arcgis.realtime.velocity.input.XMLFormat`].
                                An instance that contains the data format configuration for this feed. Configure only
                                allowed formats. If this is not set right during initialization, a format will be
                                auto-detected and set from a sample of the incoming data. This sample will be fetched
                                from the configuration provided so far in the init.
    ---------------------       ----------------------------------------------------------------------------------------
    track_id_field              String. name of the field from the incoming data that should be set as
                                track ID.
    ---------------------       ----------------------------------------------------------------------------------------
    geometry                    [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                                that will be used to create geometry objects from the incoming data.
    ---------------------       ----------------------------------------------------------------------------------------
    time                        [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                will be used to create time information from the incoming data.
    =====================       ========================================================================================

    :return: A data class with MQTT feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import MQTT
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

        mqtt_config = MQTT(
            label="feed_name",
            description="feed_description",
            host="Mqtt host",
            port=8883,
            topic="Mqtt topic",
            qos_level=0,
            username="Mqtt_username",
            password="Mqtt_password",
            client_id="Mqtt_client_id",
            data_format=None
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        mqtt_feed = feeds.create(mqtt_config)
        mqtt_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Broker specific properties
    host: str
    port: int
    topic: str
    qos_level: int = field(default=0)
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None

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
    _name: ClassVar[str] = "mqtt"

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
        if self.username:
            username_prop = {f"{self._name}.username": self.username}
        else:
            username_prop = {}

        if self.password:
            password_prop = {f"{self._name}.password": self.password}
        else:
            password_prop = {}

        if self.client_id:
            client_id_prop = {f"{self._name}.clientid": self.client_id}
        else:
            client_id_prop = {}

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.host": self.host,
                f"{self._name}.port": self.port,
                f"{self._name}.topic": self.topic,
                f"{self._name}.qos": self.qos_level,
                **username_prop,
                **password_prop,
                **client_id_prop,
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
