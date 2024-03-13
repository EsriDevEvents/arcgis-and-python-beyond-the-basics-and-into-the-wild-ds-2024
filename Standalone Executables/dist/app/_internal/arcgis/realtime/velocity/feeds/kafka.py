from typing import Union, Optional, ClassVar
from dataclasses import dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds.kafka_authentication_type import (
    NoAuth,
    SASLPlain,
    SaslScramSha256,
    SaslScramSha512,
)
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
class Kafka(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Receive event data from a Kafka broker. This data class can be used to define the feed configuration and to
    create the feed.

    ==================      ============================================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------------------------------
    label                   String. Unique label for this feed instance.
    ------------------      --------------------------------------------------------------------------------------------
    description             String. Feed description.
    ------------------      --------------------------------------------------------------------------------------------
    brokers                 String. Comma-separated list of Kafka brokers, including the port, such as

                            ``host1.domain.com:9092`` , ``host2.domain.com:9092`` .

                            For example:

                                kafkaServer1.hostname.com:9092,kafkaServer2.hostname.com:9092
    ------------------      --------------------------------------------------------------------------------------------
    topics                  String. Topic to which the output will send messages.
    ------------------      --------------------------------------------------------------------------------------------
    authentication          [:class:`~arcgis.realtime.velocity.feeds.NoAuth`, :class:`~arcgis.realtime.velocity.feeds.SASLPlain` ,  :class:`~arcgis.realtime.velocity.feeds.SaslScramSha512` :class:`~arcgis.realtime.velocity.feeds.SaslScramSha256`]. Kafka authentication type.
    ==================      ============================================================================================

    =====================   ============================================================================================
    **Optional Argument**   **Description**
    =====================   ============================================================================================
    consumer_group_id       String. A unique string that identifies the consumer group this feed
                            belongs to as a consumer.
    ---------------------   --------------------------------------------------------------------------------------------
    data_format             [:class:`~arcgis.realtime.velocity.input.EsriJsonFormat`, :class:`~arcgis.realtime.velocity.input.GeoJsonFormat`, :class:`~arcgis.realtime.velocity.input.DelimitedFormat`, :class:`~arcgis.realtime.velocity.input.JsonFormat`, :class:`~arcgis.realtime.velocity.input.XMLFormat`].
                            An instance that contains the data format
                            configuration for this feed. Configure only allowed formats.
                            If this is not set right during initialization, a format will be
                            auto-detected and set from a sample of the incoming data. This sample
                            will be fetched from the configuration provided so far in the init.
    ---------------------   --------------------------------------------------------------------------------------------
    track_id_field          String. name of the field from the incoming data that should be set as
                            track ID.
    ---------------------   --------------------------------------------------------------------------------------------
    geometry                [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                            that will be used to create geometry objects from the incoming data.
    ---------------------   --------------------------------------------------------------------------------------------
    time                    [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                            will be used to create time information from the incoming data.
    =====================   ============================================================================================

    :return: A data class with Kafka feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import Kafka
        from arcgis.realtime.velocity.feeds.kafka_authentication_type import NoAuth, SASLPlain
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

        kafka_config = Kafka(
            label="feed_name",
            description="feed_description",
            brokers="kafka.a4iot.com:9092",
            topics="topicName",
            authentication=NoAuth(),
            data_format=None
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        kafka_feed = feeds.create(kafka_config)
        kafka_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Kafka specific properties
    brokers: str
    topics: str
    authentication: Union[NoAuth, SASLPlain, SaslScramSha256, SaslScramSha512]
    consumer_group_id: Optional[str] = None

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
    _name: ClassVar[str] = "kafka"

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
        if self.consumer_group_id:
            consumer_group_id_prop = {
                f"{self._name}.consumerGroupId": self.consumer_group_id
            }
        else:
            consumer_group_id_prop = {}

        # kafka authentication type
        auth_properties = self.authentication._build(self._name)

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.brokers": self.brokers,
                f"{self._name}.topics": self.topics,
                **consumer_group_id_prop,
                **auth_properties,
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
