from ._feed_template import _FeedTemplate
from .run_interval import RunInterval
from .time import _HasTime, TimeInterval, TimeInstant
from .geometry import _HasGeometry, XYZGeometry, SingleFieldGeometry
from ..feeds_manager import FeedsManager
from ..velocity import Velocity
from ..http_authentication_type import (
    _HttpAuthenticationType,
    BasicAuth,
    NoAuth,
    CertificateAuth,
)
from ..input.format import RssFormat, GeoRssFormat, _format_from_config

from typing import Union, Dict, Any, Optional, ClassVar
from dataclasses import field, dataclass


@dataclass
class RSS(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Poll an HTTP endpoint for RSS events. This data class can be used to define the feed configuration and to
    create the feed.

    ==================          ========================================================================================
    **Parameter**                **Description**
    ------------------          ----------------------------------------------------------------------------------------
    label                       String. Unique label for this feed instance.
    ------------------          ----------------------------------------------------------------------------------------
    description                 String. Feed description.
    ------------------          ----------------------------------------------------------------------------------------
    rss_url                     String. Address of the HTTP endpoint providing data.
    ------------------          ----------------------------------------------------------------------------------------
    http_auth_type              [:class:`~arcgis.realtime.velocity.NoAuth`, :class:`~arcgis.realtime.velocity.BasicAuth`, :class:`~arcgis.realtime.velocity.CertificateAuth`]. An instance that contains
                                the authentication information for the feed instance.
    ------------------          ----------------------------------------------------------------------------------------
    http_headers                Dict[str, str]. A Name-Value dictionary that contains HTTP headers
                                for connecting to the RSS feed.
    ==================          ========================================================================================

    =====================       ========================================================================================
    **Optional Argument**       **Description**
    =====================       ========================================================================================
    data_format                 [:class:`~arcgis.realtime.velocity.input.RssFormat`, :class:`~arcgis.realtime.velocity.input.GeoRssFormat`]. An instance that contains the data format
                                configuration for this feed. Configure only allowed formats.
                                If this is not set right during initialization, a format will be
                                auto-detected and set from a sample of the incoming data. This sample
                                will be fetched from the configuration provided so far in the init.
    ---------------------       ----------------------------------------------------------------------------------------
    track_id_field              String. Name of the field from the incoming data that should be set as
                                track ID.
    ---------------------       ----------------------------------------------------------------------------------------
    geometry                    [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                                that will be used to create geometry objects from the incoming data.
    ---------------------       ----------------------------------------------------------------------------------------
    time                        [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                will be used to create time information from the incoming data.
    ---------------------       ----------------------------------------------------------------------------------------
    run_interval                :class:`~arcgis.realtime.velocity.feeds.RunInterval`. An instance of the scheduler configuration. The default is:
                                RunInterval(cron_expression=``0 * * ? * * *``, timezone="America/Los_Angeles")
    =====================       ========================================================================================

    :return: A data class with RSS feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import RSS
        from arcgis.realtime.velocity.http_authentication_type import (
            NoAuth,
            BasicAuth,
            CertificateAuth,
        )

        from arcgis.realtime.velocity.input.format import GeoRssFormat
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant
        from arcgis.realtime.velocity.feeds.run_interval import RunInterval

        name = "rss feed name"
        description = "rss feed description"
        url = "rss feed url"
        http_auth = NoAuth()
        # http_auth = BasicAuth(username="username", password="password")
        # http_auth = CertificateAuth(pfx_file_http_location="https://link", password="password")

        http_headers = {
            "Content-Type": "application/json"
        }

        # all properties can also be defined in the constructor as follows

        # Set data format
        data_format = GeoRssFormat()

        # Set geometry field
        geometry = XYZGeometry(
            x_field="category_longitude",
            y_field="category_latitude",
            wkid=4326,
            z_field="category_altitude",
            z_unit="Meters"
        )

        # Set time field
        time = TimeInterval(
            interval_start_field="start_field",
            interval_end_field="end_field"
        )

        # Set recurrence
        run_interval = RunInterval(
            cron_expression="0 * * ? * * *",
            timezone="America/Los_Angeles"
        )

        # Configure the RSS Feed
        rss = RSS(
            label="feed_name",
            description="feed_description",
            rss_url=url,
            http_auth_type=http_auth,
            http_headers=http_headers,
            track_id_field="track_id",
            data_format=data_format,
            geometry=geometry,
            time=time,
            run_interval=run_interval
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        RSS_feed = feeds.create(rss)
        RSS_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # RSS specific properties
    rss_url: str
    http_auth_type: Union[NoAuth, BasicAuth, CertificateAuth]
    http_headers: Dict[str, str] = field(default_factory=dict)

    # user can define these properties even after initialization
    data_format: Optional[Union[RssFormat, GeoRssFormat]] = None
    # FeedTemplate properties
    track_id_field: Optional[str] = None
    # HasGeometry properties
    geometry: Optional[Union[XYZGeometry, SingleFieldGeometry]] = None
    # HasTime properties
    time: Optional[Union[TimeInstant, TimeInterval]] = None
    # scheduler
    run_interval: RunInterval = field(
        default=RunInterval(
            cron_expression="0 * * ? * * *", timezone="America/Los_Angeles"
        )
    )

    # FeedTemplate properties
    _name: ClassVar[str] = "rss-feed"

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
        # Test connection to make sure rss can fetch schema
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

        else:
            raise AssertionError(
                "Test connection failed. Please make sure the feed has valid url"
            )

    def _build(self) -> dict:
        feed_configuration = {
            "id": "",
            "label": self.label,
            "description": self.description,
            "feed": {**self._generate_schema_transformation()},
            **self.run_interval._build(),
            "properties": {"executable": True},
        }

        feed_properties = self._generate_feed_properties()
        self._dict_deep_merge(feed_configuration["feed"], feed_properties)
        print(feed_configuration)
        return feed_configuration

    def _generate_feed_properties(self) -> dict:
        # http headers
        if bool(self.http_headers):
            http_headers_properties = {f"{self._name}.headers": self.http_headers}
        else:
            http_headers_properties = {}

        # http authentication type
        auth_properties = self.http_auth_type._build(self._name)

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.url": self.rss_url,
                **auth_properties,
                **http_headers_properties,
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
