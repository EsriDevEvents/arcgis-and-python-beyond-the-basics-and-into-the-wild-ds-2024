from typing import Union, Dict, Optional, ClassVar
from dataclasses import field, dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds._feed_template import _FeedTemplate
from arcgis.realtime.velocity.feeds.geometry import (
    _HasGeometry,
    SingleFieldGeometry,
    XYZGeometry,
)
from arcgis.realtime.velocity.feeds.run_interval import RunInterval
from arcgis.realtime.velocity.feeds.time import (
    _HasTime,
    TimeInstant,
    TimeInterval,
)
from arcgis.realtime.velocity.http_authentication_type import (
    NoAuth,
    BasicAuth,
    CertificateAuth,
)
from arcgis.realtime.velocity.input.format import (
    EsriJsonFormat,
    GeoJsonFormat,
    JsonFormat,
    DelimitedFormat,
    XMLFormat,
    _format_from_config,
)


@dataclass
class HttpPoller(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Poll an HTTP endpoint for event data. This data class can be used to define the feed configuration and to create
    the feed.

    ===================      ====================================================================
    **Parameter**             **Description**
    -------------------      --------------------------------------------------------------------
    label                    String. Unique label for this feed instance.
    -------------------      --------------------------------------------------------------------
    description              String. Feed description.
    -------------------      --------------------------------------------------------------------
    url                      String. URL of the HTTP endpoint providing data.
    -------------------      --------------------------------------------------------------------
    http_http_method         String. HTTP method. Options: GET or POST.
    -------------------      --------------------------------------------------------------------
    http_auth_type           [:class:`~arcgis.realtime.velocity.NoAuth`, :class:`~arcgis.realtime.velocity.BasicAuth`, :class:`~arcgis.realtime.velocity.CertificateAuth`,OAuth]. An instance that contains the
                             authentication information for this feed instance.
    -------------------      --------------------------------------------------------------------
    url_params               Dict[str, str]. A dictionary of URL param/value pairs that contains
                             HTTP params used to access the HTTP resource.
    -------------------      --------------------------------------------------------------------
    http_headers             Dict[str, str]. A Name-Value dictionary that contains HTTP headers
                             for connecting to the HTTP resource.
    -------------------      --------------------------------------------------------------------
    enable_long_polling      bool. The default is: False.
    ===================      ====================================================================

    =====================     ====================================================================
    **Optional Argument**     **Description**
    =====================     ====================================================================
    data_format               [:class:`~arcgis.realtime.velocity.input.EsriJsonFormat`, :class:`~arcgis.realtime.velocity.input.GeoJsonFormat`, :class:`~arcgis.realtime.velocity.input.DelimitedFormat`, :class:`~arcgis.realtime.velocity.input.JsonFormat`, :class:`~arcgis.realtime.velocity.input.XMLFormat`].
                              An instance that contains the data format
                              configuration for this feed. Configure only allowed formats.
                              If this is not set right during initialization, a format will be
                              auto-detected and set from a sample of the incoming data. This sample
                              will be fetched from the configuration provided so far in the init.
    ---------------------     --------------------------------------------------------------------
    track_id_field            String. Name of the field from the incoming data that should be set as
                              track ID.
    ---------------------     --------------------------------------------------------------------
    geometry                  [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                              that will be used to create geometry objects from the incoming data.
    ---------------------     --------------------------------------------------------------------
    time                      [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                              will be used to create time information from the incoming data.
    ---------------------     --------------------------------------------------------------------
    run_interval              :class:`~arcgis.realtime.velocity.feeds.RunInterval`. An instance of the scheduler configuration. The default is:
                              RunInterval(cron_expression="0 * * ? * * *", timezone="America/Los_Angeles")
    =====================     ====================================================================

    :return: A dataclass with Http poller feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import HttpPoller
        from arcgis.realtime.velocity.http_authentication_type import (
            NoAuth,
            BasicAuth,
            CertificateAuth,
        )
        arcgis.realtime.velocity.input.format import DelimitedFormat
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant
        from arcgis.realtime.velocity.feeds.run_interval import RunInterval

        name = "http_poller_feed_name"
        description = "http_poller_description_feed"
        url = "http_poller_url"
        http_auth = NoAuth()
        # http_auth = BasicAuth(username="username", password="password")
        # http_auth = CertificateAuth(pfx_file_http_location="http_auth_link", password="password")

        http_headers = {"Content-Type": "application/json"}
        url_params = {"f": "json"}

        http_poller = HttpPoller(
            label=name,
            description=description,
            url=url,
            http_method="GET",
            http_auth_type=http_auth,
            url_params=url_params,
            http_headers=http_headers,
            enable_long_polling=False,
            data_format=None
        )

        # Set track id field
        http_poller.set_track_id("track_id")

        # Set time field
        time = TimeInstant(time_field="time_field")
        http_poller.set_time_config(time=time)

        # Set geometry field
        geometry = XYZGeometry(
            x_field="x",
            y_field="y",
            wkid=4326
        )
        http_poller.set_geometry_config(geometry=geometry)

        # Set recurrence
        http_poller.run_interval = RunInterval(
            cron_expression="0 * * ? * * *", timezone="America/Los_Angeles"
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        http_poller_feed = feeds.create(http_poller)
        http_poller_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # HTTP Poller specific properties
    url: str
    http_method: str
    http_auth_type: Union[NoAuth, BasicAuth, CertificateAuth]
    url_params: Dict[str, str] = field(default_factory=dict)
    http_headers: Dict[str, str] = field(default_factory=dict)
    enable_long_polling: bool = field(default=False)

    # user can define these properties even after initialization
    data_format: Optional[
        Union[
            EsriJsonFormat,
            GeoJsonFormat,
            JsonFormat,
            DelimitedFormat,
            XMLFormat,
        ]
    ] = None
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
    _name: ClassVar[str] = "http-poller"

    def __post_init__(self):
        if Velocity is None:
            return
        self._util = Velocity._util

        # validation of fields
        if self._util.is_valid(self.label) == False:
            raise ValueError(
                "Label should only contain alpha numeric, _ and space only"
            )
        elif self.http_method not in ("POST", "GET"):
            raise ValueError("http_post str can either be 'POST' or 'GET'.")

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
        # url params
        if bool(self.url_params):
            url_params_properties = {f"{self._name}.urlParameters": self.url_params}
        else:
            url_params_properties = {}

        # http authentication type
        auth_properties = self.http_auth_type._build(self._name)

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.url": self.url,
                f"{self._name}.httpMethod": self.http_method,
                f"{self._name}.isLongPolling": self.enable_long_polling,
                **auth_properties,
                **http_headers_properties,
                **url_params_properties,
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
