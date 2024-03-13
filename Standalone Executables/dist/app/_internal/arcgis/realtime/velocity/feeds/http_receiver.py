from typing import AsyncGenerator, Union, Optional, ClassVar
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
class HttpReceiver(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Receive events via a dedicated HTTP endpoint. This data class can be used to define the feed configuration
    and to create the feed.

    ====================      ======================================================================
    **Parameter**              **Description**
    --------------------      ----------------------------------------------------------------------
    label                     String. Unique label for this feed instance.
    --------------------      ----------------------------------------------------------------------
    description               String. Feed description.
    --------------------      ----------------------------------------------------------------------
    authentication_type       String. Authentication type.

                              Options:

                                none or arcgis.
    --------------------      ----------------------------------------------------------------------
    sample_message            String. The sample content to auto-detect the data format.

                              For example:

                                "name,age\\nsam,23"
    ====================      ======================================================================

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
    geometry                  [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry
                              configuration that will be used to create geometry objects from the incoming data.
    ---------------------     --------------------------------------------------------------------
    time                      [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                              will be used to create time information from the incoming data.
    =====================     ====================================================================

    :return: A data class with Http receiver feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import HttpReceiver
        from arcgis.realtime.velocity.http_authentication_type import (
            NoAuth,
            BasicAuth,
            CertificateAuth,
        )

        sample_message="name,age\n
        dan,23"

        http_receiver = HttpReceiver(
            label="feed_name",
            description="feed_description",
            authentication_type="none",
            sample_message=sample_message,
            data_format=None
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        http_receiver_feed = feeds.create(http_receiver)
        http_receiver_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # HTTP Poller specific properties
    authentication_type: str
    sample_message: str

    # user can define these properties even after initialization
    data_format: Optional[
        Union[EsriJsonFormat, GeoJsonFormat, JsonFormat, DelimitedFormat, XMLFormat]
    ] = None
    # FeedTemplate properties
    track_id_field: Optional[str] = None
    # HasGeometry properties
    geometry: Optional[Union[XYZGeometry, SingleFieldGeometry]] = None
    # HasTime properties
    time: Optional[Union[TimeInstant, TimeInterval]] = None

    # FeedTemplate properties
    _name: ClassVar[str] = "http-receiver"

    def __post_init__(self):
        if Velocity is None:
            return
        self._util = Velocity._util

        # validation of fields
        if self._util.is_valid(self.label) == False:
            raise ValueError(
                "Label should only contain alpha numeric, _ and space only"
            )
        if self.authentication_type not in ("none", "arcgis"):
            raise ValueError("authentication_type must be either 'none' or 'arcgis'")
        elif not self.sample_message:
            raise ValueError("sample_message must not be empty")

        # generate dictionary of this feed object's properties that will be used to query test-connection and
        # sample-messages Rest endpoint
        feed_properties = self._generate_feed_properties()

        derived_schema = self._util.derive(sample_data=self.sample_message)
        if "schema" in derived_schema and derived_schema["schema"] is not None:
            self._set_fields(derived_schema["schema"])

            # if Format was not specified by user, use the auto-detected format from the sample messages response as this feed object's format.
            if self.data_format is None:
                self.data_format = _format_from_config(derived_schema)

            print(
                "Feature Schema retrieved from the Feed:",
                derived_schema["schema"],
            )

            # initiate actions for each of the following properties if it was set at init
            if self.track_id_field is not None:
                self.set_track_id(self.track_id_field)
            if self.geometry is not None:
                self.set_geometry_config(self.geometry)
            if self.time is not None:
                self.set_time_config(self.time)
        elif ("status", "error") in derived_schema.items():
            english_messages = map(
                lambda message: message["englishMessage"], derived_schema["messages"]
            )
            errors = "\n".join(english_messages)
            print(
                f"Could not derive schema from sample_message because of the following reasons:\n{errors}"
            )
        else:
            print("Unknown error detecting the format from sample_message")

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
            "properties": {
                f"{self._name}.httpAuthenticationType": self.authentication_type
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
