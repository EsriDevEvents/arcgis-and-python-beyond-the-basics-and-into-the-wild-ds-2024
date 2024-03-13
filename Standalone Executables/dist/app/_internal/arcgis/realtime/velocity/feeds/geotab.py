from typing import Union, Optional, ClassVar
from dataclasses import field, dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds._feed_template import _FeedTemplate
from arcgis.realtime.velocity.feeds.geometry import (
    _HasGeometry,
    SingleFieldGeometry,
    XYZGeometry,
)
from arcgis.realtime.velocity.feeds.run_interval import RunInterval
from arcgis.realtime.velocity.feeds.time import _HasTime, TimeInstant, TimeInterval
from arcgis.realtime.velocity.input.format import JsonFormat, _format_from_config


@dataclass
class Geotab(_FeedTemplate, _HasTime, _HasGeometry):
    """
    Poll Geotab for event data. This data class can be used to define the feed configuration and to create the feed.

    ==================        ====================================================================
    **Parameter**              **Description**
    ------------------        --------------------------------------------------------------------
    label                     String. Unique label for this feed instance.
    ------------------        --------------------------------------------------------------------
    description               String. Feed description.
    ------------------        --------------------------------------------------------------------
    url                       String. The URL to authenticate Geotab.
    ------------------        --------------------------------------------------------------------
    database                  String. The name of the Geotab database providing data.
    ------------------        --------------------------------------------------------------------
    username                  String. Specify the username to authenticate Geotab.
    ------------------        --------------------------------------------------------------------
    password                  String. Specify the password to authenticate Geotab.
    ==================        ====================================================================

    =====================     ==========================================================================================
    **Optional Argument**     **Description**
    =====================     ==========================================================================================
    groups                    String. List of groups to include in the feature schema. Separate multiple values with a
                              semi-colon (;).
    ---------------------     ------------------------------------------------------------------------------------------
    diagnostics_ids           String. List of diagnostic IDs to include in the feature schema. Separate multiple values
                              with a semi-colon (;).
    ---------------------     ------------------------------------------------------------------------------------------
    data_format               :class:`~arcgis.realtime.velocity.input.JsonFormat`.
                              An instance that contains the data format
                              configuration for this feed. Configure only allowed formats.
                              If this is not set right during initialization, a format will be
                              auto-detected and set from a sample of the incoming data. This sample
                              will be fetched from the configuration provided so far in the init.
    ---------------------     ------------------------------------------------------------------------------------------
    track_id_field            String. Name of the field from the incoming data that should be set as
                              track ID.
    ---------------------     ------------------------------------------------------------------------------------------
    geometry                  [:class:`~arcgis.realtime.velocity.feeds.XYZGeometry`, :class:`~arcgis.realtime.velocity.feeds.SingleFieldGeometry`]. An instance of geometry configuration
                              that will be used to create geometry objects from the incoming data.
    ---------------------     ------------------------------------------------------------------------------------------
    time                      [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                              will be used to create time information from the incoming data.
    ---------------------     ------------------------------------------------------------------------------------------
    run_interval              :class:`~arcgis.realtime.velocity.feeds.RunInterval`. An instance of the scheduler configuration. The default is:
                              RunInterval(cron_expression="0 * * ? * * *", timezone="America/Los_Angeles")
    =====================     ==========================================================================================

    :return: A data class with Geotab feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import Geotab
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

        geotab = Geotab(
            label="feed_name",
            description="feed_description",
            url="Geotab_url",
            database="Geotab_database",
            username="Geotab_user_name",
            password="Geotab_password",
            data_format=None
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        geotab_feed = feeds.create(geotab)
        geotab_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Geotab specific properties
    url: str
    database: str
    username: str
    password: str
    groups: Optional[str] = None
    diagnostics_ids: Optional[str] = None

    # user can define these properties even after initialization
    data_format: Optional[JsonFormat] = None
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
    _name: ClassVar[str] = "geotab"

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
            **self.run_interval._build(),
            "properties": {"executable": True},
        }

        feed_properties = self._generate_feed_properties()
        self._dict_deep_merge(feed_configuration["feed"], feed_properties)
        print(feed_configuration)
        return feed_configuration

    def _generate_feed_properties(self) -> dict:
        if self.groups:
            groups_prop = {f"{self._name}.groupNames": self.groups}
        else:
            groups_prop = {}
        # url params
        if self.diagnostics_ids:
            diagnostics_ids_prop = {f"{self._name}.diagnosticIds": self.diagnostics_ids}
        else:
            diagnostics_ids_prop = {}

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.url": self.url,
                f"{self._name}.databaseName": self.database,
                f"{self._name}.userName": self.username,
                f"{self._name}.password": self.password,
                **groups_prop,
                **diagnostics_ids_prop,
            },
        }

        if self.data_format is not None:
            format_dict = self.data_format._build()
            self._dict_deep_merge(feed_properties, format_dict)

        return feed_properties
