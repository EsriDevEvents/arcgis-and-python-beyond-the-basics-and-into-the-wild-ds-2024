from typing import Union, Dict, Any, Optional, ClassVar
from dataclasses import field, dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds._feed_template import _FeedTemplate
from arcgis.realtime.velocity.feeds.run_interval import RunInterval
from arcgis.realtime.velocity.feeds.time import _HasTime, TimeInstant, TimeInterval


@dataclass
class FeatureLayer(_FeedTemplate, _HasTime):
    """
    Poll a feature layer for features at a fixed schedule. This data class can be used to define the feed configuration
    and to create the feed.

    The data format is a feature layer. ArcGIS Velocity will automatically handle the location for you.

    =====================           ====================================================================
    **Parameter**                    **Description**
    ---------------------           --------------------------------------------------------------------
    label                           String. Unique label for this feed instance.
    ---------------------           --------------------------------------------------------------------
    description                     String. Feed description.
    ---------------------           --------------------------------------------------------------------
    query                           String. Feature layer query parameters. The default is: 1=1.
    ---------------------           --------------------------------------------------------------------
    fields                          String. Requested feature layer output fields.

                                    For example:

                                           "field1, field2"

                                    The default is: *.
    ---------------------           --------------------------------------------------------------------
    outSR                           int. Requested output spatial reference. The default is: 4326.

                                    .. note::
                                        To learn more about projected and geographic coordinate systems, refer to
                                        `Using spatial references <https://developers.arcgis.com/rest/services-reference/enterprise/using-spatial-references.htm>`_.
    =====================           ====================================================================

    =====================           ====================================================================
    **Optional Argument**           **Description**
    =====================           ====================================================================
    portal_item_id                  String. The Portal :class:`~arcgis.gis.Item` ID of the feature layer.

                                    .. note::
                                        Either the portal_item_id or url is required.
    ---------------------           --------------------------------------------------------------------
    extent                          Dict[str, Any]. A Geometry object that defines the spatial extent for
                                    the feature layer.

                                    .. code-block:: python

                                        # Sample Value
                                        {
                                            "spatialReference": {
                                                "latestWkid": 3857,
                                                "wkid": 102100
                                            },
                                            "xmin": -14784278.027601289,
                                            "ymin": 2604610.848073723,
                                            "xmax": -11451317.846255329,
                                            "ymax": 6852675.132049575
                                        }

    ---------------------           --------------------------------------------------------------------
    time_stamp_field                String.
                                    An optional date field for latest features.
                                    Optionally, specify a date field to be used to retrieve only the latest
                                    features from the feature layer.

                                    If a timestamp field is not specified, ArcGIS Velocity will load all
                                    features that meet the criteria of the WHERE clause when it polls the feature layer.

                                    If a timestamp field is specified, the first time ArcGIS Velocity polls the feature
                                    layer it will load all features with a timestamp field datetime within the past
                                    minute and less than the first feed poll time that also meets the criteria of the
                                    WHERE clause. With each subsequent poll, only features with a timestamp field value
                                    between the last polling time and the current polling time that also meet the
                                    criteria of the WHERE clause will be loaded.
    ---------------------           --------------------------------------------------------------------
    track_id_field                  String. Name of the field from the incoming data that should be set as
                                    track ID.
    ---------------------           --------------------------------------------------------------------
    time                            [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                    will be used to create time information from the incoming data.
    ---------------------           --------------------------------------------------------------------
    run_interval                    :class:`~arcgis.realtime.velocity.feeds.RunInterval`. An instance of the scheduler configuration. The default is:
                                    RunInterval(cron_expression="0 * * ? * * *", timezone="America/Los_Angeles")
    =====================           ====================================================================

    :return: A data class with feature layer feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import FeatureLayer
        from arcgis.realtime.velocity.http_authentication_type import (
            NoAuth,
            BasicAuth,
            CertificateAuth,
        )

        from arcgis.realtime.velocity.input.format import DelimitedFormat
        from arcgis.realtime.velocity.feeds.geometry import XYZGeometry, SingleFieldGeometry
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant
        from arcgis.realtime.velocity.feeds.run_interval import RunInterval

        extent = {
            "spatialReference": {
                "latestWkid": 3857,
                "wkid": 102100
            },
            "xmin": "xmin",
            "ymin": "ymin",
            "xmax": "xmax",
            "ymax": "ymax"
        }

        # Feature Layer Properties

        feature_layer_config = FeatureLayer(
            label="feed_name",
            description="feed_description",
            query="1=1",
            fields="*",
            outSR=4326,
            url="feed_sample_server_link",
            extent=extent,
            time_stamp_field="date_field"
        )

        feature_layer_config

        # Set recurrence
        feature_layer_config.run_interval = RunInterval(
            cron_expression="0 * * ? * * *", timezone="America/Los_Angeles"
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        feature_layer_feed = feeds.create(feature_layer_config)
        feature_layer_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Feature Layer specific properties
    query: str = field(default="1=1")
    fields: str = field(default="*")
    outSR: int = field(default=4326)
    url: Optional[str] = None
    portal_item_id: Optional[str] = None
    extent: Optional[Dict[str, Any]] = None
    time_stamp_field: Optional[str] = None

    # FeedTemplate properties
    track_id_field: Optional[str] = None
    # HasTime properties
    time: Optional[Union[TimeInstant, TimeInterval]] = None
    # scheduler
    run_interval: RunInterval = field(
        default=RunInterval(
            cron_expression="0 * * ? * * *", timezone="America/Los_Angeles"
        )
    )
    # Feature Layer is a standard format and format properties do not need to be set in the feed configuration
    data_format: Any = field(default=None, init=False)
    # FeedTemplate properties
    _name: ClassVar[str] = "feature-layer"

    def __post_init__(self):
        if Velocity is None:
            return
        self._util = Velocity._util

        # validation of fields
        if self._util.is_valid(self.label) == False:
            raise ValueError(
                "Label should only contain alpha numeric, _ and space only"
            )
        if self.url is None and self.portal_item_id is None:
            raise ValueError("Either url or portal_item_id is required")

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

            print(
                "Feature Schema retrieved from the Feed:",
                sample_messages["featureSchema"],
            )

            # initiate actions for each of the following properties if it was set at init
            if self.track_id_field is not None:
                self.set_track_id(self.track_id_field)
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
        if self.url:
            url_or_portal_item_id = {f"{self._name}.url": self.url}
        else:
            url_or_portal_item_id = {f"{self._name}.portalItemId": self.portal_item_id}

        if self.extent:
            extent_properties = {f"{self._name}.extent": self.extent}
        else:
            extent_properties = {}

        if self.time_stamp_field:
            time_stamp_property = {
                f"{self._name}.timestampField": self.time_stamp_field
            }
        else:
            time_stamp_property = {}

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.query": self.query,
                f"{self._name}.fields": self.fields,
                f"{self._name}.outSR": self.outSR,
                **url_or_portal_item_id,
                **extent_properties,
                **time_stamp_property,
            },
        }

        return feed_properties
