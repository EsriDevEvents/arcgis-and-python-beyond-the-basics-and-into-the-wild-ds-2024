from typing import Union, Dict, Any, Optional, ClassVar
from dataclasses import field, dataclass

from arcgis.realtime import Velocity
from arcgis.realtime.velocity.feeds._feed_template import _FeedTemplate
from arcgis.realtime.velocity.feeds.time import _HasTime, TimeInstant, TimeInterval


@dataclass
class StreamLayer(_FeedTemplate, _HasTime):
    """
    Receive features from a stream layer. This data class can be used to define the feed configuration and to
    create the feed.

    Data format is Esri stream layer. ArcGIS Velocity will automatically handle the location for you.

    ==================              ====================================================================
    **Parameter**                    **Description**
    ------------------              --------------------------------------------------------------------
    label                           String. Unique label for the feed instance.
    ------------------              --------------------------------------------------------------------
    description                     String. Feed description.
    ------------------              --------------------------------------------------------------------
    portal_item_id                  String. Portal item ID of the stream layer.
    ------------------              --------------------------------------------------------------------
    query                           String. Stream layer query parameters. The default is: "1=1"
    ------------------              --------------------------------------------------------------------
    fields                          String. Requested stream layer output fields.

                                    For example:

                                            "field1,field2"

                                    The default is: "*".
    ------------------              --------------------------------------------------------------------
    outSR                           int. Requested output spatial reference. The default is: 4326.

                                    .. note::
                                        To learn more about projected and geographic coordinate systems, refer to
                                        `Using spatial references <https://developers.arcgis.com/rest/services-reference/enterprise/using-spatial-references.htm>`_.
    ------------------              --------------------------------------------------------------------
    data_format                     String. Specifies the overall format of the incoming data.
    ==================              ====================================================================

    ==========================      ==================================================================================
    **Optional Argument**           **Description**
    ==========================      ==================================================================================
    WHERE clause                    String. Query to retrieve a subset of features.
    --------------------------      ----------------------------------------------------------------------------------
    Out fields                      String. Comma-separated list of fields to use for processing.
    --------------------------      ----------------------------------------------------------------------------------
    Output spatial reference        String. Spatial reference in which queried features should return.
    --------------------------      ----------------------------------------------------------------------------------
    extent                          Dict[str, Any]. JSON representing an envelope as defined by the ArcGIS
                                    REST API's JSON geometry schema.

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

    --------------------------      ----------------------------------------------------------------------------------
    track_id_field                  String. Name of the field from the incoming data that should be set as
                                    track ID.
    --------------------------      ----------------------------------------------------------------------------------
    time                            [:class:`~arcgis.realtime.velocity.feeds.TimeInstant`, :class:`~arcgis.realtime.velocity.feeds.TimeInterval`]. An instance of time configuration that
                                    will be used to create time information from the incoming data.
    ==========================      ==================================================================================

    :return: A data class with stream layer feed configuration.

    .. code-block:: python

        # Usage Example

        from arcgis.realtime.velocity.feeds import StreamLayer
        from arcgis.realtime.velocity.feeds.time import TimeInterval, TimeInstant

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

        stream_layer_config = StreamLayer(
            label="feed_name",
            description="feed_description",
            portal_item_id="portal_id",
            query="1=1",
            fields="*",
            outSR=4326,
            extent=extent
        )

        # use velocity object to get the FeedsManager instance
        feeds = velocity.feeds

        # use the FeedsManager object to create a feed from this feed configuration
        stream_layer_feed = feeds.create(stream_layer_config)
        stream_layer_feed.start()
        feeds.items

    """

    # fields that the user sets during init
    # Stream Layer specific properties
    portal_item_id: str
    query: str = field(default="1=1")
    fields: str = field(default="*")
    outSR: int = field(default=4326)
    extent: Optional[Dict[str, Any]] = None

    # FeedTemplate properties
    track_id_field: Optional[str] = None
    # HasTime properties
    time: Optional[Union[TimeInstant, TimeInterval]] = None
    # Stream Layer is a standard format and format properties do not need to be set in the feed configuration
    data_format: Any = field(default=None, init=False)

    # FeedTemplate properties
    _name: ClassVar[str] = "stream-layer"

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
            "properties": {"executable": True},
        }

        feed_properties = self._generate_feed_properties()
        self._dict_deep_merge(feed_configuration["feed"], feed_properties)
        print(feed_configuration)
        return feed_configuration

    def _generate_feed_properties(self) -> dict:
        if self.extent:
            extent_properties = {f"{self._name}.extent": self.extent}
        else:
            extent_properties = {}

        feed_properties = {
            "name": self._name,
            "properties": {
                f"{self._name}.portalItemId": self.portal_item_id,
                f"{self._name}.query": self.query,
                f"{self._name}.fields": self.fields,
                f"{self._name}.outSR": self.outSR,
                **extent_properties,
            },
        }

        return feed_properties
