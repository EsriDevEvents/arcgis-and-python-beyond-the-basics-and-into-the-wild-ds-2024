from typing import Optional, Dict, Union, List

from arcgis import GIS

from ._feed import Feed
from ._util import _Util
import logging

_LOGGER = logging.getLogger(__name__)


class FeedsManager:
    """
    Used to manage a feed item.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    url                    URL of the ArcGIS Velocity organization.
    ------------------     --------------------------------------------------------------------
    gis                    An authenticated :class:`~arcgis.gis.GIS` object.
    ==================     ====================================================================

    """

    _gis = None
    _util = None

    def __init__(self, url: str, gis: GIS):
        self._gis = gis
        self._util = _Util(gis, url)

    @property
    def items(self) -> List[Feed]:
        """
        Get all feeds.

        :return: returns a collection of all configured feed tasks with feed id and feed label.

        .. code-block:: python

            # Get all feeds item

            all_feeds = feeds.items
            all_feeds

        """
        all_feeds_response = self._util._get_request("feeds")
        if all_feeds_response is not None and type(all_feeds_response) is list:
            feed_items = [
                Feed(self._gis, self._util, feed) for feed in all_feeds_response
            ]
            return feed_items
        elif all_feeds_response is None:
            _LOGGER.warning("No Feed items found for the user.")
            return []
        else:
            raise Exception(
                f"Error retrieving Feed items. Velocity response: ${all_feeds_response}"
            )

    def get(self, id) -> Feed:
        """
        Get feed by ID.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Unique ID of a feed.
        ===============     ====================================================================

        :return: endpoint response of feed for the given id and label.

        .. code-block:: python

            # Get feed by id
            # Method: <item>.get(id)

            sample_feed = feeds.get("id")

        """
        feed_item = self._util._get("feed", id)
        return Feed(self._gis, self._util, feed_item)

    def create(self, feed=None) -> Feed:
        """
        Creates a new feed configuration.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        feed                An instance of a feed such as RSS or HTTP Poller.
        ===============     ====================================================================

        :return: Id and label of the newly created feed

        .. code-block:: python

            # Usage Example of creating a feature layer feed

            # Connect to a Velocity instance

            from arcgis import GIS
            from arcgis.realtime.velocity.feeds_manager import Feed

            gis = GIS(
                url="https://url.link",
                username="user_name",
                password="user_password",
            )

            velocity = gis.velocity
            feeds = gis.velocity.feeds
            feeds

            # Configure the Feature Layer Feed

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

            # feature layer properties

            name = "feature_layer_name"
            description = "feature_layer_description"
            url = "feature_layer_url"
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

            # Set time field

            time = TimeInterval(
                interval_start_field="start_field",
                interval_end_field="end_field"
                # time instant
                # time = TimeInstant(time_field="pubDate")
                # feature_layer_config.set_time_config(time=time)
            )

            # Set recurrence

            run_interval = RunInterval(
                cron_expression="0 * * ? * * *",
                timezone="America/Los_Angeles"
            )

            # Set geometry field - configuring X,Y and Z fields

            geometry = XYZGeometry(
                x_field = "x",
                y_field = "y",
                wkid = 4326
            )

            # a single field geometry could also be configured
            # geometry = SingleFieldFeometry(
                # geometry_field="geometry_field"
                # geometry_type="esriGeometryPoint",
                # geometry_format="esrijson",
                # wkid=4326
            # )
            # feature_layer.set_geometry_config(geometry=geometry)

            feature_layer_config = FeatureLayer(
                label=name,
                description=description,
                query="1=1",
                fields="*",
                outSR=4326,
                url=url,
                extent=extent,
                time_stamp_field=time
            )

            # Manipulate the schema - rename or remove fields, change field data-type

            feature_layer_config.rename_field("org_field_name", "new_field_name")
            feature_layer_config.remove_field("description")

            # Set track id

            feature_layer_config.set_track_id("track_id")

            # Set recurrence

            feature_layer_config.run_interval = RunInterval(
                cron_expression="0 * * ? * * *", timezone="America/Los_Angeles"
            )

            # Create the feed and start it
            feature_layer_feed = feeds.create(feature_layer_config)
            feature_layer_feed.start()
            feeds.items

        """
        if feed is None:
            raise "Feed not found"
        else:
            feed_configuration = feed._build()
            response = self._util._post_request(
                "feed", id=None, payload=feed_configuration
            )
            if response is not None:
                return self.get(response["id"])
