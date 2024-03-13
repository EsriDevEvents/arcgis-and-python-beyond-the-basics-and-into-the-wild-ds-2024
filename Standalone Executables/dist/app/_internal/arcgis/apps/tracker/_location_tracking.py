import arcgis
import datetime as _dt
from arcgis._impl.common._utils import _lazy_property
from arcgis.apps.tracker import LocationTrackingError


class LocationTrackingManager:
    """
    This manages Location Sharing for an organization.
    It can be accessed from the gis as `location_tracking`

    Additional information can be found `here <https://doc.arcgis.com/en/tracker/help/configure-location-tracking.htm>`_

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    Required :class:`~arcgis.gis.GIS`. The GIS to configure location
                           sharing for.
    ==================     ====================================================================
    """

    def __init__(self, gis):
        self._gis = gis

    def _validate_environment(self, location_tracking_enabled=True):
        if self._gis is None:
            raise LocationTrackingError("You must use a valid GIS")
        if float(self._gis.properties.get("currentVersion", "0")) < 7.1:
            raise LocationTrackingError(
                "Location Tracking requires ArcGIS Enterprise 10.7+ or ArcGIS Online"
            )
        if self._gis.users.me.role != "org_admin":
            raise LocationTrackingError(
                "You must be an administrator to configure Location Tracking"
            )
        if self._gis.properties.get("isPortal", False) and not self._gis.properties.get(
            "hasSpatioTemporalArcGISDataStore", False
        ):
            raise LocationTrackingError(
                "Location Tracking requires the Spatiotemporal Big Data Store"
            )
        if (
            location_tracking_enabled
            and "locationTracking" not in self._gis.properties.helperServices
        ):
            raise LocationTrackingError("Location Tracking is not enabled.")
        if (
            not location_tracking_enabled
            and "locationTracking" in self._gis.properties.helperServices
        ):
            raise LocationTrackingError("Location Tracking is already enabled.")

    @property
    def _use_location_sharing(self):
        return float(self._gis.properties.get("currentVersion", "0")) >= 10.1

    def enable(
        self,
        tracks_layer_shards: int = 6,
        lkl_layer_shards: int = 3,
        tracks_layer_rolling_index_strategy: str = "Monthly",
    ):
        """
        Enables location sharing for the organization.

        ===================================       ===============================================================
        **Parameter**                              **Description**
        -----------------------------------       ---------------------------------------------------------------
        tracks_layer_shards                       The number of shards to use for the tracks layer. This only
                                                  applies for ArcGIS Enterprise.
        -----------------------------------       ---------------------------------------------------------------
        lkl_layer_shards                          The number of shards to use for the last known location layer.
                                                  This only applies for ArcGIS Enterprise.
        -----------------------------------       ---------------------------------------------------------------
        tracks_layer_rolling_index_strategy       The rolling index strategy for the tracks layer
                                                  ["Daily", "Weekly", "Monthly", "Yearly", "Decade", "Century"]
                                                  This only applies for ArcGIS Enterprise.
        ===================================       ===============================================================

        :return: True if successful, False otherwise
        """
        if self.status == "enabled":
            return False
        self._validate_environment(location_tracking_enabled=False)
        if tracks_layer_rolling_index_strategy not in [
            "Daily",
            "Weekly",
            "Monthly",
            "Yearly",
            "Decade",
            "Century",
        ]:
            raise ValueError(
                f"Invalid rolling index strategy '{tracks_layer_rolling_index_strategy}'"
            )
        if float(
            self._gis.properties.get("currentVersion", "0")
        ) <= 7.3 and tracks_layer_rolling_index_strategy in ["Century", "Decade"]:
            raise ValueError(
                f"'{tracks_layer_rolling_index_strategy}' is not supported for this version of Enterprise"
            )
        folder_title = (
            "Location Sharing" if self._use_location_sharing else "Location Tracking"
        )
        for folder in self._gis.users.me.folders:
            if folder["title"] == folder_title:
                break
        else:
            self._gis.content.create_folder(folder_title)
        service_name = "location_tracking"
        if not self._gis.content.is_service_name_available(
            service_name, service_type="featureService"
        ):
            service_name = f"{service_name}{int(_dt.datetime.now().timestamp())}"
        create_params = {
            "name": f"{service_name}",
            "layers": [
                {
                    "adminLayerInfo": {
                        "tableMetadata": {
                            "numberOfShards": f"{tracks_layer_shards}",
                            "rollingIndexStrategy": f"{tracks_layer_rolling_index_strategy}",
                            "dataRetentionStrategy": "30",
                            "dataRetentionStrategyUnits": "DAYS",
                            "dataRetention": "true",
                        }
                    }
                },
                {
                    "adminLayerInfo": {
                        "tableMetadata": {
                            "numberOfShards": f"{lkl_layer_shards}",
                            "rollingIndexStrategy": "Yearly",
                            "dataRetentionStrategy": "30",
                            "dataRetentionStrategyUnits": "DAYS",
                            "dataRetention": "false",
                        }
                    }
                },
            ],
            "description": f"Location {'Sharing' if self._use_location_sharing else 'Tracking'} Service",
            "snippet": f"Location {'Sharing' if self._use_location_sharing else 'Tracking'} Service",
        }
        # Use a longer rolling index strategy if 10.8.1 or later
        if float(self._gis.properties.get("currentVersion", "0")) > 7.3:
            create_params["layers"][1]["adminLayerInfo"]["tableMetadata"][
                "rollingIndexStrategy"
            ] = "Century"
        item = self._gis.content.create_service(
            "location_tracking",
            create_params=create_params,
            folder=folder_title,
            service_type="locationTrackingService",
        )
        item.protect(True)
        item.update(
            {
                "description": f"The location {'Sharing' if self._use_location_sharing else 'Tracking'} service stores the last known location of each mobile user, "
                "as well as full historical tracks of where the mobile user has been. It is part of an "
                "organization-wide capability that is managed by an administrator.",
                "snippet": f"Location {'Sharing' if self._use_location_sharing else 'Tracking'} Service",
            }
        )
        item.share(org=True)
        self._gis.update_properties(
            {"locationTrackingService": {"url": item.url, "id": item.itemid}}
        )
        return True

    def pause(self):
        """
        Pauses location sharing for the organization.

        :return: True if successful, False otherwise
        """
        self._validate_environment()
        if self.status == "enabled":
            return bool(
                self._service.manager.update_definition({"capabilities": "Query"})[
                    "success"
                ]
            )
        else:
            return False

    def resume(self):
        """
        Resumes location sharing for the organization after it was paused.

        :return: True if successful, False otherwise
        """
        self._validate_environment()
        if self.status == "paused":
            return bool(
                self._service.manager.update_definition(
                    {"capabilities": "Query,Create,Update"}
                )["success"]
            )
        else:
            return False

    def disable(self):
        """
        Disables location sharing for the organization.

        THIS WILL DELETE ALL LOCATION SHARING VIEWS, SERVICES, AND DATA.

        :return: True if successful, False otherwise
        """
        item = self.item
        if self.status == "disabled":
            possible_lts = self._gis.content.search(
                query='typekeywords:"Location Tracking Service" NOT typekeywords:"Location Tracking View"'
            )
            if len(possible_lts) == 0:
                return False
            else:
                item = possible_lts[0]
        folder_id = item.ownerFolder
        views = item.related_items("Service2Service", "forward")
        for view in views:
            if (
                hasattr(view, "properties")
                and view.properties is not None
                and "trackViewGroup" in view.properties
            ):
                group = self._gis.groups.get(view.properties["trackViewGroup"])
                if group:
                    group.protected = False
                    group.delete()
            view.protect(False)
            view.delete()
        item.protect(False)
        item.delete()
        self._gis.update_properties({"locationTrackingService": "null"})
        if (
            len(
                self._gis.content.search(
                    """owner:"{}" ownerfolder:{}""".format(item.owner, folder_id)
                )
            )
            == 0
        ):
            folder_title = (
                "Location Sharing"
                if self._use_location_sharing
                else "Location Tracking"
            )
            for folder in self._gis.users.get(item.owner).folders:
                if folder["title"] == folder_title:
                    self._gis.content.delete_folder(folder["title"], owner=item.owner)
                    break
        return True

    def create_track_view(self, title: str):
        """
        This creates a :class:`~arcgis.apps.tracker.TrackView`.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title                  Required String. The title of the Track View to create.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.tracker.TrackView`
        """
        self._validate_environment()
        folder = None
        if self.item.ownerFolder is not None:
            for f in self._gis.users.get(self.item.owner).folders:
                if f["id"] == self.item.ownerFolder:
                    folder = f["title"]
                    break
        group = self._gis.groups.create(
            title,
            f"Location {'Sharing' if self._use_location_sharing else 'Tracking'} Group",
            is_view_only=True,
            is_invitation_only=True,
            access="private",
        )
        group.protected = True
        item = self._gis.content.create_service(
            "{}_Track_View".format(group.id),
            create_params={"name": "{}_Track_View".format(group.id), "isView": True},
            folder=folder,
            service_type="locationTrackingService",
            is_view=True,
            owner=self.item.owner,
        )
        item.protect(True)
        item.update({"title": title, "properties": {"trackViewGroup": group.id}})
        definition = {"viewDefinitionQuery": "created_user in ('')"}
        # Workaround for 10.7 bug where time wasn't enabled
        if self._gis.properties.isPortal and self._gis.version <= [7, 1]:
            definition["timeInfo"] = {"startTimeField": "location_timestamp"}
        item.layers[0].manager.update_definition(definition)
        item.layers[1].manager.update_definition(definition)
        # Older versions of the LTS may not have this layer
        if len(item.layers) >= 3:
            item.layers[2].manager.update_definition(definition)
        if self._gis.properties.isPortal:
            # Set allowOthersToQuery to True - workaround for missing feature in Enterprise 10.7/10.7.1
            arcgis.features.FeatureLayerCollection(
                item.url, self._gis
            ).manager.update_definition(
                {
                    "editorTrackingInfo": {
                        "enableOwnershipAccessControl": True,
                        "enableEditorTracking": True,
                        "allowOthersToQuery": True,
                        "allowOthersToUpdate": False,
                        "allowOthersToDelete": False,
                    }
                }
            )
        item.share(groups=[group])
        if group.owner != self.item.owner:
            group.reassign_to(self.item.owner)
            group.remove_users([self._gis.users.me])
        return arcgis.apps.tracker.TrackView(item)

    @property
    def retention_period(self):
        """
        The retention period of the Location Sharing Tracks Layer.
        This is a positive integer whose units are defined by :attr:`~arcgis.apps.tracker.LocationTrackingManager.retention_period_units`
        """
        try:
            return int(
                self.tracks_layer.manager.properties["adminLayerInfo"]["tableMetadata"][
                    "dataRetentionStrategy"
                ]
            )
        except:
            return None

    @retention_period.setter
    def retention_period(self, value):
        self._validate_environment()
        if isinstance(value, str):
            if not value.isdigit():
                raise LocationTrackingError(
                    "Invalid Retention Policy Setting: '{}' expected an integer greater than 0".format(
                        value
                    )
                )
        elif isinstance(value, int):
            if value < 0:
                raise LocationTrackingError(
                    "Invalid Retention Policy Setting: '{}' expected an integer greater than 0".format(
                        value
                    )
                )
        else:
            raise LocationTrackingError(
                "Invalid Retention Policy Setting: '{}' expected an integer greater than 0".format(
                    value
                )
            )
        self.tracks_layer.manager.update_definition(
            {"tableMetadata": {"dataRetentionStrategy": "{}".format(value)}}
        )

    @property
    def retention_period_units(self):
        """The retention period units ("HOURS", "DAYS", "MONTHS", "YEARS") of the Location Sharing Tracks Layer"""
        try:
            return self.tracks_layer.manager.properties["adminLayerInfo"][
                "tableMetadata"
            ]["dataRetentionStrategyUnits"]
        except:
            return None

    @retention_period_units.setter
    def retention_period_units(self, value):
        self._validate_environment()
        if not isinstance(value, str):
            raise LocationTrackingError(
                "Invalid Retention Policy Setting: '{}' expected HOURS, DAYS, MONTHS, or YEARS".format(
                    value
                )
            )
        elif value.upper() not in ("HOURS", "DAYS", "MONTHS", "YEARS"):
            raise LocationTrackingError(
                "Invalid Retention Policy Setting: '{}' expected HOURS, DAYS, MONTHS, or YEARS".format(
                    value
                )
            )
        self.tracks_layer.manager.update_definition(
            {"tableMetadata": {"dataRetentionStrategyUnits": "{}".format(value)}}
        )

    @property
    def retention_period_enabled(self):
        """A boolean indicating if the retention period is enabled"""
        try:
            return (
                self.tracks_layer.manager.properties["adminLayerInfo"]["tableMetadata"][
                    "dataRetention"
                ].lower()
                == "true"
            )
        except:
            return None

    @retention_period_enabled.setter
    def retention_period_enabled(self, value):
        self._validate_environment()
        if isinstance(value, str):
            if value.lower() not in ("false", "true"):
                raise LocationTrackingError(
                    "Invalid Retention Policy Setting: '{}' expected True or False".format(
                        value
                    )
                )
        elif not isinstance(value, bool):
            raise LocationTrackingError(
                "Invalid Retention Policy Setting: '{}' expected True or False".format(
                    value
                )
            )
        self.tracks_layer.manager.update_definition(
            {"tableMetadata": {"dataRetention": "{}".format(str(value).lower())}}
        )

    @_lazy_property
    def item(self):
        """The Location Sharing :class:`~arcgis.gis.Item`"""
        try:
            return self._gis.content.get(
                self._gis.properties.helperServices["locationTracking"]["id"]
            )
        except:
            return None

    @_lazy_property
    def tracks_layer(self):
        """The tracks :class:`~arcgis.features.FeatureLayer`"""
        try:
            return self.item.layers[0]
        except:
            return None

    @_lazy_property
    def last_known_locations_layer(self):
        """The last known locations :class:`~arcgis.features.FeatureLayer`"""
        try:
            return self.item.layers[1]
        except:
            return None

    @_lazy_property
    def track_lines_layer(self):
        """The track lines :class:`~arcgis.features.FeatureLayer`"""
        try:
            return self.item.layers[2]
        except IndexError:
            return None

    @property
    def status(self):
        """The status of location sharing ("disabled", "paused", "enabled")"""
        try:
            if "locationTracking" not in self._gis.properties.helperServices:
                return "disabled"
            if "Create" in self._service.properties["capabilities"]:
                return "enabled"
            else:
                return "paused"
        except:
            return "disabled"

    @_lazy_property
    def _service(self):
        try:
            return arcgis.features.FeatureLayerCollection(
                self._gis.properties.helperServices["locationTracking"]["url"],
                self._gis,
            )
        except:
            return None
