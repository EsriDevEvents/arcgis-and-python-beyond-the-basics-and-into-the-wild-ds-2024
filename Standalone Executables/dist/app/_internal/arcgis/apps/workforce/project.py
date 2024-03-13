""" Defines the Project object.
"""
import arcgis
from arcgis.features import FeatureLayer, Table
from arcgis.gis import Group
from arcgis._impl.common._utils import _lazy_property
from warnings import warn
import json

from ._schemas import *
from .managers import *
from arcgis.apps.workforce.exceptions import WorkforceError


class Project:
    """
    A Workforce Project

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    item                   Required :class:`~arcgis.gis.Item`. The item that
                           the contains the project.

                           For a version 1 Workforce project, this is an item of type
                           `Workforce Project`. For a version 2 Workforce project, this is an
                           item of type `Feature Service` with typeKeyword `Workforce Project`
    ==================     ====================================================================

    .. code-block:: python

        # Get a Project and search the assignments and workers.

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        assignments = project.assignments.search()
        workers = project.workers.search()

        # Create v1 "Classic" Workforce project and v2 "offline-enabled" project
        v1_project = arcgis.apps.workforce.create_project('v1_project', major_version=1)
        v2_project = arcgis.apps.workforce.create_project('v2_project', major_version=2)


    """

    def __init__(self, item):
        """
        :param item: The project's arcigs.gis.Item
        """
        self.gis = item._gis
        if "Workforce Project" in item.typeKeywords:
            self._item = item
        else:
            raise WorkforceError("Incorrect item passed into Project class")
        if self._is_v2_project:
            self._item_data = item.properties
        else:
            self._item_data = item.get_data()
        self._assignment_schema = AssignmentSchema(self.assignments_layer)
        if self._supports_tracks:
            self._track_schema = TrackSchema(self.tracks_layer)
        else:
            self._track_schema = None
        self._worker_schema = WorkerSchema(self.workers_layer)
        if self._is_v2_project:
            self._assignment_types = AssignmentTypeSchema(self.assignment_types_table)
            self._integration_schema = IntegrationSchema(self.integrations_table)
        self._dispatcher_schema = DispatcherSchema(self.dispatchers_layer)
        self._update_cached_objects()

    def _update_cached_assignment_types(self):
        """
        Updates the cached assignment types
        """
        self._cached_assignment_types = {
            a.code: a for a in self.assignment_types.search()
        }

    def _update_cached_objects(self):
        """
        Caches the types, workers, and dispatchers for quicker assignment creation when querying
        Should be called when querying assignments
        """
        self._update_cached_assignment_types()
        self._cached_workers = {w.id: w for w in self.workers.search()}
        self._cached_dispatchers = {d.id: d for d in self.dispatchers.search()}
        for d in self._cached_dispatchers.values():
            if d.user_id == self.gis.users.me.username:
                self._cached_dispatcher = d
                break
        else:
            raise WorkforceError(
                "'{}' is not a dispatcher, please authenticate as a dispatcher".format(
                    self.gis.users.me.username
                )
            )

    def __str__(self):
        return self.title

    def __repr__(self):
        return "<Project {}>".format(self.title)

    @staticmethod
    def _delete_item_with_related_views(item):
        """
        Recursively fetches all related feature service items and then will delete any child services before deleting the parent service(s).
        :param item: An item
        """
        try:
            # Related Items API might not exist for older Enterprise versions
            related_items = item.related_items("Service2Service", "forward")
        except Exception:
            related_items = []
        for v in related_items:
            Project._delete_item_with_related_views(v)
        item.protect(False)
        item.delete()

    def delete(self):
        """
        Deletes the project, group, folder, layers, and webmaps.
        Assumes the currently signed in user owns the project or is an admin.
        """
        title = self.title
        owner = self._item.owner
        self._delete_item_with_related_views(self.assignments_item)
        if self._supports_tracks:
            self.tracks_item.protect(False)
            self.tracks_item.delete()
        if not self._is_v2_project:
            self.workers_item.protect(False)
            self.workers_item.delete()
            self.dispatchers_item.protect(False)
            self.dispatchers_item.delete()
            self._item.protect(False)
            self._item.delete()
        self.dispatcher_webmap.item.protect(False)
        self.dispatcher_webmap.item.delete()
        self.worker_webmap.item.protect(False)
        self.worker_webmap.item.delete()
        self.group.protected = False
        self.group.delete()
        for folder in self.gis.users.get(owner).folders:
            if folder["title"] == title:
                self.gis.content.delete_folder(folder["title"], owner=owner)

    def _update_data(self):
        # this function is used by v1 projects only
        self._item.update({"text": json.dumps(self._item_data)})

    def update(self, summary=None):
        """
        Updates the project on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        summary                  Optional :class:`String`. The summary of the project.
        ==================     ====================================================================
        """
        item_properties = {}
        if summary:
            item_properties["snippet"] = summary
        self._item.update(item_properties)
        if self._is_v2_project:
            self.gis.content.get(self.dispatcher_web_map_id).update(item_properties)
            self.gis.content.get(self.worker_web_map_id).update(item_properties)

    @property
    def _supports_tracks(self):
        return not self._is_v2_project and "tracks" in self._item_data

    @property
    def _is_v2_project(self):
        try:
            return (
                self._item.type == "Feature Service"
                and int(self._item.properties["workforceProjectVersion"].split(".")[0])
                >= 2
            )
        except:
            return False

    @property
    def _tracking_enabled(self):
        if self._supports_tracks:
            return self._item_data["tracks"]["enabled"]

    @_tracking_enabled.setter
    def _tracking_enabled(self, value):
        if self._supports_tracks:
            self._item_data["tracks"]["enabled"] = value
            self._update_data()

    @property
    def _tracking_interval(self):
        if self._supports_tracks:
            return self._item_data["tracks"]["updateInterval"]

    @_tracking_interval.setter
    def _tracking_interval(self, value):
        if self._supports_tracks:
            self._item_data["tracks"]["updateInterval"] = value
            self._update_data()

    @property
    def integrations(self):
        """The :class:`~arcgis.apps.workforce.managers.AssignmentIntegrationManager` for the project"""
        return AssignmentIntegrationManager(self)

    @property
    def id(self):
        """The item id of the project"""
        return self._item["id"]

    @property
    def title(self):
        """Gets the title of the project"""
        return self._item["title"]

    @property
    def summary(self):
        """The title of the project"""
        return self._item["snippet"]

    @summary.setter
    def summary(self, value):
        self._item["snippet"] = value

    @property
    def owner_user_id(self):
        """The user id of the project owner."""
        return self._item["owner"]

    @property
    def version(self):
        """The version of the project"""
        if self._is_v2_project:
            return self._item_data["workforceProjectVersion"]
        else:
            return self._item_data["version"]

    @_lazy_property
    def assignments_item(self):
        """The assignments :class:`~arcgis.gis.Item`"""
        if self._is_v2_project:
            # this is the same item as workers_item, dispatchers_item for a v2 project - each points to the one FS
            return self.gis.content.get(self._item.id)
        else:
            return self.gis.content.get(self._item_data["assignments"]["serviceItemId"])

    @property
    def assignments_layer_url(self):
        """The assignments feature layer url"""
        if self._is_v2_project:
            return self._item.url + "/0"
        else:
            return self._item_data["assignments"]["url"]

    @_lazy_property
    def assignment_types_item(self):
        """The assignment types :class:`~arcgis.gis.Item`"""
        if self._is_v2_project:
            return self.gis.content.get(self._item.id)
        else:
            warn(
                "This Workforce Project does not have an assignment types item",
                WorkforceWarning,
            )

    @property
    def assignment_types_table_url(self):
        """The assignment types table url"""
        if self._is_v2_project:
            return self._item.url + "/3"
        else:
            warn(
                "This Workforce Project does not have an assignment types table",
                WorkforceWarning,
            )

    @_lazy_property
    def dispatchers_item(self):
        """The dispatchers :class:`~arcgis.gis.Item`"""
        if self._is_v2_project:
            return self.gis.content.get(self._item.id)
        else:
            return self.gis.content.get(self._item_data["dispatchers"]["serviceItemId"])

    @property
    def dispatchers_layer_url(self):
        """The dispatchers layer url"""
        if self._is_v2_project:
            return self._item.url + "/2"
        else:
            return self._item_data["dispatchers"]["url"]

    @_lazy_property
    def integrations_table_url(self):
        """The integrations table url :class:`~arcgis.features.Table`"""
        if self._is_v2_project:
            return self._item.url + "/4"
        else:
            warn(
                "This Workforce Project does not have an integrations table",
                WorkforceWarning,
            )

    @_lazy_property
    def tracks_item(self):
        """The tracks :class:`~arcgis.gis.Item`"""
        if self._supports_tracks:
            return self.gis.content.get(self._item_data["tracks"]["serviceItemId"])
        else:
            warn("This Workforce Project does not support tracks.", WorkforceWarning)

    @property
    def tracks_layer_url(self):
        """The tracks feature layer url"""
        if self._supports_tracks:
            return self._item_data["tracks"]["url"]
        else:
            warn("This Workforce Project does not support tracks.", WorkforceWarning)

    @_lazy_property
    def workers_item(self):
        """The workers :class:`~arcgis.gis.Item`"""
        if self._is_v2_project:
            return self.gis.content.get(self._item.id)
        else:
            return self.gis.content.get(self._item_data["workers"]["serviceItemId"])

    @property
    def workers_layer_url(self):
        """The workers feature layer url"""
        if self._is_v2_project:
            return self._item.url + "/1"
        else:
            return self._item_data["workers"]["url"]

    @property
    def dispatcher_web_map_id(self):
        """The dispatcher webmap item id"""
        if self._is_v2_project:
            # not all systems will support this new "Workforce2MapFeatureService" so we try/except. If the rel does
            # not exist, we can get the webmap out of the metadata
            try:
                related_items = self._item.related_items(
                    "WorkforceMap2FeatureService", "reverse"
                )
                for item in related_items:
                    if "Workforce Dispatcher" in item.typeKeywords:
                        return item.id
                return self._item_data["workforceDispatcherMapId"]
            except Exception:
                return self._item_data["workforceDispatcherMapId"]
        else:
            return self._item_data["dispatcherWebMapId"]

    @property
    def worker_web_map_id(self):
        """The worker webmap item id"""
        if self._is_v2_project:
            try:
                related_items = self._item.related_items(
                    "WorkforceMap2FeatureService", "reverse"
                )
                for item in related_items:
                    if "Workforce Worker" in item.typeKeywords:
                        return item.id
                return self._item_data["workforceWorkerMapId"]
            except Exception:
                return self._item_data["workforceWorkerMapId"]
        else:
            return self._item_data["workerWebMapId"]

    @property
    def group_id(self):
        """The group id that all project items are part of"""
        if self._is_v2_project:
            return self._item_data["workforceProjectGroupId"]
        else:
            return self._item_data["groupId"]

    @_lazy_property
    def owner(self):
        """The owner :class:`~arcgis.gis.User` of the project"""
        return self.gis.users.get(self.owner_user_id)

    @_lazy_property
    def assignments_layer(self):
        """The assignments :class:`~arcgis.features.FeatureLayer`"""
        return FeatureLayer(self.assignments_layer_url, self.gis)

    @_lazy_property
    def dispatchers_layer(self):
        """The dispatchers :class:`~arcgis.features.FeatureLayer`"""
        if self._is_v2_project:
            return Table(self.dispatchers_layer_url, self.gis)
        else:
            return FeatureLayer(self.dispatchers_layer_url, self.gis)

    @_lazy_property
    def assignment_types_table(self):
        """The assignment types :class:`~arcgis.features.Table`"""
        if self._is_v2_project:
            return Table(self.assignment_types_table_url, self.gis)
        else:
            warn(
                "This Workforce Project does not have an assignment types table",
                WorkforceWarning,
            )

    @_lazy_property
    def integrations_table(self):
        """The integrations :class:`~arcgis.features.Table`"""
        if self._is_v2_project:
            return Table(self.integrations_table_url, self.gis)
        else:
            warn(
                "This Workforce Project does not have an integrations table",
                WorkforceWarning,
            )

    @_lazy_property
    def tracks_layer(self):
        """The tracks :class:`~arcgis.features.FeatureLayer`"""
        if self._supports_tracks:
            return FeatureLayer(self.tracks_layer_url, self.gis)
        else:
            warn("This Workforce Project does not support tracks.", WorkforceWarning)

    @_lazy_property
    def workers_layer(self):
        """The workers :class:`~arcgis.features.FeatureLayer`"""
        return FeatureLayer(self.workers_layer_url, self.gis)

    @_lazy_property
    def dispatcher_webmap(self):
        """The dispatcher :class:`~arcgis.mapping.WebMap` for the project"""
        return arcgis.mapping.WebMap(self.gis.content.get(self.dispatcher_web_map_id))

    @_lazy_property
    def worker_webmap(self):
        """The worker :class:`~arcgis.mapping.WebMap` for the project"""
        return arcgis.mapping.WebMap(self.gis.content.get(self.worker_web_map_id))

    @_lazy_property
    def group(self):
        """The :class:`~arcgis.gis.Group` that the project resources are part of"""
        return Group(self.gis, self.group_id)

    @property
    def assignments(self):
        """The :class:`~arcgis.apps.workforce.managers.AssignmentManager` for the project"""
        return AssignmentManager(self)

    @property
    def workers(self):
        """The :class:`~arcgis.apps.workforce.managers.WorkerManager` for the project"""
        return WorkerManager(self)

    @property
    def dispatchers(self):
        """The :class:`~arcgis.apps.workforce.managers.DispatcherManager` for the project"""
        return DispatcherManager(self)

    @property
    def tracks(self):
        """The :class:`~arcgis.apps.workforce.managers.TrackManager` for the project"""
        if self._supports_tracks:
            return TrackManager(self)
        else:
            warn("This Workforce Project does not support tracks.", WorkforceWarning)

    @property
    def assignment_types(self):
        """The :class:`~arcgis.apps.workforce.managers.AssignmentTypeManager` for the project"""
        return AssignmentTypeManager(self)
