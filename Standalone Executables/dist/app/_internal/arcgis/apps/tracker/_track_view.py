import arcgis
from arcgis._impl.common._utils import _lazy_property
from arcgis.apps.tracker import LocationTrackingError
import re
import math


class TrackView:
    """
    A Track View

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    item                   Required :class:`~arcgis.gis.Item`. The item that represents the
                           Track View.
    ==================     ====================================================================

    .. code-block:: python

        # Get a Track View and list mobile users.

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        track_view = arcgis.apps.tracker.TrackView(item)
        mobile_users = track_view.mobile_users.list()

    """

    def __init__(self, item):
        if "Location Tracking View" not in item.typeKeywords:
            raise LocationTrackingError("Item is not a Track View")
        self._item = item
        self._gis = item._gis

    def delete(self):
        """
        Deletes the Track View, including the group and view service.
        """
        if self.group:
            self.group.protected = False
            self.group.delete()
        self._item.protect(False)
        self._item.delete()

    @_lazy_property
    def viewers(self):
        """The :class:`~arcgis.apps.tracker.TrackViewerManager` for the Track View"""
        return TrackViewerManager(self)

    @_lazy_property
    def mobile_users(self):
        """The :class:`~arcgis.apps.tracker.MobileUserManager` for the Track View"""
        return MobileUserManager(self)

    @_lazy_property
    def group(self):
        """The group that contains the Track Viewers and Layers"""
        try:
            return self._gis.groups.get(self.item.properties["trackViewGroup"])
        except:
            return None

    @property
    def item(self):
        """The Track View :class:`~arcgis.gis.Item`"""
        return self._item

    @_lazy_property
    def tracks_layer(self):
        """The tracks :class:`~arcgis.features.FeatureLayer`"""
        return self._item.layers[0]

    @_lazy_property
    def last_known_locations_layer(self):
        """The last known locations :class:`~arcgis.features.FeatureLayer`"""
        return self._item.layers[1]

    @_lazy_property
    def track_lines_layer(self):
        """The track lines :class:`~arcgis.features.FeatureLayer`"""
        try:
            return self._item.layers[2]
        except IndexError:
            return None


class TrackViewerManager:
    """
    A class that manages the Track Viewers in the Track View.

    It can be accessed from the TrackView as :py:attr:`~arcgis.apps.tracker.TrackView.viewers`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    track_view             Required :class:`~arcgis.apps.tracker.TrackView`. The Track View to
                           configure Track Viewers for.
    ==================     ====================================================================
    """

    def __init__(self, track_view):
        self._track_view = track_view

    def add(self, viewers):
        """
        Adds the specified usernames as Track Viewers

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        viewers                Required List of strings or :class:`~arcgis.gis.User`.
                               The list of usernames/users to add as Track Viewers.
        ==================     ====================================================================
        """
        if isinstance(viewers, (str, arcgis.gis.User)):
            viewers = [viewers]
        max_add_per_call = 25
        for i in range(0, math.ceil(len(viewers) / max_add_per_call)):
            self._track_view.group.add_users(
                viewers[
                    i * max_add_per_call : (i * max_add_per_call) + max_add_per_call
                ]
            )

    def delete(self, viewers):
        """
        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        viewers                Required List of strings or :class:`~arcgis.gis.User`.
                               The list of usernames/users to remove as Track Viewers.
        ==================     ====================================================================
        """
        if isinstance(viewers, (str, arcgis.gis.User)):
            viewers = [viewers]
        if isinstance(viewers[0], str):
            if self._track_view._item["owner"] in viewers:
                raise LocationTrackingError(
                    "Cannot remove track view owner from being a track viewer. Please try again without the owner included"
                )
        else:
            if any(
                viewer.username == self._track_view._item["owner"] for viewer in viewers
            ):
                raise LocationTrackingError(
                    "Cannot remove track view owner from being a track viewer. Please try again without the owner included"
                )
        max_add_per_call = 25
        for i in range(0, math.ceil(len(viewers) / max_add_per_call)):
            self._track_view.group.remove_users(
                viewers[
                    i * max_add_per_call : (i * max_add_per_call) + max_add_per_call
                ]
            )

    def list(self):
        """
        List of all the Track Viewers

        :return: List of strings representing usernames
        """
        results = self._track_view.group.get_members()
        usernames = results["users"]
        usernames.append(results["owner"])
        return usernames


class MobileUserManager:
    """
    A class that manages the Mobile Users in the Track View.

    It can be accessed from the TrackView as :py:attr:`~arcgis.apps.tracker.TrackView.mobile_users`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    track_view             Required :class:`~arcgis.apps.tracker.TrackView`. The Track View to
                           configure Mobile Users for.
    ==================     ====================================================================
    """

    _VDQ_RE = r"""^created_user\s+(?:in|IN)\s+\(\s*((?:'[^']*')(?:\s*,\s*'[^']*')*)\s*\)(?:\s+((?:and|AND).*))?$"""
    _DEFAULT_VDQ = """created_user in ('')"""

    def __init__(self, track_view):
        self._track_view = track_view

    def add(self, users):
        """
        Adds the specified usernames as Mobile Users

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List of strings or :class:`~arcgis.gis.User`.
                               The list of usernames/users to add as Mobile Users
        ==================     ====================================================================
        """
        if isinstance(users, str):
            users = {users}
        elif isinstance(users, arcgis.gis.User):
            users = {users.username}
        elif isinstance(users, (set, list)):
            if len(users) > 0 and isinstance(next(iter(users)), arcgis.gis.User):
                users = {u.username for u in users}
            else:
                users = set(users)
        self._update_vdq(users.union(self.list()))

    def delete(self, users):
        """
        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        users                  Required List of strings or :class:`~arcgis.gis.User`.
                               The list of usernames/users to remove as Mobile Users.
        ==================     ====================================================================
        """
        if isinstance(users, str):
            users = {users}
        elif isinstance(users, arcgis.gis.User):
            users = {users.username}
        elif isinstance(users, (set, list)):
            if len(users) > 0 and isinstance(next(iter(users)), arcgis.gis.User):
                users = {u.username for u in users}
        self._update_vdq(set(self.list()).difference(users))

    def list(self):
        """
        List of all the Track Viewers

        :return: List of strings representing usernames
        """
        pattern = re.compile(self._VDQ_RE)
        match = pattern.match(self.view_definition_query)
        if match is None:
            raise LocationTrackingError("Unable to parse viewDefinitionQuery")
        return [
            username.strip()[1:-1]
            for username in match[1].split(",")
            if username.strip()[1:-1]
        ]

    def _generate_users_where_clause(self, usernames):
        if len(usernames) > 0:
            return "created_user in ({})".format(
                ",".join(["'{}'".format(u) for u in usernames])
            )
        return self._DEFAULT_VDQ

    def _update_vdq(self, usernames):
        # we do not validate the usernames because, they could represent people who used to be in
        # the org and have historical tracks
        new_users_clause = self._generate_users_where_clause(usernames)
        pattern = re.compile(self._VDQ_RE)
        match = pattern.match(self.view_definition_query)
        custom_section = match.group(2)
        if custom_section is None:
            new_vdq = new_users_clause
        else:
            new_vdq = "{} {}".format(new_users_clause, custom_section)
        self._track_view.tracks_layer.manager.update_definition(
            {"viewDefinitionQuery": new_vdq}
        )
        self._track_view.last_known_locations_layer.manager.update_definition(
            {"viewDefinitionQuery": new_vdq}
        )
        # Older versions of the LTS do not have this layer
        if self._track_view.track_lines_layer:
            self._track_view.track_lines_layer.manager.update_definition(
                {"viewDefinitionQuery": new_vdq}
            )
            self._track_view.track_lines_layer._hydrate()
        # update cached values
        self._track_view.tracks_layer._hydrate()
        self._track_view.last_known_locations_layer._hydrate()

    @property
    def view_definition_query(self):
        """The View Definition Query of the tracks layer"""
        vdq = self._track_view.tracks_layer.properties["viewDefinitionQuery"]
        # Enterprise wraps VDQ in parentheses, but online doesn't
        if vdq[0] == "(" and vdq[-1] == ")":
            return vdq[1:-1]
        return vdq
