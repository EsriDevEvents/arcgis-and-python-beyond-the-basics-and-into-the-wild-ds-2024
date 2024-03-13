""" Defines the Track object.
"""

from .feature_model import FeatureModel
from ._store import *
from ._schemas import TrackSchema
from .exceptions import WorkforceError


class Track(FeatureModel):
    """Represents a track feature, which describes the historical location of a worker. V1 Projects
    only.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    feature                Optional :class:`~arcgis.features.Feature`.
                           A feature containing the assignments attributes. Mostly intended for
                           internal usage. If supplied, other parameters are ignored.
    ------------------     --------------------------------------------------------------------
    geometry               Optional :class:`Dict`.
                           A dictionary containing the assignment geometry
    ------------------     --------------------------------------------------------------------
    accuracy               Optional :class:`Float`. The accuracy of the point
    ==================     ====================================================================
    """

    def __init__(self, project, feature=None, geometry=None, accuracy=None):
        if project.tracks is None:
            raise WorkforceError("This Workforce Project does not support tracks.")
        super().__init__(project, project.tracks_layer, feature)
        self._schema = TrackSchema(project.tracks_layer)
        if not feature:
            self.accuracy = accuracy
            self.geometry = geometry

    def __str__(self):
        return "{} at {}, {}".format(
            self.creator, self.geometry["x"], self.geometry["y"]
        )

    def __repr__(self):
        return "<Track {}>".format(self.object_id)

    def update(self, geometry=None, accuracy=None):
        """
        Updates the track point on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        geometry               Optional :class:`Dict`.
                               A dictionary containing the assignment geometry
        ------------------     --------------------------------------------------------------------
        accuracy               Optional :class:`Float`. The accuracy of the point
        ==================     ====================================================================
        """
        update_track(self.project, self, geometry, accuracy)

    def delete(self):
        """Deletes the track point on the server"""
        delete_tracks(self.project, [self])

    @FeatureModel.geometry.setter
    def geometry(self, value):
        self._feature.geometry = value

    @property
    def accuracy(self):
        """The horizontal accuracy of the location measurement, in meters."""
        return self._feature.attributes[self._schema.accuracy]

    @accuracy.setter
    def accuracy(self, value):
        self._feature.attributes[self._schema.accuracy] = value
