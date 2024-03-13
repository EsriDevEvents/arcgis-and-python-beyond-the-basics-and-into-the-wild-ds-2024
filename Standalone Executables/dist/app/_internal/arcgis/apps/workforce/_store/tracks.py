""" Defines store functions for working with Tracks.
"""
from ... import workforce
from .utils import remove_features, validate, update_features, add_features


def get_track(project, object_id=None, global_id=None):
    """Gets the identified track.  Exactly one form of identification should be provided.
    :param project:
    :param object_id: The track's OBJECTID.
    :param global_id: The track's GlobalID.
    :returns: Track
    """
    if object_id is not None:
        where = "{}={}".format(project._track_schema.object_id, object_id)
    elif global_id is not None:
        where = "{}='{}'".format(project._track_schema.global_id, global_id)
    else:
        where = "1=0"
    tracks = query_tracks(project, where)
    return tracks[0] if tracks else None


def get_tracks(project):
    """Gets all tracks in the project.
    :param project:
    :returns: list of Tracks
    """
    return query_tracks(project, "1=1")


def add_tracks(project, tracks):
    """Adds Tracks to a project.

    Side effect: Upon successful addition on the server, the object_id and global_id fields of
    each Track in tracks will be updated to the values assigned by the server.

    :param project:
    :param tracks: list of tracks
    :raises ValidationError: Indicates that one or more tracks failed validation.
    :raises ServerError: Indicates that the server rejected the tracks.
    """
    if tracks:
        use_global_ids = True
        for track in tracks:
            validate(track._validate_for_add)

            if track.global_id is None:
                use_global_ids = False

        features = [track.feature for track in tracks]
        add_features(project.tracks_layer, features, use_global_ids)
    return tracks


def add_track(project, feature=None, geometry=None, accuracy=None):
    """Adds a track to the project.

    Side effect: Upon successful addition on the server, the object_id and global_id fields of
    each Track in tracks will be updated to the values assigned by the server.

    :param project:
    :param feature: the feature
    :param geometry: the geometry
    :param accuracy: the accuracy of the track
    :raises ValidationError: Indicates that one or more tracks failed validation.
    :raises ServerError: Indicates that the server rejected the tracks.
    """
    track = workforce.Track(project, feature, geometry, accuracy)
    return add_tracks(project, [track])[0]


def update_tracks(project, tracks):
    """Updates Tracks.
    :param project:
    :param tracks: list of tracks to update
    :raises ValidationError: Indicates that one or more tracks failed validation.
    :raises ServerError: Indicates that the server rejected the tracks.
    """
    if tracks:
        for track in tracks:
            validate(track._validate_for_update)
        features = [track.feature for track in tracks]
        update_features(project.tracks_layer, features)
    return tracks


def update_track(project, track, geometry=None, accuracy=None):
    """Updates individual track and submits changes to the server"""
    if accuracy:
        track.accuracy = accuracy
    if geometry:
        track.geometry = geometry
    return update_tracks(project, [track])[0]


def query_tracks(project, where):
    """Executes a query against the tracks feature layer.
    :param project: The project in which to query tracks.
    :param where: An ArcGIS where clause.
    :returns: list of Tracks
    """
    track_features = project.tracks_layer.query(where, return_all_records=True).features
    return [workforce.Track(project, feature) for feature in track_features]


def delete_tracks(project, tracks):
    """Removes tracks from the project.
    :param project:
    :param tracks: list of Tracks
    :raises ValidationError: Indicates that one or more tracks failed validation.
    :raises ServerError: Indicates that the server rejected the removal.
    """
    if tracks:
        for track in tracks:
            validate(track._validate_for_remove)
        remove_features(project.tracks_layer, [track.feature for track in tracks])
