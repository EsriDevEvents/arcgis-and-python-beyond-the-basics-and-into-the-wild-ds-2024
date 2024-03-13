""" Defines store functions for working with Dispatchers.
"""

import math
from ... import workforce
from .utils import add_features, update_features, remove_features, validate


def get_dispatcher(project, object_id=None, global_id=None, user_id=None):
    """Gets the identified Dispatcher.  Exactly one form of identification should be provided.
    :param project:
    :param object_id: The dispatcher's OBJECTID.
    :param global_id: The dispatcher's GlobalID.
    :param user_id: The dispatcher's named user user_id.
    :returns: Dispatcher
    """
    if object_id:
        where = "{} = {}".format(project._dispatcher_schema.object_id, object_id)
    elif global_id:
        where = "{} = '{}'".format(project._dispatcher_schema.global_id, global_id)
    elif user_id:
        where = "{} = '{}'".format(project._dispatcher_schema.user_id, user_id)
    else:
        where = "1=0"
    dispatchers = query_dispatchers(project, where)
    return dispatchers[0] if dispatchers else None


def get_dispatchers(project):
    """Gets all Dispatchers in the project.
    :param project:
    :returns: list of Dispatchers
    """
    return query_dispatchers(project, "1=1")


def query_dispatchers(project, where):
    """Executes a query against the dispatchers feature layer.
    :param project: The project in which to query dispatchers.
    :param where: An ArcGIS where clause.
    :returns: list of Dispatchers
    """
    features = project.dispatchers_layer.query(where, return_all_records=True).features
    return [workforce.Dispatcher(project, feature) for feature in features]


def add_dispatcher(project, feature=None, contact_number=None, name=None, user_id=None):
    """
    Adds a new dispatcher to the project
    """
    project._update_cached_objects()
    dispatcher = workforce.Dispatcher(project, feature, contact_number, name, user_id)
    return add_dispatchers(project, [dispatcher])[0]


def update_dispatcher(
    project, dispatcher, contact_number=None, name=None, user_id=None
):
    """
    Updates a dispatcher and submits changes to the server
    """
    project._update_cached_objects()
    if contact_number:
        dispatcher.contact_number = contact_number
    if name:
        dispatcher.name = name
    if user_id and dispatcher.user_id != project.owner_user_id:
        dispatcher.user_id = user_id
    return update_dispatchers(project, [dispatcher])[0]


def add_dispatchers(project, dispatchers):
    """Adds Dispatchers to a project.

    Side effect: Upon successful addition on the server, the object_id and global_id fields of
    each Dispatcher in dispatchers will be updated to the values assigned by the server.

    :param project:
    :param dispatchers: list of Dispatchers
    :raises ValidationError: Indicates that one or more dispatchers failed validation.
    :raises ServerError: Indicates that the server rejected the dispatchers.
    """
    project._update_cached_objects()
    if dispatchers:
        use_global_ids = True
        for dispatcher in dispatchers:
            validate(dispatcher._validate_for_add)

            if dispatcher.global_id is None:
                use_global_ids = False

        features = [dispatcher.feature for dispatcher in dispatchers]
        add_features(project.dispatchers_layer, features, use_global_ids)

    # add dispatcher named users to the project's group.
    max_add_per_call = 25
    for i in range(0, math.ceil(len(dispatchers) / max_add_per_call)):
        project.group.add_users(
            [
                d.user_id
                for d in dispatchers[
                    i * max_add_per_call : (i * max_add_per_call) + max_add_per_call
                ]
            ]
        )
    return dispatchers


def update_dispatchers(project, dispatchers):
    """Updates Dispatchers.
    :param project:
    :param dispatchers: list of Dispatchers to update
    :raises ValidationError: Indicates that one or more dispatchers failed validation.
    :raises ServerError: Indicates that the server rejected the dispatchers.
    """
    project._update_cached_objects()
    if dispatchers:
        for dispatcher in dispatchers:
            validate(dispatcher._validate_for_update)
        features = [dispatcher.feature for dispatcher in dispatchers]
        update_features(project.dispatchers_layer, features)
    return dispatchers


def delete_dispatchers(project, dispatchers):
    """Removes Dispatchers from the project.
    :param project:
    :param dispatchers: list of Dispatchers
    :raises ValidationError: Indicates that one or more dispatchers failed validation.
    :raises ServerError: Indicates that the server rejected the removal.
    """
    project._update_cached_objects()
    if dispatchers:
        for dispatcher in dispatchers:
            validate(dispatcher._validate_for_remove)
        features = [dispatcher.feature for dispatcher in dispatchers]
        remove_features(project.dispatchers_layer, features)

        # Remove dispatcher named users from the project's group, unless they are also workers.
        user_ids = [dispatcher.user_id for dispatcher in dispatchers]
        where = "{} in ({})".format(
            project._worker_schema.user_id,
            ",".join(["'{}'".format(user_id) for user_id in user_ids]),
        )
        workers = workforce._store.query_workers(project, where)
        for worker in workers:
            user_ids.remove(worker.user_id)
        if user_ids:
            project.group.remove_users(user_ids)
