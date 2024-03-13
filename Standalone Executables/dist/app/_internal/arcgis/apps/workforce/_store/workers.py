""" Defines store functions for working with Workers.
"""

import math
from ... import workforce
from .utils import add_features, update_features, remove_features, validate


def get_worker(project, object_id=None, global_id=None, user_id=None):
    """Gets the identified worker.  Exactly one form of identification should be provided.
    :param project:
    :param object_id: The worker's OBJECTID.
    :param global_id: The worker's GlobalID.
    :param user_id: The worker's named user user_id.
    :returns: Worker
    """
    if object_id:
        where = "{} = {}".format(project._worker_schema.object_id, object_id)
    elif global_id:
        where = "{} = '{}'".format(project._worker_schema.global_id, global_id)
    elif user_id:
        where = "{} = '{}'".format(project._worker_schema.user_id, user_id)
    else:
        where = "1=0"
    workers = query_workers(project, where)
    return workers[0] if workers else None


def get_workers(project):
    """Gets all workers in the project.
    :param project:
    :returns: list of Workers
    """
    return query_workers(project, "1=1")


def query_workers(project, where):
    """Executes a query against the workers feature layer.
    :param project: The project in which to query workers.
    :param where: An ArcGIS where clause.
    :returns: list of Workers
    """
    worker_features = project.workers_layer.query(
        where, return_all_records=True
    ).features
    return [workforce.Worker(project, feature) for feature in worker_features]


def add_worker(
    project,
    feature=None,
    geometry=None,
    contact_number=None,
    name=None,
    notes=None,
    status=None,
    title=None,
    user_id=None,
):
    """
    Creates and adds a worker to the project
    """
    project._update_cached_objects()
    worker = workforce.Worker(
        project, feature, geometry, contact_number, name, notes, status, title, user_id
    )
    return add_workers(project, [worker])[0]


def add_workers(project, workers):
    """Adds Workers to a project.

    Side effect: Upon successful addition on the server, the object_id and global_id fields of
    each Worker in workers will be updated to the values assigned by the server.

    :param project:
    :param workers: list of Workers
    :raises ValidationError: Indicates that one or more workers failed validation.
    :raises ServerError: Indicates that the server rejected the workers.
    """
    project._update_cached_objects()
    if workers:
        use_global_ids = True
        for worker in workers:
            validate(worker._validate_for_add)

            if worker.global_id is None:
                use_global_ids = False

        features = [worker.feature for worker in workers]
        add_features(project.workers_layer, features, use_global_ids)

        # add worker named users to the project's group.
        max_add_per_call = 25
        for i in range(0, math.ceil(len(workers) / max_add_per_call)):
            project.group.add_users(
                [
                    w.user_id
                    for w in workers[
                        i * max_add_per_call : (i * max_add_per_call) + max_add_per_call
                    ]
                ]
            )
    return workers


def update_worker(
    project,
    worker,
    geometry=None,
    contact_number=None,
    name=None,
    notes=None,
    status=None,
    title=None,
    user_id=None,
):
    """
    Updates a worker and submits the changes to the server
    """
    project._update_cached_objects()
    if geometry:
        worker.geometry = geometry
    if contact_number:
        worker.contact_number = contact_number
    if name:
        worker.name = name
    if notes:
        worker.notes = notes
    if status:
        worker.status = status
    if title:
        worker.title = title
    if user_id:
        worker.user_id = user_id
    return update_workers(project, [worker])[0]


def update_workers(project, workers):
    """Updates Workers.
    :param project:
    :param workers: list of Workers to update
    :raises ValidationError: Indicates that one or more workers failed validation.
    :raises ServerError: Indicates that the server rejected the workers.
    """
    project._update_cached_objects()
    if workers:
        for worker in workers:
            validate(worker._validate_for_update)
        update_features(project.workers_layer, [worker.feature for worker in workers])
    return workers


def delete_workers(project, workers):
    """Removes workers from the project.
    :param project:
    :param workers: list of Workers
    :raises ValidationError: Indicates that one or more workers failed validation.
    :raises ServerError: Indicates that the server rejected the removal.
    """
    project._update_cached_objects()
    if workers:
        for worker in workers:
            validate(worker._validate_for_remove)
        remove_features(project.workers_layer, [worker.feature for worker in workers])

        # Remove worker named users from the project's group, unless they are also dispatchers.
        user_ids = [worker.user_id for worker in workers]
        where = "{} in ({})".format(
            project._dispatcher_schema.user_id,
            ",".join(["'{}'".format(user_id) for user_id in user_ids]),
        )
        dispatchers = workforce._store.query_dispatchers(project, where)
        for dispatcher in dispatchers:
            user_ids.remove(dispatcher.user_id)
        if user_ids:
            project.group.remove_users(user_ids)
