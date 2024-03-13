""" Defines store functions for working with assignments.
"""

from ... import workforce
from .dispatchers import query_dispatchers
from .utils import add_features, remove_features, update_features, validate


def get_assignment(project, object_id=None, global_id=None):
    """Gets the identified Assignment.  Exactly one form of identification should be provided.
    :param project:
    :param object_id: The assignment's OBJECTID.
    :param global_id: The assignment's GlobalID.
    :returns: Assignment
    """
    if object_id:
        where = "{} = {}".format(project._assignment_schema.object_id, object_id)
    elif global_id:
        where = "{} = '{}'".format(project._assignment_schema.global_id, global_id)
    else:
        where = "1=0"
    assignments = query_assignments(project, where)
    if assignments:
        return assignments[0]
    return None


def get_assignments(project):
    """Gets all Assignments in the project.
    :param project:
    :returns: list of Assignments
    """
    return query_assignments(project, "1=1")


def query_assignments(project, where="1=1"):
    """Executes a query against the assignments feature layer.
    :param project: The project in which to query assignments.
    :param where: An ArcGIS where clause.
    :returns: list of Assignments
    """
    assignments = []
    assignment_features = project.assignments_layer.query(
        where, return_all_records=True
    ).features
    # fetch assignment types, dispatchers, and workers
    project._update_cached_objects()
    # refresh cached objects
    for feature in assignment_features:
        assignments.append(workforce.Assignment(project, feature))
    return assignments


def add_assignments(project, assignments):
    """Adds Assignments to a project.

    Side effect: Upon successful addition on the server, the object_id and global_id fields of
    each Assignment in assignments will be updated to the values assigned by the server.

    :param project:
    :param assignments: list of Assignments
    :returns the list of Assignments
    :raises ValidationError: Indicates that one or more assignments failed validation.
    :raises ServerError: Indicates that the server rejected the assignments.
    """
    project._update_cached_objects()
    use_global_ids = True
    for assignment in assignments:
        validate(assignment._validate_for_add)
        if assignment.global_id is None:
            use_global_ids = False

    features = [assignment.feature for assignment in assignments]
    add_features(project.assignments_layer, features, use_global_ids)
    return assignments


def add_assignment(
    project,
    feature=None,
    geometry=None,
    assignment_type=None,
    assigned_date=None,
    assignment_read=None,
    completed_date=None,
    declined_comment=None,
    declined_date=None,
    description=None,
    dispatcher=None,
    due_date=None,
    in_progress_date=None,
    location=None,
    notes=None,
    paused_date=None,
    priority=None,
    status=None,
    work_order_id=None,
    worker=None,
):
    """
    Adds a new assignment to the project
    """
    project._update_cached_objects()
    assignment = workforce.Assignment(
        project,
        feature,
        geometry,
        assignment_type,
        assigned_date,
        assignment_read,
        completed_date,
        declined_comment,
        declined_date,
        description,
        dispatcher,
        due_date,
        in_progress_date,
        location,
        notes,
        paused_date,
        priority,
        status,
        work_order_id,
        worker,
    )

    return add_assignments(project, [assignment])[0]


def update_assignments(project, assignments):
    """Updates Assignments.
    :param project:
    :param assignments: list of Assignments to update
    :raises ValidationError: Indicates that one or more assignments failed validation.
    :raises ServerError: Indicates that the server rejected the updates.
    """
    project._update_cached_objects()
    for assignment in assignments:
        validate(assignment._validate_for_update)
    features = [assignment.feature for assignment in assignments]
    update_features(project.assignments_layer, features)
    return assignments


def update_assignment(
    project,
    assignment,
    geometry=None,
    assignment_type=None,
    assigned_date=None,
    assignment_read=None,
    completed_date=None,
    declined_comment=None,
    declined_date=None,
    description=None,
    dispatcher=None,
    due_date=None,
    in_progress_date=None,
    location=None,
    notes=None,
    paused_date=None,
    priority=None,
    status=None,
    work_order_id=None,
    worker=None,
):
    """
    Sets the properties of an assignment and updates the item on the server
    """
    project._update_cached_objects()
    if status:
        # logic for resetting stale dates if reassigning
        if assignment.status == "declined" and worker and worker != assignment.worker:
            assignment.declined_date = None
        if assignment.status == "paused" and worker and worker != assignment.worker:
            assignment.paused_date = None
        assignment.status = status
    if geometry:
        assignment.geometry = geometry
    if assigned_date:
        assignment.assigned_date = assigned_date
    if assignment_read:
        assignment.assignment_read = assignment_read
    if completed_date:
        assignment.completed_date = completed_date
    if declined_comment:
        assignment.declined_comment = declined_comment
    if declined_date:
        assignment.declined_date = declined_date
    if description:
        assignment.description = description
    if due_date:
        assignment.due_date = due_date
    if in_progress_date:
        assignment.in_progress_date = in_progress_date
    if location:
        assignment.location = location
    if notes:
        assignment.notes = notes
    if paused_date:
        assignment.paused_date = paused_date
    if priority:
        assignment.priority = priority
    if status:
        assignment.status = status
    if work_order_id:
        assignment.work_order_id = work_order_id
    if assignment_type:
        assignment.assignment_type = assignment_type
    if dispatcher:
        assignment.dispatcher = dispatcher
    if worker:
        assignment.worker = worker
    return update_assignments(project, [assignment])[0]


def delete_assignments(project, assignments):
    """Removes assignments from the project.
    :param project:
    :param assignments: list of Assignments
    :raises ValidationError: Indicates that one or more assignments failed validation.
    :raises ServerError: Indicates that the server rejected the removal.
    """
    project._update_cached_objects()
    for assignment in assignments:
        validate(assignment._validate_for_remove)
    features = [assignment.feature for assignment in assignments]
    remove_features(project.assignments_layer, features)


def dispatchers_for_assignment_features(project, features):
    dispatcher_ids = [
        feature.attributes.get(project._assignment_schema.dispatcher_id)
        for feature in features
    ]
    dispatcher_ids = [did for did in dispatcher_ids if did is not None]
    if dispatcher_ids:
        dispatchers_where = "{} IN ({})".format(
            project._dispatcher_schema.object_id, ",".join(map(str, dispatcher_ids))
        )
        return query_dispatchers(project, dispatchers_where)
    return []


def workers_for_assignment_features(project, features):
    worker_ids = [
        feature.attributes.get(project._assignment_schema.worker_id)
        for feature in features
    ]
    worker_ids = [wid for wid in worker_ids if wid is not None]
    if worker_ids:
        workers_where = "{} IN ({})".format(
            project._worker_schema.object_id, ",".join(map(str, worker_ids))
        )
        return workforce._store.query_workers(project, workers_where)
    return []
