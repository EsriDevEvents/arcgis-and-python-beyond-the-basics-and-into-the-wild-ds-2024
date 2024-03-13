""" Defines store functions for working with AssignmentTypes in Version 2 projects.
"""

from ... import workforce
from .utils import validate, add_features, update_features, remove_features


def get_assignment_type_v2(project, code=None, name=None):
    """Gets the identified AssignmentType. Exactly one form of identification should be provided.
    :param project:
    :param code: The AssignmentType GlobalID.
    :param name: The AssignmentType name.
    """
    assignment_types = get_assignment_types_v2(project)
    if code:
        return next((at for at in assignment_types if at.code == code), None)
    elif name:
        return next((at for at in assignment_types if at.name == name), None)
    return None


def get_assignment_types_v2(project):
    """Gets all AssignmentTypes for the project.
    :param project:
    :returns: A list of AssignmentTypes.
    """
    return query_assignment_types(project, "1=1")


def query_assignment_types(project, where="1=1"):
    """Executes a query against the assignment types table.
    :param project: The project in which to query assignment types.
    :param where: An ArcGIS where clause.
    :returns: list of Assignment Types
    """
    assignment_type_features = project.assignment_types_table.query(
        where, return_all_records=True
    ).features
    return [
        workforce.AssignmentType(project, feature)
        for feature in assignment_type_features
    ]


def add_assignment_type_v2(project, name):
    """
    Adds a new assignment type
    """
    assignment_type = workforce.AssignmentType(project, name=name)
    return add_assignment_types_v2(project, [assignment_type])[0]


def add_assignment_types_v2(project, assignment_types):
    """Adds an AssignmentType to a project.
    :param project:
    :param assignment_types: list of AssignmentTypes
    :raises ValidationError: Indicates that one or more assignment types failed validation.
    """
    for assignment_type in assignment_types:
        validate(assignment_type._validate)

    use_global_ids = False
    features = [assignment_type.feature for assignment_type in assignment_types]
    add_features(project.assignment_types_table, features, use_global_ids)
    project._update_cached_assignment_types()
    return assignment_types


def update_assignment_types_v2(project, assignment_types):
    """Updates the AssignmentTypes.
    :param project:
    :param assignment_types: list of AssignmentTypes
    :raises ValidationError: Indicates that one or more assignment types failed validation.
    """
    for assignment_type in assignment_types:
        validate(assignment_type._validate_for_update)
    features = [assignment_type.feature for assignment_type in assignment_types]
    update_features(project.assignment_types_table, features)
    project._update_cached_assignment_types()
    return assignment_types


def update_assignment_type_v2(project, assignment_type, name=None):
    """Updates an assignment type's name"""
    if name:
        assignment_type.name = name
    return update_assignment_types_v2(project, [assignment_type])[0]


def delete_assignment_types_v2(project, assignment_types):
    """Removes AssignmentTypes from the project.
    :param project:
    :param assignment_types: list of AssignmentTypes.
    """
    for assignment_type in assignment_types:
        validate(assignment_type._validate_for_remove)
    features = [assignment_type.feature for assignment_type in assignment_types]
    remove_features(project.assignment_types_table, features)
    project._update_cached_assignment_types()
