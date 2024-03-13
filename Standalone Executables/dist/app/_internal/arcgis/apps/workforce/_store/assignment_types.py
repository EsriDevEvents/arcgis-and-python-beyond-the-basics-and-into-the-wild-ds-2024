""" Defines store functions for working with AssignmentTypes.
"""

from ... import workforce
from .utils import validate


def get_assignment_type(project, code=None, name=None):
    """Gets the identified AssignmentType. Exactly one form of identification should be provided.
    :param project:
    :param code: The AssignmentType code.
    :param name: The AssignmentType name.
    """
    assignment_types = get_assignment_types(project)
    if code:
        return next((at for at in assignment_types if at.code == code), None)
    elif name:
        return next((at for at in assignment_types if at.name == name), None)
    return None


def get_assignment_types(project):
    """Gets all AssignmentTypes for the project.
    :param project:
    :returns: A list of AssignmentTypes.
    """
    assignment_type_field = get_assignment_type_field(
        project, project.assignments_layer
    )
    return coded_values_to_assignment_types(
        project, assignment_type_field["domain"]["codedValues"]
    )


def add_assignment_type(project, coded_value=None, name=None):
    """
    Adds a new assignment type
    """
    assignment_type = workforce.AssignmentType(
        project, coded_value=coded_value, name=name
    )
    return add_assignment_types(project, [assignment_type])[0]


def add_assignment_types(project, assignment_types):
    """Adds an AssignmentType to a project.

    Side effect: Each AssignmentType in assignment_types will be assigned a unique code.

    :param project:
    :param assignment_types: list of AssignmentTypes
    :raises ValidationError: Indicates that one or more assignment types failed validation.
    """
    if assignment_types:
        assignment_type_field = get_assignment_type_field(
            project, project.assignments_layer
        )
        coded_values = assignment_type_field["domain"]["codedValues"]
        existing_assignment_types = coded_values_to_assignment_types(
            project, coded_values
        )
        if existing_assignment_types:
            max_code = max(
                [assignment_type.code for assignment_type in existing_assignment_types]
            )
        else:
            max_code = 0
        for assignment_type in assignment_types:
            validate(
                assignment_type._validate_for_add,
                assignment_types=existing_assignment_types,
            )
            assignment_type.coded_value["code"] = max_code + 1
            max_code = assignment_type.code
            coded_values.append(assignment_type.coded_value)
            project.assignments_layer.manager.update_definition(
                {"fields": [assignment_type_field]}
            )
        project._update_cached_assignment_types()
    return assignment_types


def update_assignment_types(project, assignment_types):
    """Updates the AssignmentTypes.
    :param project:
    :param assignment_types: list of AssignmentTypes
    :raises ValidationError: Indicates that one or more assignment types failed validation.
    """
    if assignment_types:
        assignment_type_field = get_assignment_type_field(
            project, project.assignments_layer
        )
        coded_values = assignment_type_field["domain"]["codedValues"]
        existing_assignment_types = coded_values_to_assignment_types(
            project, coded_values
        )
        for assignment_type in assignment_types:
            validate(
                assignment_type._validate_for_update,
                assignment_types=existing_assignment_types,
            )
            for i, coded_value in enumerate(coded_values):
                if assignment_type.code == coded_value["code"]:
                    coded_values[i] = assignment_type.coded_value
        project.assignments_layer.manager.update_definition(
            {"fields": [assignment_type_field]}
        )
        project._update_cached_assignment_types()
    return assignment_types


def update_assignment_type(project, assignment_type, name=None):
    """Updates an assignment types name"""
    if name:
        assignment_type.name = name
    return update_assignment_types(project, [assignment_type])[0]


def delete_assignment_types(project, assignment_types):
    """Removes AssignmentTypes from the project.
    :param project:
    :param assignment_types: list of AssignmentTypes.
    """
    if assignment_types:
        assignment_type_codes = [a.code for a in assignment_types]
        where = "{} IN ({})".format(
            project._assignment_schema.assignment_type,
            ",".join(map(str, assignment_type_codes)),
        )
        affected_assignments = workforce._store.query_assignments(project, where)
        for assignment_type in assignment_types:
            validate(
                assignment_type._validate_for_remove, assignments=affected_assignments
            )
        assignment_type_field = get_assignment_type_field(
            project, project.assignments_layer
        )
        coded_values = assignment_type_field["domain"]["codedValues"]
        coded_values = [
            coded_value
            for coded_value in coded_values
            if coded_value["code"] not in assignment_type_codes
        ]
        assignment_type_field["domain"]["codedValues"] = coded_values
        project.assignments_layer.manager.update_definition(
            {"fields": [assignment_type_field]}
        )
        project._update_cached_assignment_types()


def get_assignment_type_field(project, feature_layer):
    for field in feature_layer.properties["fields"]:
        if field["name"] == project._assignment_schema.assignment_type:
            return field


def coded_values_to_assignment_types(project, coded_values):
    assignment_types = []
    for assignment_type_json in coded_values:
        assignment_type = workforce.AssignmentType(
            project, coded_value=assignment_type_json
        )
        assignment_types.append(assignment_type)
    return assignment_types
