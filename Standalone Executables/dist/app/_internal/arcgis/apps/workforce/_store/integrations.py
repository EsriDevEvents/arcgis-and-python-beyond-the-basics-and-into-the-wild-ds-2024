""" Defines store functions for working with Integrations.
"""

from ... import workforce
from .utils import add_features, update_features, remove_features, validate


def get_integration(project, integration_id=None):
    """Gets the identified Integration.  Exactly one integration_id should be provided.
    :param project:
    :param integration_id
    :returns: Integration
    """
    if integration_id:
        where = "{} = '{}'".format(
            project._integration_schema.integration_id, integration_id
        )
    else:
        where = "1=0"
    integrations = query_integrations(project, where)
    return integrations if integrations else None


def query_integrations(project, where):
    """Executes a query against the integrations table.
    :param project: The project in which to query integrations.
    :param where: An ArcGIS where clause.
    :returns: list of Integrations
    """
    features = project.integrations_table.query(where, return_all_records=True).features
    return [workforce.Integration(project, feature) for feature in features]


def add_integration(
    project,
    feature=None,
    integration_id=None,
    prompt=None,
    url_template=None,
    assignment_type=None,
):
    """
    Adds a new integration to the project
    """
    project._update_cached_objects()
    integration = workforce.Integration(
        project, feature, integration_id, prompt, url_template, assignment_type
    )
    return add_integrations(project, [integration])[0]


def add_integrations(project, integrations):
    """Adds Integrations to a project.

    :param project:
    :param integrations: list of Integrations
    :raises ValidationError: Indicates that one or more integrations failed validation.
    :raises ServerError: Indicates that the server rejected the integrations.
    """
    project._update_cached_objects()
    if integrations:
        use_global_ids = True
        for integration in integrations:
            validate(integration._validate)
            if integration.global_id is None:
                use_global_ids = False
                break

        features = [integration.feature for integration in integrations]
        add_features(project.integrations_table, features, use_global_ids)

    return integrations


def update_integration(
    project,
    integration,
    integration_id=None,
    prompt=None,
    url_template=None,
    assignment_type=None,
):
    """
    Updates an integration and submits changes to the server
    """
    project._update_cached_objects()
    if integration_id:
        integration.integration_id = integration_id
    if prompt:
        integration.prompt = prompt
    if url_template:
        integration.url_template = url_template
    if assignment_type:
        # GUIDs are cast to upper for internal storage purposes
        integration.assignment_type = assignment_type.upper()
    return update_integrations(project, [integration])[0]


def update_integrations(project, integrations):
    """Updates Integrations to a project.
    :param project:
    :param integrations: list of Integrations
    :raises ValidationError: Indicates that one or more dispatchers failed validation.
    :raises ServerError: Indicates that the server rejected the dispatchers.
    """
    project._update_cached_objects()
    if integrations:
        for integration in integrations:
            validate(integration._validate)
        features = [integration.feature for integration in integrations]
        update_features(project.integrations_table, features)

    return integrations


def delete_integrations(project, integrations):
    """Removes Integrations from the project.
    :param project:
    :param integrations: list of Integrations
    :raises ValidationError: Indicates that one or more integrations failed validation.
    :raises ServerError: Indicates that the server rejected the removal.
    """
    project._update_cached_objects()
    if integrations:
        features = [integration.feature for integration in integrations]
        remove_features(project.integrations_table, features)
