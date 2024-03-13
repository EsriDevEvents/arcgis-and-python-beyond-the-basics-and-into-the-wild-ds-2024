""" Defines the Integration class.
"""

from .exceptions import ValidationError
from .feature_model import FeatureModel
from ._store import *
from ._schemas import IntegrationSchema
from .assignment_type import AssignmentType


class Integration(FeatureModel):
    """
    Represents an integration in a project. Version 2 Workforce projects only.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project that
                           the dispatcher belongs to.
    ------------------     --------------------------------------------------------------------
    feature                Optional :class:`~arcgis.features.Feature`. The feature representing
                           the dispatcher. Mostly intended for
                           internal usage. If supplied, other parameters are ignored.
    ------------------     --------------------------------------------------------------------
    integration_id         Optional :class:`String`. The id for the integration
    ------------------     --------------------------------------------------------------------
    prompt                 Optional :class:`String`. The prompt in the mobile app for the
                           integration
    ------------------     --------------------------------------------------------------------
    url_template           Optional :class:`String`. The url that the prompt links to
    ------------------     --------------------------------------------------------------------
    assignment_type        Optional :class:`String`. The assignment type for the integration
    ==================     ====================================================================

    .. code-block:: python

        # Get an integration, update it, delete it

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        integration = project.integrations.search()[0]
        integration.update(integration_id="arcgis-navigator",prompt="Navigate to Assignment")
        integration.delete()


    """

    def __init__(
        self,
        project,
        feature=None,
        integration_id=None,
        prompt=None,
        url_template=None,
        assignment_type=None,
    ):
        super().__init__(project, project.integrations_table, feature)
        self._schema = IntegrationSchema(project.integrations_table)
        if not feature:
            self.integration_id = integration_id
            self.prompt = prompt
            self.url_template = url_template
            if assignment_type:
                if isinstance(assignment_type, AssignmentType):
                    self.assignment_type = assignment_type.code
                else:
                    self.assignment_type = assignment_type.upper()

    def __str__(self):
        return "<Integration {}>".format(self.integration_id)

    def __repr__(self):
        return "<Integration {}>".format(self.integration_id)

    def update(
        self, integration_id=None, prompt=None, url_template=None, assignment_type=None
    ):
        """
        Updates the dispatcher on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        integration_id         Optional :class:`String`. The id for the integration
        ------------------     --------------------------------------------------------------------
        prompt                 Optional :class:`String`. The prompt in the mobile app for the
                               integration
        ------------------     --------------------------------------------------------------------
        url_template           Optional :class:`String`. The url that the prompt links to
        ------------------     --------------------------------------------------------------------
        assignment_type        Optional :class:`String`. The assignment type for the integration
        ==================     ====================================================================

        """
        update_integration(
            self.project, self, integration_id, prompt, url_template, assignment_type
        )

    def delete(self):
        """Deletes the integration from the server"""
        delete_integrations(self.project, [self])

    @property
    def integration_id(self):
        """Gets/Sets the id of the integration"""
        return self._feature.attributes.get(self._schema.integration_id)

    @integration_id.setter
    def integration_id(self, value):
        self._feature.attributes[self._schema.integration_id] = value

    @property
    def prompt(self):
        """Gets/Sets the prompt of the integration"""
        return self._feature.attributes.get(self._schema.prompt)

    @prompt.setter
    def prompt(self, value):
        self._feature.attributes[self._schema.prompt] = value

    @property
    def url_template(self):
        """Gets/Sets the url template of the integration"""
        return self._feature.attributes.get(self._schema.url_template)

    @url_template.setter
    def url_template(self, value):
        self._feature.attributes[self._schema.url_template] = value

    @property
    def assignment_type(self):
        """Gets/Sets the assignment type of the integration"""
        return self._feature.attributes.get(self._schema.assignment_type)

    @assignment_type.setter
    def assignment_type(self, value):
        self._feature.attributes[self._schema.assignment_type] = value

    def _validate(self):
        if not self.integration_id:
            raise ValidationError("Integration must have an id", self)
        elif not self.prompt:
            raise ValidationError("Assignment integration must contain a prompt", self)
        elif not self.url_template:
            raise ValidationError(
                "Assignment integration must contain a URL template", self
            )
        elif self.assignment_type:
            if self.assignment_type.upper() not in [
                at.code.upper() for at in self.project.assignment_types.search()
            ]:
                raise ValidationError("Invalid assignment type in integration", self)
            # Don't enforce this when doing an update (which means the integration already has a global_id)
            for integration in self.project.integrations.search():
                if (
                    self.global_id is None
                    and integration.integration_id == self.integration_id
                    and not integration.assignment_type
                ):
                    raise ValidationError(
                        "Cannot add an integration with an assignment type when project level integration of same id exists",
                        self,
                    )
                if (
                    self.global_id is None
                    and integration.integration_id == self.integration_id
                    and integration.assignment_type.upper()
                    == self.assignment_type.upper()
                ):
                    raise ValidationError(
                        "Cannot add an integration with the same id and assignment type",
                        self,
                    )
        else:
            # Don't enforce this when doing an update (which means the integration already has a global_id)
            if self.global_id is None and self.integration_id in [
                integration.integration_id
                for integration in self.project.integrations.search()
            ]:
                raise ValidationError(
                    "Cannot add project level integration when same id integration exists",
                    self,
                )
