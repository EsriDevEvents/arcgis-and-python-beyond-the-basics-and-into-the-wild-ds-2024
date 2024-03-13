"""
Defines the AssignmentType class.
"""
from .exceptions import ValidationError
from .feature_model import FeatureModel
from ._store import *
from ._store.assignment_types_v2 import *
from ._schemas import AssignmentTypeSchema


class AssignmentType(FeatureModel):
    """
    Defines the acceptable values for :class:`~arcgis.apps.workforce.AssignmentType` types.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project that
                           this assignment belongs to.
    ------------------     --------------------------------------------------------------------
    coded_value            Optional :class:`dict`. The dictionary storing the code and
                           name of the type. Only works for v1 projects.
    ------------------     --------------------------------------------------------------------
    name                   Optional :class:`String`. The name of the assignment type.
    ==================     ====================================================================

    .. code-block:: python

        # Get an assignment type, update it, delete it

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        assignment_type = project.assignment_types.search()[0]
        assignment_type.update(name="Manhole Inspection")
        assignment_type.delete()


    """

    def __init__(self, project, feature=None, coded_value=None, name=None):
        if project._is_v2_project:
            super().__init__(
                project=project,
                feature_layer=project.assignment_types_table,
                feature=feature,
            )
            self._schema = AssignmentTypeSchema(project.assignment_types_table)
            self._coded_value = None
            if not feature:
                self.name = name
        else:
            super().__init__()
            if coded_value:
                self._coded_value = coded_value
            else:
                self._coded_value = {"code": None, "name": name}
        self.project = project

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<AssignmentType {}>".format(self.name)

    def update(self, name=None):
        """
        Updates the assignment type on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`String`.
                               The name of the assignment type
        ==================     ====================================================================
        """
        if self.project._is_v2_project:
            update_assignment_type_v2(self.project, self, name)
        else:
            update_assignment_type(self.project, self, name)

    def delete(self):
        """Deletes the assignment type from the server"""
        if self.project._is_v2_project:
            delete_assignment_types_v2(self.project, [self])
        else:
            delete_assignment_types(self.project, [self])

    @property
    def id(self):
        """Gets the id of the assignment type"""
        return self.code

    @property
    def code(self):
        """Gets the internal code that uniquely identifies the assignment type"""
        if not self.project._is_v2_project:
            return self._coded_value["code"]
        elif self._feature.attributes.get(self._schema.global_id):
            return self._feature.attributes.get(self._schema.global_id).upper()
        else:
            return None

    @property
    def name(self):
        """Gets/Sets The name of the assignment type"""
        if not self.project._is_v2_project:
            return self._coded_value["name"]
        else:
            return self._feature.attributes.get(self._schema.description)

    @name.setter
    def name(self, value):
        if not self.project._is_v2_project:
            self._coded_value["name"] = value
        else:
            self._feature.attributes[self._schema.description] = value

    @property
    def coded_value(self):
        """Gets the coded value"""
        return self._coded_value

    def _validate(self, **kwargs):
        errors = super()._validate(**kwargs)
        errors += self._validate_name()
        errors += self._validate_name_uniqueness(**kwargs)
        return errors

    def _validate_for_update(self, **kwargs):
        if not self.project._is_v2_project:
            return super()._validate_for_update(**kwargs) + self._validate_code()
        else:
            return super()._validate_for_update(**kwargs)

    def _validate_for_remove(self, **kwargs):
        if not self.project._is_v2_project:
            assignments = kwargs["assignments"]
            errors = super()._validate_for_remove(**kwargs) + self._validate_code()
            if assignments is None:
                schema = self.project._assignment_schema
                where = "{}={}".format(schema.assignment_type, self.code)
                assignments = self.project.assignments.search(where=where)
            else:
                assignments = [
                    a for a in assignments if a.assignment_type.code == self.code
                ]
        else:
            assignments = self.project.assignments.search(where="1=1")
            errors = super()._validate_for_remove(**kwargs)
            assignments = [
                a for a in assignments if a.assignment_type.code == self.code
            ]

        if assignments:
            errors.append(
                ValidationError("Cannot remove an in-use AssignmentType", self)
            )
        return errors

    def _validate_name(self):
        errors = []
        if self.name is None or self.name.isspace():
            errors.append(ValidationError("AssignmentType must have a name", self))
        elif ">" in self.name or "<" in self.name or "%" in self.name:
            errors.append(
                ValidationError("AssignmentType name contains invalid characters", self)
            )
        return errors

    def _validate_name_uniqueness(self, assignment_types=None):
        errors = []
        if assignment_types is None:
            assignment_types = self.project.assignment_types.search()
        for assignment_type in assignment_types:
            # note that code is AT guid for v2 projects
            if (
                self.name
                and assignment_type.name.lower() == self.name.lower()
                and assignment_type.code != self.code
            ):
                errors.append(
                    ValidationError("AssignmentType name must be unique", self)
                )
        return errors

    def _validate_code(self):
        errors = []
        if not isinstance(self.code, int):
            errors.append(ValidationError("Code must be a unique integer", self))
        return errors
