""" Defines the Dispatcher class.
"""

from .. import workforce
from .exceptions import ValidationError
from .feature_model import FeatureModel
from ._store import *
from ._schemas import DispatcherSchema


class Dispatcher(FeatureModel):
    """
    Represents a dispatcher in a project.

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
    contact_number         Optional :class:`String`. The contact number of the dispatcher
    ------------------     --------------------------------------------------------------------
    name                   Optional :class:`String`. The name of the dispatcher
    ------------------     --------------------------------------------------------------------
    user_id                Optional :class:`String`. The user id of the dispatcher
    ==================     ====================================================================

    .. code-block:: python

        # Get a dispatcher, update it, delete it

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        dispatcher = project.dispatchers.search()[0]
        dispatcher.update(name="Dispatcher Name", contact_number="1234567890")
        dispatcher.delete()


    """

    def __init__(
        self, project, feature=None, contact_number=None, name=None, user_id=None
    ):
        super().__init__(project, project.dispatchers_layer, feature)
        self._schema = DispatcherSchema(project.dispatchers_layer)
        if not feature:
            self.contact_number = contact_number
            self.name = name
            self.user_id = user_id

    def __str__(self):
        return "{} ({})".format(self.name, self.user_id)

    def __repr__(self):
        return "<Dispatcher {}>".format(self.name)

    def update(self, contact_number=None, name=None, user_id=None):
        """
        Updates the dispatcher on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        contact_number         Optional :class:`String`. The contact number of the dispatcher
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`String`. The name of the dispatcher
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`String`. The user id of the dispatcher
        ==================     ====================================================================

        """
        update_dispatcher(self.project, self, contact_number, name, user_id)

    def delete(self):
        """Deletes the dispatcher from the server"""
        delete_dispatchers(self.project, [self])

    @property
    def name(self):
        """Gets/Sets the name of the dispatcher"""
        return self._feature.attributes.get(self._schema.name)

    @name.setter
    def name(self, value):
        self._feature.attributes[self._schema.name] = value

    @property
    def contact_number(self):
        """Gets/Sets the contact number of the dispatcher"""
        return self._feature.attributes.get(self._schema.contact_number)

    @contact_number.setter
    def contact_number(self, value):
        self._feature.attributes[self._schema.contact_number] = value

    @property
    def user_id(self):
        """Gets/Sets the user id of the dispatcher"""
        return self._feature.attributes.get(self._schema.user_id)

    @user_id.setter
    def user_id(self, value):
        self._feature.attributes[self._schema.user_id] = value

    def _validate(self, **kwargs):
        errors = super()._validate(**kwargs)
        errors += self._validate_name()
        errors += self._validate_user_id()
        return errors

    def _validate_for_add(self, **kwargs):
        errors = super()._validate_for_add(**kwargs)
        errors += self._validate_user_id_on_server()
        return errors

    def _validate_for_update(self, **kwargs):
        errors = super()._validate_for_update(**kwargs)
        errors += self._validate_user_id_on_server()
        return errors

    def _validate_for_remove(self, **kwargs):
        errors = super()._validate_for_remove(**kwargs)
        where = "{} = '{}'".format(
            self.project._assignment_schema.dispatcher_id, self.id
        )
        assignments = workforce._store.query_assignments(self.project, where=where)
        if assignments:
            errors.append(
                ValidationError("Cannot remove a Dispatcher that has assignments", self)
            )
        if self.user_id == self.project.owner_user_id:
            errors.append(
                ValidationError("Cannot remove the project owner's Dispatcher", self)
            )
        return errors

    def _validate_name(self):
        errors = []
        if not self.name or self.name.isspace():
            errors.append(ValidationError("Dispatcher cannot have an empty name", self))
        return errors

    def _validate_user_id(self):
        errors = []
        if not self.user_id or self.user_id.isspace():
            errors.append(
                ValidationError("Dispatcher cannot have an empty user_id", self)
            )
        return errors

    def _validate_user_id_on_server(self):
        errors = []
        user = self.project.gis.users.get(self.user_id)
        if user is None:
            message = "The Dispatcher must have an accessible named user_id"
            errors.append(ValidationError(message, self))

        dispatchers = [
            d
            for d in self.project._cached_dispatchers.values()
            if d.user_id == self.user_id
        ]
        duplicate_dispatchers = [
            d for d in dispatchers if d.object_id != self.object_id
        ]
        if duplicate_dispatchers:
            message = "There cannot be multiple Dispatchers with the same user_id"
            errors.append(ValidationError(message, self))
        return errors
