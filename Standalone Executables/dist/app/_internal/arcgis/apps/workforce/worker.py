""" Defines the Worker class.
"""

from ..workforce import _store
from .exceptions import ValidationError
from .feature_model import FeatureModel
from ._store import *
from ._schemas import WorkerSchema


class Worker(FeatureModel):
    """
    Represents a worker in a Workforce Project

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project that
                           the worker belongs to.
    ------------------     --------------------------------------------------------------------
    feature                Optional :class:`~arcgis.features.Feature`. The feature representing
                           the worker. Mostly intended for
                           internal usage. If supplied, other parameters are ignored.
    ------------------     --------------------------------------------------------------------
    geometry               Optional :class:`Dict`. The geometry of the worker.
    ------------------     --------------------------------------------------------------------
    contact_number         Optional :class:`String`. The contact number of the worker.
    ------------------     --------------------------------------------------------------------
    name                   Optional :class:`String`. The name of the worker.
    ------------------     --------------------------------------------------------------------
    notes                  Optional :class:`String`. The notes about the worker.
    ------------------     --------------------------------------------------------------------
    status                 Optional :class:`String`. The status of the worker.

                           `not_working`, `working`, `on_break`
    ------------------     --------------------------------------------------------------------
    title                  Optional :class:`String`. The title of the worker.
    ------------------     --------------------------------------------------------------------
    user_id                Optional :class:`String`. The user id of the worker
    ==================     ====================================================================

    .. code-block:: python

        # Get a worker, update it, delete it

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        worker = project.workers.search()[0]
        worker.update(title="Inspector",status="not_working")


    """

    def __init__(
        self,
        project,
        feature=None,
        geometry=None,
        contact_number=None,
        name=None,
        notes=None,
        status="not_working",
        title=None,
        user_id=None,
    ):
        super().__init__(project, project.workers_layer, feature)
        self._schema = WorkerSchema(project.workers_layer)
        if not feature:
            self.geometry = geometry
            self.contact_number = contact_number
            self.name = name
            self.notes = notes
            self.status = status
            self.title = title
            self.user_id = user_id

    def __str__(self):
        return "{} ({})".format(self.name, self.user_id)

    def __repr__(self):
        return "<Worker {}>".format(self.id)

    def update(
        self,
        geometry=None,
        contact_number=None,
        name=None,
        notes=None,
        status=None,
        title=None,
        user_id=None,
    ):
        """
        Updates the worker on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        geometry               Optional :class:`Dict`. The geometry of the worker.
        ------------------     --------------------------------------------------------------------
        contact_number         Optional :class:`String`. The contact number of the worker.
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`String`. The name of the worker.
        ------------------     --------------------------------------------------------------------
        notes                  Optional :class:`String`. The notes about the worker.
        ------------------     --------------------------------------------------------------------
        status                 Optional :class:`String`. The status of the worker.

                               `not_working`, `working`, `on_break`
        ------------------     --------------------------------------------------------------------
        title                  Optional :class:`String`. The title of the worker.
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`String`. The user id of the worker
        ==================     ====================================================================

        """
        update_worker(
            self.project,
            self,
            geometry,
            contact_number,
            name,
            notes,
            status,
            title,
            user_id,
        )

    def delete(self):
        """Deletes the worker from the server"""
        delete_workers(self.project, [self])

    @property
    def name(self):
        """Gets/Sets the name of the worker"""
        return self._feature.attributes.get(self._schema.name)

    @name.setter
    def name(self, value):
        self._feature.attributes[self._schema.name] = value

    @property
    def contact_number(self):
        """Gets/Sets the contact number of the worker"""
        return self._feature.attributes.get(self._schema.contact_number)

    @contact_number.setter
    def contact_number(self, value):
        self._feature.attributes[self._schema.contact_number] = value

    @property
    def title(self):
        """Gets/Sets the title of the worker"""
        return self._feature.attributes.get(self._schema.title)

    @title.setter
    def title(self, value):
        self._feature.attributes[self._schema.title] = value

    @property
    def notes(self):
        """Gets/Sets the notes of the worker"""
        return self._feature.attributes.get(self._schema.notes)

    @notes.setter
    def notes(self, value):
        self._feature.attributes[self._schema.notes] = value

    @property
    def user_id(self):
        """Gets/Sets the user id of the worker"""
        return self._feature.attributes.get(self._schema.user_id)

    @user_id.setter
    def user_id(self, value):
        self._feature.attributes[self._schema.user_id] = value

    @property
    def status(self):
        """
        Gets/Sets the :class:`String` status of the worker

        `not_working`, `working`, `on_break`
        """
        lut = {
            0: "not_working",
            1: "working",
            2: "on_break",
        }
        if self._feature.attributes[self._schema.status] is not None:
            return lut[self._feature.attributes[self._schema.status]]
        else:
            return None

    @status.setter
    def status(self, value):
        if (isinstance(value, int) and value >= 0 and value <= 2) or value is None:
            self._feature.attributes[self._schema.status] = value
        elif isinstance(value, str):
            reduced_str = value.lower().replace(" ", "").replace("_", "")
            if reduced_str == "notworking":
                self._feature.attributes[self._schema.status] = 0
            elif reduced_str == "working":
                self._feature.attributes[self._schema.status] = 1
            elif reduced_str == "onbreak":
                self._feature.attributes[self._schema.status] = 2
            else:
                raise ValidationError("Invalid status", self)
        else:
            raise ValidationError("Invalid status", self)

    @FeatureModel.geometry.setter
    def geometry(self, value):
        self._feature.geometry = value

    def _validate(self, **kwargs):
        """ """
        errors = super()._validate(**kwargs)
        errors += self._validate_name()
        errors += self._validate_status()
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
        assignments = _store.query_assignments(
            self.project,
            "{} = '{}'".format(self.project._assignment_schema.worker_id, self.id),
        )
        if assignments:
            errors.append(
                ValidationError("Cannot remove a Worker that has assignments", self)
            )
        return errors

    def _validate_name(self):
        errors = []
        if not self.name or self.name.isspace():
            errors.append(ValidationError("Worker cannot have an empty name", self))
        return errors

    def _validate_status(self):
        errors = []
        if self.status is None:
            errors.append(ValidationError("Worker must have a status", self))
        return errors

    def _validate_user_id(self):
        errors = []
        if not self.user_id or self.user_id.isspace():
            errors.append(ValidationError("Worker cannot have an empty user_id", self))
        return errors

    def _validate_user_id_on_server(self):
        errors = []
        user = self.project.gis.users.get(self.user_id)
        if user is None:
            message = "The Worker user_id must match an accessible named user id"
            errors.append(ValidationError(message, self))

        workers = [
            w
            for w in self.project._cached_workers.values()
            if w.user_id == self.user_id
        ]
        duplicate_workers = [w for w in workers if w.object_id != self.object_id]
        if duplicate_workers:
            message = "There cannot be multiple Workers with the same user_id"
            errors.append(ValidationError(message, self))
        return errors
