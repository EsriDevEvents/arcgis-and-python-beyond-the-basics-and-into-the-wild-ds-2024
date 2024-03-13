""" Defines the Assignment object.
"""
from .feature_model import FeatureModel
from .managers import *
from ._schemas import AssignmentSchema
from .assignment_type import AssignmentType
from .exceptions import WorkforceWarning, ValidationError
from warnings import warn
import datetime


class Assignment(FeatureModel):
    """
    Represents an assignment

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project that
                           this assignment belongs to.
    ------------------     --------------------------------------------------------------------
    feature                Optional :class:`~arcgis.features.Feature`.
                           A feature containing the assignments attributes. Mostly intended for
                           internal usage. If supplied, other parameters are ignored.
    ------------------     --------------------------------------------------------------------
    geometry               Optional :class:`Dict`.
                           A dictionary containing the assignment geometry
    ------------------     --------------------------------------------------------------------
    assignment_type        Optional :class:`~arcgis.apps.workforce.AssignmentType`.
                           The assignment type that represents this assignment.
    ------------------     --------------------------------------------------------------------
    assigned_date          Optional :class:`Date`
                           The date and time the assignment was assigned
    ------------------     --------------------------------------------------------------------
    assignment_read        Optional :class:`Bool`.
                           A flag indicating that the mobile worker has seen the assignment.
                           Version 1 Projects Only
    ------------------     --------------------------------------------------------------------
    completed_date         Optional :class:`Date`.
                           The date the assignment was completed
    ------------------     --------------------------------------------------------------------
    declined_comment       Optional :class:`String`.
                           The comment submitted by the mobile worker.
    ------------------     --------------------------------------------------------------------
    declined_date          Optional :class:`Date`.
                           The date the assignment was declined.
    ------------------     --------------------------------------------------------------------
    description            Optional :class:`Description`.
                           The description associated with the assignment.
    ------------------     --------------------------------------------------------------------
    dispatcher             Optional :class:`~arcgis.apps.workforce.Dispatcher`.
                           The dispatcher that assigned/created the assignment.
    ------------------     --------------------------------------------------------------------
    due_date               Optional :class:`Date`.
                           The date the assignment is due.
    ------------------     --------------------------------------------------------------------
    in_progress_date       Optional :class:`Date`.
                           The date the assignment was started.
    ------------------     --------------------------------------------------------------------
    location               Optional :class:`String`.
                           The location or address of the assignment.
    ------------------     --------------------------------------------------------------------
    notes                  Optional :class:`String`.
                           The notes associated with the assignment.
    ------------------     --------------------------------------------------------------------
    paused_date            Optional :class:`Date`.
                           The date and time the assignment was paused.
    ------------------     --------------------------------------------------------------------
    priority               Optional :class:`String`.
                           The priority of the assignment

                           `none`, `low`, `medium`, `high`, `critical`
    ------------------     --------------------------------------------------------------------
    status                 Optional :class:`String`.
                           The status of the assignment.

                           `unassigned`, `assigned`, `in_progress`, `completed`, `declined`,
                           `paused`, `canceled`
    ------------------     --------------------------------------------------------------------
    work_order_id          Optional :class:`String`.
                           The work order id associated with the assignment.
    ------------------     --------------------------------------------------------------------
    worker                 Optional :class:`~arcgis.apps.workforce.Worker`.
                           The worker assigned to the assignment
    ==================     ====================================================================

    .. code-block:: python

        # Get an assignment and update it

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        assignment = project.assignments.search()[0]
        assignment.update(priority="high",description="new assignment",location="100 Commercial Street, Portland, ME")
        assignment.delete()


    """

    def __init__(
        self,
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
        priority="none",
        status=None,
        work_order_id=None,
        worker=None,
    ):
        super().__init__(project, project.assignments_layer, feature)
        self._schema = AssignmentSchema(project.assignments_layer)
        if feature:
            # create required objects from the feature
            # uses cached objects that are set when the assignments were last queried
            # speeds up construction by not querying FS for every assignment instantiated
            # forces GUIDs to upper case
            if feature.attributes[project._assignment_schema.worker_id]:
                if self.project._is_v2_project:
                    self.worker = self.project._cached_workers[
                        feature.attributes[project._assignment_schema.worker_id].upper()
                    ]
                else:
                    self.worker = self.project._cached_workers[
                        feature.attributes[project._assignment_schema.worker_id]
                    ]
            else:
                self.worker = None
            if feature.attributes[project._assignment_schema.dispatcher_id]:
                # in case dispatcher for an existing assignment has been deleted
                try:
                    if self.project._is_v2_project:
                        self.dispatcher = self.project._cached_dispatchers[
                            feature.attributes[
                                project._assignment_schema.dispatcher_id
                            ].upper()
                        ]
                    else:
                        self.dispatcher = self.project._cached_dispatchers[
                            feature.attributes[project._assignment_schema.dispatcher_id]
                        ]
                except KeyError:
                    self.dispatcher = None
            else:
                self.dispatcher = None
            if feature.attributes[project._assignment_schema.assignment_type]:
                if self.project._is_v2_project:
                    self.assignment_type = self.project._cached_assignment_types[
                        feature.attributes[
                            project._assignment_schema.assignment_type
                        ].upper()
                    ]
                else:
                    self.assignment_type = self.project._cached_assignment_types[
                        feature.attributes[project._assignment_schema.assignment_type]
                    ]
            else:
                self.assignment_type = None
        else:
            self.geometry = geometry
            self.assigned_date = assigned_date
            if not project._is_v2_project:
                self.assignment_read = assignment_read
            self.completed_date = completed_date
            self.declined_comment = declined_comment
            self.declined_date = declined_date
            self.description = description
            self.due_date = due_date
            self.in_progress_date = in_progress_date
            self.location = location
            self.notes = notes
            self.paused_date = paused_date
            self.priority = priority
            self.status = status
            self.work_order_id = work_order_id
            self.assignment_type = assignment_type
            self.dispatcher = dispatcher
            self.worker = worker

    def __str__(self):
        if self.assignment_type is None:
            type_name = "no type"
        else:
            type_name = self.assignment_type.name
        location = self.location if self.location is not None else "no location"
        return "{} at {}".format(type_name, location)

    def __repr__(self):
        return "<Assignment {}>".format(self.object_id)

    def update(
        self,
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
        Updates the assignment on the server

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        geometry               Optional :class:`Dict`.
                               A dictionary containing the assignment geometry
        ------------------     --------------------------------------------------------------------
        assignment_type        Optional :class:`~arcgis.apps.workforce.AssignmentType`.
                               The assignment type that represents this assignment.
        ------------------     --------------------------------------------------------------------
        assigned_date          Optional :class:`Date`
                               The date and time the assignment was assigned
        ------------------     --------------------------------------------------------------------
        assignment_read        Optional :class:`Bool`.
                               A flag indicating that the mobile worker has seen the assignment.
                               Version 1 Projects Only
        ------------------     --------------------------------------------------------------------
        completed_date         Optional :class:`Date`.
                               The date the assignment was completed
        ------------------     --------------------------------------------------------------------
        declined_comment       Optional :class:`String`.
                               The comment submitted by the mobile worker.
        ------------------     --------------------------------------------------------------------
        declined_date          Optional :class:`Date`.
                               The date the assignment was declined.
        ------------------     --------------------------------------------------------------------
        description            Optional :class:`Description`.
                               The description associated with the assignment.
        ------------------     --------------------------------------------------------------------
        dispatcher             Optional :class:`~arcgis.apps.workforce.Dispatcher`.
                               The dispatcher that assigned/created the assignment.
        ------------------     --------------------------------------------------------------------
        due_date               Optional :class:`Date`.
                               The date the assignment is due.
        ------------------     --------------------------------------------------------------------
        in_progress_date       Optional :class:`Date`.
                               The date the assignment was started.
        ------------------     --------------------------------------------------------------------
        location               Optional :class:`String`.
                               The location or address of the assignment.
        ------------------     --------------------------------------------------------------------
        notes                  Optional :class:`String`.
                               The notes associated with the assignment.
        ------------------     --------------------------------------------------------------------
        paused_date            Optional :class:`Date`.
                               The date and time the assignment was paused.
        ------------------     --------------------------------------------------------------------
        priority               Optional :class:`String`.
                               The priority of the assignment

                               `none`, `low`, `medium`, `high`, `critical`
        ------------------     --------------------------------------------------------------------
        status                 Optional :class:`String`.
                               The status of the assignment.

                               `unassigned`, `assigned`, `in_progress`, `completed`, `declined`,
                               `paused`, `canceled`
        ------------------     --------------------------------------------------------------------
        work_order_id          Optional :class:`String`.
                               The work order id associated with the assignment.
        ------------------     --------------------------------------------------------------------
        worker                 Optional :class:`~arcgis.apps.workforce.Worker`.
                               The worker assigned to the assignment
        ==================     ====================================================================
        """
        update_assignment(
            self.project,
            self,
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

    def delete(self):
        """Deletes the assignment from the server"""
        delete_assignments(self.project, [self])

    @property
    def _supports_assignment_read_field(self):
        return bool(self._schema.assignment_read)

    @property
    def attachments(self):
        """Gets the :class:`~arcgis.apps.workforce.managers.AssignmentAttachmentManager` of the assignment"""
        return AssignmentAttachmentManager(self)

    @property
    def assigned_date(self):
        """Gets/Sets the assigned :class:`datetime` of the assignment"""
        return self._get_datetime_attr(self._schema.assigned_date)

    @assigned_date.setter
    def assigned_date(self, value):
        self._set_datetime_attr(self._schema.assigned_date, value)

    @property
    def assignment_read(self):
        """Gets/Sets the assignment read field"""
        if self._supports_assignment_read_field:
            return bool(self._feature.attributes.get(self._schema.assignment_read))
        else:
            warn(
                "This Workforce Project does not support the assignment_read field.",
                WorkforceWarning,
            )

    @assignment_read.setter
    def assignment_read(self, value):
        if self._supports_assignment_read_field:
            self._feature.attributes[self._schema.assignment_read] = 1 if value else 0
        else:
            warn(
                "This Workforce Project does not support the assignment_read field.",
                WorkforceWarning,
            )

    @property
    def assignment_type_code(self):
        if self.project._is_v2_project and self._assignment_type is not None:
            value = self._feature.attributes.get(self._schema.assignment_type)
            if value:
                return value.upper()
            return None
        else:
            return self._feature.attributes.get(self._schema.assignment_type)

    @property
    def assignment_type(self):
        """Gets/Sets the :class:`~arcgis.apps.workforce.AssignmentType`"""
        return self._assignment_type

    @assignment_type.setter
    def assignment_type(self, value):
        if isinstance(value, AssignmentType) or value is None:
            self._assignment_type = value
        elif isinstance(value, int):
            if value in self.project._cached_assignment_types:
                self._assignment_type = self.project._cached_assignment_types[value]
            else:
                raise ValidationError("Invalid Assignment Type", self)
        elif isinstance(value, str):
            for at in self.project._cached_assignment_types.values():
                if at.name.lower() == value.lower():
                    self._assignment_type = at
                    break
            else:
                raise ValidationError("Invalid Assignment Type", self)
        else:
            raise ValidationError("Invalid Assignment Type", self)
        if self._assignment_type:
            self._feature.attributes[
                self._schema.assignment_type
            ] = self._assignment_type.code
        else:
            self._feature.attributes[self._schema.assignment_type] = None

    @property
    def completed_date(self):
        """Gets/Sets the completed :class:`datetime` of the assignment"""
        return self._get_datetime_attr(self._schema.completed_date)

    @completed_date.setter
    def completed_date(self, value):
        self._set_datetime_attr(self._schema.completed_date, value)

    @property
    def declined_comment(self):
        """Gets/Sets the declined comment of the assignment"""
        return self._feature.attributes.get(self._schema.declined_comment)

    @declined_comment.setter
    def declined_comment(self, value):
        self._feature.attributes[self._schema.declined_comment] = value

    @property
    def declined_date(self):
        """Gets/Sets the declined :class:`datetime` of the assignment"""
        return self._get_datetime_attr(self._schema.declined_date)

    @declined_date.setter
    def declined_date(self, value):
        self._set_datetime_attr(self._schema.declined_date, value)

    @property
    def description(self):
        """Gets/Sets the description for the assignment"""
        return self._feature.attributes.get(self._schema.description)

    @description.setter
    def description(self, value):
        self._feature.attributes[self._schema.description] = value

    @property
    def dispatcher_id(self):
        """Gets the dispatcher id of the assignment"""
        if self.project._is_v2_project and self.dispatcher is not None:
            value = self._feature.attributes.get(self._schema.dispatcher_id)
            if value:
                return value.upper()
            return None
        else:
            return self._feature.attributes.get(self._schema.dispatcher_id)

    @property
    def dispatcher(self):
        """Gets/Sets the :class:`~arcgis.apps.workforce.Dispatcher` of the assignment"""
        return self._dispatcher

    @dispatcher.setter
    def dispatcher(self, value):
        if value is not None:
            self._dispatcher = value
        else:
            self._dispatcher = self.project._cached_dispatcher
        if self.project._is_v2_project:
            self._feature.attributes[
                self._schema.dispatcher_id
            ] = self._dispatcher.global_id
        else:
            self._feature.attributes[
                self._schema.dispatcher_id
            ] = self._dispatcher.object_id

    @property
    def due_date(self):
        """Gets/Sets the due :class:`datetime` of the assignment"""
        return self._get_datetime_attr(self._schema.due_date)

    @due_date.setter
    def due_date(self, value):
        self._set_datetime_attr(self._schema.due_date, value)

    @FeatureModel.geometry.setter
    def geometry(self, value):
        self._feature.geometry = value

    @property
    def in_progress_date(self):
        """Gets/Sets the in progress :class:`datetime` for the assignment"""
        return self._get_datetime_attr(self._schema.in_progress_date)

    @in_progress_date.setter
    def in_progress_date(self, value):
        self._set_datetime_attr(self._schema.in_progress_date, value)

    @property
    def location(self):
        """Gets/Sets the location of the assignment"""
        return self._feature.attributes.get(self._schema.location)

    @location.setter
    def location(self, value):
        self._feature.attributes[self._schema.location] = value

    @property
    def notes(self):
        """Gets/Sets the notes of the assignment"""
        return self._feature.attributes.get(self._schema.notes)

    @notes.setter
    def notes(self, value):
        self._feature.attributes[self._schema.notes] = value

    @property
    def paused_date(self):
        """Gets/Sets the paused :class:`datetime` for the assignment"""
        return self._get_datetime_attr(self._schema.paused_date)

    @paused_date.setter
    def paused_date(self, value):
        self._set_datetime_attr(self._schema.paused_date, value)

    @property
    def priority(self):
        """
        Gets/Sets the :class:`String` priority of the assignment

        `none`, `low`, `medium`, `high`, `critical`
        """
        lut = {0: "none", 1: "low", 2: "medium", 3: "high", 4: "critical"}
        if self._feature.attributes[self._schema.priority] is not None:
            return lut[self._feature.attributes[self._schema.priority]]
        else:
            return None

    @priority.setter
    def priority(self, value):
        if (isinstance(value, int) and value >= 0 and value <= 4) or value is None:
            self._feature.attributes[self._schema.priority] = value
        elif isinstance(value, str):
            reduced_str = value.lower().replace(" ", "").replace("_", "")
            if reduced_str == "none":
                self._feature.attributes[self._schema.priority] = 0
            elif reduced_str == "low":
                self._feature.attributes[self._schema.priority] = 1
            elif reduced_str == "medium":
                self._feature.attributes[self._schema.priority] = 2
            elif reduced_str == "high":
                self._feature.attributes[self._schema.priority] = 3
            elif reduced_str == "critical":
                self._feature.attributes[self._schema.priority] = 4
            else:
                raise ValidationError("Invalid priority", self)
        else:
            raise ValidationError("Invalid priority", self)

    @property
    def status(self):
        """
         Gets/Sets the :class:`String` status of the assignment

        `unassigned`, `assigned`, `in_progress`, `completed`, `declined`,
        `paused`, `canceled`
        """
        lut = {
            0: "unassigned",
            1: "assigned",
            2: "in_progress",
            3: "completed",
            4: "declined",
            5: "paused",
            6: "canceled",
        }
        if self._feature.attributes[self._schema.status] is not None:
            try:
                return lut[self._feature.attributes[self._schema.status]]
            # this supports systems that add to the assignment CVD
            except KeyError:
                return self._feature.attributes[self._schema.status]
        else:
            return None

    @status.setter
    def status(self, value):
        if isinstance(value, str):
            value = value.lower().replace(" ", "").replace("_", "")
        if value == "unassigned" or value == 0:
            self._feature.attributes[self._schema.status] = 0
        elif value == "assigned" or value == 1:
            self._feature.attributes[self._schema.status] = 1
            if self.assigned_date is None:
                self.assigned_date = datetime.datetime.now()
        elif value == "inprogress" or value == 2:
            self._feature.attributes[self._schema.status] = 2
            if self.in_progress_date is None:
                self.in_progress_date = datetime.datetime.now()
        elif value == "completed" or value == 3:
            self._feature.attributes[self._schema.status] = 3
            if self.completed_date is None:
                self.completed_date = datetime.datetime.now()
        elif value == "declined" or value == 4:
            self._feature.attributes[self._schema.status] = 4
            if self.declined_date is None:
                self.declined_date = datetime.datetime.now()
        elif value == "paused" or value == 5:
            self._feature.attributes[self._schema.status] = 5
            if self.paused_date is None:
                self.paused_date = datetime.datetime.now()
        elif value == "canceled" or value == "cancelled" or value == 6:
            self._feature.attributes[self._schema.status] = 6
        elif value is None:
            self._feature.attributes[self._schema.status] = None
        else:
            raise ValidationError("Invalid status", self)

    @property
    def web_app_link(self):
        """Returns a link to the assignment in the Workforce web app"""
        if self.project.gis.properties["isPortal"]:
            portal_url = self.project.gis.properties["portalHostname"]
            # AGOL and Enterprise 10.9+ no longer use hash routing
            if self.project.gis.version >= [8, 4]:
                projects_route = "/apps/workforce/projects/"
            else:
                projects_route = "/apps/workforce/#/projects/"
            return (
                "https://"
                + portal_url
                + projects_route
                + self.project.id
                + "/dispatch/assignments/"
                + str(self.object_id)
            )

    @property
    def work_order_id(self):
        """Gets/Sets the work order id of the assignment"""
        return self._feature.attributes.get(self._schema.work_order_id)

    @work_order_id.setter
    def work_order_id(self, value):
        self._feature.attributes[self._schema.work_order_id] = value

    @property
    def worker_id(self):
        """Gets the worker id of the assignment"""
        if self.project._is_v2_project and self.worker is not None:
            value = self._feature.attributes.get(self._schema.worker_id)
            if value:
                return value.upper()
        else:
            return self._feature.attributes.get(self._schema.worker_id)

    @property
    def worker(self):
        """Gets the :class:`~arcgis.apps.workforce.Worker` of the assignment"""
        return self._worker

    @worker.setter
    def worker(self, value):
        self._worker = value
        if value is None:
            self._feature.attributes[self._schema.worker_id] = None
        elif self.project._is_v2_project:
            self._feature.attributes[self._schema.worker_id] = self._worker.global_id
        else:
            self._feature.attributes[self._schema.worker_id] = self._worker.object_id

    def _validate(self, **kwargs):
        errors = super()._validate(**kwargs)
        errors += self._validate_assignment_type()
        errors += self._validate_location()
        errors += self._validate_geometry()
        errors += self._validate_worker()
        errors += self._validate_dispatcher()
        errors += self._validate_status()
        errors += self._validate_priority()
        return errors

    def _validate_for_add(self, **kwargs):
        errors = super()._validate_for_add(**kwargs)
        errors += self._validate_assignment_type_on_server()
        errors += self._validate_worker_on_server()
        errors += self._validate_dispatcher_on_server()
        return errors

    def _validate_for_update(self, **kwargs):
        errors = super()._validate_for_update(**kwargs)
        errors += self._validate_assignment_type_on_server()
        errors += self._validate_worker_on_server()
        errors += self._validate_dispatcher_on_server()
        return errors

    def _validate_assignment_type(self):
        errors = []
        if self.assignment_type is None:
            errors.append(
                ValidationError("An assignment must have an assignment type", self)
            )
        else:
            if self.assignment_type.name is None:
                errors.append(
                    ValidationError(
                        "Invalid assignment type name: cannot be None", self
                    )
                )
            if self.assignment_type.code is None:
                errors.append(
                    ValidationError(
                        "Invalid assignment type code: cannot be None", self
                    )
                )
        return errors

    def _validate_assignment_type_on_server(self):
        errors = []
        assignment_type = self.project._cached_assignment_types.get(
            self.assignment_type_code, None
        )
        if assignment_type is None:
            errors.append(ValidationError("Unrecognized assignment type", self))
        return errors

    def _validate_location(self):
        errors = []
        if not self.location or self.location.isspace():
            errors.append(
                ValidationError("Assignment cannot have an empty location", self)
            )
        return errors

    def _validate_geometry(self):
        errors = []
        if not self.geometry:
            errors.append(ValidationError("Assignment must have geometry", self))
        return errors

    def _validate_worker(self):
        errors = []
        if self.worker is not None and self.worker.object_id is None:
            errors.append(ValidationError("Worker object_id cannot be None", self))
        return errors

    def _validate_worker_on_server(self):
        errors = []
        if self.worker is not None:
            worker = self.project._cached_workers.get(self.worker_id)
            if not worker:
                errors.append(ValidationError("Unrecognized worker object_id", self))
        return errors

    def _validate_dispatcher(self):
        errors = []
        if self.dispatcher is None:
            errors.append(ValidationError("An assignment must have a dispatcher", self))
        elif self.dispatcher.object_id is None:
            errors.append(ValidationError("Dispatcher object_id cannot be None", self))
        return errors

    def _validate_dispatcher_on_server(self):
        errors = []
        if self.dispatcher is not None:
            dispatcher = self.project._cached_dispatchers.get(self.dispatcher_id)
            if not dispatcher:
                errors.append(
                    ValidationError("Unrecognized dispatcher object_id", self)
                )
        return errors

    def _validate_status(self):
        errors = []
        if self.status is None:
            errors.append(ValidationError("Assignment status cannot be None", self))
        elif self.status == "unassigned":
            if self.worker is not None:
                message = "An UNASSIGNED assignment cannot have a worker"
                errors.append(ValidationError(message, self))
        elif self.status == "assigned":
            if self.worker is None:
                message = "An ASSIGNED assignment must have a worker"
                errors.append(ValidationError(message, self))
            if self.assigned_date is None:
                message = "An ASSIGNED assignment must have an assigned_date"
                errors.append(ValidationError(message, self))
        elif self.status == "in_progress":
            if self.worker is None:
                message = "An IN PROGRESS assignment must have a worker"
                errors.append(ValidationError(message, self))
            if self.assigned_date is None:
                message = "An IN PROGRESS assignment must have an assigned_date"
                errors.append(ValidationError(message, self))
            if self.in_progress_date is None:
                message = "An IN PROGRESS assignment must have an in_progress_date"
                errors.append(ValidationError(message, self))
        elif self.status == "paused":
            if self.worker is None:
                message = "A PAUSED assignment must have a worker"
                errors.append(ValidationError(message, self))
            if self.assigned_date is None:
                message = "A PAUSED assignment must have an assigned_date"
                errors.append(ValidationError(message, self))
            if self.in_progress_date is None:
                message = "A PAUSED assignment must have an in_progress_date"
                errors.append(ValidationError(message, self))
            if self.paused_date is None:
                message = "A PAUSED assignment must have a paused_date"
                errors.append(ValidationError(message, self))
        elif self.status == "completed":
            if self.worker is None:
                message = "A COMPLETED assignment must have a worker"
                errors.append(ValidationError(message, self))
            if self.assigned_date is None:
                message = "A COMPLETED assignment must have an assigned_date"
                errors.append(ValidationError(message, self))
            if self.in_progress_date is None:
                message = "A COMPLETED assignment must have an in_progress_date"
                errors.append(ValidationError(message, self))
            if self.completed_date is None:
                message = "A COMPLETED assignment must have a completed_date"
                errors.append(ValidationError(message, self))
        elif self.status == "declined":
            if self.worker is None:
                message = "A DECLINED assignment must have a worker"
                errors.append(ValidationError(message, self))
            if self.declined_date is None:
                message = "A DECLINED assignment must have a declined_date"
                errors.append(ValidationError(message, self))
            if self.declined_comment is None or self.declined_comment.isspace():
                message = "A DECLINED assignment must have a declined_comment"
                errors.append(ValidationError(message, self))
        return errors

    def _validate_priority(self):
        errors = []
        if not self.priority:
            errors.append(ValidationError("Assignment must have priority", self))
        return errors
