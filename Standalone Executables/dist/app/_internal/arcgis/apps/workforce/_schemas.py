class FeatureSchema:
    """Describes the schema used to persist a FeatureModel, which may vary based on the particular
    backend in use (AGO vs Enterprise).
    """

    def __init__(self, feature_layer):
        """Initializes a new FeatureSchema.
        :param feature_layer: The FeatureLayer used to persist the FeatureModel.
        :type feature_layer: arcgis.features.FeatureLayer
        """
        self._feature_layer = feature_layer

    @property
    def object_id(self):
        """Gets the object id field name"""
        return self._field_called("OBJECTID")["name"]

    @property
    def global_id(self):
        """Gets the global id field name"""
        return self._field_called("GlobalID")["name"]

    @property
    def creator(self):
        """Gets the creator field name"""
        return self._editor_tracking_field("creatorField")["name"]

    @property
    def creation_date(self):
        """Gets the creation date field name"""
        return self._editor_tracking_field("creationDateField")["name"]

    @property
    def editor(self):
        """Gets the editor field name"""
        return self._editor_tracking_field("editorField")["name"]

    @property
    def edit_date(self):
        """Gets the edit date field name"""
        return self._editor_tracking_field("editDateField")["name"]

    def _field_called(self, name):
        """Finds a field with a name similar to name, using a case-insensitive comparison."""
        for field in self._feature_layer.properties.fields:
            if field["name"].lower() == name.lower():
                return field

    def _editor_tracking_field(self, name):
        """Finds the editor tracking field identified by the name.  The name corresponds to one of
        the universal editor tracking field names used in the FeatureLayer metadata, including
        'creatorField', 'creationDateField', 'editorField', and 'editDateField'.
        """
        return self._field_called(self._feature_layer.properties.editFieldsInfo[name])


class AssignmentSchema(FeatureSchema):
    """Describes the schema for an assignment :py:attr:`~arcgis.apps.workforce.Assignment.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def assigned_date(self):
        """Gets the assigned date field name"""
        return self._field_called("assignedDate")["name"]

    @property
    def assignment_read(self):
        """Gets the assignment read field name"""
        field = self._field_called("assignmentRead")
        if field is not None:
            return field["name"]

    @property
    def assignment_type(self):
        """Gets the assignment type field name"""
        return self._field_called("assignmentType")["name"]

    @property
    def completed_date(self):
        """Gets the completed date field name"""
        return self._field_called("completedDate")["name"]

    @property
    def declined_comment(self):
        """Gets the comment field name"""
        return self._field_called("declinedComment")["name"]

    @property
    def declined_date(self):
        """Gets the declined date field name"""
        return self._field_called("declinedDate")["name"]

    @property
    def description(self):
        """Gets the description field name"""
        return self._field_called("description")["name"]

    @property
    def dispatcher_id(self):
        """Gets the dispatcher id field name"""
        return self._field_called("dispatcherId")["name"]

    @property
    def due_date(self):
        """Gets the due date field name"""
        return self._field_called("dueDate")["name"]

    @property
    def in_progress_date(self):
        """Gets the in progress date field name"""
        return self._field_called("inProgressDate")["name"]

    @property
    def location(self):
        """Gets the location field name"""
        return self._field_called("location")["name"]

    @property
    def notes(self):
        """Gets the notes field name"""
        return self._field_called("notes")["name"]

    @property
    def paused_date(self):
        """Gets the paused date field name"""
        return self._field_called("pausedDate")["name"]

    @property
    def priority(self):
        """Gets the priority field name"""
        return self._field_called("priority")["name"]

    @property
    def status(self):
        """Gets the status field name"""
        return self._field_called("status")["name"]

    @property
    def work_order_id(self):
        """Gets the work order id field name"""
        return self._field_called("workOrderId")["name"]

    @property
    def worker_id(self):
        """Gets the worker id field name"""
        return self._field_called("workerId")["name"]


class DispatcherSchema(FeatureSchema):
    """Describes the schema for a dispatcher :py:attr:`~arcgis.apps.workforce.Dispatcher.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def contact_number(self):
        """Gets the contact number field name"""
        return self._field_called("contactNumber")["name"]

    @property
    def name(self):
        """Gets the name field name"""
        return self._field_called("name")["name"]

    @property
    def user_id(self):
        """Gets the user id field name"""
        return self._field_called("userId")["name"]


class AssignmentTypeSchema(FeatureSchema):
    """Describes the schema for an assignment type :py:attr:`~arcgis.apps.workforce.AssignmentType.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def description(self):
        """Gets the assignment type field name"""
        return self._field_called("description")["name"]


class IntegrationSchema(FeatureSchema):
    """Describes the schema for an integration :py:attr:`~arcgis.apps.workforce.Integration.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def integration_id(self):
        """Gets the integration app id"""
        return self._field_called("appid")["name"]

    @property
    def prompt(self):
        """Gets the prompt"""
        return self._field_called("prompt")["name"]

    @property
    def url_template(self):
        """Gets the url template"""
        return self._field_called("urltemplate")["name"]

    @property
    def assignment_type(self):
        """Gets the assignment type"""
        return self._field_called("assignmenttype")["name"]


class TrackSchema(FeatureSchema):
    """Describes the schema for a track :py:attr:`~arcgis.apps.workforce.Track.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def accuracy(self):
        """Gets the accuracy field name"""
        return self._field_called("accuracy")["name"]


class WorkerSchema(FeatureSchema):
    """Describes the schema for a worker :py:attr:`~arcgis.apps.workforce.Worker.schema`
    This is useful for getting the field names, which may be different depending on if
    ArcGIS Workforce is hosted on ArcGIS Online or on an Enterprise Deployment.
    """

    @property
    def contact_number(self):
        """Gets the contact number field name"""
        return self._field_called("contactNumber")["name"]

    @property
    def name(self):
        """Gets the name field name"""
        return self._field_called("name")["name"]

    @property
    def notes(self):
        """Gets the notes field name"""
        return self._field_called("notes")["name"]

    @property
    def status(self):
        """Gets the status field name"""
        return self._field_called("status")["name"]

    @property
    def title(self):
        """Gets the title field name"""
        return self._field_called("title")["name"]

    @property
    def user_id(self):
        """Gets the user id field name"""
        return self._field_called("userId")["name"]
