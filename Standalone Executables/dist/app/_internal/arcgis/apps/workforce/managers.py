import os
from ._store import *
from .exceptions import *


class AssignmentManager(object):
    """
    This manages the assignments in the project.
    It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.assignments`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                           manage.
    ==================     ====================================================================

    .. code-block:: python

        # Add / get assignments using assignment manager

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        assignment = project.assignments.add(assignment_type=type_1, location="100 Commercial St",
                                            geometry={'x': -7820308, 'y': 5412450}, status=0, priority=0)
        assignment_2 = project.assignments.get(object_id=2)


    """

    def __init__(self, project):
        self.project = project

    def get(self, object_id=None, global_id=None):
        """
        Gets the identified assignment by either an object id or global id.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        object_id              Optional :class:`integer`. The object id of the assignment to get
        ------------------     --------------------------------------------------------------------
        global_id              Optional :class:`string`. The global id of the assignment to get.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Assignment`
        """
        return get_assignment(self.project, object_id, global_id)

    def search(self, where="1=1"):
        """
        Searches the assignments in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        where                  Optional :class:`string`. The where clause to use to query
                               assignments. Defaults to '1=1'
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Assignment`
        """
        return query_assignments(self.project, where)

    def batch_add(self, assignments):
        """
        Adds the list of assignments to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignments            Required :class:`List` of :class:`~arcgis.apps.workforce.Assignment`.
                               The list of assignments to add.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Assignment`
        """
        return add_assignments(self.project, assignments)

    def add(
        self,
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
        """
        Creates and adds a new assignment to the project

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        feature                Optional :class:`~arcgis.features.Feature`.
                               A feature containing the assignments attributes. If this is provided
                               the other parameters are all ignored.
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
                               A flag indicating that the mobile worker has seen the assignment
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
        ------------------     --------------------------------------------------------------------
        status                 Optional :class:`String`.
                               The status of the assignment.
        ------------------     --------------------------------------------------------------------
        work_order_id          Optional :class:`String`.
                               The work order id associated with the assignment.
        ------------------     --------------------------------------------------------------------
        worker                 Optional :class:`~arcgis.apps.workforce.Worker`.
                               The worker assigned to the assignment
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Assignment`
        """
        return add_assignment(
            self.project,
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

    def batch_update(self, assignments):
        """
        Updates the list of assignments in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignments            Required :class:`List` of :class:`~arcgis.apps.workforce.Assignment`.
                               The list of assignments to update.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Assignment`
        """
        return update_assignments(self.project, assignments)

    def batch_delete(self, assignments):
        """
        Removes the list of assignments from the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignments            Required :class:`List` of :class:`~arcgis.apps.workforce.Assignment`.
                               The list of assignments to remove.
        ==================     ====================================================================
        """
        return delete_assignments(self.project, assignments)


class AssignmentTypeManager:
    """
    This manages the assignment types in the project.
    It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.assignment_types`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                           manage.
    ==================     ====================================================================

    .. code-block:: python

        # Add / get assignment types using assignment type manager

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        assignment_type = project.assignment_types.add(name="Tree Inspection")
        assignment_type_2 = project.assignment_types.get(name="Fix Sign")


    """

    def __init__(self, project):
        self.project = project

    def get(self, code=None, name=None):
        """
        Gets the identified assignment type by either the name or code.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        code                   Optional :class:`integer`. The code of the assignment type.
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`string`. The name of the assignment type.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.AssignmentType`
        """
        if not self.project._is_v2_project:
            return get_assignment_type(self.project, code=code, name=name)
        else:
            return get_assignment_type_v2(self.project, code=code, name=name)

    def search(self):
        """
        Gets all of the assignment types in the project.

        :return: :class:`List` of :class:`~arcgis.apps.workforce.AssignmentType`
        """
        if not self.project._is_v2_project:
            return get_assignment_types(self.project)
        else:
            return get_assignment_types_v2(self.project)

    def add(self, coded_value=None, name=None):
        """
        Adds an assignment type to the project.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        coded_value            Optional :class:`dict`. The dictionary storing the code and
                               name of the type. Only works for v1 projects.
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`String`. The name of the assignment type.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.AssignmentType`
        """
        if not self.project._is_v2_project:
            return add_assignment_type(self.project, coded_value=coded_value, name=name)
        else:
            return add_assignment_type_v2(self.project, name=name)

    def batch_add(self, assignment_types):
        """
        Adds the list of assignment types to the project.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignment_types       Required :class:`List` of :class:`~arcgis.apps.workforce.AssignmentTypes`.
                               The list of assignment types to add.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.AssignmentTypes`
        """
        if not self.project._is_v2_project:
            return add_assignment_types(self.project, assignment_types)
        else:
            return add_assignment_types_v2(self.project, assignment_types)

    def batch_update(self, assignment_types):
        """
        Updates the list of assignment types to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignment_types       Required :class:`List` of :class:`~arcgis.apps.workforce.AssignmentTypes`.
                               The list of assignment types to update.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.AssignmentType`
        """
        if not self.project._is_v2_project:
            return update_assignment_types(self.project, assignment_types)
        else:
            return update_assignment_types_v2(self.project, assignment_types)

    def batch_delete(self, assignment_types):
        """
        Removes the list of assignment types to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        assignment_types       Required :class:`List` of :class:`~arcgis.apps.workforce.AssignmentTypes`.
                               The list of assignment types to remove.
        ==================     ====================================================================
        """
        if not self.project._is_v2_project:
            return delete_assignment_types(self.project, assignment_types)
        else:
            return delete_assignment_types_v2(self.project, assignment_types)


class AssignmentAttachmentManager(object):
    """
    This manages the attachments associated with an assignment.
    It can be accessed from the assignment as :py:attr:`~arcgis.apps.workforce.Assignment.attachments`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    assignment             Required :class:`~arcgis.apps.workforce.Assignment`. The assignment to
                           manage.
    ==================     ====================================================================

    """

    def __init__(self, assignment):
        self.assignment = assignment

    def get(self):
        """
        This gets all of the Attachments belonging to the assignment.

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Attachment`
        """
        return get_attachments(self.assignment)

    def add(self, file_path):
        """
        Adds the file as an attachment to the assignment.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        file_path              Required :class:`string` The file to upload.
        ==================     ====================================================================
        """
        return add_attachment(self.assignment, file_path)

    def batch_delete(self, attachments):
        """
        Removes the list of attachments from the assignment.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        attachments            Required :class:`List` of :class:`~arcgis.apps.workforce.Attachment`.
                               The list of attachments to delete.
        ==================     ====================================================================
        """
        return delete_attachments(self.assignment, attachments)

    def download(self, out_folder=None):
        """
        Downloads all of an assignments attachments.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        out_folder             Required :class:`string`. The folder to download the attachments to.
        ==================     ====================================================================

        :return: A :class:`List` of file path strings
        """
        if not out_folder:
            out_folder = os.getcwd()
        attachments = get_attachments(self.assignment)
        paths = []
        for attachment in attachments:
            paths.append(attachment.download(out_folder))
        return paths


class DispatcherManager:
    """
     This manages the dispatchers in the project.
     It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.dispatchers`

     ==================     ====================================================================
     **Parameter**           **Description**
     ------------------     --------------------------------------------------------------------
     project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                            manage.
     ==================     ====================================================================

    .. code-block:: python

        # Add / get dispatchers using dispatcher manager

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        dispatcher = project.dispatchers.add(name="New Dispatcher",user_id="dispatcher_username")
        dispatcher_2 = project.dispatchers.get(user_id="dispatcher2_username")


    """

    def __init__(self, project):
        self.project = project

    def get(self, object_id=None, global_id=None, user_id=None):
        """
        This gets a dispatcher by their object id, global id, or user id.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        object_id              Optional :class:`integer`. The object id of the dispatcher to get
        ------------------     --------------------------------------------------------------------
        global_id              Optional :class:`string`. The global id of the dispatcher to get.
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`string`. The user id of the dispatcher to get.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Dispatcher`

        """
        return get_dispatcher(self.project, object_id, global_id, user_id)

    def search(self, where="1=1"):
        """
        Searches the dispatchers in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        where                  Optional :class:`string`. The where clause to use to query
                               dispatchers. Defaults to '1=1'
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`
        """
        return query_dispatchers(self.project, where)

    def add(self, feature=None, contact_number=None, name=None, user_id=None):
        """
        Creates and adds a dispatcher to the project.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        project                Required :class:`~arcgis.apps.workforce.Project`. The project that
                               the dispatcher belongs to.
        ------------------     --------------------------------------------------------------------
        feature                Optional :class:`~arcgis.features.Feature`. The feature representing
                               the dispatcher.
        ------------------     --------------------------------------------------------------------
        contact_number         Optional :class:`String`. The contact number of the dispatcher
        ------------------     --------------------------------------------------------------------
        name                   Optional :class:`String`. The name of the dispatcher
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`String`. The user id of the dispatcher
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Dispatcher`
        """
        return add_dispatcher(self.project, feature, contact_number, name, user_id)

    def batch_add(self, dispatchers):
        """
        Adds the list of dispatchers to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        dispatchers            Required :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`.
                               The list of dispatchers to add.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`
        """
        return add_dispatchers(self.project, dispatchers)

    def batch_update(self, dispatchers):
        """
        Adds the list of dispatchers to update in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        dispatchers            Required :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`.
                               The list of dispatchers to update.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`
        """
        return update_dispatchers(self.project, dispatchers)

    def batch_delete(self, dispatchers):
        """
        Removes the list of dispatchers to remove from the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        dispatchers            Required :class:`List` of :class:`~arcgis.apps.workforce.Dispatcher`.
                               The list of dispatchers to remove.
        ==================     ====================================================================
        """
        return delete_dispatchers(self.project, dispatchers)


class TrackManager:
    """
    This manages the tracks in the project.
    It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.tracks`

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                           manage.
    ==================     ====================================================================
    """

    def __init__(self, project):
        if not project._supports_tracks:
            raise WorkforceError("This Workforce Project does not support tracks.")
        self.project = project

    def get(self, object_id=None, global_id=None):
        """
        This gets a track by their object id or global id.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        object_id              Optional :class:`integer`. The object id of the track to get
        ------------------     --------------------------------------------------------------------
        global_id              Optional :class:`string`. The global id of the track to get.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Track`

        """
        return get_track(self.project, object_id, global_id)

    def search(self, where="1=1"):
        """
        Searches the tracks in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        where                  Optional :class:`string`. The where clause to use to query
                               tracks. Defaults to '1=1'
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Track`
        """
        return query_tracks(self.project, where)

    def add(self, feature=None, geometry=None, accuracy=None):
        """
        Adds a track to the project.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        feature                Optional :class:`~arcgis.features.Feature`.
                               The feature to use.
        ------------------     --------------------------------------------------------------------
        geometry               Optional :class:`Dict`.
                               A dictionary containing the assignment geometry
        ------------------     --------------------------------------------------------------------
        accuracy               Optional :class:`Float`.
                               The accuracy to use.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Track`
        """
        return add_track(self.project, feature, geometry, accuracy)

    def batch_add(self, tracks):
        """
        Adds the list of tracks to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        tracks                 Required :class:`List` of :class:`~arcgis.apps.workforce.Track`.
                               The list of tracks to add.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Track`
        """
        return add_tracks(self.project, tracks)

    def batch_delete(self, tracks):
        """
        Removes the list of tracks to remove from the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        tracks                 Required :class:`List` of :class:`~arcgis.apps.workforce.Track`.
                               The list of tracks to remove.
        ==================     ====================================================================
        """
        return delete_tracks(self.project, tracks)

    def batch_update(self, tracks):
        """
        Updates the list of tracks in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        tracks                 Required :class:`List` of :class:`~arcgis.apps.workforce.Track`.
                               The list of tracks to update.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Track`
        """
        return update_tracks(self.project, tracks)

    @property
    def enabled(self):
        """Gets/sets if tracking is enabled for the project"""
        return self.project._tracking_enabled

    @enabled.setter
    def enabled(self, value):
        self.project._tracking_enabled = value

    @property
    def interval(self):
        """Gets/sets the tracking interval for the project (in seconds)"""
        return self.project._tracking_interval

    @interval.setter
    def interval(self, value):
        self.project._tracking_interval = value


class WorkerManager:
    """
     This manages the workers in the project
     It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.workers`

     ==================     ====================================================================
     **Parameter**           **Description**
     ------------------     --------------------------------------------------------------------
     project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                            manage.
     ==================     ====================================================================

    .. code-block:: python

        # Add / get workers using worker manager

        import arcgis
        gis = arcgis.gis.GIS("https://arcgis.com", "<username>", "<password>")
        item = gis.content.get("<item-id>")
        project = arcgis.apps.workforce.Project(item)
        worker = project.workers.add(user_id="worker_username", name="Worker One", status=0)


    """

    def __init__(self, project):
        self.project = project

    def get(self, object_id=None, global_id=None, user_id=None):
        """
        This gets a worker by their object id, global id, or user id.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        object_id              Optional :class:`integer`. The object id of the worker to get
        ------------------     --------------------------------------------------------------------
        global_id              Optional :class:`string`. The global id of the worker to get.
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`string`. The user id of the worker to get.
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Worker`

        """
        return get_worker(self.project, object_id, global_id, user_id)

    def search(self, where="1=1"):
        """
        Searches the workers in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        where                  Optional :class:`string`. The where clause to use to query
                               workers. Defaults to '1=1'
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Worker`
        """
        return query_workers(self.project, where)

    def batch_add(self, workers):
        """
        Adds the list of workers to the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        workers                Required :class:`List` of :class:`~arcgis.apps.workforce.Worker`.
                               The list of workers to add.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Worker`
        """
        return add_workers(self.project, workers)

    def add(
        self,
        feature=None,
        geometry=None,
        contact_number=None,
        name=None,
        notes=None,
        status="not_working",
        title=None,
        user_id=None,
    ):
        """
        Creates and adds a new worker to the project.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        feature                Optional :class:`~arcgis.features.Feature`. The feature representing
                               the worker.
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
        ------------------     --------------------------------------------------------------------
        title                  Optional :class:`String`. The title of the worker.
        ------------------     --------------------------------------------------------------------
        user_id                Optional :class:`String`. The user id of the worker
        ==================     ====================================================================

        :return: :class:`~arcgis.apps.workforce.Worker`
        """
        return add_worker(
            self.project,
            feature,
            geometry,
            contact_number,
            name,
            notes,
            status,
            title,
            user_id,
        )

    def batch_update(self, workers):
        """
        Adds the list of workers to update in the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        workers                Required :class:`List` of :class:`~arcgis.apps.workforce.Worker`.
                               The list of workers to update.
        ==================     ====================================================================

        :return: :class:`List` of :class:`~arcgis.apps.workforce.Worker`
        """
        return update_workers(self.project, workers)

    def batch_delete(self, workers):
        """
        Adds the list of workers to remove from the project.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        workers                Required :class:`List` of :class:`~arcgis.apps.workforce.Worker`.
                               The list of workers to remove.
        ==================     ====================================================================
        """
        return delete_workers(self.project, workers)


class AssignmentIntegrationManager:
    """
    This manages the assignment integrations in the project
    It can be accessed from the project as :py:attr:`~arcgis.apps.workforce.Project.integrations`

    For a version 2 (offline-enabled project), an integration is represented by an object
    :class:`~arcgis.apps.workforce.Integration`
    object and can be updated in the same fashion as Assignment, AssignmentType, Dispatcher, Project,
    and Worker objects.

    For a version 1 project, an integration in Workforce consists of a formatted dictionary.
    Two examples are shown below:

    .. code-block:: python

           navigator_integration = {
               "id": "default-navigator",
               "prompt": "Navigate to Assignment",
               "urlTemplate": "arcgis-navigator://?stop=${assignment.latitude},${assignment.longitude}&stopname=${assignment.location}&callback=arcgis-workforce://&callbackprompt=Workforce"
           }

           explorer_integration = {
               "id": "default-explorer",
               "prompt": "Explore at Assignment",
               "assignmentTypes": {
                   "1": {
                       "urlTemplate": "arcgis-explorer://?itemID=651324c8661b42c897657f8afbe846qe&center=${assignment.latitude},${assignment.longitude}&scale=9000"
                   }
           }

    The urlTemplate can be generated by using the :py:mod:`~arcgis.apps.integrations` module

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    project                Required :class:`~arcgis.apps.workforce.Project`. The project to
                           manage.
    ==================     ====================================================================
    """

    def __init__(self, project):
        self.project = project
        # A version 2 Workforce project stores integrations in a table in its base feature layer collection
        if self.project._is_v2_project:
            self.integration_table = project.integrations_table
        elif "assignmentIntegrations" not in self.project._item_data:
            self.project._item_data["assignmentIntegrations"] = []

    def get(self, integration_id):
        """
        This gets an integration dictionary by its id

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        integration_id         Required :class:`string`. The id of the integration. This is field
                               'appid' for a Version 2 Workforce project.
        ==================     ====================================================================

        :returns: Version 1: :class:`dict` or :class:`None`, Version 2: :class:`Integration`

        """
        if self.project._is_v2_project:
            return get_integration(self.project, integration_id)
        else:
            for integration in self.project._item_data["assignmentIntegrations"]:
                if integration_id == integration["id"]:
                    return integration
            return None

    def search(self, where="1=1"):
        """
        This returns all of the assignment integrations for the project

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        where                  Optional :class:`string`. ArcGIS where clause - version 2 projects
                               only. Defaults to "1=1"
        ==================     ====================================================================

        :returns: :class:`List` A list of the integrations.
        """
        if self.project._is_v2_project:
            return query_integrations(self.project, where=where)
        else:
            return self.project._item_data["assignmentIntegrations"]

    def add(self, integration_id, prompt, url_template=None, assignment_types=None):
        """
        This adds an integration to the project

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        integration_id         Required :class:`string`. The id of the integration
        ------------------     --------------------------------------------------------------------
        prompt                 Required: :class:`string`. The prompt to display.
        ------------------     --------------------------------------------------------------------
        url_template           Required for version 2 Workforce projects. Optional for version 1.
                               :class:`string`. The url template that is used for app linking.
        ------------------     --------------------------------------------------------------------
        assignment_types       Optional: :class:`string` or :class:`list`
                               Version 2 Projects:
                               String which is a globalid representing an assignment type. This is
                               stored at assignment_type.code. If you pass a list for this object,
                               it will add multiple integrations, with identical integration_id,
                               prompt, and url_template values, with differring assignment_types
                               values.

                               Version 1 Projects: :class:`dict`.
                               A dictionary containing assignment type
                               codes as keys and a dictionaries that contains a "urlTemplate" for each
                               code as values. If provided, this will override any general url_template specified.
        ==================     ====================================================================

        :returns: :class:`dict` Version 1: dict representing the integration, Version 2: :class:`Integration`
        """
        if self.project._is_v2_project:
            if isinstance(assignment_types, list):
                integration = [
                    add_integration(
                        project=self.project,
                        integration_id=integration_id,
                        prompt=prompt,
                        url_template=url_template,
                        assignment_type=a,
                    )
                    for a in assignment_types
                ]
            else:
                integration = add_integration(
                    project=self.project,
                    integration_id=integration_id,
                    prompt=prompt,
                    url_template=url_template,
                    assignment_type=assignment_types,
                )
            return integration
        else:
            integration = dict()
            if integration_id:
                integration["id"] = integration_id
            if prompt:
                integration["prompt"] = prompt
            if assignment_types:
                integration["assignmentTypes"] = assignment_types
            if url_template:
                integration["urlTemplate"] = url_template
            new_integration = self._validate(integration)
            self.project._item_data["assignmentIntegrations"].append(new_integration)
            self.project._update_data()
            return new_integration

    def batch_add(self, integrations):
        """
        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        integrations           Required :class:`List` of :class:`dict`. The integrations to add
        ==================     ====================================================================

        :returns: :class:`List` The list of integrations that were added
        """
        if self.project._is_v2_project:
            return add_integrations(self.project, integrations)
        else:
            for integration in integrations:
                new_integration = self._validate(integration)
                self.project._item_data["assignmentIntegrations"].append(
                    new_integration
                )
            self.project._update_data()
            return integrations

    def batch_delete(self, integrations):
        """
        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        integrations            Required :class:`List` of :class:`dict`. The integrations to delete
        ==================     ====================================================================
        """
        if self.project._is_v2_project:
            delete_integrations(project=self.project, integrations=integrations)
        else:
            self.project._item_data["assignmentIntegrations"] = [
                e
                for e in self.project._item_data["assignmentIntegrations"]
                if e not in integrations
            ]
            self.project._update_data()

    def _validate(self, integration):
        """Validates an integration for a version 1 project before adding it. Returns integration (that may have
        been modified in this validation method"""
        if "id" not in integration:
            raise ValidationError("Assignment integration must contain an id", self)
        elif integration["id"] in [
            at["id"] for at in self.project._item_data["assignmentIntegrations"]
        ]:
            raise ValidationError(
                "Assignment integration contains duplicate id for version 1 project",
                self,
            )
        if "prompt" not in integration:
            raise ValidationError("Assignment integration must contain a prompt", self)
        if "assignmentTypes" in integration:
            copy_dict = integration["assignmentTypes"].copy()
            for key, value in copy_dict.items():
                if isinstance(key, str):
                    if "urlTemplate" not in value:
                        raise ValidationError(
                            "Assignment integration must contain a urlTemplate", self
                        )
                    if key not in [
                        at.name for at in self.project.assignment_types.search()
                    ]:
                        raise ValidationError(
                            "Invalid assignment type in integration", self
                        )
                    # swap the name for the code
                    integration["assignmentTypes"][
                        self.project.assignment_types.get(name=key).code
                    ] = integration["assignmentTypes"].pop(key)
                elif isinstance(key, int):
                    if key not in [
                        at.code for at in self.project.assignment_types.search()
                    ]:
                        raise ValidationError(
                            "Invalid assignment type in integration", self
                        )
                    elif "urlTemplate" not in value:
                        raise ValidationError(
                            "Assignment integration must contain a urlTemplate", self
                        )
                else:
                    raise ValidationError("Invalid assignment type", self)
        elif "urlTemplate" not in integration:
            raise ValidationError(
                "Assignment integration must contain a urlTemplate", self
            )
        return integration
