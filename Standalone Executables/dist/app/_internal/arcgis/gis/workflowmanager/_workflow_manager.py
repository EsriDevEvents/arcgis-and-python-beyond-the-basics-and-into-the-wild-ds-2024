import datetime
import json
import sys
from typing import Optional
import urllib.parse

from arcgis.geometry import Geometry
import arcgis.gis
from arcgis.gis import Item
from arcgis.geoprocessing._tool import _camelCase_to_underscore


def _underscore_to_camelcase(name):
    def camelcase():
        yield str.lower
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else "_" for x in name.split("_"))


def _check_license(gis):
    is_portal = gis.properties.get("isPortal", False)
    portal_version = float(gis.properties.get("currentVersion", "0"))
    if is_portal and portal_version < 10.3:  # < ArcGIS Enterprise 11.1
        user_url = f"{gis._portal.resturl}community/self"
        raw_user = gis._con.get(user_url, {"returnUserLicenseTypeExtensions": True})
        if "userLicenseTypeExtensions" in raw_user:
            licenses = raw_user["userLicenseTypeExtensions"]
            has_license = "workflow" in licenses
        else:
            has_license = False

        if has_license is False:
            raise ValueError(
                "No Workflow Manager license is available for the current user"
            )


def _initialize(instance, gis):
    instance._gis = gis
    if instance._gis.users.me is None:
        raise ValueError("An authenticated `GIS` is required.")

    instance._url = instance._wmx_server_url[0]
    if instance._url is None:
        raise ValueError("No WorkflowManager Registered with your Organization")


class WorkflowManagerAdmin:
    """
    Represents a series of CRUD functions for Workflow Manager Items

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    gis                 Optional GIS. The connection to the Enterprise.
    ===============     ====================================================================
    """

    def __init__(self, gis):
        _initialize(self, gis)
        _check_license(gis)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl

        # Set org_id
        info_result = self._gis.properties
        self.is_enterprise = info_result["isPortal"]

        if self.is_enterprise:
            self.org_id = "workflow"
            res = self._gis.servers
            for s in res["servers"]:
                server_functions = [
                    x.strip() for x in s.get("serverFunction", "").lower().split(",")
                ]
                if "workflowmanager" in server_functions:
                    self._url = s.get("url", None)
                    self._private_url = s.get("adminUrl", None)
                    if self._url is None:
                        raise RuntimeError("Cannot find a WorkflowManager Server")
                    self._url += f"/{self.org_id}"
                    self._private_url += f"/{self.org_id}"
                    return self._url, self._private_url
            raise RuntimeError(
                "Unable to locate Workflow Manager Server. Please contact your ArcGIS Enterprise "
                "Administrator to ensure Workflow Manager Server is properly configured."
            )
        # is Arcgis Online
        else:
            self.org_id = info_result["id"]

            helper_services = info_result["helperServices"]
            if helper_services is None:
                raise RuntimeError("Cannot find helper functions")

            self._url = helper_services["workflowManager"]["url"]
            if self._url is None:
                raise RuntimeError("Cannot get Workflow Manager url")

            self._url += f"/{self.org_id}"
            self._private_url = f"/{self.org_id}"
            return self._url, self._private_url

        return None

    def create_item(self, name: str) -> tuple:
        """
        Creates a `Workflow Manager` schema that stores all the configuration
        information and location data in the data store on Portal. This can
        be run by any user assigned to the administrator role in Portal.

        For users that do not belong to the administrator role, the
        following privileges are required to run Create Workflow Item:

        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        name                Required String. The name of the new schema.
        ==================  =========================================================

        :return:
            string (item_id)
        """

        url = "{base}/admin/createWorkflowItem?name={name}".format(
            base=self._url, name=name
        )
        params = {"name": name}
        return_obj = json.loads(
            self._gis._con.post(
                url, params=params, try_json=False, json_encode=False, post_json=True
            )
        )
        return_obj = return_obj["itemId"]
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def upgrade_item(self, item: Item):
        """
        Upgrades an outdated Workflow Manager schema. Requires the Workflow Manager
        Advanced Administrator privilege or the Portal Admin Update Content privilege.

        ==================  =========================================================
        **Parameter**        **Description**
        ------------------  ---------------------------------------------------------
        item                Required Item. The Workflow Manager Item to be upgraded
        ==================  =========================================================

        :return:
            success object

        """

        url = "{base}/admin/{id}/upgrade".format(base=self._url, id=item.id)
        return_obj = json.loads(
            self._gis._con.post(url, try_json=False, json_encode=False, post_json=True)
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def delete_item(self, item: Item):
        """
        Delete a Workflow Manager schema. Does not delete the Workflow Manager Admin group.
        Requires the administrator or publisher role. If the user has the publisher role,
        the user must also be the owner of the item to delete.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required Item. The Workflow Manager Item to be deleted
        ===============     ====================================================================

        :return:
            success object

        """

        url = "{base}/admin/{id}?".format(base=self._url, id=item.id)

        return_obj = json.loads(self._gis._con.delete(url, try_json=False))
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    @property
    def server_status(self):
        """
        Gets the current status of the Workflow Manager Server

        :return:
            Boolean

        """

        url = "{base}/checkStatus".format(base=self._url)

        return_obj = self._gis._con.get(url)
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    @property
    def health_check(self):
        """
        Checks the health of Workflow Manager Server and if the cluster is active (if applicable).

        :return:
            Boolean

        """

        url = "{base}/healthCheck".format(base=self._url)

        return_obj = self._gis._con.get(url)
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def export_item(
        self,
        item: Item,
        job_template_ids: Optional[str] = None,
        diagram_ids: Optional[str] = None,
        include_other_configs: bool = True,
        passphrase: Optional[str] = None,
    ):
        """
        Exports a new Workflow Manager configuration (.wmc) file based on the indicated item. This configuration file
        includes the version, job templates, diagrams, roles, role-group associations, lookup tables, charts and
        queries, templates, and user settings of the indicated item. This file can be used with the import endpoint
        to update other item configurations. Configurations from Workflow items with a server that is on a more
        recent version will not import due to incompatability.

        =====================  =========================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------
        item                   Required Item. The Workflow Manager Item to be exported
        ---------------------  ---------------------------------------------------------
        job_template_ids       Optional. The job template(s) to be exported. If job template is exported,
                               the associated diagram must be included to be exported.
        ---------------------  ---------------------------------------------------------
        diagram_ids            Optional. The diagram(s) to be exported. If not defined, all diagrams are exported.
                               If defined as empty, no diagram is exported
        ---------------------  ---------------------------------------------------------
        include_other_configs  Optional. If false other configurations are not exported including templates,
                               User defined settings, shared searches, shared queries, email settings etc.
        ---------------------  ---------------------------------------------------------
        passphrase             Optional. If exporting encrypted user defined settings, define a passphrase.
                               If no passphrase is specified, the keys for encrypted user defined settings will be
                               exported without their values.
        =====================  =========================================================

        :return:
            success object

        """
        params = {"includeOtherConfiguration": include_other_configs}
        if job_template_ids is not None:
            params["jobTemplateIds"] = job_template_ids
        if diagram_ids is not None:
            params["diagramIds"] = diagram_ids
        if passphrase is not None:
            params["passphrase"] = passphrase

        url = "{base}/admin/{id}/export".format(base=self._url, id=item.id)
        return_obj = self._gis._con.post(
            url, params=params, try_json=False, json_encode=False, post_json=True
        )

        if "error" in return_obj:
            return_obj = json.loads(return_obj)
            self._gis._con._handle_json_error(return_obj["error"], 0)
        return return_obj

    def import_item(self, item: Item, config_file, passphrase: Optional[str] = None):
        """
        Imports a new Workflow Manager configuration from the selected .wmc file. Configurations from Workflow
        items with a server that is on a more recent version will not import due to incompatability. This will
        completely replace the version, job templates, diagrams, roles, role-group associations, lookup tables,
        charts and queries, templates, and user settings of the indicated item, and it is recommended to back
        up configurations before importing. Any encrypted settings included will only have their key imported
        and will need the value updated. Importing will fail if any jobs exist in the destination item.
        Excess scheduled tasks will be dropped based on the portal limit.

        ==================  =========================================================
        **Argument**        **Description**
        ------------------  ---------------------------------------------------------
        item                Required Item. The Workflow Manager Item that to import the configuration to.
        ------------------  ---------------------------------------------------------
        config_file         Required. The file path to the workflow manager configuration file.
        ------------------  ---------------------------------------------------------
        passphrase          Optional. If importing encrypted user defined settings, specify the same passphrase
                            used when exporting the configuration file. If no passphrase is specified, the keys for
                            encrypted user defined settings will be imported without their values.
        ==================  =========================================================

        :return:
            success object

        """

        url = "{base}/admin/{id}/import".format(base=self._url, id=item.id)
        data = {}
        if passphrase is not None:
            data["passphrase"] = passphrase

        return_obj = self._gis._con.post(
            url,
            files={"file": config_file},
            params=data,
            try_json=False,
            json_encode=False,
            post_json=False,
        )
        return_obj = json.loads(return_obj)

        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj


class JobManager:
    """
    Represents a helper class for workflow manager jobs. Accessible as the
    :attr:`~arcgis.gis.workflowmanager.WorkflowManager.jobs` property of the
    :class:`~arcgis.gis.workflowmanager.WorkflowManager`.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                The Workflow Manager Item
    ===============     ====================================================================

    """

    def __init__(self, item):
        """initializer"""
        if item is None:
            raise ValueError("Item cannot be None")
        self._item = item
        _initialize(self, item._gis)

    def _handle_error(self, info):
        """Basic error handler - separated into a function to allow for expansion in future releases"""
        error_class = info[0]
        error_text = info[1]
        raise Exception(error_text)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl

        # Set org_id
        info_result = self._gis.properties
        self.is_enterprise = info_result["isPortal"]

        if self.is_enterprise:
            self.org_id = "workflow"
            res = self._gis.servers
            for s in res["servers"]:
                server_functions = [
                    x.strip() for x in s.get("serverFunction", "").lower().split(",")
                ]
                if "workflowmanager" in server_functions:
                    self._url = s.get("url", None)
                    self._private_url = s.get("adminUrl", None)
                    if self._url is None:
                        raise RuntimeError("Cannot find a WorkflowManager Server")
                    self._url += f"/{self.org_id}/{self._item.id}"
                    self._private_url += f"/{self.org_id}/{self._item.id}"
                    return self._url, self._private_url
            raise RuntimeError(
                "Unable to locate Workflow Manager Server. Please contact your ArcGIS Enterprise "
                "Administrator to ensure Workflow Manager Server is properly configured."
            )
        # is Arcgis Online
        else:
            self.org_id = info_result["id"]

            helper_services = info_result["helperServices"]
            if helper_services is None:
                raise RuntimeError("Cannot find helper services")

            wm_service = helper_services["workflowManager"]
            self._url = wm_service["url"]
            if self._url is None:
                raise RuntimeError("Cannot get Workflow Manager url")

            self._url += f"/{self.org_id}/{self._item.id}"
            self._private_url = f"/{self.org_id}/{self._item.id}"
            return self._url, self._private_url

        return None

    def close(self, job_ids: list):
        """
        Closes a single or multiple jobs with specific Job IDs

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             Required list of job ID strings
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/jobs/manage".format(base=self._url)
            return Job.manage_jobs(self, self._gis, url, job_ids, "Close")
        except:
            self._handle_error(sys.exc_info())

    def reopen(self, job_ids):
        """
        Reopens a single or multiple jobs with specific Job IDs

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             Required list of job ID strings
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/jobs/manage".format(base=self._url)
            return Job.manage_jobs(self, self._gis, url, job_ids, "Reopen")
        except:
            self._handle_error(sys.exc_info())

    def create(
        self,
        template: str,
        count: int = 1,
        name: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        group: Optional[str] = None,
        assigned: Optional[str] = None,
        complete: Optional[str] = None,
        notes: Optional[str] = None,
        parent: Optional[str] = None,
        location: Optional[Geometry] = None,
        extended_properties: Optional[dict] = None,
        related_properties: Optional[dict] = None,
        job_id: Optional[str] = None,
    ):
        """
        Adds a job to the Workflow Manager instance given a user-defined template

        ===================         ====================================================================
        **Parameter**                **Description**
        -------------------         --------------------------------------------------------------------
        template                    Required object. Workflow Manager Job Template ID
        -------------------         --------------------------------------------------------------------
        count                       Optional Integer Number of jobs to create
        -------------------         --------------------------------------------------------------------
        name                        Optional string. Job Name
        -------------------         --------------------------------------------------------------------
        start                       Optional string. Job Start Date
        -------------------         --------------------------------------------------------------------
        end                         Optional string. Job End Date
        -------------------         --------------------------------------------------------------------
        priority                    Optional string. Job Priority Level
        -------------------         --------------------------------------------------------------------
        description                 Optional string. Job Description
        -------------------         --------------------------------------------------------------------
        owner                       Optional string. Job Owner
        -------------------         --------------------------------------------------------------------
        group                       Optional string Job Group
        -------------------         --------------------------------------------------------------------
        assigned                    Optional string. Initial Job Assignee
        -------------------         --------------------------------------------------------------------
        complete                    Optional Integer Percentage Complete
        -------------------         --------------------------------------------------------------------
        notes                       Optional string. Job Notes
        -------------------         --------------------------------------------------------------------
        parent                      Optional string Parent Job
        -------------------         --------------------------------------------------------------------
        location                    Optional Geometry or Workflow Manager :class:`~arcgis.gis.workflowmanager.JobLocation`
                                    Define an area of location for your job.
        -------------------         --------------------------------------------------------------------
        extended_properties         Optional Dict. Define additional properties on a job template
                                    specific to your business needs.
        -------------------         --------------------------------------------------------------------
        related_properties          Optional Dict. Define additional 1-M properties on a job template
                                    specific to your business needs.
        -------------------         --------------------------------------------------------------------
        job_id                      Optional string. Define the unique jobId of the job to be created.
                                    Once defined, only one job can be created.
        ===================         ====================================================================

        :return:
            List of newly created job ids

        """
        location_obj = location
        if location is not None and type(location) is not dict:
            location_obj = {"geometryType": location.type}
            if location.type == "Polygon":
                location_obj["geometry"] = json.dumps(
                    {
                        "rings": location.rings,
                        "spatialReference": location.spatial_reference,
                    }
                )
            elif location.type == "Polyline":
                location_obj["geometry"] = json.dumps(
                    {
                        "paths": location.paths,
                        "spatialReference": location.spatial_reference,
                    }
                )
            elif location.type == "Multipoint":
                location_obj["geometry"] = json.dumps(
                    {
                        "points": location.points,
                        "spatialReference": location.spatial_reference,
                    }
                )
        job_object = {
            "numberOfJobs": count,
            "jobName": name,
            "startDate": start,
            "dueDate": end,
            "priority": priority,
            "description": description,
            "ownedBy": owner,
            "assignedType": group,
            "assignedTo": assigned,
            "percentComplete": complete,
            "notes": notes,
            "parentJob": parent,
            "location": location_obj,
            "extendedProperties": extended_properties,
            "relatedProperties": related_properties,
            "jobId": job_id,
        }
        filtered_object = {}
        for key in job_object:
            if job_object[key] is not None:
                filtered_object[key] = job_object[key]
        url = "{base}/jobTemplates/{template}/job".format(
            base=self._url, template=template
        )
        return_obj = json.loads(
            self._gis._con.post(
                url,
                filtered_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj["jobIds"]

    def delete_attachment(self, job_id: str, attachment_id: str):
        """
        Deletes a job attachment given a job ID and attachment ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Required string. Job ID
        ---------------     --------------------------------------------------------------------
        attachment_id       Required string. Attachment ID
        ===============     ====================================================================

        :return:
            status code

        """
        try:
            res = Job.delete_attachment(
                self,
                self._gis,
                "{base}/jobs/{jobId}/attachments/{attachmentId}".format(
                    base=self._url,
                    jobId=job_id,
                    attachmentId=attachment_id,
                    item=self._item.id,
                ),
            )
            return res
        except:
            self._handle_error(sys.exc_info())

    def diagram(self, id: str):
        """
        Returns the job diagram for the user-defined job

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Job ID
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`Job Diagram <arcgis.gis.workflowmanager.JobDiagram>` object

        """
        try:
            return JobDiagram.get(
                self,
                self._gis,
                "{base}/jobs/{job}/diagram".format(base=self._url, job=id),
                {},
            )
        except:
            self._handle_error(sys.exc_info())

    def get(self, id: str, get_ext_props: bool = True, get_holds: bool = True):
        """
        Returns an active job with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Job ID
        ---------------     --------------------------------------------------------------------
        get_ext_props       Optional Boolean. If set to false the object will not include the jobs extended properties.
        ---------------     --------------------------------------------------------------------
        get_holds           Optional Boolean. If set to false the object will not include the jobs holds.
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`Job <arcgis.gis.workflowmanager.Job>` Object

        """
        try:
            url = f"{self._url}/jobs/{id}"
            job_dict = self._gis._con.get(
                url, {"extProps": get_ext_props, "holds": get_holds}
            )
            return Job(job_dict, self._gis, self._url)
        except:
            self._handle_error(sys.exc_info())

    def search(
        self,
        query: Optional[str] = None,
        search_string: Optional[str] = None,
        fields: Optional[str] = None,
        display_names: Optional[str] = [],
        sort_by: Optional[str] = [],
        num: int = 10,
        start_num: int = 0,
    ):
        """
        Runs a search against the jobs stored inside the Workflow Manager instance

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        query               Required string. SQL query to search against (e.g. "priority='High'")
        ---------------     --------------------------------------------------------------------
        search_str          Optional string. Search string to search against (e.g. "High")
        ---------------     --------------------------------------------------------------------
        fields              Optional string. Field list to return
        ---------------     --------------------------------------------------------------------
        display_names       Optional string. Display names for the return fields
        ---------------     --------------------------------------------------------------------
        sort_by             Optional string. Field to sort by (e.g. {'field': 'priority', 'sortOrder': 'Asc'})
        ---------------     --------------------------------------------------------------------
        num                 Optional Integer. Number of return results
        ---------------     --------------------------------------------------------------------
        start_num           Optional string. Index of first return value
        ===============     ====================================================================

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_ of search results

        """
        try:
            search_object = {
                "q": query,
                "search": search_string,
                "num": num,
                "displayNames": display_names,
                "start": start_num,
                "sortFields": sort_by,
                "fields": fields,
            }
            url = "{base}/jobs/search".format(base=self._url)
            return Job.search(self, self._gis, url, search_object)
        except:
            self._handle_error(sys.exc_info())

    def update(self, job_id: str, update_object):
        """
        Updates a job object by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Required string. ID for the job to update
        ---------------     --------------------------------------------------------------------
        update_object       Required object. An object containing the fields and new values to add to the job
        ===============     ====================================================================

        :return:
            success object


        .. code-block:: python

            # USAGE EXAMPLE: Updating a Job's properties

            # create a WorkflowManager object from the workflow item
            workflow_manager = WorkflowManager(wf_item)

            updates = { 'priority': 'High' }
            updates['extended_properties']: [
                {
                    "identifier": "table_name.prop1",
                    "value": "updated_123"
                },
                {
                    "identifier": "table_name.prop2",
                    "value": "updated_456"
                },
            ]

            workflow_manager.jobs.update(job_id, updates)

        """
        try:
            current_job = self.get(job_id).__dict__
            for k in update_object.keys():
                current_job[k] = update_object[k]
            url = "{base}/jobs/{jobId}/update".format(base=self._url, jobId=job_id)
            new_job = Job(current_job, self._gis, url)
            # remove existing properties if not updating.
            if "extended_properties" not in update_object:
                new_job.extended_properties = None
            if "related_properties" not in update_object:
                new_job.related_properties = None

            # temporary fix for error in privileges
            delattr(new_job, "percent_complete")
            delattr(new_job, "parent_job")
            return new_job.post()
        except:
            self._handle_error(sys.exc_info())

    def upgrade(self, job_ids: list):
        """
        Upgrades a single or multiple jobs with specific JobIDs

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             Required list. A list of job ID strings
        ===============     ====================================================================

        :return:
          success object

        """
        try:
            url = "{base}/jobs/manage".format(base=self._url)
            return Job.manage_jobs(self, self._gis, url, job_ids, "Upgrade")
        except:
            self._handle_error(sys.exc_info())

    def set_job_location(self, job_id, geometry):
        """
        Set a location of work for an existing job. jobUpdateLocation privilege is required to set a location on a job.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_id              Required string. ID for the job to update
        ---------------     --------------------------------------------------------------------
        geometry            Required ArcGIS.Geometry.Geometry or Workflow Manager :class:`~arcgis.gis.workflowmanager.JobLocation`
                            that describes a Job's Location. Must be a Polygon, Polyline, or Multipoint geometry type
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/jobs/{jobId}/location".format(
                base=self._url, jobId=job_id, item=self._item
            )
            if type(geometry) is dict:
                location = geometry
            else:
                location = {"geometryType": geometry.type}
                if geometry.type == "Polygon":
                    location["geometry"] = json.dumps(
                        {
                            "rings": geometry.rings,
                            "spatialReference": geometry.spatial_reference,
                        }
                    )
                elif geometry.type == "Polyline":
                    location["geometry"] = json.dumps(
                        {
                            "paths": geometry.paths,
                            "spatialReference": geometry.spatial_reference,
                        }
                    )
                elif geometry.type == "Multipoint":
                    location["geometry"] = json.dumps(
                        {
                            "points": geometry.points,
                            "spatialReference": geometry.spatial_reference,
                        }
                    )

            return_obj = json.loads(
                self._gis._con.put(
                    url,
                    {"location": location},
                    post_json=True,
                    try_json=False,
                    json_encode=False,
                )
            )
            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]
            return_obj = {
                _camelCase_to_underscore(k): v
                for k, v in return_obj.items()
                if v is not None and not k.startswith("_")
            }
            return return_obj
        except:
            self._handle_error(sys.exc_info())

    def delete(self, job_ids: list):
        """
        Deletes a single or multiple jobs with specific JobIDs

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        job_ids             Required list. A list of job ID strings
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/jobs/manage".format(base=self._url)
            return Job.manage_jobs(self, self._gis, url, job_ids, "Delete")
        except:
            self._handle_error(sys.exc_info())


class WorkflowManager:
    """
    Represents a connection to a Workflow Manager instance or item.

    Users create, update, delete workflow diagrams, job templates and jobs
    or the various other properties with a workflow item.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Required string. The Workflow Manager Item
    ===============     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE: Creating a WorkflowManager object from a workflow item

        from arcgis.gis.workflowmanager import WorkflowManager
        from arcgis.gis import GIS

        # connect to your GIS and get the web map item
        gis = GIS(url, username, password)
        wf_item = gis.content.get('1234abcd_workflow item id')

        # create a WorkflowManager object from the workflow item
        wm = WorkflowManager(wf_item)
        type(wm)
        >> arcgis.gis.workflowmanager.WorkflowManager

        # explore the users in this workflow using the 'users' property
        wm.users
        >> [{}...{}]  # returns a list of dictionaries representing each user
    """

    def __init__(self, item):
        if item is None:
            raise ValueError("Item cannot be None")
        self._item = item
        _initialize(self, item._gis)
        _check_license(item._gis)

        self.job_manager = JobManager(item)
        self.saved_searches_manager = SavedSearchesManager(item)

    def _handle_error(self, info):
        """Basic error handler - separated into a function to allow for expansion in future releases"""
        error_class = info[0]
        error_text = info[1]
        raise Exception(error_text)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl

        # Set org_id
        info_result = self._gis.properties
        self.is_enterprise = info_result["isPortal"]

        if self.is_enterprise:
            self.org_id = "workflow"
            res = self._gis.servers
            for s in res["servers"]:
                server_functions = [
                    x.strip() for x in s.get("serverFunction", "").lower().split(",")
                ]
                if "workflowmanager" in server_functions:
                    self._url = s.get("url", None)
                    self._private_url = s.get("adminUrl", None)
                    if self._url is None:
                        raise RuntimeError("Cannot find a WorkflowManager Server")
                    self._url += f"/{self.org_id}/{self._item.id}"
                    self._private_url += f"/{self.org_id}/{self._item.id}"
                    return self._url, self._private_url
            raise RuntimeError(
                "Unable to locate Workflow Manager Server. Please contact your ArcGIS Enterprise "
                "Administrator to ensure Workflow Manager Server is properly configured."
            )
        # is Arcgis Online
        else:
            self.org_id = info_result["id"]

            helper_services = info_result["helperServices"]
            if helper_services is None:
                raise RuntimeError("Cannot find helper functions")

            self._url = helper_services["workflowManager"]["url"]
            if self._url is None:
                raise RuntimeError("Cannot get Workflow Manager url")

            self._url += f"/{self.org_id}/{self._item.id}"
            self._private_url = f"/{self.org_id}/{self._item.id}"
            return self._url, self._private_url

        return None

    @property
    def jobs(self):
        """
        The job manager for a workflow item.

        :return:
            :class:`~arcgis.gis.workflowmanager.JobManager` object

        """

        return self.job_manager

    def evaluate_arcade(
        self,
        expression: str,
        context: Optional[str] = None,
        context_type: str = "BaseContext",
        mode: str = "Standard",
    ):
        """
        Evaluates an arcade expression

        ======================  ===============================================================
        **Parameter**            **Description**
        ----------------------  ---------------------------------------------------------------
        expression              Required String.
        ----------------------  ---------------------------------------------------------------
        context                 Optional String.
        ----------------------  ---------------------------------------------------------------
        context_type            Optional String.
        ----------------------  ---------------------------------------------------------------
        mode                    Optional String.
        ======================  ===============================================================

        :return: String
        """
        url = f"{self._url}/evaluateArcade"
        params = {
            "expression": expression,
            "contextType": context_type,
            "context": context,
            "parseMode": mode,
        }
        res = self._gis._con.post(url, params=params, json_encode=False, post_json=True)
        return res.get("result", None)

    @property
    def wm_roles(self):
        """
        Returns a list of user :class:`roles <arcgis.gis.workflowmanager.WMRole>` available
        in the local Workflow Manager instance.

        :return: List
        """
        try:
            role_array = self._gis._con.get(
                "{base}/community/roles".format(base=self._url)
            )["roles"]
            return_array = [WMRole(r) for r in role_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def users(self):
        """
        Returns an list of all user profiles stored in Workflow Manager

        :return: List of :attr:`~arcgis.gis.workflowmanager.WorkflowManager.user` profiles
        """
        try:
            user_array = self._gis._con.get(
                "{base}/community/users".format(base=self._url)
            )["users"]
            return_array = [self.user(u["username"]) for u in user_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def assignable_users(self):
        """
        Get all assignable users for a user in the workflow system

        :return:
            A `list <https://docs.python.org/3/library/stdtypes.html#list>`_ of the assignable :attr:`~assarcgis.gis.workflowmanager.WorkflowManager.user` objects

        """
        try:
            user_array = self._gis._con.get(
                "{base}/community/users".format(base=self._url)
            )["users"]
            return_array = [
                self.user(u["username"]) for u in user_array if u["isAssignable"]
            ]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def assignable_groups(self):
        """
        Get portal groups associated with Workflow Manager roles, to which the current user
        can assign work based on their Workflow Manager assignment privileges.

        :return:
            A `list <https://docs.python.org/3/library/stdtypes.html#list>`_ of
            the assignable :class:`~arcgis.gis.workflowmanager.Group` objects

        """
        try:
            group_array = self._gis._con.get(
                "{base}/community/groups".format(base=self._url)
            )["groups"]
            return_array = [
                self.group(g["id"]) for g in group_array if g["isAssignable"]
            ]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def settings(self):
        """
        Returns a list of all settings for the Workflow Manager instance

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_

        """
        try:
            return self._gis._con.get("{base}/settings".format(base=self._url))[
                "settings"
            ]
        except:
            self._handle_error(sys.exc_info())

    @property
    def groups(self):
        """
        Returns an list of all user :class:`groups <arcgis.gis.workflowmanager.Group>`
        stored in Workflow Manager

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_

        """
        try:
            group_array = self._gis._con.get(
                "{base}/community/groups".format(base=self._url)
            )["groups"]
            return_array = [self.group(g["id"]) for g in group_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    def searches(self, search_type: Optional[str] = None):
        """
        Returns a list of all saved searches.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        search_type         Optional string. The search type for returned saved searches.
                            The accepted values are `Standard`, `Chart`, and `All`. If not
                            defined, the Standard searches are returned.
        ===============     ====================================================================

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_

        """
        params = {}
        if search_type is not None:
            params["searchType"] = search_type

        try:
            return self._gis._con.get(
                "{base}/searches".format(base=self._url), params=params
            )["searches"]
        except:
            self._handle_error(sys.exc_info())

    @property
    def job_templates(self):
        """
        Gets all the job templates in a workflow item.

        :return:
            List of all current :class:`job templates <arcgis.gis.workflowmanager.JobTemplate>`
            in the Workflow Manager (required information for create_job call).

        """
        try:
            template_array = self._gis._con.get(
                "{base}/jobTemplates".format(base=self._url)
            )["jobTemplates"]
            return_array = [
                JobTemplate(t, self._gis, self._url) for t in template_array
            ]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    @property
    def diagrams(self):
        """
        Gets the workflow diagrams within the workflow item.

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_ of all current
            :class:`diagrams <arcgis.gis.workflowmanager.JobDiagram>` in the Workflow Manager

        """
        try:
            diagram_array = self._gis._con.get(
                "{base}/diagrams".format(base=self._url)
            )["diagrams"]
            return_array = [JobDiagram(d, self._gis, self._url) for d in diagram_array]
            return return_array
        except:
            self._handle_error(sys.exc_info())

    def update_settings(self, props: list):
        """
        Returns an active job with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        props               Required list. A list of Props objects to update
                            (Prop object example: {'propName': 'string', 'value': 'string'})
        ===============     ====================================================================

        :return:
            success object

        """
        url = "{base}/settings".format(base=self._url)
        params = {"settings": props}
        return_obj = json.loads(
            self._gis._con.post(
                url,
                params,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def wm_role(self, name: str):
        """
        Returns an active role with the given name

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. Role Name
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`Role <arcgis.gis.workflowmanager.WMRole>` Object

        """
        try:
            return WMRole.get(
                self,
                self._gis,
                "{base}/community/roles/{role}".format(
                    base=self._url, role=urllib.parse.quote(name), item=self._item.id
                ),
                params={},
            )
        except:
            self._handle_error(sys.exc_info())

    def job_template(self, id: str):
        """
        Returns a job template with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Job Template ID
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`JobTemplate <arcgis.gis.workflowmanager.JobTemplate>` Object

        """
        try:
            return JobTemplate.get(
                self,
                self._gis,
                "{base}/jobTemplates/{jobTemplate}".format(
                    base=self._url, jobTemplate=id
                ),
                params={},
            )
        except:
            self._handle_error(sys.exc_info())

    def delete_job_template(self, id: str):
        """
        Deletes a job template with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Job Template ID
        ===============     ====================================================================

        :return:
            status code

        """
        try:
            res = JobTemplate.delete(
                self,
                self._gis,
                "{base}/jobTemplates/{jobTemplate}".format(
                    base=self._url, jobTemplate=id, item=self._item.id
                ),
            )
            return res
        except:
            self._handle_error(sys.exc_info())

    def user(self, username: str):
        """
        Returns a user profile with the given username

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        username            Required string. Workflow Manager Username
        ===============     ====================================================================

        :return:
            Workflow Manager user profile

        """
        try:
            return arcgis.gis.User(self._gis, username)
        except:
            self._handle_error(sys.exc_info())

    def group(self, group_id: str):
        """
        Returns group information with the given group ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        group_id            Required string. Workflow Manager Group ID
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`~arcgis.gis.workflowmanager.Group` Object

        """
        try:
            wmx_group = Group.get(
                self,
                self._gis,
                "{base}/community/groups/{groupid}".format(
                    base=self._url, groupid=group_id, item=self._item.id
                ),
                params={},
            )
            arcgis_group = arcgis.gis.Group(self._gis, group_id)
            arcgis_group.roles = wmx_group.roles
            return arcgis_group
        except:
            self._handle_error(sys.exc_info())

    def update_group(self, group_id: str, update_object):
        """
        Update the information to the portal group. The adminAdvanced privilege is required.
        New roles can be added to the portal group. Existing roles can be deleted from the portal group.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        group_id            Required string. :class:`Workflow Manager Group <arcgis.gis.workflowmanager.Group>` ID
        ---------------     --------------------------------------------------------------------
        update_object       Required object. Object containing the updated actions of the information to be taken to the portal group.
        ===============     ====================================================================

        :return:
            Boolean

        """
        url = "{base}/community/groups/{groupid}".format(
            base=self._url, groupid=group_id
        )

        return_obj = json.loads(
            self._gis._con.post(
                url,
                update_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )

        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]

        return return_obj

    def diagram(self, id: str):
        """
        Returns the :class:`diagram <arcgis.gis.workflowmanager.JobDiagram>` with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Diagram ID
        ===============     ====================================================================

        :return:
             Workflow Manager :class:`~arcgis.gis.workflowmanager.JobDiagram` Object

        """
        try:
            return JobDiagram.get(
                self,
                self._gis,
                "{base}/diagrams/{diagram}".format(base=self._url, diagram=id),
                params={},
            )
        except:
            self._handle_error(sys.exc_info())

    def diagram_version(self, diagram_id: str, version_id: str):
        """
        Returns the :class:`diagram <arcgis.gis.workflowmanager.JobDiagram>` with the given version ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        diagram_id          Required string. Diagram ID
        ---------------     --------------------------------------------------------------------
        version_id          Required string. Diagram Version ID
        ===============     ====================================================================

        :return:
             Specified version of the Workflow Manager :class:`~arcgis.gis.workflowmanager.JobDiagram` object

        """
        try:
            return JobDiagram.get(
                self,
                self._gis,
                "{base}/diagrams/{diagram}/{diagramVersion}".format(
                    base=self._url, diagram=diagram_id, diagramVersion=version_id
                ),
                params={},
            )
        except:
            self._handle_error(sys.exc_info())

    def create_wm_role(self, name, description="", privileges=[]):
        """
        Adds a role to the Workflow Manager instance given a user-defined name

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. Role Name (required)
        ---------------     --------------------------------------------------------------------
        description         Required string. Role Description
        ---------------     --------------------------------------------------------------------
        privileges          Required list. List of privileges associated with the role
        ===============     ====================================================================

        :return:
            Workflow Manager :class:`~arcgis.gis.workflowmanager.WMRole` Object

        """
        try:
            url = "{base}/community/roles/{name}".format(base=self._url, name=name)
            post_role = WMRole(
                {"roleName": name, "description": description, "privileges": privileges}
            )
            return post_role.post(self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    def create_job_template(
        self,
        name: str,
        priority: str,
        id: str = None,
        category: str = "",
        job_duration: int = 0,
        assigned_to: str = "",
        default_due_date: Optional[str] = None,
        default_start_date: Optional[str] = None,
        start_date_type: str = "CreationDate",
        diagram_id: str = "",
        diagram_name: str = "",
        assigned_type: str = "Unassigned",
        description: str = "",
        default_description: str = "",
        state: str = "Draft",
        last_updated_by: str = "",
        last_updated_date: Optional[str] = None,
        extended_property_table_definitions: list = [],
    ):
        """
        Adds a job template to the Workflow Manager instance given a user-defined name and default priority level

        ====================================     ====================================================================
        **Parameter**                             **Description**
        ------------------------------------     --------------------------------------------------------------------
        name                                     Required string. Job Template Name
        ------------------------------------     --------------------------------------------------------------------
        priority                                 Required string. Default Job Template Priority Level
        ------------------------------------     --------------------------------------------------------------------
        id                                       Optional string. Job Template ID
        ------------------------------------     --------------------------------------------------------------------
        category                                 Optional string. Job Template Category
        ------------------------------------     --------------------------------------------------------------------
        job_duration                             Optional int. Default Job Template Duration
        ------------------------------------     --------------------------------------------------------------------
        assigned_to                              Optional string. Job Owner
        ------------------------------------     --------------------------------------------------------------------
        default_due_date                         Optional string. Due Date for Job Template
        ------------------------------------     --------------------------------------------------------------------
        default_start_date                       Optional string. Start Date for Job Template
        ------------------------------------     --------------------------------------------------------------------
        start_date_type                          Optional string. Type of Start Date (e.g. creationDate)
        ------------------------------------     --------------------------------------------------------------------
        diagram_id                               Optional string. Job Template Diagram ID
        ------------------------------------     --------------------------------------------------------------------
        diagram_name                             Optional string. Job Template Diagram Name
        ------------------------------------     --------------------------------------------------------------------
        assigned_type                            Optional string. Type of Job Template Assignment
        ------------------------------------     --------------------------------------------------------------------
        description                              Optional string. Job Template Description
        ------------------------------------     --------------------------------------------------------------------
        default_description                      Optional string. Default Job Template Description
        ------------------------------------     --------------------------------------------------------------------
        state                                    Optional string. Default Job Template State
        ------------------------------------     --------------------------------------------------------------------
        last_updated_by                          Optional string. User Who Last Updated Job Template
        ------------------------------------     --------------------------------------------------------------------
        last_updated_date                        Optional string. Date of Last Job Template Update
        ------------------------------------     --------------------------------------------------------------------
        extended_property_table_definitions      Optional list. List of Extended Properties for Job Template
        ====================================     ====================================================================

        :return:
            Workflow Manager :class:`~arcgis.gis.workflowmanager.JobTemplate` ID

        """
        try:
            if default_due_date is None:
                default_due_date = datetime.datetime.now().strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            if default_start_date is None:
                default_start_date = datetime.datetime.now().strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            if last_updated_date is None:
                last_updated_date = datetime.datetime.now().strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            url = "{base}/jobTemplates".format(base=self._url)

            post_job_template = JobTemplate(
                {
                    "jobTemplateId": id,
                    "jobTemplateName": name,
                    "category": category,
                    "defaultJobDuration": job_duration,
                    "defaultAssignedTo": assigned_to,
                    "defaultDueDate": default_due_date,
                    "defaultStartDate": default_start_date,
                    "jobStartDateType": start_date_type,
                    "diagramId": diagram_id,
                    "diagramName": diagram_name,
                    "defaultPriorityName": priority,
                    "defaultAssignedType": assigned_type,
                    "description": description,
                    "defaultDescription": default_description,
                    "state": state,
                    "extendedPropertyTableDefinitions": extended_property_table_definitions,
                    "lastUpdatedBy": last_updated_by,
                    "lastUpdatedDate": last_updated_date,
                }
            )

            return post_job_template.post(self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    def update_job_template(self, template):
        """
        Updates a job template object by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        template            Required object. :class:`Job Template <arcgis.gis.workflowmanger.JobTemplate>`
                            body. Existing Job Template object that inherits required/optional fields.
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/jobTemplates/{jobTemplate}".format(
                base=self._url,
                jobTemplate=template["job_template_id"],
                item=self._item.id,
            )
            template_object = JobTemplate(template)
            res = template_object.put(self._gis, url)
            return res
        except:
            self._handle_error(sys.exc_info())

    def create_diagram(
        self,
        name: str,
        steps: list,
        display_grid: bool,
        description: str = "",
        active: bool = False,
        annotations: list = [],
        data_sources: list = [],
        diagram_id: Optional[str] = None,
    ):
        """
        Adds a diagram to the Workflow Manager instance given a user-defined name and array of steps

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. Diagram Name
        ---------------     --------------------------------------------------------------------
        steps               Required list. List of Step objects associated with the Diagram
        ---------------     --------------------------------------------------------------------
        display_grid        Required boolean. Boolean indicating whether the grid will be displayed in the Diagram
        ---------------     --------------------------------------------------------------------
        description         Optional string. Diagram description
        ---------------     --------------------------------------------------------------------
        active              Optional Boolean. Indicates whether the Diagram is active
        ---------------     --------------------------------------------------------------------
        annotations         Optinal list. List of Annotation objects associated with the Diagram
        ---------------     --------------------------------------------------------------------
        data_sources        Optional list. List of Data Source objects associated with the Diagram
        ---------------     --------------------------------------------------------------------
        diagram_id          Optional string. The unique ID of the diagram to be created.
        ===============     ====================================================================

        :return:
            :class:`Workflow Manager Diagram <arcgis.gis.workflowmanager.JobDiagram>` ID

        """
        try:
            url = "{base}/diagrams".format(base=self._url)

            post_diagram = JobDiagram(
                {
                    "diagramId": diagram_id,
                    "diagramName": name,
                    "description": description,
                    "active": active,
                    "initialStepId": "",
                    "initialStepName": "",
                    "steps": steps,
                    "dataSources": data_sources,
                    "annotations": annotations,
                    "displayGrid": display_grid,
                }
            )
            return post_diagram.post(self._gis, url)["diagram_id"]
        except:
            self._handle_error(sys.exc_info())

    def update_diagram(self, body, delete_draft: bool = True):
        """
        Updates a diagram object by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        body                Required object. Diagram body - existing Diagram object that inherits required/optional
                            fields.
        ---------------     --------------------------------------------------------------------
        delete_draft        Optional Boolean - option to delete the Diagram draft (optional)
        ===============     ====================================================================

        :return:
            success object

        """
        try:
            url = "{base}/diagrams/{diagramid}".format(
                base=self._url, diagramid=body["diagram_id"]
            )
            post_diagram = JobDiagram(
                {
                    "diagramId": body["diagram_id"],
                    "diagramName": body["diagram_name"],
                    "description": (
                        body["description"] if "description" in body else ""
                    ),
                    "active": (body["active"] if "active" in body else False),
                    "initialStepId": (
                        body["initial_step_id"] if "initial_step_id" in body else ""
                    ),
                    "initialStepName": (
                        body["initial_step_name"] if "initial_step_name" in body else ""
                    ),
                    "steps": body["steps"],
                    "dataSources": (
                        body["data_sources"] if "data_sources" in body else []
                    ),
                    "annotations": (
                        body["annotations"] if "annotations" in body else ""
                    ),
                    "displayGrid": body["display_grid"],
                }
            )
            res = post_diagram.update(self._gis, url, delete_draft)

            return res
        except:
            self._handle_error(sys.exc_info())

    def delete_diagram(self, id: str):
        """
        Deletes a diagram object by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Diagram id
        ===============     ====================================================================

        :return:
            :class:`Workflow Manager Diagram <arcgis.gis.workflowmanager.JobDiagram>` ID

        """
        try:
            url = "{base}/diagrams/{diagramid}".format(base=self._url, diagramid=id)
            return JobDiagram.delete(self, self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    def delete_diagram_version(self, diagram_id, version_id):
        """
        Deletes a diagram version by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        diagram_id          Required string. Diagram ID
        ---------------     --------------------------------------------------------------------
        version_id          Required string. Diagram Version ID
        ===============     ====================================================================

        :return:
            Boolean

        """
        try:
            url = "{base}/diagrams/{diagramid}/{diagramVersion}".format(
                base=self._url, diagramid=diagram_id, diagramVersion=version_id
            )
            return JobDiagram.delete(self, self._gis, url)
        except:
            self._handle_error(sys.exc_info())

    @property
    def saved_searches(self):
        """
        The Saved Searches manager for a workflow item.

        :return:
            :class:`~arcgis.gis.workflowmanager.SavedSearchesManager`

        """

        return self.saved_searches_manager

    @property
    def table_definitions(self):
        """
        Get the definitions of each extended properties table in a workflow item. The response will consist of a list
        of table definitions. If the extended properties table is a feature service, its definition will include a
        dictionary of feature service properties. Each table definition will also include definitions of the properties
        it contains and list the associated job templates. This requires the adminBasic or adminAdvanced privileges.

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_

        """

        url = "{base}/tableDefinitions".format(base=self._url)

        return_obj = self._gis._con.get(url)
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]

        return return_obj["tableDefinitions"]

    def lookups(self, lookup_type):
        """
        Returns LookUp Tables by given type

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        lookup_type         Required string. The type of lookup table stored in the workflow item.
        ===============     ====================================================================

        :return:
           Workflow Manager :class:`LookUpTable <arcgis.gis.workflowmanager.LookUpTable>` Object

        """
        try:
            return LookUpTable.get(
                self,
                self._gis,
                "{base}/lookups/{lookupType}".format(
                    base=self._url, lookupType=lookup_type
                ),
                params={},
            )
        except:
            self._handle_error(sys.exc_info())

    def delete_lookup(self, lookup_type):
        """
        Deletes a job template with the given ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        lookup_type         Required string. The type of lookup table stored in the workflow item.
        ===============     ====================================================================

        :return:
            status code

        """
        try:
            res = LookUpTable.delete(
                self,
                self._gis,
                "{base}/lookups/{lookupType}".format(
                    base=self._url, lookupType=lookup_type, item=self._item.id
                ),
            )
            return res
        except:
            self._handle_error(sys.exc_info())

    def create_lookup(self, lookup_type, lookups):
        """
        Adds a diagram to the Workflow Manager instance given a user-defined name and array of steps

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        lookup_type         Required string. The type of lookup table stored in the workflow item.
        ---------------     --------------------------------------------------------------------
        lookups             Required list. List of lookups to be created / updated
        ===============     ====================================================================

        :return:
            Boolean

        .. code-block:: python

            # USAGE EXAMPLE: Creating a Lookup Table

            # create a WorkflowManager object from the workflow item
            wm = WorkflowManager(wf_item)

            # create the lookups object
            lookups = [{"lookupName": "Low", "value": 0},
                       {"lookupName": "Medium", "value": 5},
                       {"lookupName": "High", "value": 10},
                       {"lookupName": "EXTRA", "value": 15},
                       {"lookupName": "TEST", "value": 110}]

            wm.create_lookup("priority", lookups)
            >> True  # returns true if created successfully
        """
        try:
            url = "{base}/lookups/{lookupType}".format(
                base=self._url, lookupType=lookup_type
            )

            post_lookup = LookUpTable({"lookups": lookups})

            return post_lookup.put(self._gis, url)
        except:
            self._handle_error(sys.exc_info())


class LookUpTable(object):
    """
    Represents a Workflow Manager Look Up object with accompanying GET, POST, and DELETE methods.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object containing the relevant properties for a LookUpTable to complete REST calls
    ===============     ====================================================================
    """

    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url

    def __getattr__(self, item):
        gis = object.__getattribute__(self, "_gis")
        url = object.__getattribute__(self, "_url")
        id = object.__getattribute__(self, "job_template_id")
        full_object = gis._con.get(url, {})
        try:
            setattr(self, _camelCase_to_underscore(item), full_object[item])
            return full_object[item]
        except KeyError:
            raise KeyError(f'The attribute "{item}" is invalid for LookUpTables')

    def get(self, gis, url, params):
        lookup_dict = gis._con.get(url, params)
        return LookUpTable(lookup_dict, gis, url)

    def put(self, gis, url):
        put_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        return_obj = json.loads(
            gis._con.put(
                url,
                put_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def delete(self, gis, url):
        return_obj = json.loads(gis._con.delete(url, try_json=False))
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj


class SavedSearchesManager:
    """
    Represents a helper class for workflow manager saved searches. Accessible as the
    :attr:`~arcgis.gis.workflowmanager.WorkflowManager.saved_searches` property.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                The Workflow Manager Item
    ===============     ====================================================================

    """

    def __init__(self, item):
        """initializer"""
        if item is None:
            raise ValueError("Item cannot be None")
        self._item = item
        _initialize(self, item._gis)

    def _handle_error(self, info):
        """Basic error handler - separated into a function to allow for expansion in future releases"""
        error_class = info[0]
        error_text = info[1]
        raise Exception(error_text)

    @property
    def _wmx_server_url(self):
        """locates the WMX server"""
        baseurl = self._gis._portal.resturl

        # Set org_id
        info_result = self._gis.properties
        self.is_enterprise = info_result["isPortal"]

        if self.is_enterprise:
            self.org_id = "workflow"
            res = self._gis.servers
            for s in res["servers"]:
                server_functions = [
                    x.strip() for x in s.get("serverFunction", "").lower().split(",")
                ]
                if "workflowmanager" in server_functions:
                    self._url = s.get("url", None)
                    self._private_url = s.get("adminUrl", None)
                    if self._url is None:
                        raise RuntimeError("Cannot find a WorkflowManager Server")
                    self._url += f"/{self.org_id}/{self._item.id}"
                    self._private_url += f"/{self.org_id}/{self._item.id}"
                    return self._url, self._private_url
            raise RuntimeError(
                "Unable to locate Workflow Manager Server. Please contact your ArcGIS Enterprise "
                "Administrator to ensure Workflow Manager Server is properly configured."
            )
        # is Arcgis Online
        else:
            self.org_id = info_result["id"]

            helper_services = info_result["helperServices"]
            if helper_services is None:
                raise RuntimeError("Cannot find helper functions")

            self._url = helper_services["workflowManager"]["url"]
            if self._url is None:
                raise RuntimeError("Cannot get Workflow Manager url")

            self._url += f"/{self.org_id}/{self._item.id}"
            self._private_url = f"/{self.org_id}/{self._item.id}"
            return self._url, self._private_url

        return None

    def create(
        self,
        name: str,
        search_type: str,
        folder: Optional[str] = None,
        definition: Optional[str] = None,
        color_ramp: Optional[str] = None,
        sort_index: Optional[str] = None,
        search_id: Optional[str] = None,
    ):
        """
        Create a saved search or chart by specifying the search parameters in the json body.
        All search properties except for optional properties must be passed in the body to save the search or chart.
        The adminAdvanced or adminBasic privilege is required.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. The display name for the saved search or chart.
        ---------------     --------------------------------------------------------------------
        search_type         Required string. The type for the saved search or chart. The accepted values are Standard, Chart and All.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. The folder the saved search or chart will be categorized under.
        ---------------     --------------------------------------------------------------------
        definition          Required string. if the searchType is Standard. The search definition to be saved.
        ---------------     --------------------------------------------------------------------
        color_ramp          Required string. if the searchType is Chart. The color ramp for the saved chart.
        ---------------     --------------------------------------------------------------------
        sort_index          Optional string. The sorting order for the saved search or chart.
        ---------------     --------------------------------------------------------------------
        search_id           Optional string. The unique ID of the search or chart to be created.
        ===============     ====================================================================

        :return:
            Saved Search ID

        """
        try:
            url = "{base}/searches".format(base=self._url, id=search_id)
            post_dict = {
                "name": name,
                "folder": folder,
                "definition": definition,
                "searchType": search_type,
                "colorRamp": color_ramp,
                "sortIndex": sort_index,
                "searchId": search_id,
            }
            post_dict = {k: v for k, v in post_dict.items() if v is not None}
            return_obj = json.loads(
                self._gis._con.post(
                    url,
                    post_dict,
                    post_json=True,
                    try_json=False,
                    json_encode=False,
                )
            )

            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]

            return return_obj["searchId"]
        except:
            self._handle_error(sys.exc_info())

    def delete(self, id: str):
        """
        Deletes a saved search by ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        id                  Required string. Saved Search id
        ===============     ====================================================================

        :return:
            Boolean
        """
        try:
            url = "{base}/searches/{searchid}".format(base=self._url, searchid=id)

            return_obj = json.loads(self._gis._con.delete(url, try_json=False))

            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]
        except:
            self._handle_error(sys.exc_info())

    def update(self, search):
        """
        Update a saved search or chart by specifying the update values in the json body.
        All the properties except for optional properties must be passed in the body
        to update the search or chart. The searchId cannot be updated once it is created.
        The adminAdvanced or adminBasic privilege is required.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        search              Required object. An object defining the properties of the search to be updated.
        ===============     ====================================================================

        :return: success object

        .. code-block:: python

            # USAGE EXAMPLE: Updating a search's properties

            # create a WorkflowManager object from the workflow item
            workflow_manager = WorkflowManager(wf_item)

            workflow_manager.create_saved_search(name="name",
                                                 definition={
                                                     "start": 0,
                                                     "fields": ["job_status"],
                                                     "displayNames": ["Status"  ],
                                                     "sortFields": [{"field": "job_status",
                                                                     "sortOrder": "Asc:}]
                                                             },
                                                 search_type='Chart',
                                                 color_ramp='Flower Field Inverse',
                                                 sort_index=2000)

            search_lst = workflow_manager.searches("All")
            search = [x for x in search_lst if x["searchId"] == searchid][0]

            search["colorRamp"] = "Default"
            search["name"] = "Updated search"

            actual = workflow_manager.update_saved_search(search)

        """
        try:
            url = "{base}/searches/{searchId}".format(
                base=self._url, searchId=search["searchId"]
            )
            return_obj = json.loads(
                self._gis._con.put(
                    url,
                    search,
                    post_json=True,
                    try_json=False,
                    json_encode=False,
                )
            )

            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]
            return return_obj
        except:
            self._handle_error(sys.exc_info())

    def share(self, search_id, group_ids):
        """
        Shares a saved search with the list of groups

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        search_id           Required string. Saved Search id
        ---------------     --------------------------------------------------------------------
        group_ids           Required list. List of Workflow Group Ids
        ===============     ====================================================================

        :return:
            Boolean
        """
        try:
            url = "{base}/searches/{searchId}/shareWith".format(
                base=self._url, searchId=search_id
            )
            post_dict = {"groupIds": group_ids}

            return_obj = json.loads(
                self._gis._con.post(
                    url,
                    post_dict,
                    post_json=True,
                    try_json=False,
                    json_encode=False,
                )
            )

            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]
        except:
            self._handle_error(sys.exc_info())

    def share_details(self, search_id):
        """
        Returns the list of groups that the saved search is shared with by searchId.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        search_id           Search ID
        ===============     ====================================================================

        :return:
            List of :class:`~arcgis.gis.workflowmanager.Group` ID

        """

        url = "{base}/searches/{searchId}/shareWith".format(
            base=self._url, searchId=search_id
        )

        return_obj = self._gis._con.get(url)

        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj["groupIds"]


class Job(object):
    """
    Helper class for managing Workflow Manager jobs in a workflow item. This class is
    not created by users directly. An instance of this class, can be created by calling
    the :meth:`get <arcgis.gis.workflowmanager.JobManager.get>` method of the
    :class:`~arcgis.gis.workflowmanager.JobManager` with the appropriate job ID. The
    :class:`~arcgis.gis.workflowmanager.JobManager` is accessible as the
    :attr:`~arcgis.gis.workflowmanager.WorkflowManager.jobs` property of the
    :class:`~arcgis.gis.workflowmanager.WorkflowManager`.

    """

    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        self.job_status = (
            self.notes
        ) = (
            self.diagram_id
        ) = (
            self.end_date
        ) = (
            self.due_date
        ) = (
            self.description
        ) = (
            self.started_date
        ) = (
            self.current_steps
        ) = (
            self.job_template_name
        ) = (
            self.job_template_id
        ) = (
            self.extended_properties
        ) = (
            self.holds
        ) = (
            self.diagram_name
        ) = (
            self.parent_job
        ) = (
            self.job_name
        ) = (
            self.diagram_version
        ) = (
            self.active_versions
        ) = (
            self.percent_complete
        ) = (
            self.priority
        ) = (
            self.job_id
        ) = (
            self.created_date
        ) = (
            self.created_by
        ) = (
            self.closed
        ) = (
            self.owned_by
        ) = self.start_date = self._location = self.related_properties = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url

    def post(self):
        post_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None and not k.startswith("_")
        }
        return_obj = json.loads(
            self._gis._con.post(
                self._url,
                post_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def search(self, gis, url, search_object):
        return_obj = json.loads(
            gis._con.post(
                url,
                search_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def get_attachment(self, attachment_id: str):
        """
        Returns an embedded job attachment given an attachment ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment_id       Attachment ID
        ===============     ====================================================================

        :return:
            Job Attachment

        """

        url = "{base}/jobs/{jobId}/attachments/{attachmentId}".format(
            base=self._url, jobId=self.job_id, attachmentId=attachment_id
        )
        return_obj = self._gis._con.get(url, {}, try_json=False)
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj

    def add_attachment(
        self, attachment: str, alias: Optional[str] = None, folder: Optional[str] = None
    ):
        """
        Adds an attachment to the job

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment          Filepath to attachment
        ---------------     --------------------------------------------------------------------
        alias               Optional string. Alias for the attachment
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Folder for the attachment
        ===============     ====================================================================

        :return:
            Job Attachment

        """
        url = "{base}/jobs/{jobId}/attachments".format(
            base=self._url, jobId=self.job_id
        )
        return_obj = json.loads(
            self._gis._con.post(
                url,
                params={"alias": alias, "folder": folder},
                files={"attachment": attachment},
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        return {"id": return_obj["url"].split("/")[-1], "alias": return_obj["alias"]}

    def add_linked_attachment(self, attachments: list):
        """
        Add linked attachments to a job to provide additional or support information related to the job.
        Linked attachments can be links to a file on a local or shared file system or a URL.
        jobUpdateAttachments privilege is required to add an attachment to a job.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        attachments         List of linked attachments to associate with the job.
                            Each attachment should define the url, alias and folder
        ===============     ====================================================================

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_ list of job attachments

        """
        url = "{base}/jobs/{jobId}/attachmentslinked".format(
            base=self._url, jobId=self.job_id
        )

        post_object = {"attachments": attachments}
        return_obj = json.loads(
            self._gis._con.post(
                url,
                params=post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        return return_obj["attachments"]

    def update_attachment(self, attachment_id: str, alias: str):
        """
        Updates an attachment alias given a Job ID and attachment ID

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        attachment_id       Attachment ID
        ---------------     --------------------------------------------------------------------
        alias               Alias
        ===============     ====================================================================

        :return:
            success

        """
        url = "{base}/jobs/{jobId}/attachments/{attachmentid}".format(
            base=self._url, jobId=self.job_id, attachmentid=attachment_id
        )
        post_object = {"alias": alias}
        return_obj = json.loads(
            self._gis._con.post(
                url, params=post_object, try_json=False, json_encode=False
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def delete_attachment(self, gis, url):
        return_obj = json.loads(gis._con.delete(url, try_json=False))
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def update_step(self, step_id: str, assigned_type: str, assigned_to: str):
        """
        Update the assignment of the current step in a job based on the current user's Workflow Manager assignment privileges

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        step_id             Required String. Active Step ID
        ---------------     --------------------------------------------------------------------
        assigned_type       Required String. Type of assignment designated
                            Values: "user" | "group" | "unassigned"
        ---------------     --------------------------------------------------------------------
        assigned_to         Required String. User id to which the active step is assigned
        ===============     ====================================================================

        :return:
            success object

        """

        if step_id is None:
            step_id = self.currentSteps[0]["step_id"]
        url = "{base}/jobs/{jobId}/{stepId}".format(
            base=self._url,
            jobId=self.job_id,
            stepId=step_id,
        )
        post_object = {"assignedType": assigned_type, "assignedTo": assigned_to}
        return_obj = json.loads(
            self._gis._con.post(
                url,
                params=post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def set_current_step(self, step_id: str):
        """
        Sets a single step to be the active step on the job. The ability to set a step as current is controlled by the **workflowSetStepCurrent** privilege.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        step_id             Active Step ID
        ===============     ====================================================================

        :return:
            success object

        """

        url = "{base}/jobs/{jobId}/action".format(base=self._url, jobId=self.job_id)
        post_object = {"type": "SetCurrentStep", "stepIds": [step_id]}
        return_obj = json.loads(
            self._gis._con.post(
                url,
                params=post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    @property
    def attachments(self):
        """
        Gets the attachments of a job given job ID

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_ of attachments

        """

        url = "{base}/jobs/{jobId}/attachments".format(
            base=self._url, jobId=self.job_id
        )
        return_obj = self._gis._con.get(url)
        return return_obj["attachments"]

    @property
    def history(self):
        """
        Gets the history of a job given job ID

        :return:
            success object

        """

        url = "{base}/jobs/{jobId}/history".format(base=self._url, jobId=self.job_id)
        return_obj = self._gis._con.get(url)
        if "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    @property
    def location(self):
        """
        Get/Set the job location for the user-defined job

        :return:
            Workflow Manager :class:`~arcgis.gis.workflowmanager.JobLocation` object
        """

        if self._location is None:
            self._location = JobLocation.get(
                self,
                self._gis,
                "{base}/jobs/{job}/location".format(base=self._url, job=self.job_id),
                {},
            )
        return self._location

    @location.setter
    def location(self, value):
        self._location = value

    def manage_jobs(self, gis, url, ids, action):
        post_object = {"jobIds": ids, "type": action}
        return_obj = json.loads(
            gis._con.post(
                url,
                params=post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def add_comment(self, comment: str):
        """
        Adds a comment to the job

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        comment             Required string. Comment to add to job
        ===============     ====================================================================

        :return:
            Workflow Manager Comment Id

        """
        url = "{base}/jobs/{jobId}/comments".format(base=self._url, jobId=self.job_id)
        post_obj = {"comment": comment}

        return_obj = json.loads(
            self._gis._con.post(
                url,
                post_obj,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        return return_obj["commentId"]

    @property
    def comments(self):
        """
        Gets the comments of a job given job ID

        :return:
            `List <https://docs.python.org/3/library/stdtypes.html#list>`_ of comments

        """

        url = "{base}/jobs/{jobId}/comments".format(base=self._url, jobId=self.job_id)
        return_obj = self._gis._con.get(url)
        return return_obj["jobComments"]

    def set_job_version(
        self, data_source_name, version_guid=None, version_name=None, administered=False
    ):
        """
        Sets the version of the job.

        ================    ===================================================================
        **Argument**        **Description**
        ----------------    -------------------------------------------------------------------
        data_source_name    Required. The name of the data source for the job version to be set.
        ----------------    -------------------------------------------------------------------
        version_guid        Optional. The guid of the version to be set. If the value is null or not defined,
                            the versionName must be defined. versionGuid is preferred to be defined for better
                            performance.
        ----------------    -------------------------------------------------------------------
        version_name        Optional. The name of the version to be set. If the value is null or not defined,
                            the versionGuid must be defined.
        ----------------    -------------------------------------------------------------------
        administered        Optional. If true, the version can be claimed. If not defined, the default value is false.
        ================    ===================================================================

        :return:
            success object

        """

        url = "{base}/jobs/{jobId}/update".format(base=self._url, jobId=self.job_id)

        params = {
            "dataSourceName": data_source_name,
            "workflowAdministered": administered,
        }
        if version_guid is not None:
            params["versionGuid"] = version_guid
        if version_name is not None:
            params["versionName"] = version_name

        post_object = {"versions": [params]}

        return_obj = json.loads(
            self._gis._con.post(
                url,
                params=post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj


class WMRole(object):
    """
    Represents a Workflow Manager Role object with accompanying GET, POST, and DELETE methods

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """

    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data):
        self.privileges = self.roleName = self.description = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(self, gis, url, params):
        role_dict = gis._con.get(url, params)
        return WMRole(role_dict)

    def post(self, gis, url):
        post_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        return_obj = json.loads(
            gis._con.post(
                url,
                post_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj


class JobTemplate(object):
    """
    Represents a Workflow Manager Job Template object with accompanying GET, POST, and DELETE methods

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """

    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url

    def __getattr__(self, item):
        possible_fields = [
            "default_assigned_to",
            "last_updated_by",
            "diagram_id",
            "extended_property_table_definitions",
            "description",
            "job_template_name",
            "job_template_id",
            "default_start_date",
            "default_priority_name",
            "last_updated_date",
            "job_start_date_type",
            "diagram_name",
            "default_job_duration",
            "default_due_date",
            "state",
            "category",
            "default_assigned_type",
            "default_description",
        ]
        gis = object.__getattribute__(self, "_gis")
        url = object.__getattribute__(self, "_url")
        id = object.__getattribute__(self, "job_template_id")
        full_object = gis._con.get(url, {})
        try:
            setattr(self, _camelCase_to_underscore(item), full_object[item])
            return full_object[item]
        except KeyError:
            if item in possible_fields:
                setattr(self, _camelCase_to_underscore(item), None)
                return None
            else:
                raise KeyError(f'The attribute "{item}" is invalid for Job Templates')

    def get(self, gis, url, params):
        job_template_dict = gis._con.get(url, params)
        return JobTemplate(job_template_dict, gis, url)

    def put(self, gis, url):
        put_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        return_obj = json.loads(
            gis._con.put(
                url,
                put_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def post(self, gis, url):
        post_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        return_obj = json.loads(
            gis._con.post(
                url,
                post_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj["jobTemplateId"]

    def delete(self, gis, url):
        return_obj = json.loads(gis._con.delete(url, try_json=False))
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def share(self, group_ids):
        """
        Shares a job template with the list of groups

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        group_ids           Required list. List of Workflow Group Ids
        ===============     ====================================================================

        :return:
            boolean
        """
        try:
            url = "{base}/shareWith".format(
                base=self._url,
                templateId=self.job_template_id,
            )
            post_dict = {"groupIds": group_ids}

            return_obj = json.loads(
                self._gis._con.post(
                    url,
                    post_dict,
                    post_json=True,
                    try_json=False,
                    json_encode=False,
                )
            )

            if "error" in return_obj:
                self._gis._con._handle_json_error(return_obj["error"], 0)
            elif "success" in return_obj:
                return return_obj["success"]
        except:
            self._handle_error(sys.exc_info())

    @property
    def share_details(self):
        """
        Returns the list of groups that the job_template is shared with by template_id.

        :return:
            list of :class:`~arcgis.gis.workflowmanager.Group` ID

        """

        url = "{base}/shareWith".format(base=self._url, templateId=self.job_template_id)
        return_obj = self._gis._con.get(url)

        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj["groupIds"]

    @property
    def automated_creations(self):
        """
        Retrieve the list of created automations for a job template, including scheduled job creation and webhook.

        :return:
            list of automatedCreations associated with the JobTemplate

        """
        try:
            return_obj = self._gis._con.get(
                "{base}/automatedCreation".format(
                    base=self._url, jobTemplateId=self.job_template_id
                ),
                params={},
            )
            return return_obj["automations"]
        except:
            self._handle_error(sys.exc_info())

    def automated_creation(self, automation_id):
        """
        Returns the specified automated creation

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        automation_id       Required string. Automation Creation Id
        ===============     ====================================================================

        :return:
            automated creation object.

        """
        try:
            return_obj = self._gis._con.get(
                "{base}/automatedCreation/{automationId}".format(
                    base=self._url,
                    jobTemplateId=self.job_template_id,
                    automationId=automation_id,
                ),
                params={},
            )
            return return_obj
        except:
            self._handle_error(sys.exc_info())

    def update_automated_creation(self, adds=None, updates=None, deletes=None):
        """
        Creates an automated creation

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        adds                Optional List. The list of automated creations to create.
        ---------------     --------------------------------------------------------------------
        updates             Optional List. The list of automated creations to update
        ---------------     --------------------------------------------------------------------
        deletes             Optional List. The list of automated creation ids to delete
        ===============     ====================================================================

        :return:
            success object

        .. code-block:: python

            # USAGE EXAMPLE: Creating an automated creation for a job template

            # create a WorkflowManager object from the workflow item
            wm = WorkflowManager(wf_item)

            # create the props object with the required automation properties
            adds = [{
                        "automationName": "auto_mation",
                        "automationType": "Scheduled",
                        "enabled": True,
                        "details": "{\"timeType\":\"NumberOfDays\",\"dayOfMonth\":1,\"hour\":8,\"minutes\":0}"
                    }]
            updates = [
                    {
                      "automationId": "abc123",
                      "automationName": "automation_updated"
                    }
                  ]
            deletes =  ["def456"]

            wm.update_automated_creation(adds, updates, deletes)
            >> True  # returns true if created successfully

        """
        if adds is None:
            adds = []
        if deletes is None:
            deletes = []
        if updates is None:
            updates = []

        props = {"adds": adds, "updates": updates, "deletes": deletes}
        url = "{base}/automatedCreation".format(
            base=self._url, jobTemplateId=self.job_template_id
        )

        return_obj = json.loads(
            self._gis._con.post(
                url,
                props,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            self._gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return return_obj


class Group(object):
    """
    Represents a Workflow Manager Group object with accompanying GET, POST, and DELETE methods

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           data object representing relevant parameters for GET or POST calls
    ===============     ====================================================================
    """

    _camelCase_to_underscore = _camelCase_to_underscore

    def __init__(self, init_data):
        self.roles = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(self, gis, url, params):
        group_dict = gis._con.get(url, params)
        return Group(group_dict)


class JobDiagram(object):
    """
    Helper class for managing Workflow Manager :class:`job diagrams <arcgis.gis.workflowmanager.JobDiagram>`
    in a workflow :class:`item <arcgis.gis.Item>`. This class is not created directly. An instance
    can be created by calling the :attr:`~arcgis.gis.workflowmanager.WorkflowManager.diagrams` property
    of the :class:`~arcgis.gis.workflowmanager.WorkflowManager` to retrieve a list of diagrams. Then
    the :meth:`~arcgis.gis.workflowmanager.WorkflowManager.diagram` method can be used with the appropriate
    ID of the digram to retrieve the :class:`job diagram <arcgis.gis.workflowmanager.JobDiagram>`.

    """

    _camelCase_to_underscore = _camelCase_to_underscore
    _underscore_to_camelcase = _underscore_to_camelcase

    def __init__(self, init_data, gis=None, url=None):
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])
        self._gis = gis
        self._url = url

    def __getattr__(self, item):
        possible_fields = [
            "display_grid",
            "diagram_version",
            "diagram_name",
            "diagram_id",
            "description",
            "annotations",
            "initial_step_id",
            "data_sources",
            "steps",
            "initial_step_name",
        ]
        gis = object.__getattribute__(self, "_gis")
        url = object.__getattribute__(self, "_url")
        id = object.__getattribute__(self, "diagram_id")
        full_object = gis._con.get(url, {})
        try:
            setattr(self, _camelCase_to_underscore(item), full_object[item])
            return full_object[item]
        except KeyError:
            if item in possible_fields:
                setattr(self, _camelCase_to_underscore(item), None)
                return None
            else:
                raise KeyError(f'The attribute "{item}" is invalid for Diagrams')

    def get(self, gis, url, params):
        job_diagram_dict = gis._con.get(url, params)
        return JobDiagram(job_diagram_dict, gis, url)

    def post(self, gis, url):
        post_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        return_obj = json.loads(
            gis._con.post(
                url,
                post_dict,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def update(self, gis, url, delete_draft):
        clean_dict = {
            _underscore_to_camelcase(k): v
            for k, v in self.__dict__.items()
            if v is not None
        }
        post_object = {"deleteDraft": delete_draft, "diagram": clean_dict}
        return_obj = json.loads(
            gis._con.post(
                url,
                post_object,
                post_json=True,
                try_json=False,
                json_encode=False,
            )
        )
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj

    def delete(self, gis, url):
        return_obj = json.loads(gis._con.delete(url, try_json=False))
        if "error" in return_obj:
            gis._con._handle_json_error(return_obj["error"], 0)
        elif "success" in return_obj:
            return return_obj["success"]
        return_obj = {
            _camelCase_to_underscore(k): v
            for k, v in return_obj.items()
            if v is not None and not k.startswith("_")
        }
        return return_obj


class JobLocation(object):
    """
    Represents a Workflow Manager Job Location object with accompanying GET, POST, and DELETE methods

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    init_data           Required object. Represents. relevant parameters for GET or POST calls
    ===============     ====================================================================
    """

    _camelCase_to_underscore = _camelCase_to_underscore

    def __init__(self, init_data):
        self.geometry = self.geometry_type = None
        for key in init_data:
            setattr(self, _camelCase_to_underscore(key), init_data[key])

    def get(self, gis, url, params):
        job_location_dict = gis._con.get(url, params)
        return JobLocation(job_location_dict)
