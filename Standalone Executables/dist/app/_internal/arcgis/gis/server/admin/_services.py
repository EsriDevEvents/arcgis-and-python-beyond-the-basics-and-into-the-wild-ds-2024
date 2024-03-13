"""
Classes and objects used to manage published services.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import annotations
from typing import Any
import os
import json
import tempfile
from .._common import BaseServer
from .parameters import Extension
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import GIS
from arcgis.gis._impl._con import Connection
import datetime as _datetime
from typing import Optional
from arcgis.features.managers import WebHookScheduleInfo, WebHookEvents


########################################################################
class ServiceManager(BaseServer):
    """
    Helper class for managing services. This class is not created by users directly. An instance of this class,
    called 'services', is available as a property of the Server object. Users call methods on this 'services' object to
    managing services.
    """

    _currentURL = None
    _url = None
    _con = None
    _json_dict = None
    _currentFolder = None
    _folderName = None
    _folders = None
    _foldersDetail = None
    _folderDetail = None
    _webEncrypted = None
    _description = None
    _isDefault = None
    _services = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(
        self,
        url: str,
        gis: GIS,
        initialize: bool = False,
        sm: "Server" = None,
    ):
        """Constructor

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required string. The administration url endpoint.
        ---------------     --------------------------------------------------------------------
        gis                 Required GIS or Server object. This handles the credential management.
        ===============     ====================================================================
        """
        if sm is not None:
            self._sm = sm
        super(ServiceManager, self).__init__(gis=gis, url=url, sm=sm)
        self._con = gis
        self._url = url
        self._currentURL = url
        self._currentFolder = "/"
        if initialize:
            self._init(gis)

    # ----------------------------------------------------------------------
    def _init(self, connection: Connection = None):
        """loads the properties into the class"""
        if connection is None:
            connection = self._con
        params = {"f": "json"}
        try:
            result = connection.get(path=self._currentURL, params=params)
            if isinstance(result, dict):
                self._json_dict = result
                self._properties = PropertyMap(result)
            else:
                self._json_dict = {}
                self._properties = PropertyMap({})
        except:
            self._json_dict = {}
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    @property
    def _folder(self) -> str:
        """Get/Set current folder"""
        return self._folderName

    # ----------------------------------------------------------------------
    @_folder.setter
    def _folder(self, folder: str):
        """gets/set the current folder"""

        if folder == "" or folder == "/" or folder is None:
            self._currentURL = self._url
            self._services = None
            self._description = None
            self._folderName = None
            self._webEncrypted = None
            self._init()
            self._folderName = folder
        elif folder.lower() in [f.lower() for f in self.folders]:
            self._currentURL = self._url + "/%s" % folder
            self._services = None
            self._description = None
            self._folderName = None
            self._webEncrypted = None
            self._init()
            self._folderName = folder

    # ----------------------------------------------------------------------
    @property
    def folders(self) -> list:
        """returns a list of all folders"""
        if self._folders is None:
            self._init()
            self._folders = self.properties["folders"]
        if "/" not in self._folders:
            self._folders.append("/")
        return self._folders

    # ----------------------------------------------------------------------
    def list(self, folder: Optional[str] = None, refresh: bool = True) -> list:
        """
        returns a list of services in the specified folder

        ===============     ===========================================================================================
        **Parameter**        **Description**
        ---------------     -------------------------------------------------------------------------------------------
        folder              Required string. The name of the folder to list services from.
        ---------------     -------------------------------------------------------------------------------------------
        refresh             Optional boolean. Default is False. If True, the list of services will be
                            requested to the server, else the list will be returned from cache.
        ===============     ===========================================================================================


        :return: list

        """
        if folder is None:
            folder = "/"
        if folder != self._currentFolder or self._services is None or refresh:
            self._currentFolder = folder
            self._folder = folder
            return self._services_list()

        return self._services_list()

    # ----------------------------------------------------------------------
    def _export_services(self, folder: str) -> str:
        """
        Export services allows for the backup and storage of non-hosted services.

        =================   ====================================================
        **Parameter**        **Description**
        -----------------   ----------------------------------------------------
        folder              required string.  This is the path to the save folder.
                            The ArcGIS Account must have access to the location
                            to write the backup file.
        =================   ====================================================

        :return: string to the save location.

        """
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        url = self._url + "/exportServices"
        params = {
            "f": "json",
            "location": folder,
            "csrfPreventToken": self._con.token,
        }
        res = self._con.post(path=url, postdata=params)
        if "location" in res:
            return res["location"]
        return None

    # ----------------------------------------------------------------------
    def _import_services(self, file_path: str) -> bool:
        """
        Import services allows for the backup and storage of non-hosted services.

        =================   ====================================================
        **Parameter**        **Description**
        -----------------   ----------------------------------------------------
        file_path           required string.  File path with extension
                            .agssiteservices.
        =================   ====================================================

        :return: boolean

        """
        folder = os.path.dirname(file_path)
        if os.path.isdir(folder) == False:
            os.makedirs(folder)
        url = self._url + "/importServices"
        params = {"f": "json", "csrfPreventToken": self._con.token}
        files = {"location": file_path}
        res = self._con.post(path=url, files=files, postdata=params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def _services_list(self) -> list:
        """returns the services in the current folder"""
        self._services = []
        params = {"f": "json"}
        json_dict = self._con.get(path=self._currentURL, params=params)
        if "services" in json_dict.keys():
            for s in json_dict["services"]:
                from urllib.parse import quote, quote_plus, urlparse, urljoin

                u_url = self._currentURL + "/%s.%s" % (
                    s["serviceName"],
                    s["type"],
                )
                parsed = urlparse(u_url)
                u_url = "{scheme}://{netloc}{path}".format(
                    scheme=parsed.scheme,
                    netloc=parsed.netloc,
                    path=quote(parsed.path),
                )
                self._services.append(Service(url=u_url, gis=self._con))
        return self._services

    # ----------------------------------------------------------------------
    @property
    def _extensions(self) -> dict:
        """
        This resource is a collection of all the custom server object
        extensions that have been uploaded and registered with the server.
        You can register new server object extensions using the register
        extension operation. When updating an existing extension, you need
        to use the update extension operation. If an extension is no longer
        required, you can use the unregister operation to remove the
        extension from the site.
        """
        url = self._url + "/types/extensions"
        params = {"f": "json"}
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def publish_sd(
        self,
        sd_file: str,
        folder: Optional[str] = None,
        service_config: Optional[dict] = None,
    ) -> bool:
        """
        publishes a service definition file to arcgis server

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        sd_file             Required string. File path to the .sd file
        ---------------     --------------------------------------------------------------------
        folder              Optional string. This parameter allows for the override of the
                            folder option set in the SD file.
        ===============     ====================================================================


        :return: Boolean

        """
        return self._sm.publish_sd(sd_file, folder, service_config=service_config)

    # ----------------------------------------------------------------------
    def _find_services(self, service_type: str = "*") -> list:
        """
            returns a list of a particular service type on AGS

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        service_type        Required string. Type of service to find.  The allowed types
                             are: ("GPSERVER", "GLOBESERVER", "MAPSERVER",
                             "GEOMETRYSERVER", "IMAGESERVER",
                             "SEARCHSERVER", "GEODATASERVER",
                             "GEOCODESERVER", "*").  The default is *
                             meaning find all service names.
        ===============     ====================================================================


        :return: list of service name as folder/name.type

        """
        allowed_service_types = (
            "GPSERVER",
            "GLOBESERVER",
            "MAPSERVER",
            "GEOMETRYSERVER",
            "IMAGESERVER",
            "SEARCHSERVER",
            "GEODATASERVER",
            "GEOCODESERVER",
            "*",
        )
        lower_types = [l.lower() for l in service_type.split(",")]
        for v in lower_types:
            if v.upper() not in allowed_service_types:
                return {"message": "%s is not an allowed service type." % v}
        params = {"f": "json"}
        type_services = []
        folders = self.folders
        folders.append("")
        baseURL = self._url
        for folder in folders:
            if folder == "":
                url = baseURL
            else:
                url = baseURL + "/%s" % folder
            res = self._con.get(path=url, params=params)
            if res.has_key("services"):
                for service in res["services"]:
                    if service["type"].lower() in lower_types:
                        service["URL"] = url + "/%s.%s" % (
                            service["serviceName"],
                            service_type,
                        )
                        type_services.append(service)
                    del service
            del res
            del folder
        return type_services

    # ----------------------------------------------------------------------
    def _examine_folder(self, folder: Optional[str] = None) -> dict:
        """
        A folder is a container for GIS services. ArcGIS Server supports a
        single level hierarchy of folders.
        By grouping services within a folder, you can conveniently set
        permissions on them as a single unit. A folder inherits the
        permissions of the root folder when it is created, but you can
        change those permissions at a later time.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Required string. name of folder to examine.
        ===============     ====================================================================


        :return: dict

        """
        params = {"f": "json"}
        if folder:
            url = self._url + "/" + folder
        else:
            url = self._url
        return self._con.get(path=url, params=params)

    # ----------------------------------------------------------------------
    def _can_create_service(
        self,
        service: dict,
        options: Optional[dict] = None,
        folder_name: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> bool:
        """
        Use canCreateService to determine whether a specific service can be
        created on the ArcGIS Server site.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service configuration in JSON format. For more
                            information about the service configuration options, see
                            createService. This is an optional parameter, though either the
                            service_type or service parameter must be used.
        ---------------     --------------------------------------------------------------------
        options             optional dict. This is an optional parameter that provides additional
                            information about the service, such as whether it is a hosted
                            service.
        ---------------     --------------------------------------------------------------------
        folder_name         Optional string. This is an optional parameter to indicate the folder
                            where can_create_service will check for the service.
        ---------------     --------------------------------------------------------------------
        service_type        Optional string. The type of service that can be created. This is an
                            optional parameter, though either the service type or service
                            parameter must be used.
        ===============     ====================================================================


        :return: boolean

        """
        url = self._url + "/canCreateService"
        params = {"f": "json", "service": service}
        if options:
            params["options"] = options
        if folder_name:
            params["folderName"] = folder_name
        if service_type:
            params["serviceType"] = service_type
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _add_folder_permission(
        self,
        principal: str,
        is_allowed: bool = True,
        folder: Optional[str] = None,
    ) -> dict:
        """
           Assigns a new permission to a role (principal). The permission
           on a parent resource is automatically inherited by all child
           resources

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of role to assign/disassign accesss.
        ---------------     --------------------------------------------------------------------
        is_allowed          Optional boolean. True means grant access, False means revoke.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Name of folder to assign permissions to.
        ===============     ====================================================================


        :return: dict

        """
        if folder is not None:
            u_url = self._url + "/%s/%s" % (folder, "/permissions/add")
        else:
            u_url = self._url + "/permissions/add"
        params = {
            "f": "json",
            "principal": principal,
            "isAllowed": is_allowed,
        }
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _folder_permissions(self, folder_name: str) -> dict:
        """
        Lists principals which have permissions for the folder.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name              Optional string. Name of folder to examine permissions.
        ===============     ====================================================================


        :return: dict

        """
        u_url = self._url + "/%s/permissions" % folder_name
        params = {
            "f": "json",
        }
        return self._con.post(path=u_url, postdata=params)

    # ----------------------------------------------------------------------
    def _clean_permissions(self, principal: str) -> bool:
        """
        Cleans all permissions that have been assigned to a role
        (principal). This is typically used when a role is deleted.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of role to dis-assign all accesss.
        ===============     ====================================================================


        :return: boolean
        """
        u_url = self._url + "/permissions/clean"
        params = {"f": "json", "principal": principal}
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def create_folder(self, folder_name: str, description: str = "") -> bool:
        """
        Creates a unique folder name on AGS

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. Name of the new folder.
        ---------------     --------------------------------------------------------------------
        description         Optional string. Description of what the folder is.
        ===============     ====================================================================

        :return: Boolean
        """
        params = {
            "f": "json",
            "folderName": folder_name,
            "description": description,
        }
        u_url = self._url + "/createFolder"
        res = self._con.post(path=u_url, postdata=params)
        self._init()
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def delete_folder(self, folder_name: str) -> bool:
        """
        Removes a folder on ArcGIS Server

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. Name of the folder.
        ===============     ====================================================================

        :return: Boolean
        """
        params = {"f": "json"}
        if folder_name in self.folders:
            u_url = self._url + "/%s/deleteFolder" % folder_name
            res = self._con.post(path=u_url, postdata=params)
            self._init()
            if "status" in res:
                return res["status"] == "success"
            return res
        else:
            return False

    # ----------------------------------------------------------------------
    def _delete_service(
        self, name: str, service_type: str, folder: Optional[str] = None
    ) -> bool:
        """
        Deletes a service from ArcGIS Server

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string. Name of the service
        ---------------     --------------------------------------------------------------------
        service_type        Required string. Name of the service type.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of the service on ArcGIS Server.
        ===============     ====================================================================

        :return: boolean

        """
        if folder is None:
            u_url = self._url + "/%s.%s/delete" % (name, service_type)
        else:
            u_url = self._url + "/%s/%s.%s/delete" % (
                folder,
                name,
                service_type,
            )
        params = {"f": "json"}
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _service_report(self, folder: Optional[str] = None) -> dict:
        """
        Provides a report on all items in a given folder.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of the service on ArcGIS Server.
        ===============     ====================================================================

        :return: boolean
        """
        items = [
            "description",
            "status",
            "instances",
            "iteminfo",
            "properties",
        ]
        if folder is None:
            u_url = self._url + "/report"
        else:
            u_url = self._url + "/%s/report" % folder
        params = {"f": "json", "parameters": items}
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    @property
    def _types(self) -> dict:
        """returns the allowed services types"""
        params = {"f": "json"}
        u_url = self._url + "/types"
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    def _federate(self) -> dict:
        """
        This operation is used when federating ArcGIS Server with Portal
        for ArcGIS. It imports any services that you have previously
        published to your ArcGIS Server site and makes them available as
        items with Portal for ArcGIS. Beginning at 10.3, services are
        imported automatically as part of the federate process.
        If the automatic import of services fails as part of the federation
        process, the following severe-level message will appear in the
        server logs:
           Failed to import GIS services as items within portal.
        If this occurs, you can manually re-run the operation to import
        your services as items in the portal. Before you do this, obtain a
        portal token and then validate ArcGIS Server is federated with
        your portal using the portal website. This is done in
        My Organization > Edit Settings > Servers.
        After you run the Federate operation, specify sharing properties to
        determine which users and groups will be able to access each service.
        """
        params = {"f": "json"}
        url = self._url + "/federate"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _unfederate(self) -> bool:
        """
        This operation is used when unfederating ArcGIS Server with Portal
        for ArcGIS. It removes any items from the portal that represent
        services running on your federated ArcGIS Server. You typically run
        this operation in preparation for a full unfederate action. For
        example, this can be performed using
               My Organization > Edit Settings > Servers
        in the portal website or the Unregister Server operation in the
        ArcGIS REST API.
        Beginning at 10.3, services are removed automatically as part of
        the unfederate process. If the automatic removal of service items
        fails as part of the unfederate process, you can manually re-run
        the operation to remove the items from the portal.
        """
        params = {"f": "json"}
        url = self._url + "/unfederate"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _unregister_extension(self, extension_filename: str) -> bool:
        """
        Unregisters all the extensions from a previously registered server
        object extension (.SOE) file.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        extension_filename         Required string. Name of the previously registered .SOE file.
        ======================     ====================================================================

        :return: boolean

        """
        params = {"f": "json", "extensionFilename": extension_filename}
        url = self._url + "/types/extensions/unregister"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _update_extension(self, item_id: str) -> bool:
        """
        Updates extensions that have been previously registered with the
        server. All extensions in the new .SOE file must match with
        extensions from a previously registered .SOE file.
        Use this operation to update your implementations or extension
        configuration properties.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. Id of the uploaded .SOE file
        ===============     ====================================================================

        :return: boolean


        """
        params = {"f": "json", "id": item_id}
        url = self._url + "/types/extensions/update"
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _rename_service(
        self,
        name: str,
        service_type: str,
        new_name: str,
        folder: Optional[str] = None,
    ) -> bool:
        """
        Renames a published AGS Service

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        name                Required string.  Old service name.
        ---------------     --------------------------------------------------------------------
        service_type        Required string. The type of service.
        ---------------     --------------------------------------------------------------------
        new_name            Required string. The new service name.
        ---------------     --------------------------------------------------------------------
        folder              Optional string. Location of where the service lives, none means
                            root folder.
        ===============     ====================================================================

        :return: boolean

        """
        params = {
            "f": "json",
            "serviceName": name,
            "serviceType": service_type,
            "serviceNewName": new_name,
        }
        if folder is None:
            u_url = self._url + "/renameService"
        else:
            u_url = self._url + "/%s/renameService" % folder
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        self._init()
        return res

    # ----------------------------------------------------------------------
    def create_service(self, service: dict) -> dict:
        """
        Creates a new GIS service in the folder. A service is created by
        submitting a JSON representation of the service to this operation.

        The JSON representation of a service contains the following four
        sections:
         - Service Description Properties-Common properties that are shared
           by all service types. Typically, they identify a specific service.
         - Service Framework Properties-Properties targeted towards the
           framework that hosts the GIS service. They define the life cycle
           and load balancing of the service.
         - Service Type Properties -Properties targeted towards the core
           service type as seen by the server administrator. Since these
           properties are associated with a server object, they vary across
           the service types. The Service Types section in the Help
           describes the supported properties for each service.
         - Extension Properties-Represent the extensions that are enabled
           on the service. The Extension Types section in the Help describes
           the supported out-of-the-box extensions for each service type.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service is the properties to create a service.
        ===============     ====================================================================

        :return: Dict

        Output:
         dictionary status message

        """
        url = self._url + "/createService"
        params = {"f": "json"}
        if isinstance(service, str):
            params["service"] = service
        elif isinstance(service, dict):
            params["service"] = json.dumps(service)
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _stop_services(self, services: list) -> bool:
        """
        Stops serveral services on a single server.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        services            Required list.  A list of dictionary objects. Each dictionary object
                            is defined as:
                              folder_name - The name of the folder containing the
                                service, for example, "Planning". If the service
                                resides in the root folder, leave the folder
                                property blank ("folder_name": "").
                              serviceName - The name of the service, for example,
                                "FireHydrants".
                              type - The service type, for example, "MapServer".
                                Example:
                                [{
                                  "folder_name" : "",
                                  "serviceName" : "SampleWorldCities",
                                  "type" : "MapServer"
                                }]
        ===============     ====================================================================

        :return: boolean


        """
        url = self._url + "/stopServices"
        if isinstance(services, dict):
            services = [services]
        elif isinstance(services, (list, tuple)):
            services = list(services)
        else:
            Exception("Invalid input for parameter services")
        params = {"f": "json", "services": {"services": services}}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _start_services(self, services: list) -> bool:
        """
        starts serveral services on a single server

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        services            Required list.  A list of dictionary objects. Each dictionary object
                            is defined as:
                              folder_name - The name of the folder containing the
                                service, for example, "Planning". If the service
                                resides in the root folder, leave the folder
                                property blank ("folder_name": "").
                              serviceName - The name of the service, for example,
                                "FireHydrants".
                              type - The service type, for example, "MapServer".
                                Example:
                                [{
                                  "folder_name" : "",
                                  "serviceName" : "SampleWorldCities",
                                  "type" : "MapServer"
                                }]
        ===============     ====================================================================

        :return: boolean

        """
        url = self._url + "/startServices"
        if isinstance(services, dict):
            services = [services]
        elif isinstance(services, (list, tuple)):
            services = list(services)
        else:
            Exception("Invalid input for parameter services")
        params = {"f": "json", "services": {"services": services}}
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def _edit_folder(self, description: str, web_encrypted: bool = False) -> bool:
        """
        This operation allows you to change the description of an existing
        folder or change the web encrypted property.
        The web encrypted property indicates if all the services contained
        in the folder are only accessible over a secure channel (SSL). When
        setting this property to true, you also need to enable the virtual
        directory security in the security configuration.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        description         Required string. A description of the folder.
        ---------------     --------------------------------------------------------------------
        web_encrypted       Optional boolean. The boolean to indicate if the services are
                            accessible over SSL only.
        ===============     ====================================================================

        :return: boolean

        """
        url = self._url + "/editFolder"
        params = {
            "f": "json",
            "webEncrypted": web_encrypted,
            "description": "%s" % description,
        }
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def exists(
        self,
        folder_name: str,
        name: Optional[str] = None,
        service_type: Optional[str] = None,
    ) -> bool:
        """
        This operation allows you to check whether a folder or a service
        exists. To test if a folder exists, supply only a folder_name. To
        test if a service exists in a root folder, supply both serviceName
        and service_type with folder_name=None. To test if a service exists
        in a folder, supply all three parameters.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder_name         Required string. The folder name to check for.
        ---------------     --------------------------------------------------------------------
        name                Optional string. The service name to check for.
        ---------------     --------------------------------------------------------------------
        service_type        Optional string. A service type. Allowed values:
                             GeometryServer | ImageServer | MapServer | GeocodeServer |
                             GeoDataServer | GPServer | GlobeServer | SearchServer
        ===============     ====================================================================

        :return: Boolean

        """
        if folder_name and name is None and service_type is None:
            for folder in self.folders:
                if folder.lower() == folder_name.lower():
                    return True
                del folder
            return False
        url = self._url + "/exists"
        params = {
            "f": "json",
            "folderName": folder_name,
            "serviceName": name,
            "type": service_type,
        }
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        elif "exists" in res:
            return res["exists"]
        return res


########################################################################
class ServiceWebHook(BaseServer):
    """
    The webhooks resource returns a list of service webhooks configured
    for a specific geoprocessing or feature service, including
    deactivated and activated webhooks. Webhooks are an ArcGIS
    Enterprise capability that provide other applications or webhook
    receivers with event-driven information, delivered as an HTTPS POST
    request, that can be used to create automated and integrative
    workflows. For more information on how webhooks are supported in
    ArcGIS Enterprise, see Webhooks in ArcGIS Enterprise.
    """

    _url: str = None
    _con: Connection = None
    _gis: GIS = None

    def __init__(
        self,
        url: str,
        gis: GIS | Connection,
        initialize: bool = False,
        **kwargs,
    ):
        """initializer"""
        super()
        if isinstance(gis, GIS):
            con = gis._con
        else:
            con = gis

        self._url = url
        self._con = con
        if initialize:
            self._init(self._con)

    # ----------------------------------------------------------------------
    def edit(
        self,
        *,
        name: str | None = None,
        change_types: WebHookEvents | str | None = None,
        hook_url: str | None = None,
        signature_key: str | None = None,
        active: bool | None = None,
        schedule_info: WebHookScheduleInfo | dict[str, Any] | None = None,
        payload_format: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Updates the existing WebHook's Properties.

        =====================================    ===========================================================================
        **Parameter**                             **Description**
        -------------------------------------    ---------------------------------------------------------------------------
        name                                     Optional String. Use valid name for a webhook. This name needs to be unique per service.
        -------------------------------------    ---------------------------------------------------------------------------
        hook_url                                 Optional String.  The URL to which the payloads will be delivered.
        -------------------------------------    ---------------------------------------------------------------------------
        change_types                             Optional :class:`~arcgis.features.managers.WebHookEvents` or String.
                                                 The default is "*", which means all events.  This is a
                                                 comma separated list of values that will fire off the web hook.  The list
                                                 each supported type is below.
        -------------------------------------    ---------------------------------------------------------------------------
        signature_key                            Optional String. If specified, the key will be used in generating the HMAC
                                                 hex digest of value using sha256 hash function and is return in the
                                                 x-esriHook-Signature header.
        -------------------------------------    ---------------------------------------------------------------------------
        active                                   Optional bool. Enable or disable call backs when the webhook is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        schedule_info                            Optional :class:`~arcgis.features.managers.WebHookScheduleInfo` or Dict.
                                                 Allows the trigger to be used as a given schedule.

                                                 Example Dictionary:


                                                     | {
                                                     |    "name" : "Every-5seconds",
                                                     |    "startAt" : 1478280677536,
                                                     |    "state" : "enabled",
                                                     |    "recurrenceInfo" : {
                                                     |     "frequency" : "second",
                                                     |     "interval" : 5
                                                     |   }
                                                     | }

        -------------------------------------    ---------------------------------------------------------------------------
        payload_format                           Optional String. The payload can be sent in pretty format or standard.
                                                 The default is `json`.
        -------------------------------------    ---------------------------------------------------------------------------
        content_type                             Optional String. The Content Type is used to indicate the media type of the
                                                 resource. The media type is a string sent along with the file indicating
                                                 the format of the file.
        =====================================    ===========================================================================


        A list of allowed web hook triggers is shown below.

        =====================================    ===========================================================================
        **Name**                                 **Triggered When**
        -------------------------------------    ---------------------------------------------------------------------------
        `*`                                      Wildcard event. Any time any event is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesCreated`                        A new feature is created
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesUpdated`                        Any time a feature is updated
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesDeleted`                        Any time a feature is deleted
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesEdited`                         Any time a feature is edited (insert or update or delete)
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsCreated`                     Any time adding a new attachment to a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsUpdated`                     Any time updating a feature attachment
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsDeleted`                     Any time an attachment is deleted from a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerSchemaChanged`                     Any time a schema is changed in a layer
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerDefinitionChanged`                 Any time a layer definition is changed
        -------------------------------------    ---------------------------------------------------------------------------
        `FeatureServiceDefinitionChanged`        Any time a feature service is changed
        =====================================    ===========================================================================


        :return: Response of edit as a dict.

        """
        props: dict[str, Any] = dict(self.properties)
        url: str = f"{self._url}/edit"
        if isinstance(schedule_info, WebHookScheduleInfo):
            schedule_info: dict[str, Any] = schedule_info.as_dict()
        if isinstance(change_types, list):
            ctypes: list[str] = []
            for ct in change_types:
                if isinstance(ct, WebHookEvents):
                    ctypes.append(ct.value)
                else:
                    ctypes.append(ct)

            change_types: str = ",".join(ctypes)
        elif isinstance(change_types, WebHookEvents):
            change_types: str = change_types.value
        elif change_types is None:
            change_types: str = ",".join(self.properties["changeTypes"])
        params: dict[str, Any] = {
            "f": "json",
            "name": name,
            "changeTypes": change_types,
            "signatureKey": signature_key,
            "hookUrl": hook_url,
            "active": active,
            "scheduleInfo": schedule_info,
            "payloadFormat": payload_format,
            "content_type": content_type,
        }
        for k in list(params.keys()):
            if params[k] is None:
                params.pop(k)
            del k
        props.update(params)
        resp: dict[str, Any] = self._con.post(url, props)
        self._properties: PropertyMap = PropertyMap(resp)
        return resp

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes the current webhook from the system

        :return: Boolean, True if successful
        """
        url = f"{self._url}/delete"
        params = {"f": "json"}
        resp = self._con.post(url, params)
        return resp["status"] == "success"


########################################################################
class ServiceWebHookManager(BaseServer):
    """
    The webhooks resource returns a list of service webhooks configured
    for a specific geoprocessing or feature service, including
    deactivated and activated webhooks. Webhooks are an ArcGIS
    Enterprise capability that provide other applications or webhook
    receivers with event-driven information, delivered as an HTTPS POST
    request, that can be used to create automated and integrative
    workflows. For more information on how webhooks are supported in
    ArcGIS Enterprise, see Webhooks in ArcGIS Enterprise.
    """

    _url: str = None
    _con: Connection = None
    _properties: dict = None

    def __init__(self, url: str, con: Connection, **kwargs):
        """initializer"""
        super()
        self._url = url
        self._con = con

    # ----------------------------------------------------------------------
    def __str__(self):
        """returns the class as a string"""
        return f"< ServiceWebHookManager @ {self._url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        hook_url: str,
        *,
        change_types: WebHookEvents | str = WebHookEvents.ALL,
        signature_key: str | None = None,
        active: bool = False,
        schedule_info: dict[str, Any] | WebHookScheduleInfo | None = None,
        payload_format: str = "json",
        content_type: str | None = None,
    ) -> ServiceWebHook:
        """

        Creates a new Feature Collection Web Hook


        =====================================    ===========================================================================
        **Parameter**                             **Description**
        -------------------------------------    ---------------------------------------------------------------------------
        name                                     Required String. Use valid name for a webhook. This name needs to be unique per service.
        -------------------------------------    ---------------------------------------------------------------------------
        hook_url                                 Required String.  The URL to which the payloads will be delivered.
        -------------------------------------    ---------------------------------------------------------------------------
        change_types                             Optional WebHookEvents or String.  The default is "WebHookEvents.ALL", which means all events.  This is a
                                                 comma separated list of values that will fire off the web hook.  The list
                                                 each supported type is below.
        -------------------------------------    ---------------------------------------------------------------------------
        signature_key                            Optional String. If specified, the key will be used in generating the HMAC
                                                 hex digest of value using sha256 hash function and is return in the
                                                 x-esriHook-Signature header.
        -------------------------------------    ---------------------------------------------------------------------------
        active                                   Optional bool. Enable or disable call backs when the webhook is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        schedule_info                            Optional Dict or `WebHookScheduleInfo`. Allows the trigger to be used as a given schedule.

                                                 Example Dictionary:

                                                     | {
                                                     |   "name" : "Every-5seconds",
                                                     |   "startAt" : 1478280677536,
                                                     |   "state" : "enabled"
                                                     |   "recurrenceInfo" : {
                                                     |     "frequency" : "second",
                                                     |     "interval" : 5
                                                     |   }
                                                     | }

        -------------------------------------    ---------------------------------------------------------------------------
        payload_format                           Optional String. The payload can be sent in pretty format or standard.
                                                 The default is `json`.
        -------------------------------------    ---------------------------------------------------------------------------
        content_type                             Optional String. The Content Type is used to indicate the media type of the
                                                 resource. The media type is a string sent along with the file indicating
                                                 the format of the file.
        =====================================    ===========================================================================


        A list of allowed web hook triggers is shown below.

        =====================================    ===========================================================================
        **Name**                                 **Triggered When**
        -------------------------------------    ---------------------------------------------------------------------------
        `*`                                      Wildcard event. Any time any event is triggered.
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesCreated`                        A new feature is created
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesUpdated`                        Any time a feature is updated
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesDeleted`                        Any time a feature is deleted
        -------------------------------------    ---------------------------------------------------------------------------
        `FeaturesEdited`                         Any time a feature is edited (insert or update or delete)
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsCreated`                     Any time adding a new attachment to a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsUpdated`                     Any time updating a feature attachment
        -------------------------------------    ---------------------------------------------------------------------------
        `AttachmentsDeleted`                     Any time an attachment is deleted from a feature
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerSchemaChanged`                     Any time a schema is changed in a layer
        -------------------------------------    ---------------------------------------------------------------------------
        `LayerDefinitionChanged`                 Any time a layer definition is changed
        -------------------------------------    ---------------------------------------------------------------------------
        `FeatureServiceDefinitionChanged`        Any time a feature service is changed
        =====================================    ===========================================================================

        :return:
            A :class:`~arcgis.gis.server.admin._services.ServiceWebHook` object

        """
        url: str = f"{self._url}/create"
        if isinstance(change_types, list):
            ctnew: list[str] = []
            for ct in change_types:
                if isinstance(ct, str):
                    ctnew.append(ct)
                elif isinstance(ct, WebHookEvents):
                    ctnew.append(ct.value)
            change_types: str = ",".join(ctnew)
        elif isinstance(change_types, WebHookEvents):
            change_types: str = change_types.value
        if isinstance(schedule_info, WebHookScheduleInfo):
            schedule_info: dict[str, Any] = schedule_info.as_dict()
        elif schedule_info is None:
            schedule_info: dict[str, Any] = {
                "name": "",
                "state": "enabled",
                "startAt": int(_datetime.datetime.now().timestamp() * 1000),
                "recurrenceInfo": {"interval": 20, "frequency": "second"},
            }
        params: dict[str, Any] = {
            "f": "json",
            "name": name,
            "changeTypes": change_types,
            "signatureKey": signature_key,
            "hookUrl": hook_url,
            "active": active,
            "scheduleInfo": schedule_info,
            "payloadFormat": payload_format,
        }
        if content_type:
            params["contentType"] = content_type
        resp: dict[str, Any] = self._con.post(url, params)
        self._properties = None
        if "status" in resp and resp["status"] == "error":
            raise ValueError(". ".join(resp["messages"]))
        elif not "url" in resp:
            hook_id: str = resp.get("id", None) or resp.get("globalId", None)
            hook_url: str = self._url + f"/{hook_id}"
            return ServiceWebHook(url=hook_url, gis=self._con)
        else:
            return ServiceWebHook(url=resp["url"], gis=self._con)

    # ----------------------------------------------------------------------
    @property
    def list(self) -> list[ServiceWebHook]:
        """
        Lists the existing webhooks

        :return: list[ServiceWebHook]
        """
        hooks: list[ServiceWebHook] = []
        url: str = f"{self.url}"
        params: dict[str, Any] = {
            "f": "json",
            "start": 1,
            "num": 25,
            "sortField": None,
            "sortOrder": None,
        }
        res: dict[str, Any] = self._con.get(url, params=params)
        hooks.extend(
            [
                ServiceWebHook(url, gis=self._con)
                for url in ["%s/%s" % (self._url, wh["id"]) for wh in res["webhooks"]]
            ]
        )

        while res.get("nextStart", -1) > -1:
            params["start"] = res["nextStart"]
            res: dict[str, Any] = self._con.get(url, params=params)
            hooks.extend(
                [
                    ServiceWebHook(url, gis=self._gis)
                    for url in [
                        "%s/%s" % (self._url, wh["id"]) for wh in res["webhooks"]
                    ]
                ]
            )
        return hooks

    # ----------------------------------------------------------------------
    def disable_hooks(self) -> bool:
        """
        The `disable_hooks` will temporarily deactivate all configured webhooks
        for a geoprocessing or feature service. While deactivated, the service's
        webhooks will not be invoked and payloads will not be delivered.

        :return: Bool, True if successful

        """
        url: str = f"{self._url}/deactivateAll"
        params: dict[str, Any] = {"f": "json"}
        return self._con.post(url, params).get("status", "failed") == "success"

    # ----------------------------------------------------------------------
    def delete_all_hooks(self) -> bool:
        """
        The `delete_all_hooks` operation will permanently remove the specified webhook.

        :return: Bool, True if successful

        """
        url = f"{self._url}/deleteAll"
        params = {"f": "json"}
        return self._con.post(url, params).get("status", "failed") == "success"

    # ----------------------------------------------------------------------
    def enable_hooks(self) -> bool:
        """
        The `enable_hooks` operation restarts a deactivated webhook. When
        activated, payloads will be delivered to the payload URL when the
        webhook is invoked.

        :return: Bool, True if successful

        """
        url: str = f"{self._url}/activateAll"
        params: dict[str, Any] = {"f": "json"}
        return self._con.post(url, params).get("status", "failed") == "success"


########################################################################
class Service(BaseServer):
    """
    Represents a GIS administrative service

    **(This should not be created by a user)**

    """

    _ii = None
    _con = None
    _frameworkProperties = None
    _recycleInterval = None
    _instancesPerContainer = None
    _maxWaitTime = None
    _minInstancesPerNode = None
    _maxIdleTime = None
    _maxUsageTime = None
    _allowedUploadFileTypes = None
    _datasets = None
    _properties = None
    _recycleStartTime = None
    _clusterName = None
    _description = None
    _isDefault = None
    _type = None
    _serviceName = None
    _isolationLevel = None
    _capabilities = None
    _loadBalancing = None
    _configuredState = None
    _maxStartupTime = None
    _private = None
    _maxUploadFileSize = None
    _keepAliveInterval = None
    _maxInstancesPerNode = None
    _json = None
    _json_dict = None
    _interceptor = None
    _provider = None
    _portalProperties = None
    _jsonProperties = None
    _url = None
    _extensions = None
    _jm = None
    _whm = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS, initialize: bool = False, **kwargs):
        """
        Constructor

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        url                 Required string. The administration URL.
        ---------------     --------------------------------------------------------------------
        gis                 Required GIS. GIS or Server object
        ---------------     --------------------------------------------------------------------
        initialize          Optional boolean. fills all the properties at object creation is
                            true.
        ===============     ====================================================================


        """
        super()
        from arcgis.gis import GIS

        if isinstance(gis, GIS):
            con = gis._con
        else:
            con = gis
        # super(Service, self)

        self._service_manager = kwargs.pop("service_manager", None)
        self._url = url
        self._currentURL = url
        self._con = con
        # if url.lower().find('gpserver') > -1:
        #    self.jobs = self._jobs
        if initialize:
            self._init(self._con)

    # ----------------------------------------------------------------------
    def _init(self, connection: Connection = None):
        """populates server admin information"""
        from .parameters import Extension

        params = {"f": "json"}
        if connection:
            json_dict = connection.get(path=self._url, params=params)
        else:
            json_dict = self._con.get(path=self._currentURL, params=params)
        self._json = json.dumps(json_dict)
        self._json_dict = json_dict
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not attr.startswith("_")
        ]
        self._properties = PropertyMap(json_dict)
        for k, v in json_dict.items():
            if k.lower() == "extensions":
                self._extensions = []
                for ext in v:
                    self._extensions.append(Extension.fromJSON(ext))
                    del ext

            del k
            del v

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s at %s>" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def _refresh(self):
        """refreshes the object's values by re-querying the service"""
        self._init()

    # ----------------------------------------------------------------------
    def _json_properties(self) -> dict:
        """returns the jsonProperties"""
        if self._jsonProperties is None:
            self._init()
        return self._jsonProperties

    # ----------------------------------------------------------------------
    def change_provider(self, provider: str) -> bool:
        """
        Allows for the switching of the service provide and how it is hosted on the ArcGIS Server instance.

        Values:

           + 'ArcObjects' means the service is running under the ArcMap runtime i.e. published from ArcMap
           + 'ArcObjects11': means the service is running under the ArcGIS Pro runtime i.e. published from ArcGIS Pro
           + 'DMaps': means the service is running in the shared instance pool (and thus running under the ArcGIS Pro provider runtime)

        :return: Boolean

        """
        allowed_providers = ["ArcObjects", "ArcObjects11", "DMaps"]
        url = self._url + "/changeProvider"
        params = {"f": "json", "provider": provider}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    @property
    def extensions(self) -> list:
        """lists the :class:`extensions <arcgis.gis.server.Extension>` on a service"""
        if self._extensions is None:
            self._init()
        return self._extensions

    # ----------------------------------------------------------------------
    def modify_extensions(self, extension_objects: Optional[list] = None) -> bool:
        """
        enables/disables a service extension type based on the name

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        extension_objects      Required list. A list of new extensions.
        ==================     ====================================================================


        :return: Boolean

        """
        if extension_objects is None:
            extension_objects = []
        if len(extension_objects) > 0 and isinstance(extension_objects[0], Extension):
            self._extensions = extension_objects
            self._json_dict["extensions"] = [x.value for x in extension_objects]
            res = self.edit(str(self._json_dict))
            self._json = None
            self._init()
            return res
        return False

    # ----------------------------------------------------------------------
    def _has_child_permissions_conflict(self, principal: str, permission: dict) -> dict:
        """
        You can invoke this operation on the resource (folder or service)
        to determine if this resource has a child resource with opposing
        permissions. This operation is typically invoked before adding a
        new permission to determine if the new addition will overwrite
        existing permissions on the child resources.
        For more information, see the section on the Continuous Inheritance
        Model.
        Since this operation basically checks if the permission to be added
        will cause a conflict with permissions on a child resource, this
        operation takes the same parameters as the Add Permission operation.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. Name of the role for whom the permission is being
                            assigned.
        ---------------     --------------------------------------------------------------------
        permission          Required dict. The permission dict. The format is described below.
                            Format:
                                {
                                "isAllowed": <true|false>,
                                "constraint": ""
                                }
        ===============     ====================================================================


        :return: dict

        """
        params = {
            "f": "json",
            "principal": principal,
            "permission": permission,
        }
        url = self._url + "/permissions/hasChildPermissionsConflict"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def start(self) -> bool:
        """starts the specific service"""
        params = {"f": "json"}
        u_url = self._url + "/start"
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def stop(self) -> bool:
        """stops the current service"""
        params = {"f": "json"}
        u_url = self._url + "/stop"
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def restart(self) -> bool:
        """restarts the current service"""
        self.stop()
        self.start()
        return True

    # ----------------------------------------------------------------------
    def rename(self, new_name: str) -> bool:
        """
        Renames this service to the new name

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        new_name            Required string. New name of the current service.
        ===============     ====================================================================


        :return: Boolean

        """
        params = {
            "f": "json",
            "serviceName": self.properties.serviceName,
            "serviceType": self.properties.type,
            "serviceNewName": new_name,
        }

        u_url = self._url[: self._url.rfind("/")] + "/renameService"

        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """deletes a service from arcgis server"""
        params = {
            "f": "json",
        }
        u_url = self._url + "/delete"
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def status(self) -> dict:
        """returns the status of the service"""
        params = {
            "f": "json",
        }
        u_url = self._url + "/status"
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    @property
    def statistics(self) -> dict:
        """returns the stats for the service"""
        params = {"f": "json"}
        u_url = self._url + "/statistics"
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    @property
    def _permissions(self) -> dict:
        """returns the permissions for the service"""
        params = {"f": "json"}
        u_url = self._url + "/permissions"
        return self._con.get(path=u_url, param_dict=params)

    # ----------------------------------------------------------------------
    @property
    def webhook_manager(self) -> ServiceWebHookManager:
        """Returns an instance of :class:`~arcgis.gis.server.ServiceWebHookManager`,
        the feature service-based webhook manager available at
        *ArcGIS Server 11.1* and later.
        """
        if self._server_version() >= [11, 0]:
            url: str = f"{self._url}/webhooks"
            if self._whm is None:
                self._whm = ServiceWebHookManager(url, con=self._con)
        return self._whm

    # ----------------------------------------------------------------------
    @property
    def _iteminfo(self) -> dict:
        """returns the item information"""
        params = {"f": "json"}
        u_url = self._url + "/iteminfo"
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    def _register_extension(self, item_id: str) -> dict:
        """
        Registers a new server object extension file with the server.
        Before you register the file, you need to upload the .SOE file to
        the server using the Upload Data Item operation. The item_id
        returned by the upload operation must be passed to the register
        operation.
        This operation registers all the server object extensions defined
        in the .SOE file.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item_id             Required string. unique ID of the item
        ===============     ====================================================================


        :return: dict

        """
        params = {"id": item_id, "f": "json"}
        url = self._url + "/types/extensions/register"
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def _delete_item_info(self) -> dict:
        """
        Deletes the item information.
        """
        params = {"f": "json"}
        u_url = self._url + "/iteminfo/delete"
        return self._con.get(path=u_url, params=params)

    # ----------------------------------------------------------------------
    def _upload_item_info(self, folder: str, path: str) -> dict:
        """
        Allows for the upload of new itemInfo files such as metadata.xml

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        folder              Required string. Folder on ArcGIS Server.
        ---------------     --------------------------------------------------------------------
        path                Required string. Full path of the file to upload.
        ===============     ====================================================================


        :return: dict


        """
        files = {}
        url = self._url + "/iteminfo/upload"
        params = {"f": "json", "folder": folder}
        files["file"] = path
        return self._con.post(path=url, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def _edit_item_info(self, json_dict: dict) -> dict:
        """
        Allows for the direct edit of the service's item's information.
        To get the current item information, pull the data by calling
        iteminfo property.  This will return the default template then pass
        this object back into the editItemInfo() as a dictionary.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        json_dict           Required dict.  Item information dictionary.
        ===============     ====================================================================


        :return: dict

        """
        url = self._url + "/iteminfo/edit"
        params = {"f": "json", "serviceItemInfo": json.dumps(json_dict)}
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def service_manifest(self, file_type: str = "json") -> str:
        """
        The service manifest resource documents the data and other
        resources that define the service origins and power the service.
        This resource will tell you underlying databases and their location
        along with other supplementary files that make up the service.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        file_type           Required string.  This value can be json or xml.  json return the
                            manifest.json file.  xml returns the manifest.xml file.
        ===============     ====================================================================


        :return: string


        """

        url = self._url + "/iteminfo/manifest/manifest.%s" % file_type
        params = {}
        f = self._con.get(
            path=url,
            params=params,
            out_folder=tempfile.gettempdir(),
            file_name=os.path.basename(url),
        )
        return open(f, "r").read()

    # ----------------------------------------------------------------------
    def _add_permission(self, principal: str, is_allowed: bool = True) -> bool:
        """
        Assigns a new permission to a role (principal). The permission
        on a parent resource is automatically inherited by all child resources.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        principal           Required string. The role to be assigned.
        ---------------     --------------------------------------------------------------------
        is_allowed          Optional boolean. Access of resource by boolean.
        ===============     ====================================================================


        :return: boolean

        """
        u_url = self._url + "/permissions/add"
        params = {
            "f": "json",
            "principal": principal,
            "isAllowed": is_allowed,
        }
        res = self._con.post(path=u_url, postdata=params)
        if "status" in res:
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    def edit(self, service: dict) -> bool:
        """
        To edit a service, you need to submit the complete JSON
        representation of the service, which includes the updates to the
        service properties. Editing a service causes the service to be
        restarted with updated properties.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        service             Required dict. The service JSON as a dictionary.
        ===============     ====================================================================


        :return: Boolean


        """
        url = self._url + "/edit"
        params = {"f": "json"}
        if isinstance(service, str):
            params["service"] = service
        elif isinstance(service, dict):
            params["service"] = json.dumps(service)
        res = self._con.post(path=url, postdata=params)
        if "status" in res:
            self._properties = None
            return res["status"] == "success"
        return res

    # ----------------------------------------------------------------------
    @property
    def iteminformation(self) -> "ItemInformationManager":
        """
        Returns the item information

        :return:
            :class:`~arcgis.gis.server.ItemInformationManager`

        """
        if self._ii is None:
            u_url = self._url + "/iteminfo"
            self._ii = ItemInformationManager(url=u_url, con=self._con)
        return self._ii

    # ----------------------------------------------------------------------
    @property
    def jobs(self) -> "JobManager":
        """returns a :class:`~arcgis.gis.server.JobManager` to manage asynchronous geoprocessing tasks"""
        if self._jm is None:
            url = "%s/jobs" % self._url
            self._jm = JobManager(url=url, con=self._con)
        return self._jm

    # ----------------------------------------------------------------------
    @property
    def _jobs(self) -> "JobManager":
        """returns a `JobManager` to manage asynchronous geoprocessing tasks"""

        if self._jm is None:
            url = "%s/jobs" % self._url
            self._jm = JobManager(url=url, con=self._con)
        return self._jm


###########################################################################
class JobManager(BaseServer):
    """
    The `JobManager` provides operations to locate, monitor, and intervene
    in current asynchronous jobs being run by the geoprocessing service.
    """

    _con = None
    _gis = None
    _url = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, con: Connection):
        """Constructor"""
        self._url = url
        self._con = con

    # ----------------------------------------------------------------------
    def search(
        self,
        start_time: _datetime.datetime = None,
        end_time: _datetime.datetime = None,
        status: Optional[str] = None,
        username: Optional[str] = None,
        machine: Optional[str] = None,
    ) -> "Job":
        """
        This operation allows you to query the current jobs for a
        geoprocessing service, with a range of parameters to find jobs that
        meet specific conditions.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        start_time          Optional Datetime. The start date/time of the geoprocessing job.
        ---------------     --------------------------------------------------------------------
        end_time            Optional Datetime. The end date/time of the geoprocessing job.
        ---------------     --------------------------------------------------------------------
        status              Optional String. The current status of the job. The possible
                            statuses are as follows:

                            - esriJobNew
                            - esriJobSubmitted
                            - esriJobExecuting
                            - esriJobSucceeded
                            - esriJobFailed
                            - esriJobCancelling
                            - esriJobCancelled
                            - esriJobWaiting
        ---------------     --------------------------------------------------------------------
        username            Optional String. The ArcGIS Server user who submitted the job. If
                            the service is anonymous, this parameter will be unavailable.
        ---------------     --------------------------------------------------------------------
        machine             Optional String. The machine running the job.
        ===============     ====================================================================


        :return: List of geoprocessing service :class:`jobs <arcgis.gis.server.Job>`

        """
        url = "{base}/query".format(base=self._url)

        if start_time and end_time is None:
            end_time = int(_datetime.datetime.now().timestamp() * 1000)
        params = {
            "f": "json",
            "start": 1,
            "number": 10,
            "startTime": "",
            "endTime": "",
            "userName": "",
            "machineName": "",
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        if status:
            params["status"] = status
        if username:
            params["userName"] = username
        if machine:
            params["machineName"] = machine
        results = []
        res = self._con.get(url, params)
        results = [
            Job(url="%s/%s" % (self._url, key), con=self._con)
            for key in res["results"].keys()
        ]
        while res["nextStart"] > -1:
            params["start"] = res["nextStart"]
            res = self._con.get(url, params)
            results += [
                Job(url="%s/%s" % (self._url, key), con=self._con)
                for key in res["results"].keys()
            ]
        return results

    # ----------------------------------------------------------------------
    def purge(self) -> dict:
        """
        The method `purge` cancels all asynchronous jobs for the
        geoprocessing service that currently carry a status of NEW,
        SUBMITTED, or WAITING.

        :return: Boolean

        """
        url = "{base}/purgeQueue".format(base=self._url)
        params = {"f": "json"}
        return self._con.post(url, params)


###########################################################################
class Job(BaseServer):
    """
    A `Job` represents the asynchronous execution of an operation by a
    geoprocessing service.
    """

    _con = None
    _url = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, con: GIS):
        """Constructor"""
        self._con = con
        self._url = url

    # ----------------------------------------------------------------------
    def cancel(self) -> bool:
        """
        Cancels the current job from the server

        :return: Boolean

        """
        url = "{base}/cancel".format(base=self._url)
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"]
        return res

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes the current job from the server

        :return: Boolean

        """
        url = "{base}/cancel".format(base=self._url)
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"]
        return res


###########################################################################
class ItemInformationManager(BaseServer):
    """
    The item information resource stores metadata about a service.
    Typically, this information is available to clients that want to index
    or harvest information about the service.

    Item information is represented in JSON. The property `properties` allows
    users to access the schema and see the current format of the JSON.


    """

    _url = None
    _properties = None
    _con = None

    def __init__(self, url: str, con: Connection):
        """Constructor"""
        self._url = url
        self._con = con

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """Deletes the item information.

        :return: Boolean

        """
        url = "{base}/delete".format(base=self._url)
        params = {"f": "json"}
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def upload(self, info_file: str, folder: Optional[str] = None) -> dict:
        """Uploads a file associated with the item information to the server.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        info_file           Required String. The file to upload to the server.
        ---------------     --------------------------------------------------------------------
        folder              Optional String. The name of the folder on the server to which the
                            file must be uploaded.
        ===============     ====================================================================

        :return: Dict

        """
        f = {"file": info_file}
        params = {
            "f": "json",
        }
        if folder:
            params["folder"] = folder
        url = "{base}/upload".format(base=self._url)
        res = self._con.post(url, params, files=f)
        return res

    # ----------------------------------------------------------------------
    @property
    def manifest(self) -> dict:
        """
        The service manifest resource documents the data and other resources
        that define the service origins and power the service. This resource
        will tell you underlying databases and their location along with
        other supplementary files that make up the service.


        The JSON representation of the manifest has the following two sections:

        Databases

        - **byReference** - Indicates whether the service data is referenced from a registered folder or database (true) or it was copied to the server at the time the service was published (false).
        - **onPremiseConnectionString** - Path to publisher data location.
        - **onServerConnectionString** - Path to data location after publishing completes.


        When both the server machine and the publisher's machine are using
        the same folder or database, byReference is true and the
        onPremiseConnectionString and onServerConnectionString properties
        have the same value.

        When the server machine and the publisher machine are using
        different folders or databases, byReference is true and the
        onPremiseConnectionString and onServerConnectionString properties
        have different values.

        When the data is copied to the server automatically at publish time,
        byReference is false.

        Resources

        - **clientName** - Machine where ArcGIS Pro or ArcGIS Desktop was used to publish the service.
        - **onPremisePath** - Path, relative to the 'clientName' machine, where the source resource (.mxd, .3dd, .tbx files, geodatabases, and so on) originated.
        - **serverPath** - Path to the document after publishing completes.

        :return: Dict

        """
        url = "{base}/manifest/manifest.json".format(base=self._url)
        params = {"f": "json"}

        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @property
    def properties(self) -> dict:
        """
        Gets/Sets the Item Information for a serivce.

        :return: Dict

        """
        url = "{base}".format(base=self._url)
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @properties.setter
    def properties(self, value: dict):
        """
        Gets/Sets the Item Information for a serivce.

        :return: Dict

        """
        url = "{base}/edit".format(base=self._url)
        params = {"f": "json", "serviceItemInfo": value}
        return self._con.post(url, params)


###########################################################################
class ItemInforamtionManager(ItemInformationManager):
    """
    The item information resource stores metadata about a service.
    Typically, this information is available to clients that want to index
    or harvest information about the service.

    Item information is represented in JSON. The property `properties` allows
    users to access the schema and see the current format of the JSON.


    """

    pass
