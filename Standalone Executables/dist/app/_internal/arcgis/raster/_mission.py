from __future__ import annotations
from typing import Any, Optional
from arcgis.gis import GIS, Item


from arcgis.raster.orthomapping import Project


class Mission:
    """

    Mission represents a mission in an Orthomapping Project.

    .. note :: This class is not created by users directly. An instance of this class is returned as output for
      get_mission() and add_mission() methods on the Project class of arcgis.raster.orthomapping module.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    mission_name                             Required string representing the mission name.

                                             Example:

                                                mission_name='Mission_Yucaipa'
    ------------------------------------     --------------------------------------------------------------------
    project                                  Required Project object or an Orthomapping Project portal item. The orthomapping project to which the mission belongs to.
    ====================================     ====================================================================

    .. code-block:: python

        # Example Usage 1

        om_item = gis.content.get("85a54236c6364a88a7c7c2b1a31fd901")
        project = Project(om_item, gis=gis)

        mission_1 = project.add_mission(image_list,
                                        mission_name="mission_name",
                                        image_collection="img_collection",
                                        raster_type_name="UAV/UAS",
                                        raster_type_params=raster_type_params)

        # Example Usage 2

        om_item = gis.content.get("85a54236c6364a88a7c7c2b1a31fd901")
        project = Project(om_item, gis=gis)

        mission_1 = project.get_mission("mission_name")

    """

    def __init__(self, mission_name, project):
        self._mission_name = mission_name
        if isinstance(project, Project):
            self._project = project
        elif isinstance(project, Item):
            if project.type == "Ortho Mapping Project":
                self._project = Project(project, gis=project._gis)

        self._project_item = project._project_item
        self._gis = project._gis
        self._mission_json = self._get_mission_json(self._mission_name)
        self._collection = None
        self._resource_info = self._resource_info(self._mission_name)

    @property
    def products(self):
        """
        The ``products`` property returns all the products associated with the mission

        :return: A list of products of the mission
        """
        items_prods = self._mission_json.get("items", None)
        import copy

        mission_product = copy.deepcopy(items_prods)
        for key, val in items_prods.items():
            if "itemId" in val.keys():
                if key == "imageCollection":
                    key = "image_collection"
                    mission_product[key] = self._gis.content.get(val["itemId"])
                    del mission_product["imageCollection"]
                else:
                    mission_product[key] = self._gis.content.get(val["itemId"])
        return mission_product

    @property
    def image_count(self):
        """
        The ``image_count`` property returns the number of images in the mission

        :return: An integer representing the number of images
        """
        if "sourceData" in self._mission_json.keys():
            source_data = self._mission_json["sourceData"]
            if "imageCount" in source_data.keys():
                return source_data["imageCount"]
            else:
                return 0
        return 0

    @property
    def mission_date(self):
        """
        The ``mission_date`` property returns the date of the mission.

        :return: A datetime object representing the mission date
        """
        if "sourceData" in self._mission_json.keys():
            source_data = self._mission_json["sourceData"]
            if "flightDate" in source_data.keys():
                from datetime import datetime

                datetime_obj = datetime.strptime(source_data["flightDate"], "%Y-%m-%d")
                return datetime_obj
            else:
                return None
        return None

    @property
    def image_collection(self):
        """
        The ``image_collection`` property returns the image collection associated with the mission

        :return: image collection item
        """
        if self._collection is not None:
            return self._collection
        else:
            mission_product = self._mission_json.get("items", None)
            for key, val in mission_product.items():
                if key == "imageCollection":
                    item_id = val["itemId"]
                image_collection_item = self._gis.content.get(item_id)
                self._collection = image_collection_item
                return image_collection_item

    def add_image(
        self,
        input_rasters: list,
        raster_type_name: Optional[str] = None,
        raster_type_params: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """
        Add a collection of images to existing image collection of the mission. It provides provision to specify image collection properties through context parameter.

        It can be used when new data is available to be included in the same mission of the
        orthomapping project. When new data is added to the image collection
        the entire image collection must be reset to the original state.

        ==================                   ====================================================================
        **Parameter**                         **Description**
        ------------------                   --------------------------------------------------------------------
        input_rasters                        Required, the list of input images to be added to
                                             the image collection being created. This parameter can
                                             be a list of image paths or a path to a folder containing the images

                                             The function can create hosted imagery layers on enterprise from 
                                             local raster datasets by uploading the data to the server.  
        ------------------                   --------------------------------------------------------------------
        raster_type_name                     Optional string. The name of the raster type to use for adding data to
                                             the image collection.


                                             Choice list:

                                                 | [
                                                 | "Aerial", "ASTER", "DMCII", "DubaiSat-2", "GeoEye-1", "GF-1 PMS", "GF-1 WFV",
                                                 | "GF-2 PMS", "GRIB", "HDF", "IKONOS", "Jilin-1", "KOMPSAT-2", "KOMPSAT-3",
                                                 | "Landsat 1-5 MSS", "Landsat 4-5 TM", "Landsat 7 ETM+", "Landsat 8", "Landsat 9",
                                                 | "NetCDF", "PlanetScope", "Pleiades-1", "Pleiades NEO", "QuickBird", "RapidEye",
                                                 | "Raster Dataset", "ScannedAerial", "Sentinel-2", "SkySat", "SPOT 5", "SPOT 6",
                                                 | "SPOT 7", "Superview-1", "Tiled Imagery Layer", "UAV/UAS", "WordView-1",
                                                 | "WordView-2", "WordView-3", "WordView-4", "ZY3-SASMAC", "ZY3-CRESDA"
                                                 | ]
                                         

                                             Example:

                                                "QuickBird"
        ------------------                   --------------------------------------------------------------------
        raster_type_params                   Optional dict. Additional ``raster_type`` specific parameters.
        
                                             The process of add rasters to the image collection can be \
                                             controlled by specifying additional raster type arguments.

                                             The raster type parameters argument is a dictionary.
                                         
                                             Syntax:

                                                 {"gps": [["image1.jpg", "10", "2", "300"], ["image2.jpg", "10", "3", "300"], ["image3.jpg", "10", "4", "300"]],
                                                 "cameraProperties": {"Maker": "Canon", "Model": "5D Mark II", "FocalLength": 20, "PixelSize": 10, "x0": 0, "y0": 0, "columns": 4000, "rows": 3000},
                                                 "constantZ": 300,"isAltitudeFlightHeight": "True","dem": {"url": ``https://...``}

                                             The dictionary can contain productType, processingTemplate, \
                                             pansharpenType, Filter, pansharpenWeights, ConstantZ, \
                                             dem, zoffset, CorrectGeoid, ZFactor, StretchType, \
                                             ScaleFactor, ValidRange

                                             Please check the table below (Supported Raster Types), \
                                             for more details about the product types, \
                                             processing templates, pansharpen weights for each raster type. 

                                             - Possible values for pansharpenType - ["Mean", "IHS", "Brovey", "Esri", "Mean", "Gram-Schmidt"]
                                             - Possible values for filter - [None, "Sharpen", "SharpenMore"]
                                             - Value for StretchType dictionary can be as follows:

                                               - "None"
                                               - "MinMax; <min>; <max>"
                                               - "PercentMinMax; <MinPercent>; <MaxPercent>"
                                               - "StdDev; <NumberOfStandardDeviation>"
                                               Example: {"StretchType": "MinMax; <min>; <max>"}
                                             - Value for ValidRange dictionary can be as follows:

                                               - "<MaskMinValue>, <MaskMaxValue>"
                                               Example: {"ValidRange": "10, 200"}

                                             Example:

                                                {"productType":"All","processingTemplate":"Pansharpen",
                                                "pansharpenType":"Gram-Schmidt","filter":"SharpenMore",
                                                "pansharpenWeights":"0.85 0.7 0.35 1","constantZ":-9999}
        ------------------                   --------------------------------------------------------------------
        context                              Optional dict. The context parameter is used to provide additional input parameters.

                                             Syntax:

                                                {"image_collection_properties": {"imageCollectionType":"Satellite"},"byref":'True'}
                                            
                                             Use ``image_collection_properties`` key to set value for imageCollectionType.


                                             .. note::

                                                The "imageCollectionType" property is important for image collection that will later on be adjusted by orthomapping system service.
                                                Based on the image collection type, the orthomapping system service will choose different algorithm for adjustment.
                                                Therefore, if the image collection is created by reference, the requester should set this
                                                property based on the type of images in the image collection using the following keywords.
                                                If the imageCollectionType is not set, it defaults to "UAV/UAS"
 
                                             If byref is set to 'True', the data will not be uploaded. If it is not set, the default is 'False'
        ==================                   ====================================================================

        :return: The imagery layer url


        """
        flight_json_details = {}
        if isinstance(self, Mission):
            mission = self
            image_collection = self.image_collection

        from arcgis.raster.analytics import add_image

        gis = self._gis
        gpjob = add_image(
            image_collection=image_collection,
            input_rasters=input_rasters,
            raster_type_name=raster_type_name,
            raster_type_params=raster_type_params,
            context=context,
            gis=gis,
            future=True,
        )

        while not gpjob.done():
            continue

        if gpjob.done():
            try:
                gps_data = []
                gps_info_list = ["name", "lat", "long", "alt", "acq"]

                # if "gps" in raster_type_params:
                #    for ele in raster_type_params["gps"]:
                #        dict_gps = dict(zip(gps_info_list, ele))
                #        gps_data.append(dict_gps)
                if not gps_data:
                    try:
                        lyr = image_collection.layers[0]
                        gps_info = lyr.query_gps_info()["images"]
                        for img_info in gps_info:
                            from arcgis.raster._util import _to_datetime

                            acq = _to_datetime(img_info["acquisitionDate"]).isoformat()
                            gps = img_info["gps"]
                            name = img_info["name"]
                            lat = gps["latitude"]
                            long = gps["longitude"]
                            alt = gps["altitude"]
                            gps_val = [name, lat, long, alt, acq]
                            dict_gps = dict(zip(gps_info_list, gps_val))
                            gps_data.append(dict_gps)
                    except:
                        if "gps" in raster_type_params:
                            gps_data = []
                            for ele in raster_type_params["gps"]:
                                dict_gps = dict(zip(gps_info_list, ele))
                                gps_data.append(dict_gps)
                            mission_json = mission._mission_json
                            gps_data_existing = mission_json["sourceData"]["gps"]
                            gps_data = gps_data + gps_data_existing
                        pass

                from datetime import datetime

                lyr = image_collection.layers[0]

                image_count = lyr.query(return_count_only=True)
                mission_json = mission._mission_json
                mission_json["sourceData"]["gps"] = gps_data
                mission_json["sourceData"]["imageCount"] = image_count

                ## Set extent
                try:
                    gcs_extent = {}
                    extent_arr = image_collection.extent
                    if extent_arr is not None:
                        gcs_extent = {
                            "xmin": extent_arr[0][0],
                            "ymin": extent_arr[0][1],
                            "xmax": extent_arr[1][0],
                            "ymax": extent_arr[1][1],
                            "spatialReference": {"wkid": 4326},
                        }
                except:
                    gcs_extent = {}

                projected_extent = {}
                try:
                    projected_extent = dict(lyr.extent)
                except:
                    projected_extent = {}

                mission_json.update(
                    {"gcsExtent": gcs_extent, "projectedExtent": projected_extent}
                )

                try:
                    coverage_area = image_collection.layers[0].query_boundary()["area"]
                    mission_json.update({"coverage": coverage_area})
                except:
                    pass

                try:
                    import json

                    job_messages = gpjob.messages
                    rm = mission._project_item.resources
                    # mission_json = mission._mission_json
                    resource = mission._resource_info
                    resource_name = resource["resource"]

                    start_time = (
                        gpjob._gpjob._start_time.isoformat(timespec="milliseconds")
                        + "Z"
                    )
                    end_time = (
                        gpjob._gpjob._end_time.isoformat(timespec="milliseconds") + "Z"
                    )
                    job_id = gpjob._gpjob._jobid

                    mission_json["jobs"].update(
                        {
                            "addImages": {
                                "messages": job_messages,
                                "checked": True,
                                "progress": 100,
                                "success": True,
                                "startTime": start_time,
                                "completionTime": end_time,
                                "jobId": job_id,
                            }
                        }
                    )
                    resource_props = resource["properties"]
                    properties = json.loads(resource["properties"])

                    properties.update({"imageCount": image_count})

                    import tempfile, uuid, os

                    fname = resource_name.split("/")[1]
                    temp_dir = tempfile.gettempdir()
                    temp_file = os.path.join(temp_dir, fname)
                    with open(temp_file, "w") as writer:
                        json.dump(mission_json, writer)
                    del writer

                    try:
                        rm.update(
                            file=temp_file,
                            text=mission_json,
                            folder_name="flights",
                            file_name=fname,
                            properties=properties,
                        )
                    except:
                        raise RuntimeError("Error updating the mission resource")

                    prj_data = mission._project_item.get_data()

                    project_properties = mission._project_item.properties

                    project_properties["imageCount"] = image_count
                    mission._project_item.update(
                        item_properties={"properties": project_properties},
                        data=json.dumps(prj_data),
                    )

                    # project_item.update(data=json.dumps(prj_data))
                except:
                    raise RuntimeError("Error adding the mission")
            except:
                raise RuntimeError("Error updating the mission JSON")
        return image_collection.url

    def delete_image(self, where: str):
        """

        ``delete_image`` allows users to remove existing images from the image collection (mosaic dataset) of a mission.

        ==================                   ====================================================================
        **Parameter**                         **Description**
        ------------------                   --------------------------------------------------------------------
        where                                Required string. A SQL ``where`` clause for selecting the images
                                             to be deleted from the image collection
        ==================                   ====================================================================

        :return: The imagery layer url

        """

        flight_json_details = {}
        if isinstance(self, Mission):
            mission = self
            image_collection = self.image_collection

        gis = self._gis
        from arcgis.raster.analytics import delete_image

        gpjob = delete_image(
            image_collection=image_collection, where=where, gis=gis, future=True
        )

        while not gpjob.done():
            continue

        if gpjob.done():
            try:
                from datetime import datetime

                lyr = image_collection.layers[0]

                image_count = lyr.query(return_count_only=True)
                mission_json = mission._mission_json
                ##########mission_json["sourceData"]["gps"].append(gps_data)
                mission_json["sourceData"]["imageCount"] = image_count

                ## Set extent
                try:
                    gcs_extent = {}
                    extent_arr = image_collection.extent
                    if extent_arr is not None:
                        gcs_extent = {
                            "xmin": extent_arr[0][0],
                            "ymin": extent_arr[0][1],
                            "xmax": extent_arr[1][0],
                            "ymax": extent_arr[1][1],
                            "spatialReference": {"wkid": 4326},
                        }
                except:
                    gcs_extent = {}

                projected_extent = {}
                try:
                    projected_extent = dict(lyr.extent)
                except:
                    projected_extent = {}

                mission_json.update(
                    {"gcsExtent": gcs_extent, "projectedExtent": projected_extent}
                )

                try:
                    coverage_area = image_collection.layers[0].query_boundary()["area"]
                    mission_json.update({"coverage": coverage_area})
                except:
                    pass

                try:
                    import json

                    job_messages = gpjob.messages
                    rm = mission._project_item.resources
                    # mission_json = mission._mission_json
                    resource = mission._resource_info
                    resource_name = resource["resource"]

                    start_time = (
                        gpjob._gpjob._start_time.isoformat(timespec="milliseconds")
                        + "Z"
                    )
                    end_time = (
                        gpjob._gpjob._end_time.isoformat(timespec="milliseconds") + "Z"
                    )

                    job_id = gpjob._gpjob._jobid
                    mission_json["jobs"].update(
                        {
                            "deleteImages": {
                                "messages": job_messages,
                                "checked": True,
                                "progress": 100,
                                "success": True,
                                "startTime": start_time,
                                "completionTime": end_time,
                                "jobId": job_id,
                            }
                        }
                    )
                    resource_props = resource["properties"]
                    properties = json.loads(resource["properties"])

                    properties.update({"imageCount": image_count})

                    import tempfile, uuid, os

                    fname = resource_name.split("/")[1]
                    temp_dir = tempfile.gettempdir()
                    temp_file = os.path.join(temp_dir, fname)
                    with open(temp_file, "w") as writer:
                        json.dump(mission_json, writer)
                    del writer

                    try:
                        rm.update(
                            file=temp_file,
                            text=mission_json,
                            folder_name="flights",
                            file_name=fname,
                            properties=properties,
                        )
                    except:
                        raise RuntimeError("Error updating the mission resource")

                    prj_data = mission._project_item.get_data()

                    project_properties = mission._project_item.properties

                    project_properties["imageCount"] = image_count
                    mission._project_item.update(
                        item_properties={"properties": project_properties},
                        data=json.dumps(prj_data),
                    )

                    # project_item.update(data=json.dumps(prj_data))
                except:
                    raise RuntimeError("Error adding the mission")
            except:
                raise RuntimeError("Error updating the mission JSON")
        return image_collection.url

    def _resource_info(self, name):
        res_manager = self._project._project_item.resources
        res_list = res_manager.list()
        for resource in res_list:
            full_res_name = resource["resource"]
            res_name = full_res_name[
                full_res_name.find("/") + 1 : full_res_name.find(".")
            ]
            if name == res_name:
                return resource

        return {}

    def _get_mission_json(self, name):
        res_manager = self._project._project_item.resources
        res_list = res_manager.list()
        for resource in res_list:
            full_res_name = resource["resource"]
            res_name = full_res_name[
                full_res_name.find("/") + 1 : full_res_name.find(".")
            ]
            if name == res_name:
                mission_json = res_manager.get(full_res_name)
                return mission_json

        return {}

    def __repr__(self):
        return "<%s - %s>" % (type(self).__name__, self._mission_name)
