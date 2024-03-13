from __future__ import annotations
import datetime
from typing import Optional
from arcgis import GIS, env
from arcgis._impl.common._mixins import PropertyMap

########################################################################


class TopographicProductionManager(object):
    """
    The Topographic Production Service resource represents a topographic
    production server object extension (SOE). It is the entry point
    for all functionality related to creating topographic products in ArcGIS Server.
    This resource provides information about the service and enables
    the topographic production server capabilities on a map service.
    You can also view child resources and operations defined by the SOE.

    .. note::
        The use of this resource requires an ArcGIS GIS Server
        Advanced license and a Production Mapping Server or
        Defense Mapping Server license.

    =====================   ===========================================
    **Inputs**              **Description**
    ---------------------   -------------------------------------------
    url                     Required String. The web endpoint to the topographic service.
    ---------------------   -------------------------------------------
    gis                     Optional GIS. The enterprise connection to
                            the Portal site. A connection can be passed
                            in such as a Service Directory connection.
    =====================   ===========================================


    """

    _con = None
    _gis = None
    _url = None
    _property = None
    # ----------------------------------------------------------------------

    def __init__(self, url, gis=None):
        """Constructor"""
        if gis is None:
            gis = env.active_gis
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._portal.con
        elif hasattr(gis, "_con"):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("gis must be of type GIS")
        self._gis = gis
        self._url = url

    # ----------------------------------------------------------------------
    def _init(self):
        """initializer"""
        try:
            res = self._con.get(self._url, {"f": "json"})
            self._property = PropertyMap(res)
        except Exception as e:
            self._property = PropertyMap({})

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties for the service"""
        if self._property is None:
            self._init()
        return self._property

    # ----------------------------------------------------------------------
    def add_product(
        self,
        product_definition: dict,
        raster: str = None,
        ancillary_layers: list[dict] = None,
    ):
        """
        The add_product operation adds a definition of a map product to
        the Topographic Production Service resource that can be used to generate a map

        ======================      =====================================================
        **Parameter**                **Description**
        ----------------------      -----------------------------------------------------
        product_definition          Required dictionary. The definition of the map product.

                                    `See Parameters <https://developers.arcgis.com/rest/services-reference/enterprise/tps-add-product.htm#:~:text=productDefinition%20properties/>`_
        ----------------------      -----------------------------------------------------
        raster                      Optional string. The path to the raster on disk (server path).
        ----------------------      -----------------------------------------------------
        ancillary_layers            Optional list of dictionaries. Additional layers to
                                    include in the final product.

                                    Syntax:
                                    ```
                                    [{
                                       "layer": "url of the layer",
                                        "featureClass": "name of the feature class to extract to",
                                        "map": "name of the map the layer will be inserted into",
                                        "layerIndex": "insertion index of the layer"
                                        },
                                        ...
                                    ]
                                    ```

                                    .. note::
                                        This parameter supports services located in the
                                        same portal site as the server object extension (SOE)
                                        or services that are publicly available.
                                        The featureClass, map, and layerIndex properties
                                        in the array are optional. If the dataset is
                                        identifiable from the feature service, it is not
                                        necessary to provide the featureClass property.
                                        The default values are 0 for layerIndex and BaseMap
                                        for map.
        ======================      =====================================================

        :return: True if succeeded and False if failed.
        """
        url = "%s/addProduct" % self._url
        params = {
            "f": "json",
            "productDefinition": product_definition,
            "raster": raster,
            "ancillaryLayers": ancillary_layers,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def calculate_extent(
        self,
        name: str,
        version: str,
        area_interest_layer: str,
        input_geometry: dict,
        out_sr: str,
    ):
        """
        The calculate_extent operation calculates a custom area of interest (AOI)
        for a given product and version. The result can be specified as the value for the customAoi parameter of the generateProduct operation.

        ======================      =====================================================
        **Parameter**                **Description**
        ----------------------      -----------------------------------------------------
        name                        Required string. The name of the product.
        ----------------------      -----------------------------------------------------
        version                     Required string. The version of the product to generate.
        ----------------------      -----------------------------------------------------
        area_interest_layer         Required string. The url of the layer defining the
                                    product's area of interest.
        ----------------------      -----------------------------------------------------
        input_geometry              Required dictionary. A geometry defining the center
                                    of the custom extent. A point geometry.
        ----------------------      -----------------------------------------------------
        out_sr                      Required string. The coordinate system for the
                                    extent that is returned.
        ======================      =====================================================

        :return: True if success and False otherwise.
        """
        url = "%s/calculateExtent" % self._url
        params = {
            "f": "json",
            "productName": name,
            "productVersion": version,
            "aoiLayer": area_interest_layer,
            "inputGeometry": input_geometry,
            "outSpatialReference": out_sr,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def generate_product(
        self,
        name: str,
        version: str,
        area_interest_layer: str,
        area_interest_feature_id: str,
        output_type: str,
        custom_area_interest: Optional[list[dict]] = None,
        layer_exclusion: Optional[list[str]] = None,
        ancillary_layers: Optional[list[dict]] = None,
    ):
        """
        The generate_product operation automates the process of producing
        a layout or map based on an existing map product definition.


        ========================        =====================================================
        **Parameter**                    **Description**
        ------------------------        -----------------------------------------------------
        name                            Required string. The name of the product.
        ------------------------        -----------------------------------------------------
        version                         Required string. The version of the product to generate.
        ------------------------        -----------------------------------------------------
        area_interest_layer             Required string. The url of the layer defining the
                                        product's area of interest.
        ------------------------        -----------------------------------------------------
        area_interest_feature_id        Required string. The Object ID of the area of interest
                                        feature in the area_interest_layer.

                                        .. note::
                                            If you use the optional custom_area_interest
                                            parameter, then this parameter is ignored.
        ------------------------        -----------------------------------------------------
        output_type                     Required string. The type of output.

                                        `Values: "aprx" | "pagx" | "pdf"`
        ------------------------        -----------------------------------------------------
        custom_area_interest            Optional list dictionary. The features defining the
                                        product's area of interest. One feature is allowed at
                                        the current release.
        ------------------------        -----------------------------------------------------
        layer_exclusion                 Optional list of strings. The list of layer names to
                                        exclude from the product.
        ------------------------        -----------------------------------------------------
        ancillary_layers                Optional list of dictionaries. Additional layers to
                                        include in the final product.

                                        Syntax:
                                        ```
                                        [{
                                        "layer": "url of the layer",
                                            "featureClass": "name of the feature class to extract to",
                                            "map": "name of the map the layer will be inserted into",
                                            "layerIndex": "insertion index of the layer"
                                            },
                                            ...
                                        ]
                                        ```

                                        Ancillary layers are added as feature service layers
                                        of the final product and don't extract data to the
                                        local geodatabase.

                                        .. note::
                                            This parameter supports services located in the
                                            same portal site as the server object extension (SOE)
                                            or services that are publicly available.
                                            The featureClass, map, and layerIndex properties
                                            in the array are optional. If the dataset is
                                            identifiable from the feature service, it is not
                                            necessary to provide the featureClass property.
                                            The default values are 0 for layerIndex and BaseMap
                                            for map.
        ========================        =====================================================


        :return: A dictionary with the jobId and statusUrl.
        """
        url = "%s/generateProduct" % self._url
        params = {
            "f": "json",
            "productName": name,
            "productVersion": version,
            "aoiLayer": area_interest_layer,
            "aoiFeatureId": area_interest_feature_id,
            "outputType": output_type,
            "customAoi": custom_area_interest,
            "layerExclusion": layer_exclusion,
            "ancillaryLayers": ancillary_layers,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def remove_product(self, name: str):
        """
        The remove_product operation removes a product from the Topographic
        Production service and returns a standard REST success or error message.

        ========================        =====================================================
        **Parameter**                    **Description**
        ------------------------        -----------------------------------------------------
        name                            Required string. The name of the product.
        ========================        =====================================================

        :return: True if success, else False.
        """
        url = "%s/removeProduct" % self._url
        params = {
            "f": "json",
            "productName": name,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def update_product(
        self,
        name: str,
        description: Optional[str] = None,
        enabled: Optional[bool] = None,
        sheet_id_field: Optional[str] = None,
        raster: Optional[str] = None,
        versions: Optional[dict] = None,
        ancillary_layers: Optional[dict] = None,
    ):
        """
        The update_product operation updates properties of a product.

        ========================        =====================================================
        **Parameter**                    **Description**
        ------------------------        -----------------------------------------------------
        name                            Required string. The name of the product.
        ------------------------        -----------------------------------------------------
        description                     Optional string. The new description of the product.
        ------------------------        -----------------------------------------------------
        enabled                         Optional bool. Enables or disables the product. If a
                                        product is disabled, the `generate_product`
                                        operation returns an error for that product.
        ------------------------        -----------------------------------------------------
        sheet_id_field                  Optional string. Updates the sheet ID field.
        ------------------------        -----------------------------------------------------
        raster                          Optional string. The path to a raster on disk (server path).
        ------------------------        -----------------------------------------------------
        versions                        Optional dict. Indicates the type of operation and
                                        versions. There are two formats.

                                        If the operation is `add` or `update`:
                                        ```
                                        {
                                           "operation": "add | update",
                                            "versions": [
                                                {
                                                    "name":"name of version",
                                                    "template": "name of template"
                                                },
                                                ...
                                            ]
                                        }
                                        ```

                                        If the operation is `remove`:
                                        ```
                                        {
                                            "operation": "remove",
                                            "versions": [
                                                    "name of version 1",
                                                    "name of version 2",
                                                    ...
                                            ]
                                        }
        ------------------------        -----------------------------------------------------
        ancillary_layers                Optional list of dictionaries. Additional layers to
                                        include in the final product.

                                        Syntax:
                                        ```
                                        [{
                                        "layer": "url of the layer",
                                            "featureClass": "name of the feature class to extract to",
                                            "map": "name of the map the layer will be inserted into",
                                            "layerIndex": "insertion index of the layer"
                                            },
                                            ...
                                        ]
                                        ```

                                        Ancillary layers are added as feature service layers
                                        of the final product and don't extract data to the
                                        local geodatabase.

                                        .. note::
                                            This parameter supports services located in the
                                            same portal site as the server object extension (SOE)
                                            or services that are publicly available.
                                            The featureClass, map, and layerIndex properties
                                            in the array are optional. If the dataset is
                                            identifiable from the feature service, it is not
                                            necessary to provide the featureClass property.
                                            The default values are 0 for layerIndex and BaseMap
                                            for map.
        ========================        =====================================================

        :return: The product name and whether the operation was a success(True) or a failure (false)
        """
        url = "%s/updateProduct" % self._url
        params = {
            "f": "json",
            "productName": name,
            "description": description,
            "enabled": enabled,
            "sheetIDField": sheet_id_field,
            "raster": raster,
            "productVersions": versions,
            "ancillaryLayers": ancillary_layers,
        }
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def products(self, include_def: bool = True):
        """
        The products operation retrieves the products that a Topographic
        Production Service resource supports.

        ====================    ==========================================
        **Parameter**            **Description**
        --------------------    ------------------------------------------
        include_def             Optional bool. Specifies whether the full
                                json definition of the map product is
                                included.
        ====================    ==========================================

        :return: The products that are supported as a dictionary.
        """
        url = "%s/products" % self._url
        params = {"f": "json", "includeDef": include_def}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def product(self, name: str, include_def: bool = True):
        """
        The product operation retrieves a single product from the products that
        a Topographic Production Service resource supports.

        ====================    ==========================================
        **Parameter**            **Description**
        --------------------    ------------------------------------------
        name                    Required string. The name of the product.
        --------------------    ------------------------------------------
        include_def             Optional bool. Specifies whether the full
                                json definition of the map product is
                                included.
        ====================    ==========================================
        """
        url = "%s/products/%s" % (self._url, name)
        params = {"f": "json", "includeDef": include_def}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def jobs_manager(self):
        """
        Retrieve the Topographic Production Job Manager class. With this manager
        you can retrieve all jobs, a single job, query or cancel a job.

        :return:
           :class:`~arcgis.features._topographic.TopographicProductionJobManager`
        """
        url = "%s/jobs" % self._url
        return TopographicProductionJobManager(url, self._gis)


# ============================================================================
class TopographicProductionJobManager(object):
    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def jobs(self):
        """
        Retrieve all the jobs for the service.
        """
        params = {
            "f": "json",
        }
        return self._con.post(self.url, params)

    # ----------------------------------------------------------------------
    def job(self, job_id: str, msg_level: str = None):
        """
        The job operation tracks the status of a job executed by the
        generate_product REST operation and returns the status, start date,
        last modified date, and messages of the job.

        ====================    ==========================================
        **Parameter**            **Description**
        --------------------    ------------------------------------------
        job_id                  Required string. The unique job id to see.
        --------------------    ------------------------------------------
        msg_level               Optional string. The message level associated
                                with the job.

                                `Values: "info" | "warn" | "error"`

                                .. note::
                                    Regardless of the msgLevel value,
                                    any errors that a job contains are
                                    included in the messages array of
                                    the response by default.
        ====================    ==========================================

        :return:
            A Json response with syntax:

            ```
            {
                "status": <untranslated string representing general status of the job>,
                "statusCode": <code representing specific status of the job>,
                "submissionTime": <time and date of the job submission in UTC and ISO 8601 format YYYY-MM-DDThh:mm:ssZ>,
                "lastUpdatedTime": <time and date of the last job update in UTC and ISO 8601 format YYYY-MM-DDThh:mm:ssZ>,
                "percentComplete": <percent of the job completed>,
                "aoiLayer": <URL of AOI layer>,
                "aoiFeatureId": <AOI feature ID>,
                "outputUrl": <URL of output>,
                "user": <username>,
                "jobId": <job identifier>,
                "productName": <name of the product>,
                "productVersion": <version of the product>,
                "outputType": <type/format of the output>,
                "messages": {
                <informative | error | warning>
                }
            }
            ```
        """
        url = "%s/job/%s" % (self._url, job_id)
        params = {"f": "json", "msgLevel": msg_level}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def cancel_job(self, job_id: str):
        """
        The cancel operation cancels a job submitted through the `generate_product`
        REST operation. It returns a standard REST success or error message.

        ====================    ==========================================
        **Parameter**            **Description**
        --------------------    ------------------------------------------
        job_id                  Required string. The unique job id to cancel.
        ====================    ==========================================

        :return: Success (true) or Failure (false)
        """
        url = "%s/cancel" % self._url
        params = {"f": "json", "jobId": job_id}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def query_job(
        self,
        status: Optional[int] = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        msg_level: Optional[str] = None,
    ):
        """
        The query operation retrieves the status of jobs executed by the
        `generate_product` REST operation and returns the status, start date,
        last modified date, and messages for a set of jobs.

        ====================    ==========================================
        **Parameter**            **Description**
        --------------------    ------------------------------------------
        status                  Optional int. Retrieves all the jobs with
                                a particular status.

                                `Values:
                                    * New—0
                                    * Submitted—1
                                    * Waiting—2
                                    * Executing—3
                                    * Succeeded—4
                                    * Failed—5
                                    * TimedOut—6
                                    * Canceling—7
                                    * Canceled—8
                                    * Deleting—9
                                    * Deleted—10`
        --------------------    ------------------------------------------
        start_date              Optional datetime. Retrieves all the jobs
                                that started on or after the start date and
                                time specified. The date and time should be
                                UTC and in ISO 8601 format. If time is not
                                specified, then midnight is used in the
                                following format: 00:00:00.
        --------------------    ------------------------------------------
        end_date                Optional datetime. Retrieves all the jobs
                                that ended on or after the end date and
                                time specified. The date and time should be
                                UTC and in ISO 8601 format. If time is not
                                specified, then 23:59:59 is used.
        --------------------    ------------------------------------------
        msg_level               Optional string. The message level associated
                                with the job.

                                `Values: "info" | "warn" | "error"`

                                .. note::
                                    Regardless of the msgLevel value,
                                    any errors that a job contains are
                                    included in the messages array of
                                    the response by default.
        ====================    ==========================================
        """
        url = "%s/query" % self._url
        params = {
            "f": "json",
            "status": status,
            "startDate": start_date,
            "endDate": end_date,
            "msgLevel": msg_level,
        }
        return self._con.post(url, params)
