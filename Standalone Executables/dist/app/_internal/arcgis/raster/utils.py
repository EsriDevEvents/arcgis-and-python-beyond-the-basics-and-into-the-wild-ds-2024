from typing import Any, Optional, Union, Dict
from arcgis.gis import GIS
from arcgis.raster import _util


def generate_direct_access_url(
    expiration: Optional[int] = None, *, gis: Optional[GIS] = None
):
    """
    Function to get the direct access url for user's rasterStore on ArcGIS Online.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    expiration                               Optional integer. Direct access URL expiration time in minutes.
                                             (The default is 1440 ie. 24 hours)
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` . The GIS on which this function runs.
                                             If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        String. Direct Access URL
    """

    return _util._generate_direct_access_url(expiration=expiration, gis=gis)


def upload_imagery_to_agol_userstore(
    files: Union[str, list],
    direct_access_url: Optional[str] = None,
    auto_renew: bool = True,
    upload_properties: Optional[dict] = None,
    *,
    gis: Optional[GIS] = None,
):
    """
    Uploads file/files to the user's rasterstore on ArcGIS Online and returns the list of urls.
    
    The list of urls can then be used with :meth:`~arcgis.raster.analytics.copy_raster` or :meth:`~arcgis.raster.analytics.create_image_collection`
    method to create imagery layers on ArcGIS Online.
    
    For this functionality to work, Azure library packages for Python (Azure SDK for Python - azure-storage-blob: 12.1<= version <=12.17)
    needs to be pre-installed. Refer https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#install-the-package

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    files                                    Required. It can be a folder, list of files or single file that needs to be uploaded.
    ------------------------------------     --------------------------------------------------------------------
    direct_access_url                        Optional string. The direct access url generated using :meth:`~arcgis.raster.utils.generate_direct_access_url` .
                                             If not specified, the function would generate the direct access url internally which is valid for 1440 minutes.
    ------------------------------------     --------------------------------------------------------------------
    auto_renew                               Optional boolean. If set to True, function would continue uploading 
                                             until the entire data is uploaded by auto renewing the direct access url.
                                             (The default is True)
    ------------------------------------     --------------------------------------------------------------------
    upload_properties                        | Optional dictionary. ``upload_properties`` can be used to control specific \
                                             upload parameters. 

                                             Available options:

                                             - ``maxUploadConcurrency``: Optional integer. Maximum number of parallel connections \
                                                to use for large uploads (when individual file/blob size exceeds 64MB). \
                                                This is the **max_concurrency** parameter of the `BlobClient.upload_blob() <https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#upload-blob-data--blob-type--blobtype-blockblob---blockblob----length-none--metadata-none----kwargs->`__ method. \
                                                (The default is 6)
                                             - ``maxWorkerThreads``: Optional integer. Maximum number of threads to execute asynchronously \
                                                when uploading multiple files. This is the **max_workers** parameter of the `ThreadPoolExecutor() <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`__ class. \
                                                (The default is None)
                                             - ``displayProgress``: Optional boolean. If set to True, a progress bar will be \
                                                displayed for tracking the progress of the uploads to user's rasterstore. \
                                                (The default is False)

                                               Example:

                                                    | {"maxUploadConcurrency":8,
                                                    | "maxWorkerThreads":20,
                                                    | "displayProgress":True}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` . The GIS on which this function runs.
                                             If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        List of file paths.

    .. code-block:: python

        # Usage Example: Generates an expirable direct access url and uploads files to the user's raster store.

        sas_url = generate_direct_access_url(expiration=180, gis=gis)

        uploaded_imagery = upload_imagery_to_agol_userstore(files=r"/path/to/data", 
                                                            direct_access_url=sas_url,
                                                            upload_properties={"displayProgress":True},
                                                            gis=gis
                                                            )

        # Following snippet executes the copy_raster() function on the uploaded imagery to create imagery layer item on ArcGIS Online.

        copy_raster_op = copy_raster(input_raster=uploaded_imagery,
                                     raster_type_name="Raster Dataset",
                                     output_name="output_layer",
                                     gis=gis)

        """

    return _util._upload_imagery_agol(
        files=files,
        direct_access_url=direct_access_url,
        auto_renew=auto_renew,
        upload_properties=upload_properties,
        gis=gis,
    )


def publish_hosted_imagery_layer(
    input_data: list,
    layer_configuration: str,
    tiles_only: Optional[bool] = False,
    raster_type_name: Optional[str] = None,
    raster_type_params: Optional[Dict[str, Any]] = None,
    source_mosaic: Optional[str] = None,
    output_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    *,
    gis: Optional[GIS] = None,
    future: bool = False,
    **kwargs,
):
    """
    The function can create hosted imagery layers in ArcGIS Enterprise and ArcGIS Online 
    from local raster datasets by uploading the data to the server. 
    Multiple images are mosaicked into a single dataset to create one layer. 
    A collection can also be created from multiple input rasters.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_data                               Required list. The list of input raster paths(s) to be added to 
                                             the imagery layer being created. 
                                             
                                             .. note:: 
                                                 You can also upload an existing mosaic dataset and create an imagery layer by 
                                                 specifying the mosaic dataset path as input to source_mosaic parameter. 
                                                 The input_data param can then be used to specify local raster dataset path(s) in the mosaic dataset.
    ------------------------------------     --------------------------------------------------------------------
    layer_configuration                      Required String.

                                                - ONE_IMAGE: Uses a single, processed image or mosaics multiple \
                                                images into a single dataset to create one layer. \
                                                This option supports all common image formats and satellite \
                                                products at various levels.

                                                - IMAGE_COLLECTION: Manages a collection of images using a \
                                                single layer and mosaics images dynamically. Each image can \
                                                be accessed independently. This option supports all common \
                                                image formats and satellite products at various levels. 
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Optional boolean. If set to True, function will generate a tiles only layer
                                             otherwise will generate a dynamic imagery layer.

                                                - True - Provides imagery access as static tiles and associated metadata. \
                                                Supports client-side processing and rendering. Can be used as an input \
                                                to raster analysis.

                                                - False - Provides versatile dynamic imagery access capabilities. \
                                                Supports on-demand server-side processing and \
                                                dynamic mosaicking. Supports managing a collection of images. \
                                                Can be used as an input to raster analysis.
    ------------------------------------     --------------------------------------------------------------------
    raster_type_name                         Required string. The name of the raster type to use for adding data.

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

                                             If an existing mosaic dataset is being published as a 
                                             dynamic imagery layer using the ``source_mosaic_dataset`` parameter, the
                                             ``raster_type_name`` parameter can be set to None as it is not required.


                                             Example:

                                                "QuickBird"
    ------------------------------------     --------------------------------------------------------------------
    raster_type_params                       Optional dict. Additional ``raster_type`` specific parameters.
        
                                             The process of add rasters can be controlled by specifying \
                                             additional raster type arguments.

                                             The raster type parameters argument is a dictionary.

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
    ------------------------------------     --------------------------------------------------------------------
    source_mosaic_dataset                    Optional string. Path to the existing mosaic dataset to be published 
                                             as a hosted dynamic imagery layer.

                                             To publish an existing mosaic dataset, specify the path to the input 
                                             data of the mosaic in the ``input_rasters`` parameter. 
                                             The data will be uploaded to ArcGIS Online.

                                             ``raster_type_name`` parameter can be set to None as it is not required to 
                                             publish an imagery layer from a mosaic dataset.

                                             .. note::
                                                    Option available only on ArcGIS online

                                             Example:

                                                "./data/temp_uploaded.gdb/test"
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional String. If not provided, an Image Service is created by the method 
                                             and used as the output raster.

                                             You can pass in an existing Image Service Item from your GIS to use 
                                             that instead.

                                             Alternatively, you can pass in the name of the output Image Service 
                                             that should be created by this method to be used as the output for 
                                             the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  context contains additional settings that affect task execution.

                                             context parameter overwrites values set through arcgis.env parameter
                                         
                                             This function has the following settings:

                                             - Output Spatial Reference (outSR): The output raster will be projected into the output spatial reference.
                                                
                                               Example:

                                                    {"outSR": {spatial reference}}


                                             - Upload Properties (upload_properties): ``upload_properties`` key can be used to control specific upload parameters when trying to create hosted imagery layers in ArcGIS Online from local raster datasets.

                                               Available options:

                                               - ``maxUploadConcurrency``: Optional integer. Maximum number of parallel connections \
                                                 to use for large uploads (when individual file/blob size exceeds 64MB). \
                                                 This is the **max_concurrency** parameter of the `BlobClient.upload_blob() <https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#upload-blob-data--blob-type--blobtype-blockblob---blockblob----length-none--metadata-none----kwargs->`__ method. \
                                                 (The default is 6)
                                               - ``maxWorkerThreads``: Optional integer. Maximum number of threads to execute asynchronously \
                                                 when uploading multiple files. This is the **max_workers** parameter of the `ThreadPoolExecutor() <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`__ class. \
                                                 (The default is None)
                                               - ``displayProgress``: Optional boolean. If set to True, a progress bar will be \
                                                 displayed for tracking the progress of the uploads to user's rasterstore. \
                                                 (The default is False)

                                               Example:

                                                    | {"upload_properties": {"maxUploadConcurrency":8,
                                                    |                       "maxWorkerThreads":20,
                                                    |                       "displayProgress":True}
                                                    | }

                                             The context parameter can also be used to specify whether to
                                             build footprints, pixel value that represents the NoData,
                                             resamplingMethod etc.


                                             Example:

                                                | {"buildFootprints":True,                                            
                                                | "footprintsArguments":{"method":"RADIOMETRY","minValue":1,"maxValue":5,
                                                |                        "shrinkDistance":50,"skipOverviews":True,"updateBoundary":True,
                                                |                        "maintainEdge":False,"simplification":None,"numVertices":20,
                                                |                        "minThinnessRatio":0.05,"maxSliverSize":20,"requestSize":2000,
                                                |                        "minRegionSize":100},
                                                | "defineNodata":True,                                            
                                                | "noDataArguments":{"noDataValues":[500],"numberOfBand":99,"compositeValue":True},                                            
                                                | "buildOverview":True}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Keyword only parameter. Optional :class:`~arcgis.gis.GIS` . The GIS on which this function runs.
                                             If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
    Imagery layer item

        """
    from arcgis import env

    if output_name is None:
        from arcgis.raster._util import _id_generator

        output_name = "layer" + "_" + _id_generator()

    if layer_configuration.upper() == "ONE_IMAGE":
        gis = env.active_gis if gis is None else gis

        return gis._tools.rasteranalysis.copy_raster(
            input_raster=input_data,
            output_name=output_name,
            context=context,
            future=future,
            raster_type_name=raster_type_name,
            raster_type_params=raster_type_params,
            md_to_upload=source_mosaic,
            tiles_only=tiles_only,
            **kwargs,
        )

    elif layer_configuration.upper() == "IMAGE_COLLECTION":
        gis = env.active_gis if gis is None else gis

        return gis._tools.rasteranalysis.create_image_collection(
            image_collection=output_name,
            input_rasters=input_data,
            raster_type_name=raster_type_name,
            raster_type_params=raster_type_params,
            md_to_upload=source_mosaic,
            context=context,
            gis=gis,
            future=future,
            **kwargs,
        )

    else:
        raise RuntimeError(
            "layer_configuration should be either ONE_IMAGE or IMAGE_COLLECTION"
        )
