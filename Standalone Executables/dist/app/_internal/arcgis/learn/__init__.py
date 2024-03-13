"Functions for calling the Deep Learning Tools."
from . import _utils
from ._utils.env import _LAMBDA_TEXT_CLASSIFICATION
from arcgis.geoprocessing._support import (
    _analysis_job,
    _analysis_job_results,
    _analysis_job_status,
    _layer_input,
)
import json as _json
import arcgis as _arcgis
from arcgis.raster._layer import ImageryLayer as _ImageryLayer
from arcgis.raster._util import _set_context, _id_generator
from ._scannedmapdigitizer import ScannedMapDigitizer
from .models._timm_utils import load_timm_bckbn_pretrained
from timm.models import helpers

helpers.load_pretrained = load_timm_bckbn_pretrained

if not _LAMBDA_TEXT_CLASSIFICATION:
    from .models import (
        SingleShotDetector,
        UnetClassifier,
        FeatureClassifier,
        RetinaNet,
        PSPNetClassifier,
        MaskRCNN,
        DeepLab,
        PointCNN,
        ModelExtension,
        FasterRCNN,
        SuperResolution,
        FullyConnectedNetwork,
        MLModel,
        YOLOv3,
        HEDEdgeDetector,
        BDCNEdgeDetector,
        ImageCaptioner,
        TimeSeriesModel,
        CycleGAN,
        MultiTaskRoadExtractor,
        ChangeDetector,
        Pix2Pix,
        ConnectNet,
        SiamMask,
        Track,
        Embeddings,
        MMDetection,
        MMSegmentation,
        AutoML,
        DeepSort,
        Pix2PixHD,
        AutoDL,
        ImageryModel,
        MaXDeepLab,
        WNet_cGAN,
        DETReg,
        RandLANet,
        EfficientDet,
        SQNSeg,
        PSETAE,
        MMDetection3D,
    )

    from ._object_tracker import ObjectTracker

    from ._utils.pointcloud_data import Transform3d
from ._data import prepare_data, prepare_tabulardata, prepare_textdata
from ._process_df import process_df, add_datepart
from ._utils.evaluate_batchsize import estimate_batch_size


_point_cloud_classification_model_list = ["PointCNN", "RandLANet", "SQNSeg"]


_point_cloud_detection_model_list = ["MMDetection3D"]


def _set_param(gis, params, param_name, input_param):
    if isinstance(input_param, str):
        if "http:" in input_param or "https:" in input_param:
            params[param_name] = _json.dumps({"url": input_param})
        else:
            params[param_name] = _json.dumps({"uri": input_param})

    elif isinstance(input_param, _arcgis.gis.Item):
        params[param_name] = _json.dumps({"itemId": input_param.itemid})

    elif isinstance(input_param, dict):
        params[param_name] = input_param
    elif isinstance(input_param, Model):
        params[param_name] = input_param._model
    else:
        raise TypeError(input_param + " should be a string (service url) or Item")

    return


def _create_output_image_service(gis, output_name, task, folder=None):
    ok = gis.content.is_service_name_available(output_name, "Image Service")
    if not ok:
        raise RuntimeError(
            "An Image Service by this name already exists: " + output_name
        )

    create_parameters = {
        "name": output_name,
        "description": "",
        "capabilities": "Image",
        "properties": {"path": "@", "description": "", "copyright": ""},
    }

    output_service = gis.content.create_service(
        output_name,
        create_params=create_parameters,
        service_type="imageService",
        folder=folder,
    )
    description = "Image Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": "Analysis Image Service generated from " + task,
    }
    output_service.update(item_properties)
    return output_service


def _create_output_feature_service(
    gis,
    output_name,
    output_service_name="Analysis feature service",
    task="GeoAnalytics",
    folder=None,
):
    ok = gis.content.is_service_name_available(output_name, "Feature Service")
    if not ok:
        raise RuntimeError(
            "A Feature Service by this name already exists: " + output_name
        )

    createParameters = {
        "currentVersion": 10.2,
        "serviceDescription": "",
        "hasVersionedData": False,
        "supportsDisconnectedEditing": False,
        "hasStaticData": True,
        "maxRecordCount": 2000,
        "supportedQueryFormats": "JSON",
        "capabilities": "Query",
        "description": "",
        "copyrightText": "",
        "allowGeometryUpdates": False,
        "syncEnabled": False,
        "editorTrackingInfo": {
            "enableEditorTracking": False,
            "enableOwnershipAccessControl": False,
            "allowOthersToUpdate": True,
            "allowOthersToDelete": True,
        },
        "xssPreventionInfo": {
            "xssPreventionEnabled": True,
            "xssPreventionRule": "InputOnly",
            "xssInputRule": "rejectInvalid",
        },
        "tables": [],
        "name": output_service_name.replace(" ", "_"),
    }

    output_service = gis.content.create_service(
        output_name,
        create_params=createParameters,
        service_type="featureService",
        folder=folder,
    )
    description = "Feature Service generated from running the " + task + " tool."
    item_properties = {
        "description": description,
        "tags": "Analysis Result, " + task,
        "snippet": output_service_name,
    }
    output_service.update(item_properties)
    return output_service


def _set_output_raster(output_name, task, gis, output_properties=None):
    output_service = None
    output_raster = None

    task_name = task

    folder = None
    folderId = None

    if output_properties is not None:
        if "folder" in output_properties:
            folder = output_properties["folder"]
    if folder is not None:
        if isinstance(folder, dict):
            if "id" in folder:
                folderId = folder["id"]
                folder = folder["title"]
        else:
            owner = gis.properties.user.username
            folderId = gis._portal.get_folder_id(owner, folder)
        if folderId is None:
            folder_dict = gis.content.create_folder(folder, owner)
            folder = folder_dict["title"]
            folderId = folder_dict["id"]

    if output_name is None:
        output_name = str(task_name) + "_" + _id_generator()
        output_service = _create_output_image_service(
            gis, output_name, task, folder=folder
        )
        output_raster = {
            "serviceProperties": {
                "name": output_service.name,
                "serviceUrl": output_service.url,
            },
            "itemProperties": {"itemId": output_service.itemid},
        }
    elif isinstance(output_name, str):
        output_service = _create_output_image_service(
            gis, output_name, task, folder=folder
        )
        output_raster = {
            "serviceProperties": {
                "name": output_service.name,
                "serviceUrl": output_service.url,
            },
            "itemProperties": {"itemId": output_service.itemid},
        }
    elif isinstance(output_name, _arcgis.gis.Item):
        output_service = output_name
        output_raster = {"itemProperties": {"itemId": output_service.itemid}}
    else:
        raise TypeError("output_raster should be a string (service name) or Item")

    if folderId is not None:
        output_raster["itemProperties"].update({"folderId": folderId})
    output_raster = _json.dumps(output_raster)
    return output_raster, output_service


def detect_objects(
    input_raster,
    model,
    model_arguments=None,
    output_name=None,
    run_nms=False,
    confidence_score_field=None,
    class_value_field=None,
    max_overlap_ratio=0,
    context=None,
    process_all_raster_items=False,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function can be used to generate feature service that contains polygons on detected objects
    found in the imagery data using the designated deep learning model. Note that the deep learning
    library needs to be installed separately, in addition to the server's built in Python 3.x library.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server) and ArcGIS Image for ArcGIS Online.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required. raster layer that contains objects that needs to be detected.
    ------------------------------------     --------------------------------------------------------------------
    model                                    Required :class:`~arcgis.learn.Model` object.
    ------------------------------------     --------------------------------------------------------------------
    model_arguments                          Optional dictionary. Name-value pairs of arguments and their values that can be customized by the clients.

                                             eg: {"name1":"value1", "name2": "value2"}
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, a :class:`~arcgis.features.FeatureLayer` is created by the method and used as the output .
                                             You can pass in an existing Feature Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Feature Service that should be created by this method
                                             to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    run_nms                                  Optional bool. Default value is False. If set to True, runs the Non Maximum Suppression tool.
    ------------------------------------     --------------------------------------------------------------------
    confidence_score_field                   Optional string. The field in the feature class that contains the confidence scores as output by the object detection method.
                                             This parameter is required when you set the run_nms to True
    ------------------------------------     --------------------------------------------------------------------
    class_value_field                        Optional string. The class value field in the input feature class.
                                             If not specified, the function will use the standard class value fields
                                             Classvalue and Value. If these fields do not exist, all features will
                                             be treated as the same object class.
                                             Set only if run_nms  is set to True
    ------------------------------------     --------------------------------------------------------------------
    max_overlap_ratio                        Optional integer. The maximum overlap ratio for two overlapping features.
                                             Defined as the ratio of intersection area over union area.
                                             Set only if run_nms  is set to True
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - cellSize - Set the output raster cell size, or resolution

                                             - extent - Sets the processing extent used by the function

                                             - parallelProcessingFactor - Sets the parallel processing factor. Default is "80%"

                                             - mask: Only cells that fall within the analysis mask will be considered in the operation.

                                             Eg: {"mask": {"url": "<feature_service_url>"}}

                                             - processorType - Sets the processor type. "CPU" or "GPU"

                                             Eg: {"processorType" : "CPU"}

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.
    ------------------------------------     --------------------------------------------------------------------
    process_all_raster_items                 Optional bool. Specifies how all raster items in an image service will be processed.

                                             - False : all raster items in the image service will be mosaicked together and processed. This is the default.

                                             - True : all raster items in the image service will be processed as separate images.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
    ====================================     ====================================================================

    :return:
        The output feature layer item containing the detected objects

    """

    # task = "DetectObjectsUsingDeepLearning"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.detect_objects_using_deep_learning(
        input_raster=input_raster,
        model=model,
        output_objects=output_name,
        model_arguments=model_arguments,
        run_nms=run_nms,
        confidence_score_field=confidence_score_field,
        class_value_field=class_value_field,
        max_overlap_ratio=max_overlap_ratio,
        context=context,
        process_all_raster_items=process_all_raster_items,
        future=future,
        **kwargs,
    )

    """
    url = gis.properties.helperServices.rasterAnalytics.url
    gptool = _arcgis.gis._GISResource(url, gis)

    params = {}

    params["inputRaster"] = _layer_input(input_raster)

    if output_name is None:
        output_service_name = 'DetectObjectsUsingDeepLearning_' + _id_generator()
        output_name = output_service_name.replace(' ', '_')
    else:
        output_service_name = output_name.replace(' ', '_')

    folder = None
    folderId = None
    if kwargs is not None:
        if "folder" in kwargs:
                folder = kwargs["folder"]
        if folder is not None:
            if isinstance(folder, dict):
                if "id" in folder:
                    folderId = folder["id"]
                    folder=folder["title"]
            else:
                owner = gis.properties.user.username
                folderId = gis._portal.get_folder_id(owner, folder)
            if folderId is None:
                folder_dict = gis.content.create_folder(folder, owner)
                folder = folder_dict["title"]
                folderId = folder_dict["id"]

    output_service = _create_output_feature_service(gis, output_name, output_service_name, 'Detect Objects', folder)

    if folderId is not None:
        params["outputObjects"] = _json.dumps({"serviceProperties": {"name": output_service_name, "serviceUrl": output_service.url},
                                               "itemProperties": {"itemId": output_service.itemid}, "folderId":folderId})
    else:
        params["outputObjects"] = _json.dumps({"serviceProperties": {"name": output_service_name, "serviceUrl": output_service.url},
                                               "itemProperties": {"itemId": output_service.itemid}})

    if model is None:
        raise RuntimeError('model cannot be None')
    else:
        _set_param(gis, params, "model", model)

    if model_arguments:
        params["modelArguments"] = dict((str(k),str(v)) for k, v in model_arguments.items())

    if isinstance(run_nms, bool):
        if run_nms:
            params["runNMS"] = True

            if confidence_score_field is not None:
                params["confidenceScoreField"] = confidence_score_field

            if class_value_field is not None:
                params["classValueField"] = class_value_field
    
            if max_overlap_ratio is not None:
                params["maxOverlapRatio"] = max_overlap_ratio
        else:
            params["runNMS"] = False
    else:
        raise RuntimeError("run_nms value should be an instance of bool")
    
    _set_context(params, context)

    task_url, job_info, job_id = _analysis_job(gptool, task, params)

    job_info = _analysis_job_status(gptool, task_url, job_info)
    job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
    item_properties = {
        "properties": {
            "jobUrl": task_url + '/jobs/' + job_info['jobId'],
            "jobType": "GPServer",
            "jobId": job_info['jobId'],
            "jobStatus": "completed"
        }
    }
    output_service.update(item_properties)
    return output_service
    """


def classify_pixels(
    input_raster,
    model,
    model_arguments=None,
    output_name=None,
    context=None,
    process_all_raster_items=False,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function to classify input imagery data using a deep learning model.
    Note that the deep learning library needs to be installed separately,
    in addition to the server's built in Python 3.x library.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server) and ArcGIS Image for ArcGIS Online.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required. raster layer that needs to be classified.
    ------------------------------------     --------------------------------------------------------------------
    model                                    Required :class:`~arcgis.learn.Model` object.
    ------------------------------------     --------------------------------------------------------------------
    model_arguments                          Optional dictionary. Name-value pairs of arguments and their values that can be customized by the clients.

                                             eg: {"name1":"value1", "name2": "value2"}

    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, an imagery layer is created by the method and used as the output .
                                             You can pass in an existing Image Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method
                                             to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - outSR - (Output Spatial Reference) Saves the result in the specified spatial reference

                                             - snapRaster - Function will adjust the extent of output rasters so that they
                                               match the cell alignment of the specified snap raster.

                                             - cellSize - Set the output raster cell size, or resolution

                                             - extent - Sets the processing extent used by the function

                                             - parallelProcessingFactor - Sets the parallel processing factor. Default is "80%"

                                             - processorType - Sets the processor type. "CPU" or "GPU"

                                               Example:
                                                    {"outSR" : {spatial reference}}

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.
    ------------------------------------     --------------------------------------------------------------------
    process_all_raster_items                 Optional bool. Specifies how all raster items in an image service will be processed.

                                             - False : all raster items in the image service will be mosaicked together and processed. This is the default.

                                             - True : all raster items in the image service will be processed as separate images.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    tiles_only                               Keyword only parameter. Optional boolean.
                                             In ArcGIS Online, the default output image service for this function would be a Tiled Imagery Layer.
                                             To create Dynamic Imagery Layer as output in ArcGIS Online, set tiles_only parameter to False.

                                             Function will not honor tiles_only parameter in ArcGIS Enterprise and will generate Dynamic Imagery Layer by default.
    ====================================     ====================================================================

    :return:
        The classified imagery layer item

    """

    # task = "ClassifyPixelsUsingDeepLearning"

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.classify_pixels_using_deep_learning(
        input_raster=input_raster,
        model=model,
        model_arguments=model_arguments,
        output_classified_raster=output_name,
        context=context,
        process_all_raster_items=process_all_raster_items,
        future=future,
        **kwargs,
    )

    """
    url = gis.properties.helperServices.rasterAnalytics.url
    gptool = _arcgis.gis._GISResource(url, gis)

    output_service = None

    output_raster, output_service = _set_output_raster(output_name, task, gis, kwargs)

    params = {}

    params["outputClassifiedRaster"] = output_raster

    params["inputRaster"] = _layer_input(input_raster)

    if model is None:
        raise RuntimeError('model cannot be None')
    else:
        _set_param(gis, params, "model", model)

    if model_arguments:
        params["modelArguments"] = dict((str(k),str(v)) for k, v in model_arguments.items())

    _set_context(params, context)

    task_url, job_info, job_id = _analysis_job(gptool, task, params)

    job_info = _analysis_job_status(gptool, task_url, job_info)
    job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
    item_properties = {
        "properties": {
            "jobUrl": task_url + '/jobs/' + job_info['jobId'],
            "jobType": "GPServer",
            "jobId": job_info['jobId'],
            "jobStatus": "completed"
        }
    }
    output_service.update(item_properties)
    return output_service
    """


def export_training_data(
    input_raster,
    input_class_data=None,
    chip_format=None,
    tile_size=None,
    stride_size=None,
    metadata_format=None,
    classvalue_field=None,
    buffer_radius=None,
    output_location=None,
    context=None,
    input_mask_polygons=None,
    rotation_angle=0,
    reference_system="MAP_SPACE",
    process_all_raster_items=False,
    blacken_around_feature=False,
    fix_chip_size=True,
    additional_input_raster=None,
    input_instance_data=None,
    instance_class_value_field=None,
    min_polygon_overlap_ratio=0,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function is designed to generate training sample image chips from the input imagery data with
    labeled vector data or classified images. The output of this service tool is the data store string
    where the output image chips, labels and metadata files are going to be stored.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server)

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required :class:`~arcgis.raster.ImageryLayer`/:class:`~arcgis.raster.Raster`/:class:`~arcgis.gis.Item`/String (URL).
                                             Raster layer that needs to be exported for training.
    ------------------------------------     --------------------------------------------------------------------
    input_class_data                         Labeled data, either a feature layer or image layer.
                                             Vector inputs should follow a training sample format as
                                             generated by the ArcGIS Pro Training Sample Manager.
                                             Raster inputs should follow a classified raster format as generated by the Classify Raster tool.
    ------------------------------------     --------------------------------------------------------------------
    chip_format                              Optional string. The raster format for the image chip outputs.

                                             - ``TIFF``: TIFF format

                                             - ``PNG``: PNG format

                                             - ``JPEG``: JPEG format

                                             - ``MRF``: MRF (Meta Raster Format)
    ------------------------------------     --------------------------------------------------------------------
    tile_size                                Optional dictionary. The size of the image chips.

                                             Example: {"x": 256, "y": 256}
    ------------------------------------     --------------------------------------------------------------------
    stride_size                              Optional dictionary. The distance to move in the X and Y when creating
                                             the next image chip.
                                             When stride is equal to the tile size, there will be no overlap.
                                             When stride is equal to half of the tile size, there will be 50% overlap.

                                             Example: {"x": 128, "y": 128}
    ------------------------------------     --------------------------------------------------------------------
    metadata_format                          Optional string. The format of the output metadata labels. There are 4 options for output metadata labels for the training data,
                                             KITTI Rectangles, PASCAL VOCrectangles, Classified Tiles (a class map) and RCNN_Masks. If your input training sample data
                                             is a feature class layer such as building layer or standard classification training sample file,
                                             use the KITTI or PASCAL VOC rectangle option.
                                             The output metadata is a .txt file or .xml file containing the training sample data contained
                                             in the minimum bounding rectangle. The name of the metadata file matches the input source image
                                             name. If your input training sample data is a class map, use the Classified Tiles as your output metadata format option.

                                             - ``KITTI_rectangles``: The metadata follows the same format as the Karlsruhe Institute of Technology and Toyota echnological Institute (KITTI) Object Detection Evaluation dataset. The KITTI dataset is a vision benchmark suite. This is the default.The label files are plain text files. All values, both numerical or strings, are separated by spaces, and each row corresponds to one object. This format can be used with FasterRCNN, RetinaNet, SingleShotDetector and YOLOv3 models.

                                             - ``PASCAL_VOC_rectangles``: The metadata follows the same format as the Pattern Analysis, Statistical Modeling and
                                               Computational Learning, Visual Object Classes (PASCAL_VOC) dataset. The PASCAL VOC dataset is a standardized
                                               image data set for object class recognition.The label files are XML files and contain information about image name,
                                               class value, and bounding box(es).
                                               This format can be used with FasterRCNN, RetinaNet, SingleShotDetector and YOLOv3 models.

                                             - ``Classified_Tiles``: This option will output one classified image chip per input image chip.
                                               No other meta data for each image chip. Only the statistics output has more information on the
                                               classes such as class names, class values, and output statistics.
                                               This format can be used with BDCNEdgeDetector, DeepLab, HEDEdgeDetector, MultiTaskRoadExtractor, PSPNetClassifier and UnetClassifier models.

                                             - ``RCNN_Masks``: This option will output image chips that have a mask on the areas where the sample exists.
                                               The model generates bounding boxes and segmentation masks for each instance of an object in the image.
                                               This format can be used with MaskRCNN model.

                                             - ``Labeled_Tiles``: This option will label each output tile with a specific class.
                                               This format is used for image classification.
                                               This format can be used with FeatureClassifier model.

                                             - ``MultiLabeled_Tiles``: Each output tile will be labeled with one or more classes.
                                               For example, a tile may be labeled agriculture and also cloudy. This format is used for object classification.
                                               This format can be used with FeatureClassifier model.

                                             - ``Export_Tiles``: The output will be image chips with no label.
                                               This format is used for image enhancement techniques such as Super Resolution and Change Detection.
                                               This format can be used with ChangeDetector, CycleGAN, Pix2Pix and SuperResolution models.

                                             - ``CycleGAN``: The output will be image chips with no label. This format is used for image
                                               translation technique CycleGAN, which is used to train images that do not overlap.

                                             - ``Imagenet``: Each output tile will be labeled with a specific class. This format is used
                                               for object classification; however, it can also be used for object tracking when the Deep Sort
                                               model type is used during training.

                                             - ``Panoptic_Segmentation``: The output will be one classified image chip and one instance per
                                               input image chip. The output will also have image chips that mask the areas where the sample exists;
                                               these image chips will be stored in a different folder. This format is used for both pixel classification
                                               and instance segmentation, therefore there will be two output labels folders.
    ------------------------------------     --------------------------------------------------------------------
    classvalue_field                         Optional string. Specifies the field which contains the class values. If no field is specified,
                                             the system will look for a 'value' or 'classvalue' field. If this feature does
                                             not contain a class field, the system will presume all records belong the 1 class.
    ------------------------------------     --------------------------------------------------------------------
    buffer_radius                            Optional integer. Specifies a radius for point feature classes to specify training sample area.
    ------------------------------------     --------------------------------------------------------------------
    output_location                          This is the output location for training sample data.
                                             It can be the server data store path or a shared file system path.

                                             Example:

                                             Server datastore path -

                                                * ``/fileShares/deeplearning/rooftoptrainingsamples``
                                                * ``/rasterStores/rasterstorename/rooftoptrainingsamples``

                                             File share path -

                                                * ``\\\\servername\\deeplearning\\rooftoptrainingsamples``
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - exportAllTiles - Choose if the image chips with overlapped labeled data will be exported.

                                               * True - Export all the image chips, including those that do not overlap labeled data.
                                               * False - Export only the image chips that overlap the labelled data. This is the default.

                                             - startIndex - Allows you to set the start index for the sequence of image chips.
                                               This lets you append more image chips to an existing sequence. The default value is 0.

                                             - cellSize - cell size can be set using this key in context parameter

                                             - extent - Sets the processing extent used by the function

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.(cellSize, extent)

                                             Example:

                                                {"exportAllTiles" : False, "startIndex": 0 }
    ------------------------------------     --------------------------------------------------------------------
    input_mask_polygons                      Optional :class:`~arcgis.features.FeatureLayer`. The feature layer that delineates the area where
                                             image chips will be created.
                                             Only image chips that fall completely within the polygons will be created.
    ------------------------------------     --------------------------------------------------------------------
    rotation_angle                           Optional float. The rotation angle that will be used to generate additional
                                             image chips.

                                             An image chip will be generated with a rotation angle of 0, which
                                             means no rotation. It will then be rotated at the specified angle to
                                             create an additional image chip. The same training samples will be
                                             captured at multiple angles in multiple image chips for data augmentation.
                                             The default rotation angle is 0.
    ------------------------------------     --------------------------------------------------------------------
    reference_system                         Optional string. Specifies the type of reference system to be used to interpret
                                             the input image. The reference system specified should match the reference system
                                             used to train the deep learning model.

                                             - MAP_SPACE : The input image is in a map-based coordinate system. This is the default.

                                             - IMAGE_SPACE : The input image is in image space, viewed from the direction of the sensor
                                               that captured the image, and rotated such that the tops of buildings and trees point upward in the image.

                                             - PIXEL_SPACE : The input image is in image space, with no rotation and no distortion.
    ------------------------------------     --------------------------------------------------------------------
    process_all_raster_items                 Optional bool. Specifies how all raster items in an image service will be processed.

                                             - False : all raster items in the image service will be mosaicked together and processed. This is the default.

                                             - True : all raster items in the image service will be processed as separate images.
    ------------------------------------     --------------------------------------------------------------------
    blacken_around_feature                   Optional bool. Specifies whether to blacken the pixels around each object or feature in each image tile.
                                             This parameter only applies when the metadata format is set to Labeled_Tiles and an input feature class or classified raster has been specified.

                                             - False : Pixels surrounding objects or features will not be blackened. This is the default.

                                             - True : Pixels surrounding objects or features will be blackened.

    ------------------------------------     --------------------------------------------------------------------
    fix_chip_size                            Optional bool. Specifies whether to crop the exported tiles such that they are all the same size.
                                             This parameter only applies when the metadata format is set to Labeled_Tiles and an input feature class or classified raster has been specified.

                                             - True : Exported tiles will be the same size and will center on the feature. This is the default.

                                             - False : Exported tiles will be cropped such that the bounding geometry surrounds only the feature in the tile.
    ------------------------------------     --------------------------------------------------------------------
    additional_input_raster                  Optional :class:`~arcgis.raster.ImageryLayer`/:class:`~arcgis.raster.Raster`/:class:`~arcgis.gis.Item`/String (URL).
                                             An additional input imagery source that will be used for image translation methods.

                                             This parameter is valid when the metadata_format parameter is set to Classified_Tiles, Export_Tiles, or CycleGAN.
    ------------------------------------     --------------------------------------------------------------------
    input_instance_data                      Optional. The training sample data collected that contains classes for instance segmentation.

                                             The input can also be a point feature without a class value field or an integer raster without any class information.

                                             This parameter is only valid when the metadata_format parameter is set to Panoptic_Segmentation.
    ------------------------------------     --------------------------------------------------------------------
    instance_class_value_field               Optional string. The field that contains the class values for instance segmentation.
                                             If no field is specified, the tool will use a value or class value field, if one is present.
                                             If the feature does not contain a class field, the tool will determine that all records belong to one class.

                                             This parameter is only valid when the metadata_format parameter is set to Panoptic_Segmentation.
    ------------------------------------     --------------------------------------------------------------------
    min_polygon_overlap_ratio                Optional float. The minimum overlap percentage for a feature to be included in the training data.
                                             If the percentage overlap is less than the value specified, the feature will be excluded from the
                                             training chip, and will not be added to the label file.

                                             The percent value is expressed as a decimal. For example, to specify an overlap of 20 percent,
                                             use a value of 0.2. The default value is 0, which means that all features will be included.

                                             This parameter improves the performance of the tool and also improves inferencing.
                                             The speed is improved since less training chips are created. Inferencing is improved
                                             since the model is trained to only detect large patches of objects and ignores small
                                             corners of features.

                                             This parameter is honoured only when the input_class_data parameter value is a feature service.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
    ====================================     ====================================================================

    :return:

        Output string containing the location of the exported training data

    """

    # task = "ExportTrainingDataforDeepLearning"

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError(
            "ArcGIS Online does not support export_training_data function."
        )

    return gis._tools.rasteranalysis.export_training_data_for_deep_learning(
        input_raster=input_raster,
        input_class_data=input_class_data,
        chip_format=chip_format,
        tile_size=tile_size,
        stride_size=stride_size,
        metadata_format=metadata_format,
        class_value_field=classvalue_field,
        buffer_radius=buffer_radius,
        output_location=output_location,
        input_mask_polygons=input_mask_polygons,
        rotation_angle=rotation_angle,
        reference_system=reference_system,
        process_all_raster_items=process_all_raster_items,
        blacken_around_feature=blacken_around_feature,
        fix_chip_size=fix_chip_size,
        additional_input_raster=additional_input_raster,
        input_instance_data=input_instance_data,
        instance_class_value_field=instance_class_value_field,
        min_polygon_overlap_ratio=min_polygon_overlap_ratio,
        context=context,
        future=future,
        **kwargs,
    )

    """
    url = gis.properties.helperServices.rasterAnalytics.url
    gptool = _arcgis.gis._GISResource(url, gis)

    params = {}

    if output_location:
        params["outputLocation"] = output_location
    else:
        raise RuntimeError("output_location cannot be None")

    if input_raster:
        params["inputRaster"] = _layer_input(input_raster)
    else:
        raise RuntimeError("input_raster cannot be None")

    if input_class_data:
        params["inputClassData"] = _layer_input(input_class_data)

    if chip_format is not None:
        chipFormatAllowedValues = ['TIFF', 'PNG', 'JPEG','MRF']
        if not chip_format in chipFormatAllowedValues:
            raise RuntimeError('chip_format can only be one of the following: '+ str(chipFormatAllowedValues))
        params["chipFormat"] = chip_format

    if tile_size:
        params["tileSize"] = tile_size

    if stride_size:
        params["strideSize"] = stride_size

    if metadata_format is not None:
        metadataFormatAllowedValues = ['KITTI_rectangles', 'PASCAL_VOC_rectangles', 'Classified_Tiles', 'RCNN_Masks', 'Labeled_Tiles']
        if not metadata_format in metadataFormatAllowedValues:
            raise RuntimeError('metadata_format can only be one of the following: '+ str(metadataFormatAllowedValues))

        params['metadataFormat'] = metadata_format

    if buffer_radius is not None:
        params["bufferRadius"]= buffer_radius

    if classvalue_field is not None:
        params["classValueField"]= classvalue_field

    if input_mask_polygons:
        params["inputMaskPolygons"] = _layer_input(input_mask_polygons)

    if rotation_angle:
        params["rotationAngle"] = rotation_angle

    _set_context(params, context)

    task_url, job_info, job_id = _analysis_job(gptool, task, params)

    job_info = _analysis_job_status(gptool, task_url, job_info)
    job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
    item_properties = {
        "properties": {
            "jobUrl": task_url + '/jobs/' + job_info['jobId'],
            "jobType": "GPServer",
            "jobId": job_info['jobId'],
            "jobStatus": "completed"
        }
    }
    return job_values["outLocation"]["uri"]

    """


def list_models(*, gis=None, future=False, **kwargs):
    """
    Function is used to list all the installed deep learning models.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server)

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------     --------------------------------------------------------------------
    future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
    ==================     ====================================================================

    :return:
        list of deep learning models installed

    """

    # task = "ListDeepLearningModels"

    gis = _arcgis.env.active_gis if gis is None else gis
    if gis._con._product == "AGOL":
        raise RuntimeError("ArcGIS Online does not support list_models function.")
    return gis._tools.rasteranalysis.list_deep_learning_models(future=future, **kwargs)
    """
    url = gis.properties.helperServices.rasterAnalytics.url
    gptool = _arcgis.gis._GISResource(url, gis)
    params = {}
    task_url, job_info, job_id = _analysis_job(gptool, task, params)

    job_info = _analysis_job_status(gptool, task_url, job_info)
    job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
    item_properties = {
        "properties": {
            "jobUrl": task_url + '/jobs/' + job_info['jobId'],
            "jobType": "GPServer",
            "jobId": job_info['jobId'],
            "jobStatus": "completed"
        }
    }

    output_model_list = []
    if isinstance(job_values["deepLearningModels"], list) and job_values["deepLearningModels"] is not None:
        for element in job_values["deepLearningModels"]:
            if isinstance(element,dict):
                if "id" in element.keys():
                    item = gis.content.get(element["id"])
                    output_model_list.append(Model(item))
    return output_model_list
    """


def classify_objects(
    input_raster,
    model,
    model_arguments=None,
    input_features=None,
    class_label_field=None,
    process_all_raster_items=False,
    output_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function can be used to output feature service with assigned class label for each feature based on
    information from overlapped imagery data using the designated deep learning model.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server) and ArcGIS Image for ArcGIS Online.

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_raster                             Required. raster layer that contains objects that needs to be classified.
    ------------------------------------     --------------------------------------------------------------------
    model                                    Required :class:`~arcgis.learn.Model` object.
    ------------------------------------     --------------------------------------------------------------------
    model_arguments                          Optional dictionary. Name-value pairs of arguments and their values that can be customized by the clients.

                                             eg: {"name1":"value1", "name2": "value2"}
    ------------------------------------     --------------------------------------------------------------------
    input_features                           Optional :class:`~arcgis.features.FeatureLayer`.
                                             The point, line, or polygon input feature layer that identifies the location of each object to be
                                             classified and labelled. Each row in the input feature layer represents a single object.

                                             If no input feature layer is specified, the function assumes that each input image contains a single object
                                             to be classified. If the input image or images use a spatial reference, the output from the function is a
                                             feature layer, where the extent of each image is used as the bounding geometry for each labelled
                                             feature layer. If the input image or images are not spatially referenced, the output from the function
                                             is a table containing the image ID values and the class labels for each image.
    ------------------------------------     --------------------------------------------------------------------
    class_label_field                        Optional str. The name of the field that will contain the classification label in the output feature layer.

                                             If no field name is specified, a new field called ClassLabel will be generated in the output feature layer.

                                             Example:
                                                "ClassLabel"
    ------------------------------------     --------------------------------------------------------------------
    process_all_raster_items                 Optional bool.

                                             - If set to False, all raster items in the image service will be mosaicked together and processed. This is the default.

                                             - If set to True, all raster items in the image service will be processed as separate images.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. If not provided, a :class:`~arcgis.features.FeatureLayer` is created by the method and used as the output .
                                             You can pass in an existing Feature Service Item from your GIS to use that instead.
                                             Alternatively, you can pass in the name of the output Feature Service that should be created by this method
                                             to be used as the output for the tool.
                                             A RuntimeError is raised if a service by that name already exists
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - cellSize - Set the output raster cell size, or resolution

                                             - extent - Sets the processing extent used by the function

                                             - parallelProcessingFactor - Sets the parallel processing factor. Default is "80%"

                                             - processorType - Sets the processor type. "CPU" or "GPU"

                                             Eg: {"processorType" : "CPU"}

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        The output feature layer item containing the classified objects

    """

    gis = _arcgis.env.active_gis if gis is None else gis
    return gis._tools.rasteranalysis.classify_objects_using_deep_learning(
        input_raster=input_raster,
        input_features=input_features,
        output_feature_class=output_name,
        model=model,
        model_arguments=model_arguments,
        class_label_field=class_label_field,
        process_all_raster_items=process_all_raster_items,
        context=context,
        future=future,
        **kwargs,
    )


def compute_accuracy_for_object_detection(
    detected_features,
    ground_truth_features,
    detected_class_value_field=None,
    ground_truth_class_value_field=None,
    min_iou=None,
    mask_features=None,
    out_accuracy_table_name=None,
    out_accuracy_report_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function can be used to calculate the accuracy of a deep learning model by comparing the detected objects from
    the detect_objects function to ground truth data.
    Function available in ArcGIS Image Server 10.9 and higher (not available in ArcGIS Online).

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    detected_features                        Required. The input polygon feature layer containing the objects
                                             detected from the detect_objects function.
    ------------------------------------     --------------------------------------------------------------------
    ground_truth_features                    Required. The polygon feature layer containing ground truth data.
    ------------------------------------     --------------------------------------------------------------------
    detected_class_value_field               Optional dictionary. The field in the detected objects feature class
                                             that contains the class names or class values.

                                             If a field name is not specified, a Classvalue or Value field will
                                             be used. If these fields do not exist, all records will be
                                             identified as belonging to one class.

                                             The class values or class names must match those in the ground truth feature class exactly.

                                             Syntax: A string describing the detected class value field.

                                             Example: "class"
    ------------------------------------     --------------------------------------------------------------------
    ground_truth_class_value_field           The field in the ground truth feature class that contains the class
                                             names or class values.

                                             If a field name is not specified, a Classvalue or Value field will
                                             be used. If these fields do not exist, all records will be
                                             identified as belonging to one class.

                                             The class values or class names must match those in the detected objects feature class exactly.

                                             Example: "class"
    ------------------------------------     --------------------------------------------------------------------
    min_iou                                  The Intersection over Union (IoU) ratio to use as a threshold to
                                             evaluate the accuracy of the object-detection model. The numerator
                                             is the area of overlap between the predicted bounding box and
                                             the ground truth bounding box. The denominator is the area of
                                             union or the area encompassed by both bounding boxes.

                                             min_IoU value should be in the range 0 to 1. [0,1]
                                             Example:
                                                0.5
    ------------------------------------     --------------------------------------------------------------------
    mask_features                            Optional :class:`~arcgis.features.FeatureLayer`. A polygon feature service layer that delineates
                                             the area where accuracy will be computed. Only the image area that
                                             falls completely within the polygons will be assessed for accuracy.
    ------------------------------------     --------------------------------------------------------------------
    out_accuracy_table_name                  Optional. Name of the output accuracy table item to be created.
                                             If not provided, a random name is generated by the method and used as
                                             the output name.
    ------------------------------------     --------------------------------------------------------------------
    out_accuracy_report_name                 Optional. Accuracy report can either be added as an item to the portal.
                                             or can be written to a datastore.
                                             To add as an item, specify the name of the output report item (pdf item)
                                             to be created.
                                             Example:

                                                "accuracyReport"

                                             In order to write accuracy report to datastore, specify the datastore path as value to uri key.

                                             Example -
                                                "/fileShares/yourFileShareFolderName/accuracyReport"
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - cellSize - Set the output raster cell size, or resolution

                                             - extent - Sets the processing extent used by the function

                                             - parallelProcessingFactor - Sets the parallel processing factor. Default is "80%"

                                             - processorType - Sets the processor type. "CPU" or "GPU"

                                             Eg: {"processorType" : "CPU"}

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        The output accuracy table item or/and accuracy report item (or datastore path to accuracy report)

    .. code-block:: python

        # Usage Example: This example generates an accuracy table for a specified minimum IoU value.

        compute_accuracy_op = compute_accuracy_for_object_detection(detected_features=detected_features,
                                                                    ground_truth_features=ground_truth_features,
                                                                    detected_class_value_field="ClassValue",
                                                                    ground_truth_class_value_field="Class",
                                                                    min_iou=0.5,
                                                                    mask_features=None,
                                                                    out_accuracy_table_name="accuracy_table",
                                                                    out_accuracy_report_name="accuracy_report",
                                                                    gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError(
            "compute_accuracyfor_object_detection() is not supported on ArcGIS Online"
        )

    return gis._tools.rasteranalysis.compute_accuracyfor_object_detection(
        detected_features=detected_features,
        ground_truth_features=ground_truth_features,
        detected_class_value_field=detected_class_value_field,
        ground_truth_class_value_field=ground_truth_class_value_field,
        min_iou=min_iou,
        mask_features=mask_features,
        out_accuracy_table_name=out_accuracy_table_name,
        out_accuracy_report_name=out_accuracy_report_name,
        context=context,
        future=future,
        **kwargs,
    )


def train_model(
    input_folder,
    model_type,
    model_arguments=None,
    batch_size=2,
    max_epochs=None,
    learning_rate=None,
    backbone_model=None,
    validation_percent=None,
    pretrained_model=None,
    stop_training=True,
    freeze_model=True,
    overwrite_model=False,
    output_name=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Function can be used to train a deep learning model using the output from the
    export_training_data function.
    It generates the deep learning model package (*.dlpk) and adds it to your enterprise portal.
    train_model function performs the training using the Raster Analytics server.

    .. note::
            This function is supported with ArcGIS Enterprise (Image Server)

    ====================================     ====================================================================
    **Parameter**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    input_folder                             Required string or list. This is the input location for the training sample data.
                                             It can be the path of output location on the file share raster data store or a
                                             shared file system path.
                                             The training sample data folder needs to be the output of export_training_data function,
                                             containing "images" and "labels" folder,
                                             as well as the JSON model definition file written out together by the function.

                                             File share raster store and datastore path examples:
                                               -  /rasterStores/yourRasterStoreFolderName/trainingSampleData
                                               - /fileShares/yourFileShareFolderName/trainingSampleData

                                             Shared path example:
                                               - \\serverName\deepLearning\trainingSampleData

                                             The function also support multiple input folders. In this case,
                                             specify the list of input folders


                                             list of file share raster store and datastore path examples:
                                               -  ["/rasterStores/yourRasterStoreFolderName/trainingSampleDataA", "/rasterStores/yourRasterStoreFolderName/trainingSampleDataB"]
                                               - ["/fileShares/yourFileShareFolderName/trainingSampleDataA", "/fileShares/yourFileShareFolderName/trainingSampleDataB"]

                                             list of shared path example:
                                               - ["\\serverName\deepLearning\trainingSampleDataA", "\\serverName\deepLearning\trainingSampleDataB"]

                                             Multiple input folders are supported when all the following conditions are met:

                                             - The metadata format must be one of the following types: Classified_Tiles, Labeled_Tiles, Multi-labeled Tiles, PASCAL_VOC_rectangles, or RCNN_Masks.
                                             - All training data must have the same metadata format.
                                             - All training data must have the same number of bands.
                                             - All training data must have the same tile size.
    ------------------------------------     --------------------------------------------------------------------
    model_type                               Required string. The model type to use for training the deep learning model.
                                             Possible values:

                                             - SSD - The Single Shot Detector (SSD) is used for object detection.
                                             - UNET - U-Net is used for pixel classification.
                                             - FEATURE_CLASSIFIER - The Feature Classifier is used for object classification.
                                             - PSPNET - The Pyramid Scene Parsing Network (PSPNET) is used for pixel classification.
                                             - RETINANET - The RetinaNet is used for object detection.
                                             - MASKRCNN - The MarkRCNN is used for object detection
                                             - YOLOV3 - The YOLOv3 approach will be used to train the model. YOLOv3 is used for object detection.
                                             - DeepLabV3 - The DeepLabV3 approach will be used to train the model. DeepLab is used for pixel classification.
                                             - FASTERRCNN - The FasterRCNN approach will be used to train the model. FasterRCNN is used for object detection.
                                             - BDCN_EDGEDETECTOR -  The Bi-Directional Cascade Network (BDCN) architecture will be used to train the model.
                                               The BDCN Edge Detector is used for pixel classification. This approach is useful to improve edge detection for objects at different scales.
                                             - HED_EDGEDETECTOR -  The Holistically-Nested Edge Detection (HED) architecture will be used to train the model.
                                               The HED Edge Detector is used for pixel classification. This approach is useful to in edge and object boundary detection.
                                             - MULTITASK_ROADEXTRACTOR -  The Multi Task Road Extractor architecture will be used to train the model.
                                               The Multi Task Road Extractor is used for pixel classification. This approach is useful for road network extraction from satellite imagery.
                                             - CONNECTNET - The ConnectNet architecture will be used to train the model. ConnectNet is used for pixel classification.
                                               This approach is useful for road network extraction from satellite imagery.
                                             - PIX2PIX - The Pix2Pix approach will be used to train the model. Pix2Pix is used for image-to-image translation.
                                               This approach creates a model object that generates images of one type to another. The input training data for this
                                               model type uses the Export Tiles metadata format.
                                             - CYCLEGAN - The CycleGAN approach will be used to train the model. CycleGAN is used for image-to-image translation.
                                               This approach creates a model object that generates images of one type to another. This approach is unique in that
                                               the images to be trained do not need to overlap. The input training data for this model type uses the CycleGAN metadata format.
                                             - SUPERRESOLUTION - The Super-resolution approach will be used to train the model. Super-resolution is used for
                                               image-to-image translation. This approach creates a model object that increases the resolution and improves the
                                               quality of images. The input training data for this model type uses the Export Tiles metadata format.
                                             - CHANGEDETECTOR - The Change detector approach will be used to train the model. Change detector is used for
                                               pixel classification. This approach creates a model object that uses two spatial-temporal images to create
                                               a classified raster of the change. The input training data for this model type uses the Classified Tiles metadata format.
                                             - IMAGECAPTIONER - The Image captioner approach will be used to train the model. Image captioner is used for
                                               image-to-text translation. This approach creates a model that generates text captions for an image.
                                             - SIAMMASK - The Siam Mask approach will be used to train the model. Siam Mask is used for object detection in videos.
                                               The model is trained using frames of the video and detects the classes and bounding boxes of the objects in each frame.
                                               The input training data for this model type uses the MaskRCNN metadata format.
                                             - MMDETECTION - The MMDetection approach will be used to train the model. MMDetection is used for object detection.
                                               The supported metadata formats are PASCAL Visual Object Class rectangles and KITTI rectangles.
                                             - MMSEGMENTATION - The MMSegmentation approach will be used to train the model. MMDetection is used for pixel classification.
                                               The supported metadata format is Classified Tiles.
                                             - DEEPSORT - The Deep Sort approach will be used to train the model. Deep Sort is used for object detection in videos.
                                               The model is trained using frames of the video and detects the classes and bounding boxes of the objects in each frame.
                                               The input training data for this model type uses the Imagenet metadata format.
                                               Where Siam Mask is useful while tracking an object, Deep Sort is useful in training a model to track multiple objects.
                                             - PIX2PIXHD - The Pix2PixHD approach will be used to train the model. Pix2PixHD is used for image-to-image translation.
                                               This approach creates a model object that generates images of one type to another.
                                               The input training data for this model type uses the Export Tiles metadata format.
                                             - MAXDEEPLAB - The MAXDEEPLAB approach will be used to train the model. It is used for Panoptic Segmentation.
    ------------------------------------     --------------------------------------------------------------------
    model_arguments                          Optional dictionary. Name-value pairs of arguments and their values that can be customized by the clients.

                                             Example:
                                                {"name1":"value1", "name2": "value2"}
    ------------------------------------     --------------------------------------------------------------------
    batch_size                               Optional int.
                                             The number of training samples to be processed for training at one time.
                                             If the server has a powerful GPU, this number can be increased to 16, 36, 64, and so on.

                                             Example:
                                                4
    ------------------------------------     --------------------------------------------------------------------
    max_epochs                               Optional int. The maximum number of epochs that the model should be trained.
                                             One epoch means the whole training dataset will be passed forward and backward
                                             through the deep neural network once.

                                             Example:
                                                20
    ------------------------------------     --------------------------------------------------------------------
    learning_rate                            Optional float.
                                             The rate at which the weights are updated during the training.
                                             It is a small positive value in the range between 0.0 and 1.0.
                                             If learning rate is set to 0, it will extract the optimal learning rate
                                             from the learning curve during the training process.

                                             Example:
                                                0.0
    ------------------------------------     --------------------------------------------------------------------
    backbone_model                           Optional string.
                                             Specifies the preconfigured neural network to be used as an architecture for training the new model.
                                             Possible values: DENSENET121 , DENSENET161 , DENSENET169 , DENSENET201 , MOBILENET_V2 ,
                                             RESNET18 , RESNET34 , RESNET50 , RESNET101 , RESNET152 , VGG11 , VGG11_BN , VGG13 ,
                                             VGG13_BN , VGG16 , VGG16_BN , VGG19 , VGG19_BN , DARKNET53 , REID_V1 , REID_V2

                                             Example:
                                                RESNET34
    ------------------------------------     --------------------------------------------------------------------
    validation_percent                       Optional float.
                                             The percentage (in %) of training sample data that will be used for validating the model.

                                             Example:
                                                10
    ------------------------------------     --------------------------------------------------------------------
    pretrained_model                         Optional dlpk portal item.

                                             The pretrained model to be used for fine tuning the new model.
                                             It is a deep learning model package (dlpk) portal item.
    ------------------------------------     --------------------------------------------------------------------
    stop_training                            Optional bool.
                                             Specifies whether early stopping will be implemented.

                                             - True - The model training will stop when the model is no longer improving,
                                               regardless of the maximum epochs specified. This is the default.
                                             - False - The model training will continue until the maximum epochs is reached.
    ------------------------------------     --------------------------------------------------------------------
    freeze_model                             Optional bool.
                                             Specifies whether to freeze the backbone layers in the pretrained model,
                                             so that the weights and biases in the backbone layers remain unchanged.

                                             - True - The predefined weights and biases will not be altered in the backboneModel.
                                               This is the default.
                                             - False - The weights and biases of the backboneModel may be altered to better
                                               fit your training samples. This may take more time to process but
                                               usually could get better results.
    ------------------------------------     --------------------------------------------------------------------
    overwrite_model                          Optional bool.
                                             Overwrites an existing deep learning model package (.dlpk) portal item with the same name.

                                             If the output_name parameter uses the file share data store path, this overwriteModel parameter is not applied.

                                             - True - The portal .dlpk item will be overwritten.
                                             - False - The portal .dlpk item will not be overwritten. This is the default.
    ------------------------------------     --------------------------------------------------------------------
    output_name                              Optional. trained deep learning model package can either be added as an item
                                             to the portal or can be written to a datastore.

                                             To add as an item, specify the name of the output deep learning model package (item)
                                             to be created.

                                             Example -
                                                "trainedModel"

                                             In order to write the dlpk to fileshare datastore, specify the datastore path.

                                             Example -
                                                "/fileShares/filesharename/folder"
    ------------------------------------     --------------------------------------------------------------------
    context                                  Optional dictionary. Context contains additional settings that affect task execution.
                                             Dictionary can contain value for following keys:

                                             - cellSize - Set the output raster cell size, or resolution
                                             - extent - Sets the processing extent used by the function
                                             - parallelProcessingFactor - Sets the parallel processing factor. Default is "80%"
                                             - processorType - Sets the processor type. "CPU" or "GPU"
                                             Example -
                                                {"processorType" : "CPU"}

                                             Setting context parameter will override the values set using arcgis.env
                                             variable for this particular function.
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
    ====================================     ====================================================================

    :return:
        Returns the dlpk portal item that has properties for title, type, filename, file, id and folderId.
    """

    gis = _arcgis.env.active_gis if gis is None else gis

    if gis._con._product == "AGOL":
        raise RuntimeError("ArcGIS Online does not support train_model function.")

    return gis._tools.rasteranalysis.train_deep_learning_model(
        in_folder=input_folder,
        output_name=output_name,
        model_type=model_type,
        arguments=model_arguments,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        backbone_model=backbone_model,
        validation_percent=validation_percent,
        pretrained_model=pretrained_model,
        stop_training=stop_training,
        freeze_model=freeze_model,
        overwrite_model=overwrite_model,
        context=context,
        future=future,
        **kwargs,
    )


def detect_change_using_deep_learning(
    from_raster,
    to_raster,
    model,
    output_classified_raster=None,
    model_arguments=None,
    context=None,
    *,
    gis=None,
    future=False,
    **kwargs,
):
    """
    Runs a trained deep learning model to detect change between two rasters.
    Function available in ArcGIS Image Server 11.1 and higher.

    ====================================     ====================================================================
    **Argument**                             **Description**
    ------------------------------------     --------------------------------------------------------------------
    from_raster                              Required ImageryLayer object. The previous raster to use for change detection.
    ------------------------------------     --------------------------------------------------------------------
    to_raster                                Required ImageryLayer object. The recent raster to use for change detection.
    ------------------------------------     --------------------------------------------------------------------
    model                                    Required. The deep learning model to be used for the change detection.
                                             It can be passed as a dlpk portal item, datastore path to the Esri Model Definition (EMD)
                                             file or the EMD JSON string.
    ------------------------------------     --------------------------------------------------------------------
    output_classified_raster                 Optional String. If not provided, an Image Service is created by the method and used as the output raster.
                                             You can pass in an existing Image Service Item from your GIS to use that instead.

                                             Alternatively, you can pass in the name of the output Image Service that should be created by this method to be
                                             used as the output for the tool.

                                             A RuntimeError is raised if a service by that name already exists.
    ------------------------------------     --------------------------------------------------------------------
    model_arguments                          Optional dictionary. Name-value pairs of arguments and their values that can be customized by the clients.

                                             eg: {"name1":"value1", "name2": "value2"}
    ------------------------------------     --------------------------------------------------------------------
    context                                  Context contains additional settings that affect task execution.

                                                context parameter overwrites values set through arcgis.env parameter

                                                This function has the following settings:

                                                - Cell size (cellSize) - Set the output raster cell size, or resolution

                                                - Output Spatial Reference (outSR): The output raster will be
                                                projected into the output spatial reference.

                                                Example:
                                                    {"outSR": {spatial reference}}

                                                - Extent (extent): A bounding box that defines the analysis area.

                                                Example:
                                                    {"extent": {"xmin": -122.68,
                                                    "ymin": 45.53,
                                                    "xmax": -122.45,
                                                    "ymax": 45.6,
                                                    "spatialReference": {"wkid": 4326}}}

                                                - Parallel Processing Factor (parallelProcessingFactor): controls
                                                Raster Processing (CPU) service instances.

                                                Example:
                                                    Syntax example with a specified number of processing instances:

                                                    {"parallelProcessingFactor": "2"}

                                                    Syntax example with a specified percentage of total
                                                    processing instances:

                                                    {"parallelProcessingFactor": "60%"}
    ------------------------------------     --------------------------------------------------------------------
    gis                                      Optional GIS. The GIS on which this tool runs. If not specified, the active GIS is used.
    ------------------------------------     --------------------------------------------------------------------
    future                                   Keyword only parameter. Optional Boolean. If True, the result will be a GPJob object and
                                             results will be returned asynchronously.
    ------------------------------------     --------------------------------------------------------------------
    folder                                   Keyword only parameter. Optional str or dict. Creates a folder in the portal, if it does
                                             not exist, with the given folder name and persists the output in this folder.
                                             The dictionary returned by the gis.content.create_folder() can also be passed in as input.

                                             Example:
                                                {'username': 'user1', 'id': '6a3b77c187514ef7873ba73338cf1af8', 'title': 'trial'}
    ====================================     ====================================================================

    :return: The output imagery layer item

    .. code-block:: python

        # Usage Example 1:

        from_raster = gis.content.search("from_raster", item_type="Imagery Layer")[0].layers[0]
        to_raster = gis.content.search("to_raster", item_type="Imagery Layer")[0].layers[0]
        change_detection_model = gis.content.search("my_detection_model")[0]

        detect_change_op = detect_change_using_deep_learning(from_raster=from_raster,
                                                             to_raster=to_raster,
                                                             model=change_detection_model,
                                                             gis=gis)

    """

    gis = _arcgis.env.active_gis if gis is None else gis

    return gis._tools.rasteranalysis.detect_change_using_deep_learning(
        from_raster=from_raster,
        to_raster=to_raster,
        model=model,
        output_classified_raster=output_classified_raster,
        model_arguments=model_arguments,
        context=context,
        future=future,
        **kwargs,
    )


class Model:
    def __init__(self, model=None):
        self._model_package = False
        if isinstance(model, _arcgis.gis.Item):
            self._model = _json.dumps({"itemId": model.itemid})
            self._model_package = True
            self.item = model

    def _repr_html_(self):
        if self._model_package:
            if hasattr(self, "item"):
                self.item._repr_html_()
        else:
            self.__repr__()

    def __repr__(self):
        if self._model_package:
            model = _json.loads(self._model)
            if "url" in model.keys():
                return "<Model:%s>" % self._model
            return "<Model Title:%s owner:%s>" % (self.item.title, self.item.owner)

        else:
            try:
                return "<Model:%s>" % self._model
            except:
                return "<empty Model>"

    def from_json(self, model):
        """
        Function is used to initialize Model object from model definition JSON

        .. code-block:: python

            # Usage example

            >>> model = Model()

            >>> model.from_json({"Framework" :"TensorFlow",
                                 "ModelConfiguration":"DeepLab",
                                 "InferenceFunction":"``[functions]System\\DeepLearning\\ImageClassifier.py``",
                                 "ModelFile":"``\\\\folder_path_of_pb_file\\frozen_inference_graph.pb``",
                                 "ExtractBands":[0,1,2],
                                 "ImageWidth":513,
                                 "ImageHeight":513,
                                 "Classes": [ { "Value":0, "Name":"Evergreen Forest", "Color":[0, 51, 0] },
                                              { "Value":1, "Name":"Grassland/Herbaceous", "Color":[241, 185, 137] },
                                              { "Value":2, "Name":"Bare Land", "Color":[236, 236, 0] },
                                              { "Value":3, "Name":"Open Water", "Color":[0, 0, 117] },
                                              { "Value":4, "Name":"Scrub/Shrub", "Color":[102, 102, 0] },
                                              { "Value":5, "Name":"Impervious Surface", "Color":[236, 236, 236] } ] })

        """
        if isinstance(model, dict):
            self._model = model
            self._model_package = False

    def from_model_path(self, model):
        """
        Function is used to initialize Model object from url of model package or path of model definition file

        .. code-block:: python

            # Usage Example #1:

            >>> model = Model()
            >>> model.from_model_path("https://xxxportal.esri.com/sharing/rest/content/items/<itemId>")

            # Usage Example #2:

            >>> model = Model()
            >>> model.from_model_path("\\\\sharedstorage\\sharefolder\\findtrees.emd")
        """
        if "http:" in model or "https:" in model:
            self._model = _json.dumps({"url": model})
            self._model_package = True
        else:
            self._model = _json.dumps({"uri": model})
            self._model_package = False

    def install(self, *, gis=None, future=False, **kwargs):
        """
        Function is used to install the uploaded model package (*.dlpk). Optionally after inferencing
        the necessary information using the model, the model can be uninstalled by uninstall_model()


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        gis                    Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
        ------------------     --------------------------------------------------------------------
        future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
        ==================     ====================================================================

        :return:
            Path where model is installed

        """
        if self._model_package is False:
            raise RuntimeError(
                "model object should be created from a portal item or a portal url"
            )

        if self._model is None:
            raise RuntimeError(
                "For install/uninstall model object should be created from a portal item or portal url"
            )

        # task = "InstallDeepLearningModel"

        gis = _arcgis.env.active_gis if gis is None else gis

        if gis._con._product == "AGOL":
            raise RuntimeError(
                "ArcGIS Online does not support install method on a Model Object. The Model object can be directly used with deep learning functions without installation."
            )
        return gis._tools.rasteranalysis.install_deep_learning_model(
            model_package=self._model, future=future, **kwargs
        )

        """
        url = gis.properties.helperServices.rasterAnalytics.url
        gptool = _arcgis.gis._GISResource(url, gis)

        params = {}

        if self._model is None:
            raise RuntimeError("For install/uninstall model object should be created from a portal item or portal url")
        else:
            params["modelPackage"] = self._model

        task_url, job_info, job_id = _analysis_job(gptool, task, params)

        job_info = _analysis_job_status(gptool, task_url, job_info)
        job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
        item_properties = {
            "properties": {
                "jobUrl": task_url + '/jobs/' + job_info['jobId'],
                "jobType": "GPServer",
                "jobId": job_info['jobId'],
                "jobStatus": "completed"
            }
        }

        return job_values["installSucceed"]
        """

    def query_info(self, *, gis=None, future=False, **kwargs):
        """
        Function is used to extract the deep learning model specific settings from the model package item or model definition file.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        gis                    Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
        ------------------     --------------------------------------------------------------------
        future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
        ==================     ====================================================================

        :return:
           The key model information in dictionary format that describes what the settings are essential for this type of deep learning model.
        """

        # task = "QueryDeepLearningModelInfo"

        gis = _arcgis.env.active_gis if gis is None else gis
        if self._model is None:
            raise RuntimeError("model cannot be None")

        return gis._tools.rasteranalysis.query_deep_learning_model_info(
            model=self._model, future=future, **kwargs
        )

        """
        url = gis.properties.helperServices.rasterAnalytics.url
        gptool = _arcgis.gis._GISResource(url, gis)

        params = {}

        if self._model is None:
            raise RuntimeError('model cannot be None')
        else:
            params["model"] = self._model

        task_url, job_info, job_id = _analysis_job(gptool, task, params)

        job_info = _analysis_job_status(gptool, task_url, job_info)
        job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
        item_properties = {
            "properties": {
                "jobUrl": task_url + '/jobs/' + job_info['jobId'],
                "jobType": "GPServer",
                "jobId": job_info['jobId'],
                "jobStatus": "completed"
            }
        }
        output = job_values["outModelInfo"]
        print(output)
        try:
            dict_output =  _json.loads(output["modelInfo"])
            return dict_output
        except:
            return output
        """

    def uninstall(self, *, gis=None, future=False, **kwargs):
        """
        Function is used to uninstall the uploaded model package that was installed using the install_model()
        This function will delete the named deep learning model from the server but not the portal item.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        gis                    Optional :class:`~arcgis.gis.GIS` . The GIS on which this tool runs. If not specified, the active GIS is used.
        ------------------     --------------------------------------------------------------------
        future                 Keyword only parameter. Optional boolean. If True, the result will be a GPJob object and results will be returned asynchronously.
        ==================     ====================================================================

        :return:
            itemId of the uninstalled model package item

        """
        if self._model_package is False:
            raise RuntimeError(
                "For install/uninstall model object should be created from a portal item or a portal url"
            )

        # task = "UninstallDeepLearningModel"

        gis = _arcgis.env.active_gis if gis is None else gis
        if gis._con._product == "AGOL":
            raise RuntimeError(
                "ArcGIS Online does not support uninstall method on a Model Object."
            )

        if self._model is None:
            raise RuntimeError("model_package cannot be None")

        return gis._tools.rasteranalysis.uninstall_deep_learning_model(
            model_item_id=self._model, future=future, **kwargs
        )

        """
        url = gis.properties.helperServices.rasterAnalytics.url
        gptool = _arcgis.gis._GISResource(url, gis)

        params = {}

        if self._model is None:
            raise RuntimeError('model_package cannot be None')
        else:
            params["modelItemId"] = self._model

        task_url, job_info, job_id = _analysis_job(gptool, task, params)

        job_info = _analysis_job_status(gptool, task_url, job_info)
        job_values = _analysis_job_results(gptool, task_url, job_info, job_id)
        item_properties = {
            "properties": {
                "jobUrl": task_url + '/jobs/' + job_info['jobId'],
                "jobType": "GPServer",
                "jobId": job_info['jobId'],
                "jobStatus": "completed"
            }
        }

        return job_values["uninstallSucceed"]
        """


def export_point_dataset(
    data_path,
    output_path,
    block_size=50.0,
    max_points=8192,
    extra_features=[],
    **kwargs,
):
    """
    Note:
    This function has been deprecated starting from `ArcGIS API for
    Python` version 1.9.0. Export data using `Prepare Point Cloud Training Data` tool available
    in 3D Analyst Extension from ArcGIS Pro 2.8 onwards.

    """

    from ._utils.pointcloud_data import prepare_las_data

    prepare_las_data(
        data_path, block_size, max_points, output_path, extra_features, **kwargs
    )
