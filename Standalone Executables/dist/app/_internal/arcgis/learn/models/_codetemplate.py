code = '''
import json

import os, importlib

import numpy as np
import arcpy


def get_available_device(max_memory=0.8):
    """
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    """
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


features = {
    "displayFieldName": "",
    "fieldAliases": {"FID": "FID", "Class": "Class", "Confidence": "Confidence"},
    "geometryType": "esriGeometryPolygon",
    "fields": [
        {"name": "FID", "type": "esriFieldTypeOID", "alias": "FID"},
        {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
        {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
    ],
    "features": [],
}

fields = {
    "fields": [
        {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
        {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"},
        {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
        {"name": "Shape", "type": "esriFieldTypeGeometry", "alias": "Shape"},
    ]
}


class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISObjectDetector:
    def __init__(self):
        self.name = "Object Detector"
        self.description = "This python raster function applies deep learning model to detect objects in imagery"

    def initialize(self, **kwargs):
        if "model" not in kwargs:
            return

        model = kwargs["model"]
        model_as_file = True
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info["Framework"]
        if "ModelConfiguration" in self.json_info:
            if isinstance(self.json_info["ModelConfiguration"], str):
                ChildModelDetector = getattr(
                    importlib.import_module(
                        "{}.{}".format(framework, self.json_info["ModelConfiguration"])
                    ),
                    "ChildObjectDetector",
                )
            else:
                ChildModelDetector = getattr(
                    importlib.import_module(
                        "{}.{}".format(
                            framework, self.json_info["ModelConfiguration"]["Name"]
                        )
                    ),
                    "ChildObjectDetector",
                )
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception(
                        "PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials"
                    )
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        parameter_info = self.child_object_detector.getParameterInfo(
            required_parameters
        )
        parameter_info.extend(
            [
                {
                    "name": "test_time_augmentation",
                    "dataType": "string",
                    "required": False,
                    "value": "False"
                    if "test_time_augmentation" not in self.json_info
                    else str(self.json_info["test_time_augmentation"]),
                    "displayName": "Perform test time augmentation while predicting",
                    "description": "If True, will merge predictions from flipped and rotated images.",
                }
            ]
        )
        return parameter_info

    def getConfiguration(self, **scalars):
        configuration = self.child_object_detector.getConfiguration(**scalars)
        if "DataRange" in self.json_info:
            configuration["dataRange"] = tuple(self.json_info["DataRange"])
        configuration["inheritProperties"] = 2 | 4 | 8
        configuration["inputMask"] = True
        self.use_tta = scalars.get("test_time_augmentation", "false").lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]
        self.nms_overlap = float(scalars.get("nms_overlap", 0.1))
        return configuration

    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels

        polygon_list, scores, classes = self.tta_detect_objects(**pixelBlocks)

        features["features"] = []
        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(polygon_list[i].shape[0]):
                rings[0].append([polygon_list[i][j][1], polygon_list[i][j][0]])

            features["features"].append(
                {
                    "attributes": {
                        "OID": i + 1,
                        "Class": self.json_info["Classes"][classes[i] - 1]["Name"],
                        "Confidence": scores[i],
                    },
                    "geometry": {"rings": rings},
                }
            )

        return {"output_vectors": json.dumps(features)}

    def tta_detect_objects(self, **pixelBlocks):
        import torch
        from fastai.vision.transform import dihedral_affine, rotate
        from fastai.vision import Image

        input_image = pixelBlocks["raster_pixels"].astype(np.float32)

        tile_size = input_image.shape[1]
        pad = self.child_object_detector.padding

        allboxes = torch.empty(0,4)
        allclasses = []
        allscores = torch.empty(0)

        boxes_list, scores_list, labels_list = [], [], []
        transforms = [0]

        if self.use_tta:
            if self.json_info["ImageSpaceUsed"] == "MAP_SPACE":
                transforms = list(range(8))
            else:
                transforms = [
                    0,
                    2,
                ]  # no vertical flips for pixel space (oriented imagery)

        for k in transforms:
            out = dihedral_affine(Image(torch.tensor(input_image.copy() / 256.0)), k)
            pixelBlocksCopy = pixelBlocks.copy()
            pixelBlocksCopy["raster_pixels"] = (out.data * 256).numpy()
            polygons, scores, classes = self.detect_objects(**pixelBlocksCopy)

            bboxes = self.get_img_bbox(tile_size, polygons, scores, classes)
            if bboxes is not None:
                fixed_img_bboxes = dihedral_affine(bboxes, k)
                if k == 5 or k == 6:
                    fixed_img_bboxes = rotate(fixed_img_bboxes, 180)

                allboxes = torch.cat([allboxes, (fixed_img_bboxes.data[0]+1) / 2.0])
                allclasses = allclasses + fixed_img_bboxes.data[1].tolist()
                allscores = np.concatenate([allscores, torch.tensor(scores) * 0.01])

                boxes_list.append((fixed_img_bboxes.data[0] + 1) / 2.0)
                scores_list.append(torch.tensor(scores) * 0.01)  # normalize to [0,1]
                labels_list.append(fixed_img_bboxes.data[1].tolist())

        try:
            from ensemble_boxes import weighted_boxes_fusion

            iou_thr = self.nms_overlap
            skip_box_thr = 0.0001

            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        except:
            import warnings

            warnings.warn("Unable to perform weighted boxes fusion... use NMS")
            boxes, scores, labels = np.array(allboxes), allscores, np.array(allclasses)

        bboxes = boxes * tile_size - pad
        polygons = self.convert_bounding_boxes_to_coord_list(bboxes)

        return polygons, np.array(scores * 100).astype(float), labels.astype(int)

    def get_img_bbox(self, tile_size, polygons, scores, classes):
        from fastai.vision import ImageBBox

        pad = self.child_object_detector.padding
        bboxes = []
        for i, polygon in enumerate(polygons):
            x1, y1 = np.around(polygon).astype(int)[0]
            x2, y2 = np.around(polygon).astype(int)[2]
            bboxes.append([x1 + pad, y1 + pad, x2 + pad, y2 + pad])
        n = len(bboxes)
        if n > 0:
            return ImageBBox.create(
                tile_size,
                tile_size,
                bboxes,
                labels=classes,
                classes=["Background"] + [x["Name"] for x in self.json_info["Classes"]],
            )
        else:
            return None

    def convert_bounding_boxes_to_coord_list(self, bounding_boxes):
        """
        convert bounding box numpy array to python list of point arrays
        :param bounding_boxes: numpy array of shape [n, 4]
        :return: python array of point numpy arrays, each point array is in shape [4,2]
        """
        num_bounding_boxes = bounding_boxes.shape[0]
        bounding_box_coord_list = []
        for i in range(num_bounding_boxes):
            coord_array = np.empty(shape=(4, 2), dtype=float)
            coord_array[0][0] = bounding_boxes[i][0]
            coord_array[0][1] = bounding_boxes[i][1]

            coord_array[1][0] = bounding_boxes[i][0]
            coord_array[1][1] = bounding_boxes[i][3]

            coord_array[2][0] = bounding_boxes[i][2]
            coord_array[2][1] = bounding_boxes[i][3]

            coord_array[3][0] = bounding_boxes[i][2]
            coord_array[3][1] = bounding_boxes[i][1]

            bounding_box_coord_list.append(coord_array)

        return bounding_box_coord_list

    def detect_objects(self, **pixelBlocks):
        polygon_list, scores, classes = self.child_object_detector.vectorize(
            **pixelBlocks
        )

        padding = self.child_object_detector.padding
        keep_polygon = []
        keep_scores = []
        keep_classes = []

        chip_sz = self.json_info["ImageHeight"]

        for idx, polygon in enumerate(polygon_list):
            centroid = polygon.mean(0)
            i, j = int(centroid[0]) // chip_sz, int(centroid[1]) // chip_sz
            x, y = int(centroid[0]) % chip_sz, int(centroid[1]) % chip_sz

            x1, y1 = polygon[0]
            x2, y2 = polygon[2]

            # fix polygon by removing padded regions
            polygon[:, 0] = polygon[:, 0] - (2 * i + 1) * padding
            polygon[:, 1] = polygon[:, 1] - (2 * j + 1) * padding

            X1, Y1, X2, Y2 = (
                i * chip_sz,
                j * chip_sz,
                (i + 1) * chip_sz,
                (j + 1) * chip_sz,
            )
            t = 2.0  # within 2 pixels of edge

            # if centroid not in center, reduce confidence
            # so box can be filtered out during NMS
            if (
                x < padding
                or x > chip_sz - padding
                or y < padding
                and y > chip_sz - padding
            ):

                scores[idx] = (self.child_object_detector.thres * 100) + scores[
                    idx
                ] * 0.01

            # if not excluded due to touching edge of tile
            if not (
                self.child_object_detector.filter_outer_padding_detections
                and any(
                    [
                        abs(X1 - x1) < t,
                        abs(X2 - x2) < t,
                        abs(Y1 - y1) < t,
                        abs(Y2 - y2) < t,
                    ]
                )
            ):  # touches edge
                keep_polygon.append(polygon)
                keep_scores.append(scores[idx])
                keep_classes.append(classes[idx])

        return keep_polygon, keep_scores, keep_classes


'''

feature_classifier_prf = """

import importlib
from importlib import reload, import_module
import json
import os
import sys
import arcpy
import numpy as np


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISObjectClassifier:
    def __init__(self):
        self.name = 'Object classifier'
        self.description = 'This python raster function applies deep learning model to classify objects from overlaid imagery'

    def initialize(self, **kwargs):

        if 'model' not in kwargs:
            return

        # Read esri model definition (emd) file
        model = kwargs['model']
        model_as_file = True

        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            modelconfig = self.json_info['ModelConfiguration']
            if isinstance(modelconfig, str):
                if modelconfig not in sys.modules:
                    ChildModelDetector = getattr(import_module(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
                else:
                    ChildModelDetector = getattr(reload(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
            else:
                modelconfig = self.json_info['ModelConfiguration']['Name']
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)


    def getParameterInfo(self):

        # PRF needs values of these parameters from gp tool user,
        # either from gp tool UI or emd (a json) file.
        required_parameters = [
            {
                # To support mini batch, it is required that Classify Objects Using Deep Learning geoprocessing Tool
                # passes down a stack of raster tiles to PRF for model inference, the keyword required here is 'rasters'.
                'name': 'rasters',
                'dataType': 'rasters',
                'value': None,
                'required': True,
                'displayName': "Rasters",
                'description': 'The collection of overlapping rasters to objects to be classified'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]

        if 'BatchSize' not in self.json_info:
             required_parameters.append(
                 {
                     'name': 'batch_size',
                     'dataType': 'numeric',
                     'required': False,
                     'value': 4,
                     'displayName': 'Batch Size',
                     'description': 'Batch Size'
                 }
             )

        return self.child_object_detector.getParameterInfo(required_parameters)


    def getConfiguration(self, **scalars):


        # The information PRF returns to the GP tool,
        # the information is either from emd or defined in getConfiguration method.

        configuration = self.child_object_detector.getConfiguration(**scalars)

        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])

        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True


        return configuration

    def getFields(self):

        fields = {
                'fields': [
                    {
                        'name': 'OID',
                        'type': 'esriFieldTypeOID',
                        'alias': 'OID'
                    },
                    {
                        'name': 'Class',
                        'type': 'esriFieldTypeString',
                        'alias': 'Class'
                    },
                    {
                        'name': 'Confidence',
                        'type': 'esriFieldTypeDouble',
                        'alias': 'Confidence'
                    },
                    {
                        'name': 'Shape',
                        'type': 'esriFieldTypeGeometry',
                        'alias': 'Shape'
                    }
                ]
            }
        fields['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )

        if "MetaDataMode" in self.json_info and self.json_info["MetaDataMode"] == "MultiLabeled_Tiles":
            for item in fields['fields']:
                if item['name'] == 'Confidence':
                    item['type'] = 'esriFieldTypeString'

        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):

        # set pixel values in invalid areas to 0
        rasters_mask = pixelBlocks['rasters_mask']
        rasters_pixels = pixelBlocks['rasters_pixels']

        for i in range(0, len(rasters_pixels)):
            rasters_pixels[i][np.where(rasters_mask[i] == 0)] = 0

        pixelBlocks['rasters_pixels'] = rasters_pixels

        polygon_list, scores, labels = self.child_object_detector.vectorize(**pixelBlocks)

        features['features'] = []

        features['fieldAliases'].update({
            'Label':'Label'
        })

        features['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )

        if "MetaDataMode" in self.json_info and self.json_info["MetaDataMode"] == "MultiLabeled_Tiles":
            for item in features['fields']:
                if item['name'] == 'Confidence':
                    item['type'] = 'esriFieldTypeString'

        for i in range(len(polygon_list)):

            rings = [[]]
            for j in range(len(polygon_list[i])):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Confidence': str(scores[i]),
                    'Label': labels[i],
                    'Classname': labels[i]
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}
"""
entity_recognizer_placeholder = """
print('not implemented')
"""

image_classifier_prf = """

import arcpy
import numpy as np
import json
import sys, os, importlib
import math


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

def chunk_it(image, tile_size):
    s = image.shape
    num_rows = math.ceil(s[0]/tile_size)
    num_cols = math.ceil(s[1]/tile_size)
    r = np.array_split(image, num_rows)
    rows = []
    for x in r:
        x = np.array_split(x, num_cols, axis=1)
        rows.append(x)
    return rows, num_rows, num_cols

def crop_center(img, pad):
    if pad == 0:
        return img
    return img[pad:-pad, pad: -pad, :]

def crop_flatten(chunked, pad):
    imgs = []
    for r, row  in enumerate(chunked):
        for c, col in enumerate(row):
            col = crop_center(col, pad)
            imgs.append(col)
    return imgs

def patch_chips(imgs, n_rows, n_cols):
    h_stacks = []
    for i in range(n_rows):
        h_stacks.append(np.hstack(imgs[i*n_cols:n_cols*(i+1) ]))
    return np.vstack(h_stacks)

attribute_table = {
    'displayFieldName': '',
    'fieldAliases': {
        'OID': 'OID',
        'Value': 'Value',
        'Class': 'Class',
        'Red': 'Red',
        'Green': 'Green',
        'Blue': 'Blue'
    },
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Value',
            'type': 'esriFieldTypeInteger',
            'alias': 'Value'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Red',
            'type': 'esriFieldTypeInteger',
            'alias': 'Red'
        },
        {
            'name': 'Green',
            'type': 'esriFieldTypeInteger',
            'alias': 'Green'
        },
        {
            'name': 'Blue',
            'type': 'esriFieldTypeInteger',
            'alias': 'Blue'
        }
    ],
    'features': []
}

 

class ArcGISImageClassifier:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a pytorch image classifier'

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")
        
        self.class_values = set([row['Value'] for row in self.json_info.get('Classes', [])])
        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
            else:
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        params = self.child_image_classifier.getParameterInfo(required_parameters)
        if 0 not in self.class_values and len(self.class_values)>1:
            for param in params:
                if param['name'] == 'predict_background':
                    param['value'] = 'False'
                    break
        return params

    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = 1
        #todo: type is determined by the value range of classes in the json file
        prob_raster = getattr(self.child_image_classifier,'probability_raster',False)
        if prob_raster:
            kwargs['output_info']['pixelType'] = 'f4' # To ensure that output pixels are in prob range 0 to 1
        else:
            kwargs['output_info']['pixelType'] = 'i4'
        class_info = self.json_info['Classes']
        attribute_table['features'] = []
        for i, c in enumerate(class_info):
            attribute_table['features'].append(
                {
                    'attributes':{
                        'OID':i+1,
                        'Value':c['Value'],
                        'Class':c['Name'],
                        'Red':c['Color'][0],
                        'Green':c['Color'][1],
                        'Blue':c['Color'][2]
                    }
                }
            )
        kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0
           
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        if self.json_info['ModelName'] == 'MultiTaskRoadExtractor':
            xx = self.child_image_classifier.detectRoads(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)   
            pixelBlocks['output_pixels'] = xx
        elif hasattr(self.child_image_classifier, 'updatePixelsTTA'):
            xx = self.child_image_classifier.updatePixelsTTA(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)   
            pixelBlocks['output_pixels'] = xx
        else:
            xx = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)   
            tytx = getattr(self.child_image_classifier, 'tytx', self.json_info['ImageHeight'])
            chunks, num_rows, num_cols =  chunk_it(xx.transpose(1, 2, 0), tytx)# self.json_info['ImageHeight'])  # ImageHeight = ImageWidth
            xx = patch_chips(crop_flatten(chunks, self.child_image_classifier.padding), num_rows, num_cols)
            xx = xx.transpose(2, 0, 1)
            pixelBlocks['output_pixels'] = xx

        return pixelBlocks
"""

instance_detector_prf = """
import json
import sys, os, importlib

import numpy as np
import math
import arcpy

def get_centroid(polygon):
    polygon = np.array(polygon)
    return [polygon[:, 0].mean(), polygon[:, 1].mean()]        

def check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding):
    return ((centroid[1] >= (start_y + padding)) and  \
                (centroid[1] <= (start_y + (chip_sz - padding))) and \
                (centroid[0] >= (start_x + padding)) and \
                (centroid[0] <= (start_x + (chip_sz - padding))))

def find_i_j(centroid, n_rows, n_cols, chip_sz, padding, filter_detections):
    for i in range(n_rows):
        for j in range(n_cols):
            start_x = i * chip_sz
            start_y = j * chip_sz

            if (centroid[1] > (start_y)) and (centroid[1] < (start_y + (chip_sz))) and (centroid[0] > (start_x)) and (centroid[0] < (start_x + (chip_sz))):
                in_center = check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding)
                if filter_detections:
                    if in_center: 
                        return i, j, in_center
                else:
                    return i, j, in_center
    return None        

def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4

class ArcGISInstanceDetector:
    def __init__(self):
        self.name = 'Instance Segmentation'
        self.description = 'Instance Segmentation python raster function to inference a arcgis.learn deep learning model.'

    def initialize(self, **kwargs):

        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildInstanceDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildInstanceDetector')
            else:
                ChildInstanceDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildInstanceDetector')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_instance_detector = ChildInstanceDetector()
        self.child_instance_detector.initialize(model, model_as_file)

        
    def getParameterInfo(self):       
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]     
        return self.child_instance_detector.getParameterInfo(required_parameters)


    def getConfiguration(self, **scalars):         
        configuration = self.child_instance_detector.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):          
        return GeometryType.Polygon        

    def vectorize(self, **pixelBlocks):
           
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        masks, pred_class, pred_score = self.child_instance_detector.vectorize(**pixelBlocks)

        n_rows = int(math.sqrt(self.child_instance_detector.batch_size))
        n_cols = int(math.sqrt(self.child_instance_detector.batch_size))
        padding = self.child_instance_detector.padding
        keep_masks = []
        keep_scores = []
        keep_classes = []       

        for idx, mask in enumerate(masks):
            if mask == []:
                continue
            centroid = get_centroid(mask[0])
            tytx = getattr(self.child_instance_detector, 'tytx', self.json_info['ImageHeight'])
            grid_location = find_i_j(centroid, n_rows, n_cols, tytx, padding, True)
            if grid_location is not None:
                i, j, in_center = grid_location
                for poly_id, polygon in enumerate(mask):
                    polygon = np.array(polygon)
                    polygon[:, 0] = polygon[:, 0] - (2*i + 1)*padding  # Inplace operation
                    polygon[:, 1] = polygon[:, 1] - (2*j + 1)*padding  # Inplace operation            
                    mask[poly_id] = polygon.tolist()
                if in_center:
                    keep_masks.append(mask)
                    keep_scores.append(pred_score[idx])
                    keep_classes.append(pred_class[idx])

        masks =  keep_masks
        pred_score = keep_scores
        pred_class = keep_classes        


        features['features'] = []

        for mask_idx, mask in enumerate(masks):

            features['features'].append({
                'attributes': {
                    'OID': mask_idx + 1,
                    'Class': self.json_info['Classes'][pred_class[mask_idx] - 1]['Name'],
                    'Confidence': pred_score[mask_idx]
                },
                'geometry': {
                    'rings': mask
                }
        }) 

        return {'output_vectors': json.dumps(features)}


"""
super_resolution = """

import arcpy
import numpy as np
import json
import sys, os, importlib
import math


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

def chunk_it(image, tile_size):
    s = image.shape
    num_rows = math.ceil(s[0]/tile_size)
    num_cols = math.ceil(s[1]/tile_size)
    r = np.array_split(image, num_rows)
    rows = []
    for x in r:
        x = np.array_split(x, num_cols, axis=1)
        rows.append(x)
    return rows, num_rows, num_cols

def crop_center(img, pad):
    if pad == 0:
        return img
    return img[pad:-pad, pad: -pad, :]

def crop_flatten(chunked, pad):
    imgs = []
    for r, row  in enumerate(chunked):
        for c, col in enumerate(row):
            col = crop_center(col, pad)
            imgs.append(col)
    return imgs

def patch_chips(imgs, n_rows, n_cols):
    h_stacks = []
    for i in range(n_rows):
        h_stacks.append(np.hstack(imgs[i*n_cols:n_cols*(i+1) ]))
    return np.vstack(h_stacks)
 

class ArcGISSuperResolution:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a pytorch image classifier'

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
            else:
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        return self.child_image_classifier.getParameterInfo(required_parameters)

    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        #configuration['inheritProperties'] = 2|4|8
        #configuration['inputMask'] = True
        return configuration

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = self.json_info.get("n_channel", 3)
        kwargs['output_info']['pixelType'] = 'f4'
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0
           
        #raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        #raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        if hasattr(self.child_image_classifier, "updatePixelsSmooth"):
            xx = self.child_image_classifier.updatePixelsSmooth( tlc, shape, props, **pixelBlocks).astype(props["pixelType"], copy=False)
            pixelBlocks["output_pixels"] = xx
        else:
            xx = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)
            chunks, num_rows, num_cols =  chunk_it(xx.transpose(1, 2, 0), self.json_info['ImageHeight'])  # ImageHeight = ImageWidth
            xx = patch_chips(crop_flatten(chunks, self.child_image_classifier.padding), num_rows, num_cols)
            xx = xx.transpose(2, 0, 1)
            pixelBlocks['output_pixels'] = xx

        return pixelBlocks

"""

image_translation_prf = """
import arcpy
import numpy as np
import json
import sys, os, importlib
import math

def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0
    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id
    return available
def chunk_it(image, tile_size):
    s = image.shape
    num_rows = math.ceil(s[0]/tile_size)
    num_cols = math.ceil(s[1]/tile_size)
    r = np.array_split(image, num_rows)
    rows = []
    for x in r:
        x = np.array_split(x, num_cols, axis=1)
        rows.append(x)
    return rows, num_rows, num_cols
def crop_center(img, pad):
    if pad == 0:
        return img
    return img[pad:-pad, pad: -pad, :]
def crop_flatten(chunked, pad):
    imgs = []
    for r, row  in enumerate(chunked):
        for c, col in enumerate(row):
            col = crop_center(col, pad)
            imgs.append(col)
    return imgs
def patch_chips(imgs, n_rows, n_cols):
    h_stacks = []
    for i in range(n_rows):
        h_stacks.append(np.hstack(imgs[i*n_cols:n_cols*(i+1) ]))
    return np.vstack(h_stacks)
 
class ArcGISImageTranslation:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a pytorch image classifier'
    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return
        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
            else:
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        else:
            raise Exception("Invalid model configuration")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()
        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"
        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)
    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        return self.child_image_classifier.getParameterInfo(required_parameters)
    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        #configuration['inheritProperties'] = 2|4|8
        #configuration['inputMask'] = True
        return configuration
    def updateRasterInfo(self, **kwargs):
        direction = getattr(self.child_image_classifier, "direction", "None")
        tar_nband = self.json_info.get("n_band_c", None)
        if direction == "BtoA":
            kwargs["output_info"]["bandCount"] = int(self.json_info["n_channel_rev"])
        elif tar_nband != None:
             kwargs["output_info"]["bandCount"] = int(tar_nband)
        else:
            kwargs["output_info"]["bandCount"] = int(self.json_info["n_channel"])
        kwargs['output_info']['pixelType'] = 'f4'
        return kwargs
    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0

        # raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks["raster_pixels"]
        # raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels
        if hasattr(self.child_image_classifier, "updatePixelsSmooth"):
            xx = self.child_image_classifier.updatePixelsSmooth(
                tlc, shape, props, **pixelBlocks
            ).astype(props["pixelType"], copy=False)
            pixelBlocks["output_pixels"] = xx
        else:
            xx = self.child_image_classifier.updatePixels(
                tlc, shape, props, **pixelBlocks
            ).astype(props["pixelType"], copy=False)
            tytx = getattr(
                self.child_image_classifier, "tytx", self.json_info["ImageHeight"]
            )
            chunks, num_rows, num_cols = chunk_it(
                xx.transpose(1, 2, 0), tytx
            )  # self.json_info['ImageHeight'])  # ImageHeight = ImageWidth
            xx = patch_chips(
                crop_flatten(chunks, self.child_image_classifier.padding),
                num_rows,
                num_cols,
            )
            xx = xx.transpose(2, 0, 1)
            pixelBlocks["output_pixels"] = xx
        return pixelBlocks
"""

imagets_classifier_prf = """
import arcpy
import numpy as np
import json
import sys, os, importlib
import math

def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0
    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id
    return available
def chunk_it(image, tile_size):
    s = image.shape
    num_rows = math.ceil(s[0] / tile_size)
    num_cols = math.ceil(s[1] / tile_size)
    r = np.array_split(image, num_rows)
    rows = []
    for x in r:
        x = np.array_split(x, num_cols, axis=1)
        rows.append(x)
    return rows, num_rows, num_cols
def crop_center(img, pad):
    if pad == 0:
        return img
    return img[pad:-pad, pad:-pad, :]
def crop_flatten(chunked, pad):
    imgs = []
    for r, row in enumerate(chunked):
        for c, col in enumerate(row):
            col = crop_center(col, pad)
            imgs.append(col)
    return imgs
def patch_chips(imgs, n_rows, n_cols):
    h_stacks = []
    for i in range(n_rows):
        h_stacks.append(np.hstack(imgs[i * n_cols : n_cols * (i + 1)]))
    return np.vstack(h_stacks)
attribute_table = {
    "displayFieldName": "",
    "fieldAliases": {
        "OID": "OID",
        "Value": "Value",
        "Class": "Class"
    },
    "fields": [
        {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
        {"name": "Value", "type": "esriFieldTypeInteger", "alias": "Value"},
        {"name": "Class", "type": "esriFieldTypeString", "alias": "Class"}
    ],
    "features": [],
}
class ArcGISImageTsClassifier:
    def __init__(self):
        self.name = "ImageTS Classifier"
        self.description = "Image TS classification python raster function to inference a pytorch time-series classifier"

    def initialize(self, **kwargs):
        if "model" not in kwargs:
            return
        model = kwargs["model"]
        model_as_file = True
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        framework = self.json_info["Framework"]
        if "ModelConfiguration" in self.json_info:
            if isinstance(self.json_info["ModelConfiguration"], str):
                ChildImageClassifier = getattr(
                    importlib.import_module(
                        "{}.{}".format(framework, self.json_info["ModelConfiguration"])
                    ),
                    "ChildImageClassifier",
                )
            else:
                ChildImageClassifier = getattr(
                    importlib.import_module(
                        "{}.{}".format(
                            framework, self.json_info["ModelConfiguration"]["Name"]
                        )
                    ),
                    "ChildImageClassifier",
                )
        else:
            raise Exception("Invalid model configuration")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
            if device == -2:
                device = get_available_device()
        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception(
                        "PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials"
                    )
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"
        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)
    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        return self.child_image_classifier.getParameterInfo(required_parameters)
    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if "DataRange" in self.json_info:
            configuration["dataRange"] = tuple(self.json_info["DataRange"])
        configuration["inheritProperties"] = 4 | 8
        configuration["invalidateProperties"] = 2 | 4 | 8
        configuration["inputMask"] = True
        return configuration
    def updateRasterInfo(self, **kwargs):
        kwargs["output_info"]["bandCount"] = 1
        kwargs["output_info"]["pixelType"] = "i4"
        class_info = self.json_info.get("Class_mapping", None)
        num_class_info = self.json_info.get("Num_class_mapping", None)
        if num_class_info == None:
            num_class_info = class_info
        key_vals, num_key_vals = list(class_info.items()), list(num_class_info.items())
        attribute_table["features"] = []
        for i, c in enumerate(class_info):
            attribute_table["features"].append(
                {
                    "attributes": {
                        "OID": i + 1,
                        "Value": num_key_vals[i][1],
                        "Class": key_vals[i][1],
                    }
                }
            )
        kwargs["output_info"]["rasterAttributeTable"] = json.dumps(attribute_table)
        return kwargs
    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels
        xx = self.child_image_classifier.updatePixels(
            tlc, shape, props, **pixelBlocks
        ).astype(props["pixelType"], copy=False)
        tytx = getattr(
            self.child_image_classifier, "tytx", self.json_info["ImageHeight"]
        )
        chunks, num_rows, num_cols = chunk_it(
            xx.transpose(1, 2, 0), tytx
        )  # self.json_info['ImageHeight'])  # ImageHeight = ImageWidth
        xx = patch_chips(
            crop_flatten(chunks, self.child_image_classifier.padding),
            num_rows,
            num_cols,
        )
        xx = xx.transpose(2, 0, 1)
        pixelBlocks["output_pixels"] = xx

        return pixelBlocks
"""

image_captioning_prf = """
import importlib
from importlib import reload, import_module
import json
import os
import sys

import numpy as np
import arcpy


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISImageCaptioner:
    def __init__(self):
        self.name = 'Image Captioner'
        self.description = 'This python raster function applies deep learning model to caption objects from overlaid imagery'

    def initialize(self, **kwargs):     

        if 'model' not in kwargs:
            return

        # Read esri model definition (emd) file
        model = kwargs['model']
        model_as_file = True

        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")


        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            modelconfig = self.json_info['ModelConfiguration']
            if isinstance(modelconfig, str):
                if modelconfig not in sys.modules:
                    ChildModelDetector = getattr(import_module(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
                else:
                    ChildModelDetector = getattr(reload(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
            else:
                modelconfig = self.json_info['ModelConfiguration']['Name']
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)


    def getParameterInfo(self):

        # PRF needs values of these parameters from gp tool user,
        # either from gp tool UI or emd (a json) file.
        required_parameters = [
            {
                # To support mini batch, it is required that Classify Objects Using Deep Learning geoprocessing Tool
                # passes down a stack of raster tiles to PRF for model inference, the keyword required here is 'rasters'.
                'name': 'rasters',
                'dataType': 'rasters',
                'value': None,
                'required': True,
                'displayName': "Rasters",
                'description': 'The collection of overlapping rasters to objects to be classified'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]

        if 'BatchSize' not in self.json_info:
             required_parameters.append(
                 {
                     'name': 'batch_size',
                     'dataType': 'numeric',
                     'required': False,
                     'value': 4,
                     'displayName': 'Batch Size',
                     'description': 'Batch Size'
                 }
             )

        return self.child_object_detector.getParameterInfo(required_parameters)


    def getConfiguration(self, **scalars):


        # The information PRF returns to the GP tool,
        # the information is either from emd or defined in getConfiguration method.

        configuration = self.child_object_detector.getConfiguration(**scalars)

        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])

        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True


        return configuration

    def getFields(self):

        fields = {
                'fields': [
                    {
                        'name': 'OID',
                        'type': 'esriFieldTypeOID',
                        'alias': 'OID'
                    },
                    {
                        'name': 'Shape',
                        'type': 'esriFieldTypeGeometry',
                        'alias': 'Shape'
                    }
                ]
            }
        fields['fields'].append(
            {
                'name': 'Caption',
                'type': 'esriFieldTypeString',
                'alias': 'Caption'
            }
        )            
        
        return json.dumps(fields)

    def getGeometryType(self):       
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):        
        # set pixel values in invalid areas to 0
        rasters_mask = pixelBlocks['rasters_mask']
        rasters_pixels = pixelBlocks['rasters_pixels']

        for i in range(0, len(rasters_pixels)):
            rasters_pixels[i][np.where(rasters_mask[i] == 0)] = 0

        pixelBlocks['rasters_pixels'] = rasters_pixels

        polygon_list, labels = self.child_object_detector.vectorize(**pixelBlocks)

        features['features'] = []

        features['fieldAliases'].update({
            'Caption':'Caption'
        })

        Labelfield = {
                'name': 'Caption',
                'type': 'esriFieldTypeString',
                'alias': 'Caption'
            }

        if not Labelfield in features['fields']:
            features['fields'].append(Labelfield)        


        for i in range(len(polygon_list)):

            rings = [[]]
            for j in range(len(polygon_list[i])):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Caption': labels[i],
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}
"""
panoptic_segmenter_prf = """
import json
import sys, os, importlib
sys.path.append(os.path.dirname(__file__))

import numpy as np
import math
import arcpy

# utility functions for object detection
def check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding):
    return ((centroid[1] >= (start_y + padding)) and  \
                (centroid[1] <= (start_y + (chip_sz - padding))) and \
                (centroid[0] >= (start_x + padding)) and \
                (centroid[0] <= (start_x + (chip_sz - padding))))

def find_i_j(centroid, n_rows, n_cols, chip_sz, padding, filter_detections):
    for i in range(n_rows):
        for j in range(n_cols):
            start_x = i * chip_sz
            start_y = j * chip_sz

            if (centroid[1] > (start_y)) and (centroid[1] < (start_y + (chip_sz))) and (centroid[0] > (start_x)) and (centroid[0] < (start_x + (chip_sz))):
                in_center = check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding)
                if filter_detections:
                    if in_center:
                        return i, j, in_center
                else:
                    return i, j, in_center
    return None

def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

attribute_table = {
    'displayFieldName': '',
    'fieldAliases': {
        'OID': 'OID',
        'Value': 'Value',
        'Class': 'Class',
        'Red': 'Red',
        'Green': 'Green',
        'Blue': 'Blue'
    },
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Value',
            'type': 'esriFieldTypeInteger',
            'alias': 'Value'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Red',
            'type': 'esriFieldTypeInteger',
            'alias': 'Red'
        },
        {
            'name': 'Green',
            'type': 'esriFieldTypeInteger',
            'alias': 'Green'
        },
        {
            'name': 'Blue',
            'type': 'esriFieldTypeInteger',
            'alias': 'Blue'
        }
    ],
    'features': []
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISPanopticSegmenter:
    def __init__(self):
        self.name = 'Panoptic Segmenter'
        self.description = 'This python raster function applies deep learning model to detect object instances in imagery and segment rasters by classes'

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        # sys.path.append(os.path.dirname(__file__))
        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            ChildPanopticSegmenter = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildPanopticSegmenter')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_panoptic_segmenter = ChildPanopticSegmenter()
        self.child_panoptic_segmenter.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        return self.child_panoptic_segmenter.getParameterInfo(required_parameters)

    def getConfiguration(self, **scalars):
        configuration = self.child_panoptic_segmenter.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        # set pixel values in invalid areas to 0
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        polygon_list, scores, classes = self.child_panoptic_segmenter.vectorize(**pixelBlocks)

        n_rows = int(math.sqrt(self.child_panoptic_segmenter.batch_size))
        n_cols = int(math.sqrt(self.child_panoptic_segmenter.batch_size))
        padding = self.child_panoptic_segmenter.padding
        keep_polygon = []
        keep_scores = []
        keep_classes = []

        for idx, polygon in enumerate(polygon_list):
            centroid = polygon.mean(0)
            quadrant = find_i_j(centroid, n_rows, n_cols, self.json_info['ImageHeight'], padding, self.child_panoptic_segmenter.filter_outer_padding_detections)
            if quadrant is not None:
                i, j, in_center = quadrant
                polygon[:, 0] = polygon[:, 0] - (2*i + 1)*padding
                polygon[:, 1] = polygon[:, 1] - (2*j + 1)*padding
                keep_polygon.append(polygon)
                if not in_center:
                    scores[idx] = (self.child_panoptic_segmenter.thres * 100) + scores[idx] * 0.01
                keep_scores.append(scores[idx])
                keep_classes.append(classes[idx])

        polygon_list =  keep_polygon
        scores = keep_scores
        classes = keep_classes
        features['features'] = []
        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(polygon_list[i].shape[0]):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Class': self.json_info['Classes'][classes[i] - 1]['Name'],
                    'Confidence': scores[i]
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = 1
        #todo: type is determined by the value range of classes in the json file
        prob_raster = getattr(self.child_panoptic_segmenter,'probability_raster',False)
        if prob_raster:
            kwargs['output_info']['pixelType'] = 'f4' # To ensure that output pixels are in prob range 0 to 1
        else:
            kwargs['output_info']['pixelType'] = 'i4'
        class_info = self.json_info['Classes']
        attribute_table['features'] = []
        for i, c in enumerate(class_info):
            attribute_table['features'].append(
                {
                    'attributes':{
                        'OID':i+1,
                        'Value':c['Value'],
                        'Class':c['Name'],
                        'Red':c['Color'][0],
                        'Green':c['Color'][1],
                        'Blue':c['Color'][2]
                    }
                }
            )
        kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0

        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        xx = self.child_panoptic_segmenter.updatePixelsTTA(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)
        pixelBlocks['output_pixels'] = xx

        return pixelBlocks

"""

ml_raster_prf = """
import arcpy
import numpy as np
import json
import sys, os, importlib
import math
 
class ArcGISImageClassifier:
    def __init__(self):
        self.name = 'Image Classifier'
        self.description = 'Image classification python raster function to inference a pytorch image classifier'

    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")
        
        self.class_values = set([row['Value'] for row in self.json_info.get('Classes', [])])
        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            if isinstance(self.json_info['ModelConfiguration'], str):
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration'])), 'ChildImageClassifier')
            else:
                ChildImageClassifier = getattr(importlib.import_module(
                    '{}.{}'.format(framework, self.json_info['ModelConfiguration']['Name'])), 'ChildImageClassifier')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_image_classifier = ChildImageClassifier()
        self.child_image_classifier.initialize(model, model_as_file)

    def getParameterInfo(self):
        required_parameters = [
            {
                'name': 'raster',
                'dataType': 'raster',
                'required': True,
                'displayName': 'Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]
        params = self.child_image_classifier.getParameterInfo(required_parameters)
        if 0 not in self.class_values and len(self.class_values)>1:
            for param in params:
                if param['name'] == 'predict_background':
                    param['value'] = 'False'
                    break
        return params

    def getConfiguration(self, **scalars):
        configuration = self.child_image_classifier.getConfiguration(**scalars)
        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])
        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True
        return configuration

    def updateRasterInfo(self, **kwargs):
        kwargs['output_info']['bandCount'] = 1
        #todo: type is determined by the value range of classes in the json file
        prob_raster = getattr(self.child_image_classifier,'probability_raster',False)
        # if prob_raster:
        kwargs['output_info']['pixelType'] = 'f4' # To ensure that output pixels are in prob range 0 to 1
        # else:
        #     kwargs['output_info']['pixelType'] = 'i4'
        # kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)
        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # set pixel values in invalid areas to 0
           
        raster_mask = pixelBlocks['raster_mask']
        raster_pixels = pixelBlocks['raster_pixels']
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels

        xx = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)   

        pixelBlocks['output_pixels'] = xx

        return pixelBlocks
"""
