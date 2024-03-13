from pathlib import Path

from arcgis.learn._data import _bb_pad_collate, prepare_data
from arcgis.learn._data_utils._base_data import ArcgisData
from arcgis.learn._data_utils._pixel_classifier_data import ClassifiedTilesData

__data_classes__ = {"Classified_Tiles": ClassifiedTilesData}


class ArcgisDataFactory:
    def __init__(
        self,
        path,
        class_mapping=None,
        val_split_pct=0.1,
        batch_size=64,
        transforms=None,
        collate_fn=_bb_pad_collate,
        seed=42,
        **kwargs
    ):
        self.path = path
        self.class_mapping = class_mapping
        self.val_split_pct = val_split_pct
        self.batch_size = batch_size
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.seed = seed
        self.kwargs = kwargs

    def create(self, dataset_type):
        if dataset_type in __data_classes__.keys():
            return __data_classes__[dataset_type](
                path=self.path,
                class_mapping=self.class_mapping,
                val_split_pct=self.val_split_pct,
                batch_size=self.batch_size,
                transforms=self.transforms,
                seed=self.seed,
                dataset_type=dataset_type,
                **self.kwargs,
            )
        else:
            return None


def prepare_data_future(
    path,
    class_mapping=None,
    chip_size=224,
    val_split_pct=0.1,
    batch_size=64,
    transforms=None,
    collate_fn=_bb_pad_collate,
    seed=42,
    dataset_type=None,
    resize_to=None,
    **kwargs
):
    """
    Prepares a data object from training sample exported by the
    Export Training Data tool in ArcGIS Pro or Image Server, or training
    samples in the supported dataset formats. This data object consists of
    training and validation data sets with the specified transformations,
    chip size, batch size, split percentage, etc.
    -For object detection, use Pascal_VOC_rectangles or KITTI_rectangles format.
    -For feature categorization use Labelled Tiles or ImageNet format.
    -For pixel classification, use Classified Tiles format.
    -For entity extraction from text, use IOB, BILUO or ner_json formats.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    path                    Required string. Path to data directory.
    ---------------------   -------------------------------------------
    class_mapping           Optional dictionary. Mapping from id to
                            its string label.
                            For dataset_type=IOB, BILUO or ner_json:
                                Provide address field as class mapping
                                in below format:
                                class_mapping={'address_tag':'address_field'}.
                                Field defined as 'address_tag' will be treated
                                as a location. In cases where trained model extracts
                                multiple locations from a single document, that
                                document will be replicated for each location.

    ---------------------   -------------------------------------------
    chip_size               Optional integer, default 224. Size of the image to train the
                            model. Images are cropped to the specified chip_size. If image size is less
                            than chip_size, the image size is used as chip_size.
    ---------------------   -------------------------------------------
    val_split_pct           Optional float. Percentage of training data to keep
                            as validation.
    ---------------------   -------------------------------------------
    batch_size              Optional integer. Batch size for mini batch gradient
                            descent (Reduce it if getting CUDA Out of Memory
                            Errors).
    ---------------------   -------------------------------------------
    transforms              Optional tuple. Fast.ai transforms for data
                            augmentation of training and validation datasets
                            respectively (We have set good defaults which work
                            for satellite imagery well). If transforms is set
                            to `False` no transformation will take place and
                            `chip_size` parameter will also not take effect.
                            If the dataset_type is 'PointCloud', use
                            :class:`~arcgis.learn.Transform3d` .
    ---------------------   -------------------------------------------
    collate_fn              Optional function. Passed to PyTorch to collate data
                            into batches(usually default works).
    ---------------------   -------------------------------------------
    seed                    Optional integer. Random seed for reproducible
                            train-validation split.
    ---------------------   -------------------------------------------
    dataset_type            Optional string. :meth:`~arcgis.learn.prepare_data`  function will infer
                            the `dataset_type` on its own if it contains a
                            map.txt file. If the path does not contain the
                            map.txt file pass either of 'PASCAL_VOC_rectangles',
                            'KITTI_rectangles', 'RCNN_Masks', 'Classified_Tiles',
                            'Labeled_Tiles', 'Imagenet', 'PointCloud' and
                            'ImageCaptioning'.
    ---------------------   -------------------------------------------
    resize_to               Optional integer. Resize the image to given size.
    =====================   ===========================================

    **Keyword Arguments**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    imagery_type            Optional string. Type of imagery used to export
                            the training data, valid values are:
                                - 'naip'
                                - 'sentinel2'
                                - 'landsat8'
                                - 'ms' - any other type of imagery
    ---------------------   -------------------------------------------
    bands                   Optional list. Bands of the imagery used to export
                            training data.
                            For example ['r', 'g', 'b', 'nir', 'u']
                            where 'nir' is near infrared band and 'u' is a miscellaneous band.
    ---------------------   -------------------------------------------
    rgb_bands               Optional list. Indices of red, green and blue bands
                            in the imagery used to export the training data.
                            for example: [2, 1, 0]
    ---------------------   -------------------------------------------
    extract_bands           Optional list. Indices of bands to be used for
                            training the model, same as in the imagery used to
                            export the training data.
                            for example: [3, 1, 0] where we will not be using
                            the band at index 2 to train our model.
    ---------------------   -------------------------------------------
    norm_pct                Optional float. Percentage of training data to be
                            used for calculating imagery statistics for
                            normalizing the data.
                            Default is 0.3 (30%) of data.
    ---------------------   -------------------------------------------
    downsample_factor       Optional integer. Factor to downsample the images
                            for image SuperResolution.
                            for example: if value is 2 and image size 256x256,
                            it will create label images of size 128x128.
                            Default is 4
    ---------------------   -------------------------------------------
    sub_dataset_type        Optional string. Special sub dataset type required
                            for creating the Orientation data to use for
                            training Multi-Task Road Extractor. Currently support
                            Classified Tiles dataset format.
                            Value : "RoadOrientation"
    ---------------------   -------------------------------------------
    road_extractor_params   Optional dict. Parameters to use in Multi-Task
                                            road extractor.
                            - gaussian_thresh: set the gaussian threshold which
                                                allows to set the required road width
                            - orient_bin_size: set the bin size for orientation angles
                            - orient_theta: set the width of orientation mask
                            - is_gaussian_mask: flag to set, if given labels are converted
                                                to gaussian. if false, system will create the
                                                gaussian mask internally and convert to binary
                                                mask of appropriate road width based on
                                                the gaussian_thresh
    =====================   ===========================================

    :return: data object
    """
    has_esri_files = ArcgisData._check_esri_files(Path(path))
    if has_esri_files and dataset_type is None:
        dataset_type = ArcgisData._get_dataset_type(Path(path))
    if has_esri_files and dataset_type in __data_classes__.keys():
        dataset_type = (
            dataset_type if dataset_type else ArcgisData._get_dataset_type(Path(path))
        )
        objData = __data_classes__[dataset_type](
            path=path,
            class_mapping=class_mapping,
            chip_size=chip_size,
            val_split_pct=val_split_pct,
            batch_size=batch_size,
            transforms=transforms,
            seed=seed,
            dataset_type=dataset_type,
            resize_to=resize_to,
            **kwargs,
        )
        return objData.get_databunch(**kwargs)
    else:
        return prepare_data(
            path,
            class_mapping,
            chip_size,
            val_split_pct,
            batch_size,
            transforms,
            collate_fn,
            seed,
            dataset_type,
            resize_to,
            **kwargs,
        )
