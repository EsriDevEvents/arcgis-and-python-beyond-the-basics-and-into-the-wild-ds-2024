try:
    import torch
    import math
    import numpy as np
    import gc
    import arcgis as ag
    from collections import namedtuple
    from IPython.display import clear_output

except Exception as e:
    print(e)

object_detection_models = ["FasterRCNN", "MMDetection", "MMSegmentation", "DETReg"]
pixel_classification_models = ["MaskRCNN"]
image_translation_models = [
    "Pix2Pix",
    "Pix2PixHD",
    "CycleGAN",
    "ChangeDetector",
    "WNet_cGAN",
]
point_cloud_models = ["PointCNN", "RandLANet", "SQNSeg"]
image_captioner_models = ["ImageCaptioner"]
exception_models = [
    "MultiTaskRoadExtractor",
    "ChangeDetector",
    "SuperResolution",
    "MaXDeepLab",
    "CycleGAN",
    "ConnectNet",
    "Pix2PixHD",
]

unsupported_models = [
    "FullyConnectedNetwork",
    "MLModel",
    "TimeSeriesModel",
    "SiamMask",
    "Track",
    "Embeddings",
    "AutoML",
    "AutoDL",
    "ImageryModel",
    "EfficientDet",
    "PSETAE",
    "EntityRecognizer",
    "SequenceToSequence",
    "QuestionAnswering",
    "FillMask",
    "TextSummarizer",
    "TextGenerator",
    "TextTranslator",
    "ZeroShotClassifier",
    "_SpacyEntityRecognizer",
    "_TransformerEntityRecognizer",
    "TextClassifier",
    "MMDetection3D",
]


def estimate_batch_size(model, mode="train", **kwargs):
    """
    Function to calculate estimated batch size based on GPU capacity, size of model and data.

    =====================   ===========================================
    **Parameter**           **Description**
    ---------------------   -------------------------------------------
    model                   Required arcgis.learn imagery model. Model
                            instance for which batch size should be estimated.
                            Not supported for text, tabular, timeseries
                            or tracking models such as FullyConnectedNetwork,
                            MLModel, TimeSeriesModel, SiamMask, PSETAE
                            and EfficientDet models.
    ---------------------   -------------------------------------------
    mode                    Optional string. Default train. The mode for
                            which batch size is estimated. Supported 'train'
                            and 'eval' mode for calculating batch size in
                            training mode and evaluation mode respectively.
                            Note: max_batchsize is capped at 1024 for train
                            and eval mode and recommended_batchsize is
                            capped at 64 for train mode.
    =====================   ===========================================

    :return: Named tuple of recommended_batchsize and max_batchsize
    """

    mode = mode.lower()
    exception = None
    channel = 3
    verbose = kwargs.get("verbose", True)

    if model.__class__.__name__ in unsupported_models:
        raise Exception("unsupported model {}".format(model.__class__.__name__))

    if hasattr(model._data, "_is_multispectral") and model._data._is_multispectral:
        channel = len(model._data._band_max_values)

    height, width = model._data.chip_size, model._data.chip_size
    if (
        hasattr(model._data, "arcgis_init_kwargs")
        and model._data.arcgis_init_kwargs["resize_to"] is not None
    ):
        if isinstance(model._data.arcgis_init_kwargs["resize_to"], int):
            height, width = (
                model._data.arcgis_init_kwargs["resize_to"],
                model._data.arcgis_init_kwargs["resize_to"],
            )
        else:
            height, width = model._data.arcgis_init_kwargs["resize_to"]

    if mode == "train" or mode == "none":
        train_ds_val = math.floor(len(model._data.train_ds) * 0.2)
        max_batchsize = int(math.pow(2, (math.log(train_ds_val) // math.log(2))))
    elif mode == "eval":
        max_batchsize = 1024
    else:
        raise Exception("please select proper mode")
    breakwhile = False

    try:
        while not breakwhile:
            try:
                if mode == "train":
                    model._data.train_dl.batch_size = max_batchsize
                    x, y = model._data.one_batch(detach=False)
                    if model.__class__.__name__ in object_detection_models:
                        input_data = model._model_conf.on_batch_begin(
                            model.learn,
                            x.to(model._device),
                            [data.to(model._device) for data in y],
                        )
                        if model.__class__.__name__ == "DETReg":
                            out = model.learn.model(input_data[0])
                        elif model.__class__.__name__ == "MMSegmentation":
                            out = model.learn.model(
                                input_data[0][0],
                                input_data[0][1],
                                torch.stack(input_data[0][2]),
                            )
                        else:
                            out = model.learn.model(*input_data[0])
                    elif model.__class__.__name__ in pixel_classification_models:
                        z = model.learn.train_callback.on_batch_begin(x, y)
                        out = model.learn.model(*z["last_input"])
                    elif model.__class__.__name__ in image_translation_models:
                        if model.__class__.__name__ == "WNet_cGAN":
                            out = model.learn.model(
                                x[0].to(model._device),
                                x[0].to(model._device),
                                x[0].to(model._device),
                            )
                        elif model.__class__.__name__ == "Pix2PixHD":
                            from ..models._pix2pix_hd_utils import encode_input

                            x[0], _, x[1], _ = encode_input(
                                x[0], label_nc=model._data.label_nc, real_image=x[1]
                            )
                            model.learn.model.set_input(x)
                            model.learn.loss_func.set_input(x)
                            out = model.learn.model(
                                x[0].to(model._device), x[1].to(model._device)
                            )
                        else:
                            out = model.learn.model(
                                x[0].to(model._device), x[1].to(model._device)
                            )
                    elif model.__class__.__name__ in point_cloud_models:
                        out = model.learn.model(x[0].to(model._device))
                    elif model.__class__.__name__ in image_captioner_models:
                        out = model.learn.model(
                            x.to(model._device), [data.to(model._device) for data in y]
                        )
                    else:
                        out = model.learn.model(x.to(model._device))
                    del out

                elif mode == "eval":
                    if model.__class__.__name__ in point_cloud_models:
                        height = model.sample_point_num
                        channel = model._data.extra_dim + 3
                        blank_img = np.ones(
                            (
                                max_batchsize,
                                height,
                                channel,
                            ),
                            np.uint8,
                        )
                    else:
                        blank_img = np.ones(
                            (
                                max_batchsize,
                                channel,
                                height,
                                width,
                            ),
                            np.uint8,
                        )
                    tblank_img = torch.Tensor(blank_img).to(model._device)
                    eval_model = model.learn.model.to(model._device)
                    eval_model.eval()
                    with torch.no_grad():
                        if model.__class__.__name__ in object_detection_models:
                            eval_model(model._model_conf.transform_input(tblank_img))
                        elif model.__class__.__name__ in image_translation_models:
                            if model.__class__.__name__ == "WNet_cGAN":
                                eval_model(tblank_img, tblank_img, tblank_img)
                            else:
                                eval_model(tblank_img, tblank_img)
                        elif model.__class__.__name__ in image_captioner_models:
                            eval_model.sample(tblank_img)
                        else:
                            eval_model(tblank_img)

                elif mode == "none":
                    model._data.train_dl.batch_size = max_batchsize
                    model._data.valid_dl.batch_size = max_batchsize
                    x, y = model._data.one_batch(detach=False)
                    if "model" in model._model_kwargs:
                        nonemodel = getattr(ag.learn, model.__class__.__name__)(
                            model._data, model=model._model_kwargs["model"]
                        )
                    else:
                        nonemodel = getattr(ag.learn, model.__class__.__name__)(
                            model._data
                        )

                    if nonemodel.__class__.__name__ in object_detection_models:
                        input_data = nonemodel._model_conf.on_batch_begin(
                            nonemodel.learn,
                            x.to(nonemodel._device),
                            [data.to(nonemodel._device) for data in y],
                        )
                        if model.__class__.__name__ == "DETReg":
                            out = nonemodel.learn.model(input_data[0])
                        elif model.__class__.__name__ == "MMSegmentation":
                            out = nonemodel.learn.model(
                                input_data[0][0],
                                input_data[0][1],
                                torch.stack(input_data[0][2]),
                            )
                        else:
                            out = nonemodel.learn.model(*input_data[0])
                    elif nonemodel.__class__.__name__ in pixel_classification_models:
                        z = nonemodel.learn.train_callback.on_batch_begin(x, y)
                        out = nonemodel.learn.model(*z["last_input"])
                    elif nonemodel.__class__.__name__ in image_translation_models:
                        if model.__class__.__name__ == "WNet_cGAN":
                            out = nonemodel.learn.model(
                                x[0].to(nonemodel._device),
                                x[0].to(nonemodel._device),
                                x[0].to(nonemodel._device),
                            )
                        elif model.__class__.__name__ == "Pix2PixHD":
                            from ..models._pix2pix_hd_utils import encode_input

                            x[0], _, x[1], _ = encode_input(
                                x[0], label_nc=nonemodel._data.label_nc, real_image=x[1]
                            )
                            nonemodel.learn.model.set_input(x)
                            nonemodel.learn.loss_func.set_input(x)
                            out = nonemodel.learn.model(
                                x[0].to(nonemodel._device), x[1].to(nonemodel._device)
                            )
                        else:
                            out = nonemodel.learn.model(
                                x[0].to(nonemodel._device), x[1].to(nonemodel._device)
                            )
                    elif model.__class__.__name__ in point_cloud_models:
                        out = nonemodel.learn.model(x[0].to(nonemodel._device))
                    elif model.__class__.__name__ in image_captioner_models:
                        out = nonemodel.learn.model(
                            x.to(nonemodel._device),
                            [data.to(nonemodel._device) for data in y],
                        )
                    else:
                        out = nonemodel.learn.model(x.to(nonemodel._device))

                    del nonemodel, out

                gc.collect()
                torch.cuda.empty_cache()
                breakwhile = True

            except Exception as E:
                if (
                    "CUDA out of memory" in str(E)
                    or "non-contiguous" in str(E)
                    or "INTERNAL ASSERT FAILED" in str(E)
                ):
                    if verbose:
                        print("Out of memory with batch size:", max_batchsize)

                    gc.collect()
                    torch.cuda.empty_cache()
                    if max_batchsize > 2:
                        max_batchsize = int(max_batchsize // 2)
                        continue
                    else:
                        raise Exception(E)
                else:
                    exception = str(E)
                    breakwhile = True
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    except Exception as e:
        exception = str(e)

    if exception is not None:
        raise Exception(exception)

    output = namedtuple("batch_size", ["recommended_batchsize", "max_batchsize"])
    if model.__class__.__name__ in exception_models:
        max_batchsize = max_batchsize // 2
        if (
            model.__class__.__name__ == "MaXDeepLab"
            or model.__class__.__name__ == "Pix2PixHD"
            or model.__class__.__name__ == "ChangeDetector"
        ):
            max_batchsize = max_batchsize // 2

    if mode == "train" or mode == "none":
        model._data.train_dl.batch_size = max_batchsize

    if (mode == "train" or mode == "none") and max_batchsize > 64:
        batch_size = output(64, max_batchsize)
    else:
        if mode == "eval":
            max_batchsize = max_batchsize // 2
            batch_size = output(max_batchsize, max_batchsize)
        else:
            batch_size = output(max_batchsize, max_batchsize)

    clear_output(wait=True)

    return batch_size
