from enum import unique
from pathlib import Path
import json
from .._data import prepare_data

import numpy
from ._model_extension import ModelExtension
from ._arcgis_model import _EmptyData
import logging

from .._mmseg_config.prithvi100m_burn_scar import img_norm_burn_model
from .._mmseg_config.prithvi100m_crop_classification import img_norm_crop_model
from .._mmseg_config.prithvi100m_sen1floods import img_norm_flood_model

logger = logging.getLogger()

try:
    from fastai.vision import flatten_model
    import torch
    from fastai.torch_core import split_model_idx
    from .._utils.common import get_multispectral_data_params_from_emd, _get_emd_path

    HAS_FASTAI = True

except Exception as e:
    HAS_FASTAI = False


class MMSegmentationConfig:
    """
    Create class with following fixed function names and the number of arguents to train your model from external source
    """

    try:
        import torch
        import numpy
        import pathlib
        import os
        import types
    except:
        pass

    def get_model(self, data, backbone=None, **kwargs):
        """
        In this fuction you have to define your model with following two arguments!

        """

        import mmseg.models
        import mmcv
        import logging

        logging.disable(logging.WARNING)

        config = kwargs.get("model", False)
        checkpoint = kwargs.get("model_weight", False)
        class_weight = kwargs.get("class_weight", None)
        if config[-2:] != "py":
            config += ".py"
        if self.os.path.exists(self.pathlib.Path(config)):
            cfg = mmcv.Config.fromfile(config)
            cfg.model.pretrained = None

            # changes normalizaion layers for custom cfg since by default mmseg config consider multigpu env
            def change_norm_layer(cfg):
                for k, v in cfg.items():
                    if k == "norm_cfg":
                        cfg[k].type = "BN"
                    elif isinstance(cfg[k], dict):
                        change_norm_layer(cfg[k])

            change_norm_layer(cfg.model)
        else:
            import arcgis

            cfg_abs_path = (
                self.pathlib.Path(arcgis.__file__).parent
                / "learn"
                / "_mmseg_config"
                / config
            )
            cfg = mmcv.Config.fromfile(cfg_abs_path)
            checkpoint = cfg.get("checkpoint", False)
            if checkpoint:
                cfg.model.pretrained = None

        if isinstance(cfg.model.decode_head, list):
            for dcd_head in cfg.model.decode_head:
                dcd_head.num_classes = data.c
                dcd_head.loss_decode.class_weight = class_weight
        else:
            cfg.model.decode_head.num_classes = data.c
            cfg.model.decode_head.loss_decode.class_weight = class_weight

        if hasattr(cfg.model, "auxiliary_head"):
            if isinstance(cfg.model.auxiliary_head, list):
                for aux_head in cfg.model.auxiliary_head:
                    aux_head.num_classes = data.c
                    aux_head.loss_decode.class_weight = class_weight
            else:
                cfg.model.auxiliary_head.num_classes = data.c
                cfg.model.auxiliary_head.loss_decode.class_weight = class_weight
        if cfg.model.backbone.type == "CGNet" and getattr(
            data, "_is_multispectral", False
        ):
            cfg.model.backbone.in_channels = len(data._extract_bands)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mmseg.models.build_segmentor(cfg.model)

        if checkpoint:
            mmcv.runner.load_checkpoint(
                model, checkpoint, "cpu", False, logging.getLogger()
            )

        # default forward of the model from the original API should be modified to make it compatible with the learn module.
        from mmcv.runner import auto_fp16

        @auto_fp16(apply_to=("img",))
        def forward_modified(self, img, img_metas=None, gt_semantic_seg=None):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.training:
                    losses = self.forward_train(img, img_metas, gt_semantic_seg)
                    loss, log_vars = self._parse_losses(losses)

                    outputs = dict(loss=loss, log_vars=log_vars)

                    return outputs
                else:
                    return self.forward_test(img[0], img[1], rescale=True)

        # default simple_test of the model from the original API should be modified to correctly work in test time.
        def simple_test_modified(self, img, img_meta, rescale=True):
            seg_logit = self.encode_decode(img, img_meta)
            return seg_logit

        model.forward = self.types.MethodType(forward_modified, model)
        model.simple_test = self.types.MethodType(simple_test_modified, model)

        self.model = model
        self.cfg = cfg

        logging.disable(0)

        return model

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        image_pad_shape = model_input_batch.permute(0, 2, 3, 1).shape[1:]
        image_scale_factor = self.numpy.array(
            [1.0, 1.0, 1.0, 1.0], dtype=self.numpy.float32
        )

        img_metas_dict = {}
        img_metas_dict["pad_shape"] = image_pad_shape
        img_metas_dict["img_shape"] = image_pad_shape
        img_metas_dict["ori_shape"] = image_pad_shape
        img_metas_dict["scale_factor"] = image_scale_factor
        img_metas_dict["flip"] = False

        img_metas = [img_metas_dict] * model_input_batch.shape[0]

        if learn.model.training:
            model_input = [model_input_batch, img_metas, model_target_batch]
            return model_input, model_target_batch
        else:
            model_input = [[[model_input_batch], [img_metas]]]
            return model_input, model_target_batch

    def transform_input(self, xb):
        image_pad_shape = xb.permute(0, 2, 3, 1).shape[1:]
        image_scale_factor = self.numpy.array(
            [1.0, 1.0, 1.0, 1.0], dtype=self.numpy.float32
        )
        img_metas_dict = {}
        img_metas_dict["pad_shape"] = image_pad_shape
        img_metas_dict["img_shape"] = image_pad_shape
        img_metas_dict["ori_shape"] = image_pad_shape
        img_metas_dict["scale_factor"] = image_scale_factor
        img_metas_dict["flip"] = False

        img_metas = [img_metas_dict] * xb.shape[0]
        model_input = [[xb], [img_metas]]

        return model_input

    def transform_input_multispectral(self, xb):
        return self.transform_input(xb)

    def loss(self, model_output, *model_target):
        if not self.model.training:
            if self.cfg.model.type == "CascadeEncoderDecoder":
                losses = 0.0
                for i in range(self.cfg.model.num_stages):
                    _losses = self.model.decode_head[i].losses(
                        model_output, model_target[0]
                    )
                    losses += _losses.get("loss_ce", _losses.get("loss_seg"))
                return losses

            _losses = self.model.decode_head.losses(model_output, model_target[0])
            loss_dice = _losses.get("loss_dice")
            if loss_dice:
                return loss_dice
            else:
                return _losses.get("loss_ce", _losses.get("loss_seg"))

        return model_output["loss"]

    def post_process(self, pred, thres=0.5, thinning=True, prob_raster=False):
        """
        In this function you have to return list with appended output for each image in the batch with shape [C=1,H,W]!
        """
        if prob_raster:
            return pred
        else:
            pred = self.torch.unsqueeze(pred.argmax(dim=1), dim=1)
        return pred


def norm_prithvi(data, model):
    scaling_info = {
        "prithvi100m_burn_scar": (
            img_norm_burn_model.get("means"),
            img_norm_burn_model.get("stds"),
        ),
        "prithvi100m_sen1floods": (
            img_norm_flood_model.get("means"),
            img_norm_flood_model.get("stds"),
        ),
        "prithvi100m_crop_classification": (
            img_norm_crop_model.get("means"),
            img_norm_crop_model.get("stds"),
        ),
        "prithvi100m": (data._scaled_mean_values, data._scaled_std_values),
    }

    means, stds = scaling_info[model]
    data._scaled_mean_values, data._scaled_std_values = torch.tensor(
        means
    ), torch.tensor(stds)

    data._min_max_scaler = None

    if (data._band_max_values.mean() > 1) and (
        model != "prithvi100m_crop_classification"
    ):
        div_value = 10000
    else:
        div_value = None

    data.valid_ds.x._div = div_value
    data.train_ds.x._div = div_value

    data = data.normalize(
        stats=(data._scaled_mean_values, data._scaled_std_values), do_x=True, do_y=False
    )

    return data


class MMSegmentation(ModelExtension):
    """
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    data                    Required fastai Databunch. Returned data object from
                            :meth:`~arcgis.learn.prepare_data`  function.
    ---------------------   -------------------------------------------
    model                   Required model name or path to the configuration file
                            from :class:`~arcgis.learn.MMSegmentation` repository. The list of the
                            supported models can be queried using
                            :attr:`~arcgis.learn.MMSegmentation.supported_models`
    ---------------------   -------------------------------------------
    model_weight            Optional path of the model weight from
                            :class:`~arcgis.learn.MMSegmentation` repository.
    ---------------------   -------------------------------------------
    pretrained_path         Optional string. Path where pre-trained model is
                            saved.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    class_balancing         Optional boolean. If True, it will balance the
                            cross-entropy loss inverse to the frequency
                            of pixels per class. Default: False.
    ---------------------   -------------------------------------------
    ignore_classes          Optional list. It will contain the list of class
                            values on which model will not incur loss.
                            Default: []
    =====================   ===========================================

    :return: :class:`~arcgis.learn.MMSegmentation` Object
    """

    def __init__(self, data, model, model_weight=False, pretrained_path=None, **kwargs):
        self._check_dataset_support(data)

        if model.startswith("prithvi100m"):
            data.remove_tfm(data.norm)
            data.norm, data.denorm = None, None
            data = norm_prithvi(data, model)
        self._ignore_classes = kwargs.get("ignore_classes", [])
        self.class_balancing = kwargs.get("class_balancing", False)
        if self._ignore_classes != [] and len(data.classes) <= 2:
            raise Exception(
                f"`ignore_classes` parameter can only be used when the dataset has more than 2 classes."
            )

        data_classes = list(data.class_mapping.keys())
        if 0 not in list(data.class_mapping.values()):
            self._ignore_mapped_class = [
                data_classes.index(k) + 1 for k in self._ignore_classes if k != 0
            ]
        else:
            self._ignore_mapped_class = [
                data_classes.index(k) + 1 for k in self._ignore_classes
            ]
        if self._ignore_classes != []:
            if 0 not in self._ignore_mapped_class:
                self._ignore_mapped_class.insert(0, 0)

        class_weight = None
        if self.class_balancing:
            if data.class_weight is not None:
                # Handle condition when nodata is already at pixel value 0 in data
                if (data.c - 1) == data.class_weight.shape[0]:
                    class_weight = [
                        data.class_weight.mean()
                    ] + data.class_weight.tolist()
                else:
                    class_weight = data.class_weight.tolist()
            else:
                if getattr(data, "overflow_encountered", False):
                    logger.warning(
                        "Overflow Encountered. Ignoring `class_balancing` parameter."
                    )
                    class_weight = [1.0] * len(data.classes)
                else:
                    logger.warning(
                        "Could not find 'NumPixelsPerClass' in 'esri_accumulated_stats.json'. Ignoring `class_balancing` parameter."
                    )

        if self._ignore_classes != []:
            if not self.class_balancing:
                class_weight = [1.0] * data.c
            for idx in self._ignore_mapped_class:
                class_weight[idx] = 0.0

        self._final_class_weight = class_weight

        super().__init__(
            data,
            MMSegmentationConfig,
            pretrained_path=pretrained_path,
            model=model,
            model_weight=model_weight,
            ignore_class=self._ignore_mapped_class,
            class_weight=self._final_class_weight,
        )
        idx = self._freeze()
        self.learn.layer_groups = split_model_idx(self.learn.model, [idx])
        self.learn.create_opt(lr=3e-3)

    def unfreeze(self):
        for _, param in self.learn.model.named_parameters():
            param.requires_grad = True

    def _freeze(self):
        "Freezes the pretrained backbone."
        if self._model_conf.cfg.model.backbone.type == "CGNet":
            return 6
        for idx, i in enumerate(flatten_model(self.learn.model.backbone)):
            if isinstance(i, (torch.nn.BatchNorm2d)):
                continue
            for p in i.parameters():
                p.requires_grad = False
        return idx

    @staticmethod
    def _available_metrics():
        return ["valid_loss", "accuracy"]

    @property
    def _is_mmsegdet(self):
        return True

    @property
    def supported_datasets(self):
        """Supported dataset types for this model."""
        return MMSegmentation._supported_datasets()

    @staticmethod
    def _supported_datasets():
        return ["Classified_Tiles"]

    supported_models = [
        "ann",
        "apcnet",
        "ccnet",
        "cgnet",
        "deeplabv3",
        "deeplabv3plus",
        "dmnet",
        "dnlnet",
        "emanet",
        "fastscnn",
        "fcn",
        "gcnet",
        "hrnet",
        "mobilenet_v2",
        "nonlocal_net",
        "ocrnet",
        "psanet",
        "pspnet",
        "resnest",
        "sem_fpn",
        "unet",
        "upernet",
    ]
    """
    List of models supported by this class.
    """

    @classmethod
    def from_model(cls, emd_path, data=None):
        """
        Creates a :class:`~arcgis.learn.MMSegmentation` object from an Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to Deep Learning Package
                                (DLPK) or Esri Model Definition(EMD) file.
        ---------------------   -------------------------------------------
        data                    Required fastai Databunch or None. Returned data
                                object from :meth:`~arcgis.learn.prepare_data`  function or None for
                                inferencing.

        =====================   ===========================================

        :return: :class:`~arcgis.learn.MMSegmentation` Object
        """
        emd_path = _get_emd_path(emd_path)

        with open(emd_path) as f:
            emd = json.load(f)

        model_file = Path(emd["ModelFile"])

        if not model_file.is_absolute():
            model_file = emd_path.parent / model_file

        kwargs = emd.get("Kwargs", {})

        try:
            class_mapping = {i["Value"]: i["Name"] for i in emd["Classes"]}
            color_mapping = {i["Value"]: i["Color"] for i in emd["Classes"]}
        except KeyError:
            class_mapping = {i["ClassValue"]: i["ClassName"] for i in emd["Classes"]}
            color_mapping = {i["ClassValue"]: i["Color"] for i in emd["Classes"]}

        if data is None:
            data = _EmptyData(
                path=emd_path.parent.parent,
                loss_func=None,
                c=len(class_mapping) + 1,
                chip_size=emd["ImageHeight"],
            )
            data.class_mapping = class_mapping
            data.color_mapping = color_mapping
            data._is_empty = True
            data.emd_path = emd_path
            data.emd = emd
            data.classes = ["background"]
            for k, v in class_mapping.items():
                data.classes.append(v)
            data = get_multispectral_data_params_from_emd(data, emd)
            data.dataset_type = emd["DatasetType"]

        return cls(data, pretrained_path=str(model_file), **kwargs)

    def show_results(self, rows=5, thresh=0.5, thinning=True, **kwargs):
        """
        Displays the results of a trained model on a part of the validation set.
        """
