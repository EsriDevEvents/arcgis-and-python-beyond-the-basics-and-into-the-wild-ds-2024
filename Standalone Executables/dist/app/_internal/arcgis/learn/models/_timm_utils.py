import types
from torch import nn
from ._arcgis_model import _get_backbone_meta
import logging
import os
import sys
import warnings
import re
import timm

_logger = logging.getLogger(__name__)

try:
    from fastai.callbacks.hooks import hook_outputs, model_sizes
    from fastai.torch_core import one_param
    from fastai.vision import create_body, Image
    from fastai.layers import AdaptiveConcatPool2d
    from fastai.basic_data import DatasetType
    from fastai.callbacks import hook_output
    import torch.nn.functional as F
    from matplotlib import pyplot as plt
    import numpy as np
    import fnmatch
    from timm.models.hub import (
        has_hf_hub,
        load_state_dict_from_hf,
        load_state_dict_from_url,
    )
    from timm.models.helpers import (
        adapt_input_conv,
        build_model_with_cfg,
        overlay_external_default_cfg,
    )
    from torch.hub import get_dir
    import zipfile
    import torch
    from copy import deepcopy

    HAS_FASTAI = True
except Exception as e:
    HAS_FASTAI = False


# same function with modification fastai.vision.learner._test_cnn
def test_cnn_trnsfrmr(m):
    if not isinstance(m, nn.Sequential) or not len(m) == 2:
        return False
    if hasattr(m[1], "_transformer"):
        return True
    return isinstance(m[1][0], (AdaptiveConcatPool2d, nn.AdaptiveAvgPool2d))


def reshape_tensor(x):
    x = x.squeeze()
    if x.ndim > 2:
        return x
    # get feature size(height, width)
    feature_size = int(np.sqrt(x.shape[0]))
    # get number of tokens
    num_tokens = x.shape[0] - feature_size**2
    return x[num_tokens:, :].reshape(feature_size, feature_size, -1).permute(2, 0, 1)


# same function with modification fastai.vision.learner._cl_int_gradcam
def gradcam_trnsfrmr(
    self,
    idx,
    ds_type=None,
    heatmap_thresh=16,
    image=True,
):
    if ds_type == None:
        ds_type = DatasetType.Valid
    m = self.learn.model.eval()
    im, cl = self.learn.data.dl(ds_type).dataset[idx]
    cl = int(cl)
    xb, _ = self.data.one_item(
        im, detach=False, denorm=False
    )  # put into a minibatch of batch size = 1
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0, int(cl)].backward()
    acts = hook_a.stored[0].cpu()  # activation maps
    grad = hook_g.stored[0][0].cpu()
    if hasattr(m[1], "_transformer"):
        acts = reshape_tensor(acts)
        grad = reshape_tensor(grad)

    if (acts.shape[-1] * acts.shape[-2]) >= heatmap_thresh:
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu(((acts * grad_chan[..., None, None])).sum(0))
        if image:
            xb_im = Image(xb[0])
            _, ax = plt.subplots()
            sz = list(xb_im.shape[-2:])
            xb_im.show(
                ax,
                title=f"pred. class: {self.pred_class[idx]}, actual class: {self.learn.data.classes[cl]}",
            )
            ax.imshow(
                mult,
                alpha=0.4,
                extent=(0, *sz[::-1], 0),
                interpolation="bilinear",
                cmap="magma",
            )
        return mult


hosted_weights = {
    "ecaresnet101d": "682eb6fca513415b9c5a6be61f1bc0f1",
    "ecaresnet101d_pruned": "d676dfa5ffa04732bc8ef58d2c85dd38",
    "ecaresnet269d": "df81aea21521475eba81658496dc6ace",
    "ecaresnet50d": "9f3235316a464885b4d3e5db2d5499b9",
    "ecaresnet50d_pruned": "eca1ccf9ea3145f3ae1cd257f290ef09",
    "ecaresnetlight": "680b9da80da54bbca20a6b7446b5a2df",
    "efficientnet_b1_pruned": "ed49c5e7bb804a05b877702e7ce12f40",
    "efficientnet_b2_pruned": "8eb218d51113449181f1f25efae9ab98",
    "efficientnet_b3_pruned": "b341d13252d244c591370c082bea9696",
    "hardcorenas_a": "af3227e5f48145c2a21c8cf76c3782e4",
    "hardcorenas_b": "9c14f1b482c5465ba5cc75772aeead60",
    "hardcorenas_c": "b758a9d94cb7402c942f2a51fdd99953",
    "hardcorenas_d": "8a927ea1f2c14b2dabaae423c21c193c",
    "hardcorenas_e": "1e7d664452cf4ce596bd5d1bfe776557",
    "hardcorenas_f": "f3a1f74efe244911b166d5456f2bba68",
    "legacy_senet154": "36266e6e22444ce299d76a57a6a817df",
    "legacy_seresnext101_32x4d": "884b2dd7093e49b4884fac2d23bc2386",
    "legacy_seresnext50_32x4d": "e54dc138f33f415984ccfbf6250e1e03",
    "nasnetalarge": "91404c5ecbd842948bb1507f1e313e4b",
    "regnetx_006": "e595c123a67c4a4f87f322b1af4b293c",
    "tf_efficientnet_b6_ns": "20d17115f5db4e11837b8d43e81d59da",
}


# same function with modification timm.models.helpers.load_pretrained
def load_timm_bckbn_pretrained(
    model,
    default_cfg=None,
    num_classes=1000,
    in_chans=3,
    filter_fn=None,
    strict=True,
    progress=True,
):
    default_cfg = default_cfg or getattr(model, "default_cfg", None) or {}
    pretrained_url = default_cfg.get("url", None)
    hf_hub_id = default_cfg.get("hf_hub", None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning(
            "No pretrained weights exist for this model. Using random initialization."
        )
        return

    model_url = hosted_weights.get(default_cfg["architecture"], False)
    if model_url:
        model_dir = os.path.join(get_dir(), "checkpoints")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        cached_file = os.path.join(model_dir, pretrained_url.split("/")[-1])
        if not os.path.exists(cached_file):
            sys.stderr.write(
                'Downloading: "{}" pretrained weights to {}\n'.format(
                    default_cfg["architecture"], cached_file
                )
            )
            from arcgis.gis import GIS

            gis = GIS(set_active=False)
            item = gis.content.get(model_url)
            item.download(model_dir)
            zipped_file = os.path.join(
                model_dir, pretrained_url.split("/")[-1][:-3] + "zip"
            )
            with zipfile.ZipFile(zipped_file) as f:
                f.extractall(model_dir)
            os.remove(zipped_file)

        state_dict = torch.load(cached_file, map_location="cpu")

    elif hf_hub_id and has_hf_hub(necessary=not pretrained_url):
        _logger.info(f"Loading pretrained weights from Hugging Face hub ({hf_hub_id})")
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        _logger.info(f"Loading pretrained weights from url ({pretrained_url})")
        state_dict = load_state_dict_from_url(
            pretrained_url, progress=progress, map_location="cpu"
        )
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get("first_conv", None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + ".weight"
            try:
                state_dict[weight_name] = adapt_input_conv(
                    in_chans, state_dict[weight_name]
                )
                _logger.info(
                    f"Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)"
                )
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f"Unable to convert pretrained {input_conv_name} weights, using random init for this layer."
                )

    classifiers = default_cfg.get("classifier", None)
    label_offset = default_cfg.get("label_offset", 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg["num_classes"]:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + ".weight"]
                del state_dict[classifier_name + ".bias"]
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + ".weight"]
                state_dict[classifier_name + ".weight"] = classifier_weight[
                    label_offset:
                ]
                classifier_bias = state_dict[classifier_name + ".bias"]
                state_dict[classifier_name + ".bias"] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def _default_split(m):
    return (m[1],)


def _tresnet_split(m):
    return (m[0].feature.body.layer4,)


def _squeezenet_split(m):
    return (m[0][0][5], m[0][0][8], m[1])


def _densenet_split(m):
    return (m[0][0][7], m[0][0][9])


def _vgg_split(m):
    return m[0][0][15]


def _rep_vgg(m):
    return m[0][1][2]


def _mobilenetv2_split(m):
    return m[1]


def _darknet_split(m):
    return m[0][1][4]


def _cspres_split(m):
    return m[0][1][3]


def _nfnet_split(m):
    return m[0][1][-1]


def _dpn_split(m):
    return m[0][0][-2]


def _esevovnet_split(m):
    return m[0][1][-1]


def _gernet_split(m):
    return m[0][1][-2]


def _regnet_split(m):
    return m[0][3]


def _resnetv2_split(m):
    return m[0][1][3]


def _modified_cut(m):
    def forward_modified(self, img):
        return self.forward_features(img)

    m.forward = types.MethodType(forward_modified, m)

    class TimmBackbone(nn.Module):
        def __init__(self, m):
            super(TimmBackbone, self).__init__()
            self.feature = m

        def forward(self, x):
            return self.feature.forward_features(x)

    return TimmBackbone(m)


timm_model_meta = {
    "default": {"cut": None, "split": _default_split},
    "squeezenet": {"cut": -1, "split": _squeezenet_split},
    "densenet": {"cut": None, "split": _densenet_split},
    "repvgg": {"cut": -2, "split": _rep_vgg},
    "vgg": {"cut": -2, "split": _vgg_split},
    "mobilenet": {"cut": None, "split": _mobilenetv2_split},
    "darknet": {"cut": None, "split": _darknet_split},
    "hrnet": {"cut": _modified_cut, "split": _default_split},
    "nasnet": {"cut": _modified_cut, "split": _default_split},
    "selecsls": {"cut": _modified_cut, "split": _default_split},
    "tresnet": {"cut": _modified_cut, "split": _tresnet_split},
    "cspres": {"cut": None, "split": _cspres_split},
    "nfnet": {"cut": None, "split": _nfnet_split},
    "dpn": {"cut": None, "split": _dpn_split},
    "ese_vovnet": {"cut": None, "split": _esevovnet_split},
    "gernet": {"cut": None, "split": _gernet_split},
    "nf_regnet": {"cut": None, "split": _nfnet_split},
    "nf_resnet": {"cut": None, "split": _nfnet_split},
    "regnet": {"cut": None, "split": _regnet_split},
    "resnet51q": {"cut": None, "split": _resnetv2_split},
    "resnetv2": {"cut": None, "split": _resnetv2_split},
}


def timm_config(arch):
    model_name = arch if type(arch) is str else arch.__name__
    model_key = [key for key in timm_model_meta if key in model_name] + ["default"]
    return timm_model_meta.get(model_key[0])


def filter_timm_models(flt=[]):
    models = timm.list_models(pretrained=True)
    # remove transformer models
    flt = [
        "*cait*",
        "*coat*",
        "*convit*",
        "*deit*",
        "*gmixer*",
        "*gmlp*",
        "*levit*",
        "*mixer*",
        "*pit*",
        "*resmlp*",
        "*swin*",
        "*tnt*",
        "*twins*",
        "*visformer*",
        "vit_*",
    ] + flt
    flt_models = []
    for f in flt:
        flt_models.extend(fnmatch.filter(models, f))
    return sorted(set(models) - set(flt_models))


def _get_feature_size(arch, cut, chip_size=(64, 64), channel_in=3):
    m = nn.Sequential(*create_body(arch, False, cut).children())
    if "tresnet" in arch.__module__:
        with hook_outputs(m) as hooks:
            dummy_batch = (
                one_param(m)
                .new(1, channel_in, *chip_size)
                .requires_grad_(False)
                .uniform_(-1.0, 1.0)
            )
            x = m.eval()(dummy_batch)
            return [o.stored.shape for o in hooks]
    else:
        return model_sizes(m, chip_size)


def get_backbone(backbone_fn, pretrained):
    if "timm" in backbone_fn.__module__:
        backbone_cut = timm_config(backbone_fn)["cut"]
    elif getattr(backbone_fn, "_is_multispectral", False):
        backbone_cut = _get_backbone_meta(backbone_fn.__name__)["cut"]
    else:
        backbone_cut = None

    return create_body(backbone_fn, pretrained, backbone_cut)


def forward_VisionTransformer(self, x):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    x = self.blocks[:-1](x)

    return x


class VisionTransformerHead(nn.Module):
    def __init__(self, block, head, norm, dist_token, pre_logits, head_dist):
        super().__init__()
        self.block = block
        self.head = head
        self.norm = norm
        self.dist_token = dist_token
        self.pre_logits = pre_logits
        self.head_dist = head_dist
        self._transformer = True

    def forward(self, x):
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None:
            x = self.pre_logits(x[:, 0])

        if self.head_dist is not None:
            x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
            return x  # (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


class TransformerClassifierHead(nn.Module):
    def __init__(self, head, head_dist=None):
        super().__init__()
        self.head = head
        self.head_dist = head_dist
        self._transformer = True

    def forward(self, x):
        if self.head_dist is not None:
            x, x_dist = self.head(x[1][:, 0]), self.head_dist(x[1][:, 1])
            return x  # (x + x_dist) / 2
        else:
            x = self.head(x[1])
        return x


def checkpoint_filter_fn_Cait(state_dict, model):
    if "model" in state_dict:
        state_dict = state_dict["model"]
    checkpoint_no_module = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        if new_key == "pos_embed" and v.shape != model.pos_embed.shape:
            v = timm.models.vision_transformer.resize_pos_embed(
                v, model.pos_embed, 0, model.patch_embed.grid_size
            )
        checkpoint_no_module[new_key] = v
    return checkpoint_no_module


timm.models.cait.checkpoint_filter_fn = checkpoint_filter_fn_Cait


def forward_Cait(self, x):
    B = x.shape[0]
    x = self.patch_embed(x)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    for i, blk in enumerate(self.blocks):
        x = blk(x)

    for i, blk in enumerate(self.blocks_token_only):
        cls_tokens = blk(x, cls_tokens)

    x_grad_cam = x
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.norm(x)
    return x_grad_cam, x[:, 0]


def forward_CoaT(self, x):
    B = x.shape[0]

    # Serial blocks 1.
    x1 = self.patch_embed1(x)
    H1, W1 = self.patch_embed1.grid_size
    x1 = self.insert_cls(x1, self.cls_token1)
    for blk in self.serial_blocks1:
        x1 = blk(x1, size=(H1, W1))
    x1_nocls = self.remove_cls(x1)
    x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

    # Serial blocks 2.
    x2 = self.patch_embed2(x1_nocls)
    H2, W2 = self.patch_embed2.grid_size
    x2 = self.insert_cls(x2, self.cls_token2)
    for blk in self.serial_blocks2:
        x2 = blk(x2, size=(H2, W2))
    x2_nocls = self.remove_cls(x2)
    x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

    # Serial blocks 3.
    x3 = self.patch_embed3(x2_nocls)
    H3, W3 = self.patch_embed3.grid_size
    x3 = self.insert_cls(x3, self.cls_token3)
    for blk in self.serial_blocks3:
        x3 = blk(x3, size=(H3, W3))
    x3_nocls = self.remove_cls(x3)
    x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

    # Serial blocks 4.
    x4 = self.patch_embed4(x3_nocls)
    H4, W4 = self.patch_embed4.grid_size
    x4 = self.insert_cls(x4, self.cls_token4)
    for blk in self.serial_blocks4[:-1]:
        x4 = blk(x4, size=(H4, W4))
    x_grad_cam = x4
    x4 = self.serial_blocks4[-1](x4, size=(H4, W4))
    x4_nocls = self.remove_cls(x4)
    x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

    if self.parallel_blocks is None:
        x4 = self.norm4(x4)
        x4_cls = x4[:, 0]
        return x_grad_cam, x4_cls

    # Parallel blocks.
    for blk in self.parallel_blocks:
        x2, x3, x4 = (
            self.cpe2(x2, (H2, W2)),
            self.cpe3(x3, (H3, W3)),
            self.cpe4(x4, (H4, W4)),
        )
        x1, x2, x3, x4 = blk(
            x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)]
        )

    x2 = self.norm2(x2)
    x3 = self.norm3(x3)
    x4 = self.norm4(x4)
    x2_cls = x2[:, :1]  # [B, 1, C]
    x3_cls = x3[:, :1]
    x4_cls = x4[:, :1]
    merged_cls = torch.cat((x2_cls, x3_cls, x4_cls), dim=1)  # [B, 3, C]
    merged_cls = self.aggregate(merged_cls).squeeze(dim=1)  # Shape: [B, C]

    return x_grad_cam, merged_cls


def checkpoint_filter_fn_Convit(state_dict, model):
    if "model" in state_dict:
        state_dict = state_dict["model"]
    checkpoint_no_module = {}
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            v = timm.models.vision_transformer.resize_pos_embed(
                v, model.pos_embed, 0, model.patch_embed.grid_size
            )
        checkpoint_no_module[k] = v
    return checkpoint_no_module


def create_convit(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    return build_model_with_cfg(
        timm.models.convit.ConViT,
        variant,
        pretrained,
        default_cfg=timm.models.convit.default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn_Convit,
        **kwargs,
    )


timm.models.convit._create_convit = create_convit


def forward_ConViT(self, x):
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)

    if self.use_pos_embed:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    no_of_block = len(self.blocks)
    for u, blk in enumerate(self.blocks):
        if u == self.local_up_to_layer:
            x = torch.cat((cls_tokens, x), dim=1)
        if u + 1 == no_of_block - 1:
            x_grad_cam = x
        x = blk(x)

    x = self.norm(x)
    return x_grad_cam, x[:, 0]


def checkpoint_filter_fn_levit(state_dict, model):
    if "model" in state_dict:
        state_dict = state_dict["model"]
    D = model.state_dict()
    for k in state_dict.keys():
        if "attention_bias" in k:
            if "attention_biases" in k and D[k].shape != state_dict[k].shape:
                state_dict[k] = F.interpolate(
                    state_dict[k][None][None], size=D[k].shape, mode="bilinear"
                ).squeeze()
            else:
                state_dict[k] = D[k]
        if k in D and D[k].ndim == 4 and state_dict[k].ndim == 2:
            state_dict[k] = state_dict[k][:, :, None, None]
    return state_dict


timm.models.levit.checkpoint_filter_fn = checkpoint_filter_fn_levit


def forward_Levit(self, x):
    x = self.patch_embed(x)
    if not self.use_conv:
        x = x.flatten(2).transpose(1, 2)
    x = self.blocks(x)
    x_grad_cam = x
    x = x.mean((-2, -1)) if self.use_conv else x.mean(1)
    # concat same cls_token in dim 1 for distillation head
    x = torch.cat((x.unsqueeze(1), x.unsqueeze(1)), dim=1)
    return x_grad_cam, x


def checkpoint_filter_fn_pit(state_dict, model):
    out_dict = {}
    p_blocks = re.compile(r"pools\.(\d)\.")
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            v = F.interpolate(v, size=model.pos_embed.shape[-2:], mode="bilinear")
        k = p_blocks.sub(lambda exp: f"transformers.{int(exp.group(1))}.pool.", k)
        out_dict[k] = v
    return out_dict


timm.models.pit.checkpoint_filter_fn = checkpoint_filter_fn_pit


def forward_pit_transformer(self, x):
    x, cls_tokens = x
    B, C, H, W = x.shape
    token_length = cls_tokens.shape[1]

    x = x.flatten(2).transpose(1, 2)
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.blocks[:-1](x)
    x_grad_cam = x
    x = self.blocks[-1](x)

    cls_tokens = x[:, :token_length]
    x = x[:, token_length:]
    x = x.transpose(1, 2).reshape(B, C, H, W)

    if self.pool is not None:
        x, cls_tokens = self.pool(x, cls_tokens)
    return x_grad_cam, cls_tokens


def forward_PoolingVisionTransformer(self, x):
    x = self.patch_embed(x)
    x = self.pos_drop(x + self.pos_embed)
    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    x, cls_tokens = self.transformers((x, cls_tokens))
    cls_tokens = self.norm(cls_tokens)
    if self.head_dist is not None:
        return x, cls_tokens
    return x, cls_tokens[:, 0]


def checkpoint_filter_fn_swin(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif "relative_position" in k or "attn_mask" in k:
            # To resize embedding when using model at different size from pretrained weights
            model_attr = k.split(".")
            temp_module = model
            for attr in model_attr:
                # last temp_module will be tensor
                temp_module = temp_module.__getattr__(attr)
            if v.shape == temp_module.shape:
                continue
            if "index" in model_attr[-1]:
                v = temp_module
            elif "attn_mask" in model_attr[-1]:
                v = temp_module
            elif "bias_table" in model_attr[-1]:
                v = F.interpolate(
                    v[None][None], size=list(temp_module.shape), mode="bilinear"
                ).squeeze()

        out_dict[k] = v
    return out_dict


def create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(timm.models.swin_transformer.default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-2:]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    kwargs["window_size"] = img_size // 32

    model = build_model_with_cfg(
        timm.models.swin_transformer.SwinTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn_swin,
        **kwargs,
    )

    return model


timm.models.swin_transformer._create_swin_transformer = create_swin_transformer


def forward_SwinTransformer(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    x = self.layers(x)
    x_grad_cam = x
    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x_grad_cam, x


def forward_TNT(self, x):
    B = x.shape[0]
    pixel_embed = self.pixel_embed(x, self.pixel_pos)

    patch_embed = self.norm2_proj(
        self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1)))
    )
    patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    patch_embed = patch_embed + self.patch_pos
    patch_embed = self.pos_drop(patch_embed)

    no_of_block = len(self.blocks)
    for idx, blk in enumerate(self.blocks):
        pixel_embed, patch_embed = blk(pixel_embed, patch_embed)
        if idx + 1 == no_of_block - 1:
            x_grad_cam = patch_embed

    patch_embed = self.norm(patch_embed)
    return x_grad_cam, patch_embed[:, 0]


def forward_Twins(self, x):
    B = x.shape[0]
    for i, (embed, drop, blocks, pos_blk) in enumerate(
        zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)
    ):
        x, size = embed(x)
        x = drop(x)
        for j, blk in enumerate(blocks):
            x = blk(x, size)
            if j == 0:
                x = pos_blk(x, size)  # PEG here
        if i < len(self.depths) - 1:
            x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    x_grad_cam = x
    x = self.norm(x)
    return x_grad_cam, x.mean(dim=1)  # GAP here


def checkpoint_filter_fn_visformer(state_dict, model):
    out_dict = {}
    D = model.state_dict()
    for k, v in state_dict.items():
        if "pos_embed" in k and v.shape != D[k].shape:
            v = F.interpolate(v, size=D[k].shape[-2:], mode="bilinear")
        out_dict[k] = v
    return out_dict


def create_visformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    model = build_model_with_cfg(
        timm.models.visformer.Visformer,
        variant,
        pretrained,
        default_cfg=timm.models.visformer.default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn_visformer,
        **kwargs,
    )
    return model


timm.models.visformer._create_visformer = create_visformer


def forward_Visformer(self, x):
    if self.stem is not None:
        x = self.stem(x)

    # stage 1
    x = self.patch_embed1(x)
    if self.pos_embed:
        x = x + self.pos_embed1
        x = self.pos_drop(x)
    for b in self.stage1:
        x = b(x)

    # stage 2
    if not self.vit_stem:
        x = self.patch_embed2(x)
        if self.pos_embed:
            x = x + self.pos_embed2
            x = self.pos_drop(x)
    for b in self.stage2:
        x = b(x)

    # stage3
    if not self.vit_stem:
        x = self.patch_embed3(x)
        if self.pos_embed:
            x = x + self.pos_embed3
            x = self.pos_drop(x)
    for b in self.stage3:
        x = b(x)

    x_grad_cam = x
    x = self.norm(x)
    x = self.global_pool(x)

    return x_grad_cam, x


def get_transformer_backbone(backbone_name, num_classes, img_size, pretrained):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        transformer_backbone = timm.create_model(
            backbone_name,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
        )
        logging.disable(logging.NOTSET)

    model_name = transformer_backbone.__class__.__name__
    forward_function = globals()["forward_" + model_name]
    transformer_backbone.forward = types.MethodType(
        forward_function, transformer_backbone
    )
    return transformer_backbone


def create_transformer_FeatureClassifier(
    backbone_name, num_classes, img_size, pretrained
):
    transformer_backbone = get_transformer_backbone(
        backbone_name, num_classes, img_size, pretrained
    )

    model_name = transformer_backbone.__class__.__name__

    if model_name == "VisionTransformer":
        trnsfrmr_head = VisionTransformerHead(
            transformer_backbone.blocks[-1],
            transformer_backbone.head,
            transformer_backbone.norm,
            transformer_backbone.dist_token,
            transformer_backbone.pre_logits,
            transformer_backbone.head_dist,
        )
    else:
        head_dist = getattr(transformer_backbone, "head_dist", None)
        trnsfrmr_head = TransformerClassifierHead(transformer_backbone.head, head_dist)

    if model_name == "PoolingVisionTransformer":
        transformer_backbone.transformers[-1].forward = types.MethodType(
            forward_pit_transformer, transformer_backbone.transformers[-1]
        )

    transformer_model = nn.Sequential(transformer_backbone, trnsfrmr_head)
    return transformer_model


def shortened_transformer_backbone():
    shortened_name = [
        "cait_tiny_24",
        "cait_tiny_36",
        "cait_small_24",
        "cait_small_36",
        "cait_base_36",
        "cait_base_48",
        "deit_tiny",
        "deit_small",
        "deit_base",
        "pit_tiny",
        "pit_mini",
        "pit_small",
        "pit_base",
        "swin_tiny_window7",
        "swin_small_window7",
        "swin_base_window7",
        "swin_base_window12",
        "swin_large_window7",
        "swin_large_window12",
        "tnt_small",
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
        "vit_huge",
    ]
    return shortened_name


# backbone with unique image sizes
def map_transformer_backbone_unique(backbone):
    transformer_backbone_unique = {
        "cait_small_36": "cait_s36_384",
        "cait_base_36": "cait_m36_384",
        "cait_base_48": "cait_m48_448",
        "deit_tiny": "deit_tiny_patch16_224",
        "deit_small": "deit_small_patch16_224",
        "pit_tiny": "pit_ti_224",
        "pit_mini": "pit_xs_224",
        "pit_small": "pit_s_224",
        "pit_base": "pit_b_224",
        "swin_tiny_window7": "swin_tiny_patch4_window7_224",
        "swin_small_window7": "swin_small_patch4_window7_224",
        "swin_base_window7": "swin_base_patch4_window7_224",
        "swin_base_window12": "swin_base_patch4_window12_384",
        "swin_large_window7": "swin_large_patch4_window7_224",
        "swin_large_window12": "swin_large_patch4_window12_384",
        "tnt_small": "tnt_s_patch16_224",
        "vit_huge": "vit_huge_patch14_224_in21k",
    }
    return transformer_backbone_unique.get(backbone, None)


# backbone with different image sizes
def map_transformer_backbone(backbone):
    transformer_backbone = {
        "cait_tiny_24": "cait_xxs24",
        "cait_tiny_36": "cait_xxs36",
        "cait_small_24": "cait_s24",
        "deit_base": "deit_base_patch16",
        "vit_tiny": "vit_tiny_patch16",
        "vit_small": "vit_small_patch16",
        "vit_base": "vit_base_patch16",
        "vit_large": "vit_large_patch16",
    }

    return transformer_backbone.get(backbone, None)


def complete_transformer_backbone_name(backbone, img_size):
    if backbone is not None and "timm:" in backbone:
        bckbn = backbone.split(":")[1]
        img_size = 224 if abs(img_size - 224) < abs(img_size - 384) else 384
        shortened_backbone = shortened_transformer_backbone()
        if bckbn in shortened_backbone:
            if map_transformer_backbone_unique(bckbn) is not None:
                return f"timm:{map_transformer_backbone_unique(bckbn)}"
            bckbn = map_transformer_backbone(bckbn)
            return f"timm:{bckbn}_{img_size}"
    return backbone
