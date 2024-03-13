from pathlib import Path
import types
import os
import torch
import torch.nn.functional as F
import numpy as np
import mmcv
from mmdet3d.models import build_detector
import logging
from mmcv.runner import auto_fp16


def get_backbone_channel(model_cfg, data):
    temp_model = build_detector(model_cfg).to(data.device)

    # x_batch, _ = data.one_batch(detach=False)
    x_batch = torch.rand(
        (10000, data.num_features), dtype=torch.float32, device=data.device
    )
    x_batch[:, :3] = (x_batch[:, :3] - 0.5) / 0.5
    x_batch[:, :3] *= data.scale_factor

    voxels, num_points, coors = temp_model.voxelize([x_batch])
    voxel_features = temp_model.voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0].item() + 1
    backbone_feature = temp_model.middle_encoder(voxel_features, coors, batch_size)

    return backbone_feature.shape[1]


def get_voxel_size(voxel_parms, data):
    pc_range = np.array(data.range)
    pc_lwh = pc_range[3:] - pc_range[:3]
    # keep the minimum resolution of point cloud grid to (200, 200) in x, y direction
    tile_voxel_size = ([0.005, 0.005, 0.02] * pc_lwh).tolist()
    default_voxel_size = list(map(min, zip(tile_voxel_size, data.voxel_size)))
    voxel_size = voxel_parms.get("voxel_size", None)
    if voxel_size is None:
        voxel_size = default_voxel_size
    else:
        if len(voxel_size) != 3:
            raise Exception("voxel_size list should contain 3 items.")
        # check if given voxel_size is creating at least 64x64x8 voxels
        grid_size = pc_lwh / voxel_size
        if min(grid_size) < 64:
            raise Exception(f"The size {voxel_size} of the voxel is too big.")

    grid_size = torch.tensor(pc_lwh / voxel_size).round().long().tolist()[::-1]

    return voxel_size, grid_size


def get_max_voxels(voxel_parms, grid_size):
    no_of_voxels = np.prod(grid_size, dtype=np.uint64).tolist()
    max_voxels = voxel_parms.get("max_voxels", None)
    if max_voxels is None:
        max_voxels = list(
            map(max, [20000, 40000], [no_of_voxels // 3000, no_of_voxels // 2000])
        )
    else:
        if len(max_voxels) != 2:
            raise Exception("max_voxels list should contain 2 items.")
        if min(max_voxels) < 20000:
            raise Exception("The minimum values in max_voxel should be at least 20000.")
    return max_voxels


def set_voxel_info(voxel_parms, data):
    voxel_size, grid_size = get_voxel_size(voxel_parms, data)
    max_voxels = get_max_voxels(voxel_parms, grid_size)
    voxel_points = voxel_parms.get("voxel_points", None)
    if voxel_points is None:
        voxel_points = max(10, int(data.no_of_points_per_tile // (max_voxels[0] * 0.3)))
    if voxel_points < 10:
        raise Exception("voxel_points should be greater than or equal to 10.")

    voxel_parms["voxel_size"] = voxel_size
    voxel_parms["sparse_shape"] = grid_size
    voxel_parms["max_voxels"] = max_voxels
    voxel_parms["voxel_points"] = voxel_points

    return voxel_parms


def model_config(model_cfg, data, **kwargs):
    voxel_parms = kwargs.get("voxel_parms", {})
    voxel_parms = set_voxel_info(voxel_parms, data)

    model_cfg.voxel_layer.voxel_size = voxel_parms["voxel_size"]
    model_cfg.voxel_layer.max_voxels = voxel_parms["max_voxels"]
    model_cfg.voxel_layer.max_num_points = voxel_parms["voxel_points"]
    model_cfg.voxel_layer.point_cloud_range = data.range

    model_cfg.voxel_encoder.num_features = data.num_features
    model_cfg.middle_encoder.in_channels = data.num_features

    # set correctly otherwise RuntimeError: CUDA error: an illegal memory access was encountered
    model_cfg.middle_encoder.sparse_shape = voxel_parms["sparse_shape"]

    model_cfg.backbone.in_channels = get_backbone_channel(model_cfg, data)

    model_cfg.bbox_head.num_classes = data.c
    model_cfg.bbox_head.bbox_coder.code_size = 7
    model_cfg.bbox_head.anchor_generator.ranges = data.anchor_range
    model_cfg.bbox_head.anchor_generator.sizes = data.average_box_size
    model_cfg.bbox_head.anchor_generator.rotations = [0.0, 1.57]

    return model_cfg


@auto_fp16(apply_to=("points",))
def forward_modified(self, input):
    if not self.prediction:
        losses = self.forward_train(**input)
        loss, log_vars = self._parse_losses(losses)

        loss = dict(loss=loss, log_vars=log_vars)
        output = None
        if not self.training:
            output = self.simple_test(input["points"], input["img_metas"])
        return output, loss
    else:
        return self.simple_test(input["points"], input["img_metas"])


@auto_fp16()
def forward_neck(self, x):
    assert len(x) == len(self.in_channels)
    ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
    size = ups[0].shape[-2:]
    ups = [F.interpolate(up, size, mode="bilinear", align_corners=False) for up in ups]
    if len(ups) > 1:
        out = torch.cat(ups, dim=1)
    else:
        out = ups[0]
    return [out]


def get_model(data, **kwargs):
    logging.disable(logging.WARNING)

    config = kwargs.get("model")
    checkpoint = kwargs.get("model_weight", False)

    if os.path.exists(Path(config)):
        cfg = mmcv.Config.fromfile(config)
    else:
        import arcgis

        cfg_abs_path = (
            Path(arcgis.__file__).parent
            / "learn"
            / "_mmdet3d_config"
            / (config + ".{}".format("py"))
        )
        cfg = mmcv.Config.fromfile(cfg_abs_path)
        checkpoint = cfg.get("checkpoint", False)

    cfg.model = model_config(cfg.model, data, **kwargs)

    model = build_detector(cfg.model)

    if checkpoint:
        mmcv.runner.load_checkpoint(
            model, checkpoint, "cpu", False, logging.getLogger()
        )

    model.forward = types.MethodType(forward_modified, model)
    if cfg.model.neck.type == "SECONDFPN":
        model.neck.forward = types.MethodType(forward_neck, model.neck)

    model.prediction = False

    logging.disable(0)

    return model, cfg
