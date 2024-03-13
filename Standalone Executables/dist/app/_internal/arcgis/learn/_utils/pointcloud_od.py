try:
    import json
    import copy
    import h5py
    import random
    from pathlib import Path
    from .pointcloud_data import (
        concatenate_tensors,
        get_title_text,
        get_device,
        raise_maxpoint_warning,
        remap_labels,
        get_filter_index,
    )
    from torch.utils.data import DataLoader, Dataset, SequentialSampler
    from fastai.data_block import DataBunch
    from fastprogress.fastprogress import progress_bar
    import torch
    import numpy as np
    import types
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mmdet3d.core import LiDARInstance3DBoxes, Box3DMode
    from mmdet3d.core.points.lidar_points import LiDARPoints
    from mmdet3d.datasets.pipelines import Compose
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from IPython.display import display
    from fastai.core import listify, recurse
    from fastai.torch_core import ifnone

    HAS_FASTAI = True
except Exception as e:
    HAS_FASTAI = False


class ODTransform3D(object):
    def __init__(
        self,
        rotation_range=[-0.78539816, 0.78539816],
        scaling_range=[0.95, 1.05],
        flip_x_prob=0.5,
        flip_y_prob=0.5,
    ):
        # In LIDAR coordinates, y (horizontal) and x (vertical) axis.
        self.tfms = [
            dict(
                type="RandomFlip3D",
                sync_2d=False,
                flip_ratio_bev_vertical=flip_x_prob,
                flip_ratio_bev_horizontal=flip_y_prob,
            ),
            dict(
                type="GlobalRotScaleTrans",
                rot_range=rotation_range,
                scale_ratio_range=scaling_range,
            ),
        ]

    def _create_transforms(self, point_cloud_range):
        self.tfms.append(
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range)
        )
        self.tfms.append(
            dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range)
        )
        return Compose(self.tfms)


class PointCloudOD(Dataset):
    def __init__(
        self, path, folder="", class_mapping=None, transform_fn=False, **kwargs
    ):
        self.path = Path(path)
        with open(self.path / folder / "Statistics.json", "r") as f:
            self.statistics = json.load(f)
        self.features_to_keep = copy.copy(kwargs.get("extra_features", []))
        self.input_keys = kwargs.get("attributes", self.statistics["attributes"])
        self.input_keys.pop("rgbType", None)
        features_present = list(self.input_keys.keys())
        if "xyz" in features_present:
            features_present.remove("xyz")
        if self.features_to_keep != []:
            if not all([c in self.input_keys.keys() for c in self.features_to_keep]):
                raise Exception(
                    f"extra_features {self.features_to_keep} must be a subset of {features_present}"
                )
        self.features_to_keep += ["xyz"]
        self.input_keys = {
            k: v for k, v in self.input_keys.items() if k in self.features_to_keep
        }
        ## It is assumed that the pointcloud will have only X,Y & Z.
        extra_dim = (
            sum(
                [
                    len(v["max"]) if isinstance(v.get("max"), list) else 1
                    for _, v in self.input_keys.items()
                ]
            )
            - 3
        )
        self.extra_dim = extra_dim
        self.num_features = 3 + extra_dim  # XYZ + extra dimensions
        self.extra_features = self.input_keys

        self.scale_factor = self.statistics["parameters"]["scaleFactor"]
        self.block_size = self.statistics["parameters"]["tileSize"]

        self.max_point = self.statistics["parameters"]["maxPoints"]
        self.min_points = kwargs.get("min_points", None)
        if self.min_points is not None:
            assert (
                self.min_points < self.max_point
            ), f"min_points({self.min_points}) cannot be greater than max_points({self.max_point}) set during export"

        if "classification" in self.statistics["tileStatistics"]:
            # check if there is not any ground truth boxes in train/val folder
            if self.statistics["numberOfStoredOrientedBoundingBoxes"] == 0:
                raise Exception(
                    "Multipatch labels not found, data is not exported correctly."
                )

            class_info = self.statistics["tileStatistics"]["classification"]["table"]
            full_class_mapping = {
                int(c["classCode"]): str(c["classCode"]) for c in class_info
            }

            orig_classes = list(full_class_mapping.keys())
            self.remap_classes = kwargs.get("remap_classes", {})
            if not all([c in orig_classes for c in self.remap_classes.keys()]):
                raise Exception(
                    f"remap keys {list(self.remap_classes.keys())} must be a subset of {orig_classes}"
                )
            mapped_classes = set(self.remap_classes.values())
            unmapped_classes = set(
                [c for c in orig_classes if c not in self.remap_classes.keys()]
            )
            train_classes = sorted(mapped_classes.union(unmapped_classes))
            train_class_mapping = {int(c): str(c) for c in train_classes}

            self.classes_of_interest = kwargs.get("classes_of_interest", []).copy()

            if not all([c in train_classes for c in self.classes_of_interest]):
                raise Exception(
                    f"classes_of_interest {self.classes_of_interest} must be a subset of {train_classes}"
                )

            if class_mapping is None:
                class_mapping = train_class_mapping
            else:
                class_mapping = {
                    k: class_mapping.get(k, v) for k, v in train_class_mapping.items()
                }

            self.bg_code = kwargs.get("background_classcode", None)
            if self.classes_of_interest == [] and self.bg_code is not None:
                raise Exception(
                    "background_classcode can only be used when `classes_of_interest` is passed."
                )
            if self.bg_code is not None and type(self.bg_code) is not bool:
                raise Exception(
                    "Please enter a boolean value (True or False) for background_classcode."
                )

            if self.classes_of_interest != [] and self.bg_code:
                class_mapping = {
                    k: v
                    for k, v in class_mapping.items()
                    if k in self.classes_of_interest
                }

            class_mapping = {int(k): str(v) for k, v in class_mapping.items()}

            self.class_mapping = class_mapping

            self.c = len(self.class_mapping)
            self.classes = sorted(list(self.class_mapping.keys()))

            self.class2idx = {value: idx for idx, value in enumerate(self.classes)}
            self.idx2class = {idx: value for idx, value in enumerate(self.classes)}

            for k, v in self.remap_classes.items():
                if v in self.classes_of_interest:
                    self.classes_of_interest.append(k)
                    self.class2idx[k] = self.class2idx[v]
                if self.classes_of_interest == [] or not self.bg_code:
                    self.class2idx[k] = self.class2idx[v]

            self.classes_of_interest = sorted(list(set(self.classes_of_interest)))
            self.class2idx = {
                k: v for k, v in self.class2idx.items() if k in orig_classes
            }

            self.color_mapping = kwargs.get(
                "color_mapping",
                {
                    int(k): [random.choice(range(256)) for _ in range(3)]
                    for k, _ in self.class_mapping.items()
                },
            )

            # if classes is not continious classes = [1,3,8] or remapped
            self.remap = (
                self.classes != list(range(len(self.classes)))
                or self.remap_classes.__len__() != 0
            )

            remaped_train_classes = sorted(list(self.class2idx.keys()))

            self.z_range = self.statistics["tileStatistics"]["Z"]

            self.average_box_size = (
                np.array(
                    [
                        clas["orientedBoundingBoxAverageLength"]
                        for c in remaped_train_classes
                        for clas in class_info
                        if int(clas["classCode"]) == c
                    ]
                )
                * self.scale_factor
            ).tolist()
            # get the smallest box idx to calulate the voxel size
            box_idx = np.product(self.average_box_size, axis=1).argmin()
            box_size = self.average_box_size[box_idx]
            # taking 60 voxels in x and 20 voxels in z direction for each bbox
            self.voxel_size = [box_size[0] / 60, box_size[0] / 60, box_size[2] / 20]
            no_of_points = self.statistics["numberOfStoredRecords"]
            no_of_tiles = self.statistics["numberOfStoredTiles"]
            self.no_of_points_per_tile = no_of_points // no_of_tiles

            box_zminmax_range = [
                clas["orientedBoundingBoxZ"]
                for c in remaped_train_classes
                for clas in class_info
                if int(clas["classCode"]) == c
            ]

            box_zmin_range = [
                [box["min"][0], box["max"][0]] for box in box_zminmax_range
            ]

            anchor_zmin = [sum(box) / 2.0 for box in box_zmin_range]

            self.anchor_range = (
                np.array([[-1, -1, z_min, 1, 1, z_min] for z_min in anchor_zmin])
                * self.scale_factor
            ).tolist()

            if self.classes_of_interest != []:
                classes_of_interest = self.classes_of_interest
            else:
                classes_of_interest = None

            self._filter_box_point_percentage = kwargs.get(
                "filter_box_point_percentage", 0.2
            )
            self.transform = transform_fn
            if folder == "val":
                self._filter_box_point_percentage = 0.3
            if self.transform:
                if getattr(transform_fn, "_is_Transform3d", False):
                    self.transform = self.transform._detection_transforms()

                point_cloud_range = (
                    np.array([-1, -1, self.z_range["min"], 1, 1, self.z_range["max"]])
                    * self.scale_factor
                ).tolist()
                self.transform = self.transform._create_transforms(point_cloud_range)

        with h5py.File(self.path / folder / "ListTable.h5", "r") as f:
            self.tiles = f["Tiles"][:]
            relative_files = f["Files"][:]
            self.filenames = np.array([file.decode() for file in relative_files])

            if folder != "" and (
                self.min_points is not None or self.classes_of_interest != []
            ):
                if folder == "val":
                    self.min_points = None
                    classes_of_interest = None

                object_masks = f["ObjectMasks"][:]
                (
                    filterd_idx,
                    skip_block_min_points,
                    skip_block_COI,
                    _,
                ) = get_filter_index(
                    object_masks,
                    self.tiles,
                    self.min_points,
                    classes_of_interest,
                    True,
                )

                self._skip_block_min_points = skip_block_min_points
                self._skip_block_COI = skip_block_COI
                self._total_blocks = len(self.tiles)

                filterd_idx = np.sort(filterd_idx)
                self.tiles = self.tiles[filterd_idx]

                if len(self.tiles) == 0:
                    raise Exception(
                        f"The {folder} set is empty because everything "
                        "got filtered out."
                    )

        self.folder = folder
        if folder != "" and kwargs.get("filter_empty_tiles", False):
            self._filter()

    def _filter(self):
        idx = []
        for id, tile in enumerate(self.tiles):
            if tile[4] > 0:
                idx.append(id)
        self.tiles = self.tiles[idx]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, tile_index):
        data = {}
        tile = self.tiles[tile_index]
        filename = self.path / self.folder / self.filenames[tile[0]]

        with h5py.File(filename, "r") as read_file:
            data["points"] = concatenate_tensors(
                read_file, self.input_keys, tile, pad=False
            )
            data["points"][:, :3] *= self.scale_factor

            if "orientedBoundingBox" in read_file.keys() and self.folder != "":
                data = self._get_bbox(data, read_file, tile)
                if self.transform and random.random() > 0.5:
                    data["points"] = LiDARPoints(
                        data["points"], points_dim=data["points"].shape[-1]
                    )
                    data["bbox3d_fields"] = ["gt_bboxes_3d"]
                    data["flip"] = False
                    data["flip_direction"] = None
                    data = self.transform(data)
                    data = self._post_process(data)
            else:
                data["tile_index"] = tile_index

        data["img_metas"] = dict(
            box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR
        )
        return data

    def _post_process(self, results):
        data = {}
        data["points"] = results["points"].tensor
        data["gt_bboxes_3d"] = results["gt_bboxes_3d"]
        data["gt_labels_3d"] = results["gt_labels_3d"]
        return data

    def _get_bbox(self, data, read_file, tile):
        bboxs = read_file["orientedBoundingBox"][tile[-2] : tile[-2] + tile[-1]].astype(
            np.float32
        )

        # filter boxes with less than 20% points in it of all the points in the box
        bbox_filter = self._get_valid_boxes(read_file, tile)
        bboxs = bboxs[bbox_filter]

        center = bboxs[:, [0, 1, 6]]
        length_width = 2 * bboxs[:, [4, 5]]
        height = (bboxs[:, 7] - bboxs[:, 6])[:, None]
        drx_dry = bboxs[:, [2, 3]]
        yaw = np.arctan2(drx_dry[:, 1], drx_dry[:, 0])[:, None]
        bboxs = np.concatenate(
            (
                center,
                length_width,
                height,
                yaw,
            ),
            axis=1,
        )

        bboxs[:, :6] *= self.scale_factor
        data["gt_bboxes_3d"] = LiDARInstance3DBoxes(bboxs, box_dim=7, with_yaw=True)

        bbox_labels = read_file.get("objectCode", None)
        if bbox_labels is not None:
            bbox_labels = torch.tensor(
                bbox_labels[tile[-2] : tile[-2] + tile[-1]]
            ).long()
            bbox_labels = bbox_labels[bbox_filter]
        else:
            bbox_labels = torch.tensor(np.zeros(center.shape[0])).long()

        # if train on only classes of interest
        if self.bg_code:
            filter_classes = self._get_class_of_interest_mask(bbox_labels)
            bbox_labels = bbox_labels[filter_classes]
            data["gt_bboxes_3d"] = data["gt_bboxes_3d"][filter_classes]

        if self.remap:
            bbox_labels = remap_labels(bbox_labels, self.class2idx).long()

        data["gt_labels_3d"] = bbox_labels
        return data

    def _get_class_of_interest_mask(self, bbox_labels):
        return torch.tensor([label in self.classes for label in bbox_labels]).bool()

    def _get_valid_boxes(self, read_file, tile):
        bbox_point_counts = read_file["orientedBoundingBoxCount"][
            tile[-2] : tile[-2] + tile[-1]
        ].astype(np.uint64)
        bbox_point_perecentage = bbox_point_counts[:, 0] / bbox_point_counts[:, 1]
        bbox_mask = bbox_point_perecentage > self._filter_box_point_percentage
        return bbox_mask

    def get_batch(self, batch_size, device=None):
        idxs = np.random.randint(0, len(self), batch_size)
        data_batch = []
        for idx in idxs:
            data_batch.append(self.__getitem__(idx))

        return to_device(collate_fn(data_batch), device)


def collate_fn(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


defaults_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def to_device(b, device=None):
    device = ifnone(device, defaults_device)
    return recurse(
        lambda x: x.to(device, non_blocking=True) if type(x) is torch.Tensor else x, b
    )


def proc_batch(self, x_batch):
    x_batch = to_device(x_batch, self.device)
    for f in listify(self.tfms):
        x_batch = f(x_batch)
    y_batch = dict(
        img_metas=np.array(x_batch["img_metas"]),
        boxes_3d=x_batch["gt_bboxes_3d"],
        labels_3d=x_batch["gt_labels_3d"],
    )
    return x_batch, y_batch


def show_batch(self, rows=2, color_mapping=None, **kwargs):
    """
    This can be used to visualize the exported dataset. Colors of the PointCloud
    are only used for better visualization, and it does not depict the
    actual classcode colors. Visualization of data, exported in a geographic
    coordinate system is not yet supported.
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional rows. Number of rows to show.
                            Default: ``2``.
    ---------------------   -------------------------------------------
    color_mapping           Optional dictionary. Mapping from object id
                            to RGB values. Colors of the PointCloud via
                            color_mapping are only used for better
                            visualization, and it does not depict the
                            actual classcode colors. Default value example:
                            {0:[220,220,220], 2:[255,0,0], 6:[0,255,0]}.
    =====================   ===========================================

    **kwargs**

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    max_display_point       Optional integer. Maximum number of points
                            to display. Default is 20000. A warning will
                            be raised if the total points to display exceeds
                            this parameter. Setting this parameter will
                            randomly sample the specified number of points
                            and once set, it will be used for future uses.
    ---------------------   -------------------------------------------
    view_type               Optional string. Dataset type to display the
                            results.
                                * ``valid`` - For validation set.
                                * ``train`` - For training set.
                            Default: 'train'.
    =====================   ===========================================
    """
    max_display_point = kwargs.get("max_display_point", 20000)
    ds_type = kwargs.get("view_type", "train")
    color_mapping = self.color_mapping if color_mapping is None else color_mapping
    if ds_type == "train":
        ds = self.train_ds
    else:
        ds = self.valid_ds
    ids = list(range(len(ds)))
    random.shuffle(ids)
    for row_idx in range(rows):
        data_id = ids[row_idx]
        datapoint = ds[data_id]
        pc = datapoint["points"][:, :3]
        bbox = datapoint["gt_bboxes_3d"]
        bbox_class = datapoint["gt_labels_3d"]
        if pc.shape[0] > max_display_point:
            raise_maxpoint_warning(row_idx, kwargs, None, max_display_point)
            mask = torch.from_numpy(
                np.random.randint(0, pc.shape[0], max_display_point, dtype=np.int64)
            )
        else:
            mask = torch.arange(0, pc.shape[0])
        pc = pc[mask]
        x, y, z = pc.T

        scene = dict(aspectmode="data")
        layout = go.Layout(
            width=750,
            height=512,
            scene=scene,
        )

        color_list = np.tile(np.array([127, 127, 127]), (x.shape[0], 1)).astype(
            np.uint8
        )
        point_box_id = bbox.points_in_boxes_part(pc[:, :3].cuda()).cpu()

        mesh_objects = []
        for idx, coords in enumerate(bbox.corners):
            x_b, y_b, z_b = coords.T
            class_value = self.idx2class[bbox_class[idx].item()]
            class_name = self.class_mapping[class_value]
            class_name = f"class {class_name}"
            mesh_objects.append(
                go.Mesh3d(
                    x=x_b,
                    y=y_b,
                    z=z_b,
                    color="lightpink",
                    opacity=0.50,
                    alphahull=1,
                    text=class_name,
                )
            )

            color_list[point_box_id == idx] = color_mapping[class_value]

        fig = go.Figure(
            data=mesh_objects
            + [
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=1, color=color_list),
                )
            ],
            layout=layout,
        )
        fig.show()


def pointcloud_od(
    path,
    class_mapping,
    batch_size,
    transform_fn,
    databunch_kwargs,
    **kwargs,
):
    del databunch_kwargs["bs"]
    env_device_type = str(databunch_kwargs.get("device"))
    if env_device_type == "cpu" or not torch.cuda.is_available():
        raise Exception(f"CPU is not supported for 'dataset_type':'PointCloudOD'.")

    train_dataset = PointCloudOD(
        path,
        folder="train",
        class_mapping=class_mapping,
        transform_fn=transform_fn,
        **kwargs,
    )
    val_dataset = PointCloudOD(
        path, folder="val", class_mapping=class_mapping, **kwargs
    )
    train_dl = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=batch_size, **databunch_kwargs
    )
    valid_dl = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=batch_size, **databunch_kwargs
    )
    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device, collate_fn=collate_fn)
    data.train_dl.proc_batch = types.MethodType(proc_batch, data.train_dl)
    data.valid_dl.proc_batch = types.MethodType(proc_batch, data.valid_dl)
    data.show_batch = types.MethodType(show_batch, data)
    block_z_range = [
        min(train_dataset.z_range["min"], val_dataset.z_range["min"]),
        max(train_dataset.z_range["max"], val_dataset.z_range["max"]),
    ]
    data.range = (
        np.array([-1, -1, block_z_range[0], 1, 1, block_z_range[1]]) * data.scale_factor
    ).tolist()

    data.path = data.train_ds.path
    data.chip_size = None
    data.chip_size = None
    data._image_space_used = None

    return data


def plot_results(
    row_idx,
    points,
    gt_boxes,
    gt_boxes_labels,
    gt_points_labels,
    pred_boxes,
    pred_boxes_labels,
    pred_points_labels,
    **kwargs,
):
    color_mapping = kwargs.get("color_mapping")
    class_mapping = kwargs.get("class_mapping")
    idx2class = kwargs.get("idx2class")
    save_html = kwargs.get("save_html", False)
    save_path = kwargs.get("save_path", ".")
    max_display_point = kwargs.get("max_display_point", 20000)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])
    boxes_3d = [gt_boxes, pred_boxes]
    labels_3d = [gt_boxes_labels, pred_boxes_labels]
    points_labels_3d = [gt_points_labels, pred_points_labels]
    x, y, z = points.T
    for col_idx, (boxes, box_labels, point_labels) in enumerate(
        zip(boxes_3d, labels_3d, points_labels_3d)
    ):
        color_list = np.tile(np.array([127, 127, 127]), (points.shape[0], 1)).astype(
            np.uint8
        )
        for box_idx, coords in enumerate(boxes.corners):
            x_b, y_b, z_b = coords.detach().cpu().T
            class_value = idx2class[box_labels[box_idx].item()]
            class_name = class_mapping[class_value]
            if col_idx == 1:
                class_name = f"pred_class {class_name}"
            else:
                class_name = f"class {class_name}"
            fig.add_trace(
                go.Mesh3d(
                    x=x_b,
                    y=y_b,
                    z=z_b,
                    color="lightpink",
                    opacity=0.50,
                    alphahull=1,
                    text=class_name,
                ),
                row=1,
                col=col_idx + 1,
            )
            color_list[point_labels == box_idx] = color_mapping[class_value]
        pc_object = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=1, color=color_list),
        )
        fig.add_trace(pc_object, row=1, col=col_idx + 1)

    title_text = get_title_text(row_idx, save_html, max_display_point)
    fig.update_layout(
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        title_text=title_text,
        width=kwargs.get("width", 750),
        height=kwargs.get("height", 512),
        showlegend=False,
        title_x=0.5,
    )

    fig2 = go.FigureWidget(fig)

    def cam_change(layout, camera):
        if fig2.layout.scene2.camera == camera:
            return
        else:
            fig2.layout.scene2.camera = camera

    def cam_change2(layout, camera):
        if fig2.layout.scene.camera == camera:
            return
        else:
            fig2.layout.scene.camera = camera

    fig2.layout.scene.on_change(cam_change, "camera")
    fig2.layout.scene2.on_change(cam_change2, "camera")
    if save_html:
        save_path = Path(save_path)
        plotly.io.write_html(fig, str(save_path / "show_results.html"))
        return
    else:
        display(fig2)


def confusion_matrix3d(pred, target, n_gts, classes, iou_thresh=0.1):
    tps, p_clas, p_scores = [], [], []
    for idx in range(len(pred)):
        pred_bboxes, pred_labels, pred_scores = (
            pred[idx]["boxes_3d"],
            pred[idx]["labels_3d"].detach().cpu(),
            pred[idx]["scores_3d"].detach().cpu(),
        )
        tgt_bboxes, tgt_lables = (
            target["boxes_3d"][idx],
            target["labels_3d"][idx].detach().cpu(),
        )

        if len(pred_labels) != 0 and len(tgt_lables) != 0:
            ious = pred_bboxes.overlaps(pred_bboxes, tgt_bboxes)
            max_iou, matches = ious.max(1)
            detected = []
            for i in range(len(pred_labels)):
                if (
                    max_iou[i] >= iou_thresh
                    and matches[i] not in detected
                    and tgt_lables[matches[i]] == pred_labels[i]
                ):
                    detected.append(matches[i])
                    tps.append(1)
                else:
                    tps.append(0)
            p_clas.append(pred_labels)
            p_scores.append(pred_scores)
            n_gts += ((tgt_lables[:, None]) == classes[None, :]).sum(0)

    return tps, p_scores, p_clas, n_gts


def predict_h5(self, path, output_path, **kwargs):
    self._free_memory()
    path = Path(path)
    if output_path is None:
        output_path = path.parent / "results"
    else:
        output_path = Path(output_path)
    progressor = kwargs.pop("progressor", None)
    batch_size = kwargs.get("batch_size", 1)

    extra_features = self._data.features_to_keep.copy()
    if "xyz" in extra_features:
        extra_features.remove("xyz")

    data = PointCloudOD(path, extra_features=extra_features)

    if progressor is not None:
        progressor.set_total_blocks(len(data))

    output_path.mkdir(parents=True, exist_ok=True)
    sampler = SequentialSampler(data)

    dataloader = DataLoader(
        data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=sampler,
    )
    predict_batch_h5(self, dataloader, output_path, progressor, **kwargs)
    self._reset_thresh()

    return output_path


def predict_batch_h5(self, dl, output_path, progressor, **kwargs):
    detect_thresh = kwargs.get("detect_thresh", 0.1)
    nms_overlap = kwargs.get("nms_overlap", 0.6)
    current_file_name = ""
    for data in progress_bar(dl):
        tile_index = data.pop("tile_index")
        data = to_device(data)
        pred = self._pred_batch(data, detect_thresh, nms_overlap)

        tile = dl.dataset.tiles[tile_index]
        if len(tile.shape) < 2:
            tile = tile[None]
        fname = np.array(dl.dataset.filenames)[tile[:, 0]]
        _, unique_index = np.unique(fname, return_index=True)
        # get unique name form sorted index in actual array
        fname = [fname[i] for i in np.sort(unique_index)]
        # add batch_size for spliting prediction till last batch number
        unique_index = list(np.sort(unique_index)) + [dl.batch_size]
        for i, ufname in enumerate(fname):
            ufname = dl.dataset.path / dl.dataset.folder / ufname
            if ufname != current_file_name:
                current_file_name = ufname
                h5_file = h5py.File(current_file_name, "r")
                no_of_point, _ = h5_file["xyz"].shape
                h5_file.close()
                boxes_pred = []
                labels_pred = []
                confidence_pred = []
                boxes_tile_no = []
                low = high = 0

            predictions = split_prediction(
                self,
                pred[unique_index[i] : unique_index[i + 1]],
                data["points"][unique_index[i] : unique_index[i + 1]],
                tile_index[unique_index[i] : unique_index[i + 1]],
            )

            boxes_pred.extend(predictions[0])
            labels_pred.extend(predictions[1])
            confidence_pred.extend(predictions[2])
            boxes_tile_no.extend(predictions[3])
            high = low + predictions[4]

            if high == no_of_point:
                save_h5(
                    output_path / dl.dataset.filenames[int(tile[unique_index[i]][0])],
                    np.array(boxes_pred),
                    np.array(labels_pred, dtype=np.uint8),
                    np.array(confidence_pred),
                    np.array(boxes_tile_no, dtype=np.uint64),
                )
            low = high

        if progressor is not None:
            progressor.current_block(tile_index[0])


def export_boxes(bboxes, scale_factor):
    """
    input: [center_x, center_y, z_min, x_size, y_size, z_size, yaw]
    return: [center_x, center_y, direction_x, direction_y, half_size_x, half_size_y, z_min, z_max]
    """
    bboxes[:, :6] /= scale_factor
    cenetr_xy = bboxes[:, [0, 1]]
    z_min = bboxes[:, [2]]
    z_max = z_min + bboxes[:, [5]]
    half_size_xy = bboxes[:, [3, 4]] / 2
    direction_x, direction_y = np.cos(bboxes[:, [6]]), np.sin(bboxes[:, [6]])

    return np.concatenate(
        (cenetr_xy, direction_x, direction_y, half_size_xy, z_min, z_max), axis=1
    ).tolist()


def split_prediction(model, preds, points, tile_index):
    batch_export_bboxs = []
    batch_labels = []
    batch_confidance = []
    batch_box_tiles = []
    no_of_points_in_batch = 0
    for idx, pred in enumerate(preds):
        boxes = pred["boxes_3d"]
        labels = pred["labels_3d"]
        scores = pred["scores_3d"]
        no_of_points_in_batch += points[idx].shape[0]

        batch_export_bboxs.extend(export_boxes(boxes.tensor, model._data.scale_factor))
        batch_labels.extend(
            np.array(list(model._data.idx2class.values()))[labels.tolist()]
        )
        batch_confidance.extend(scores.tolist())
        batch_box_tiles.extend([tile_index[idx]] * scores.shape[0])

    return (
        batch_export_bboxs,
        batch_labels,
        batch_confidance,
        batch_box_tiles,
        no_of_points_in_batch,
    )


def save_h5(filename, boxes_pred, labels_pred, confidences_pred, boxes_tile_no):
    filename = Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    filename_pred = filename.parent / (filename.stem + "_pred.h5")
    with h5py.File(filename_pred, "w") as file:
        file.create_dataset("pred_boxes", data=boxes_pred)
        file.create_dataset("pred_labels", data=labels_pred)
        file.create_dataset("pred_confidence", data=confidences_pred)
        file.create_dataset("pred_boxes_block", data=boxes_tile_no)
