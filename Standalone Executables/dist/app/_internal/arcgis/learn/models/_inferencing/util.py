import torch
from torch import tensor, Tensor
import numpy as np
from torch import tensor
import torch.nn as nn
import torch
import math

from typing import Union, Tuple, Optional, List
from .._unet_utils import is_contiguous as is_cont


def A(*a):
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
std = 255 * np.array(imagenet_stats[1], dtype=np.float32)

norm = lambda x: (x - mean) / std
denorm = lambda x: x * std + mean


def scale_batch(
    image_batch, model_info, normalization_stats=None, break_extract_bands=False
):
    if normalization_stats is None:
        normalization_stats = model_info.get("NormalizationStats", None)
    if break_extract_bands:
        # Only for change detection
        # if subset of extract bands are specified fix this.
        n_bands = len(model_info["ExtractBands"]) // 2
        band_min_values = np.array(normalization_stats["band_min_values"])[
            model_info["ExtractBands"][:n_bands]
        ].reshape(1, -1, 1, 1)
        band_max_values = np.array(normalization_stats["band_max_values"])[
            model_info["ExtractBands"][:n_bands]
        ].reshape(1, -1, 1, 1)
    else:
        if normalization_stats is None:
            modtype = model_info.get("InferenceFunction", None)
            if modtype == "ArcGISImageTranslation.py":
                band_min_values = np.full((image_batch.shape[1],), 0).reshape(
                    1, -1, 1, 1
                )
                band_max_values = np.full((image_batch.shape[1],), 255).reshape(
                    1, -1, 1, 1
                )
        else:
            band_min_values = np.array(normalization_stats["band_min_values"])[
                model_info["ExtractBands"]
            ].reshape(1, -1, 1, 1)
            band_max_values = np.array(normalization_stats["band_max_values"])[
                model_info["ExtractBands"]
            ].reshape(1, -1, 1, 1)
    img_scaled = (image_batch - band_min_values) / (band_max_values - band_min_values)
    return img_scaled


def normalize_batch(
    image_batch, model_info=None, normalization_stats=None, prithvi=False
):
    if normalization_stats is None:
        normalization_stats = model_info.get("NormalizationStats", None)
    scaled_mean_values = np.array(normalization_stats["scaled_mean_values"])[
        model_info["ExtractBands"]
    ].reshape(1, -1, 1, 1)
    scaled_std_values = np.array(normalization_stats["scaled_std_values"])[
        model_info["ExtractBands"]
    ].reshape(1, -1, 1, 1)
    if prithvi:
        img_normed = (image_batch - scaled_mean_values) / scaled_std_values
        return img_normed
    else:
        img_scaled = scale_batch(image_batch, model_info)
        img_normed = (img_scaled - scaled_mean_values) / scaled_std_values
        return img_normed


def ts_normalization(x, m, s):
    x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
    x = (x - m) / s
    return torch.tensor(x[None, :, :, :, None])


def rescale_batch(
    image_batch, model_info, normalization_stats=None, break_extract_bands=False
):
    if normalization_stats is None:
        modtype = model_info.get("InferenceFunction", None)
        if modtype == "ArcGISImageTranslation.py":
            min_values = np.full((image_batch.shape[1],), 0).reshape(1, -1, 1, 1)
            max_values = np.full((image_batch.shape[1],), 255).reshape(1, -1, 1, 1)
    else:
        min_values = np.array(normalization_stats["band_min_values"])
        max_values = np.array(normalization_stats["band_max_values"])

        x = image_batch
        if x.shape[1] > min_values.shape[0]:
            res = x.shape[1] - min_values.shape[0]
            last_val = torch.tensor([min_values[min_values.shape[0] - 1]])
            for i in range(res):
                min_values = np.concatenate((min_values, last_val), axis=0)
        if x.shape[1] > max_values.shape[0]:
            res = x.shape[1] - max_values.shape[0]
            last_val = torch.tensor([max_values[max_values.shape[0] - 1]])
            for i in range(res):
                max_values = np.concatenate((max_values, last_val), axis=0)

        min_values = min_values.reshape(1, -1, 1, 1)
        max_values = max_values.reshape(1, -1, 1, 1)
    img_rescaled = (((image_batch + 1) / 2) * (max_values - min_values)) + min_values
    return img_rescaled


def pred2dict(bb_np, score, cat_str, c):
    # convert to top left x,y bottom right x,y
    return {
        "x1": bb_np[1],
        "x2": bb_np[3],
        "y1": bb_np[0],
        "y2": bb_np[2],
        "score": score,
        "category": cat_str,
        "class": c,
    }


def to_np(x):
    return x.cpu().numpy()


def load_weights(m, p):
    if p.split(".")[-1] == "h5":
        sd = torch.load(p, map_location=lambda storage, loc: storage)
    elif p.split(".")[-1] == "pth":
        # sd = torch.load(p)['model']
        sd = torch.load(p, map_location=lambda storage, loc: storage)["model"]
    m.load_state_dict(sd)
    return m


def predict_(model, images, device):
    model = model.to(device)
    images = tensor(images).to(device).float()
    clas, bbox = model(images)
    return clas, bbox


def hw2corners(ctr, hw):
    return torch.cat([ctr - hw / 2, ctr + hw / 2], dim=1)


def actn_to_bb(actn, anchors, grid_sizes):
    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:, :2] / 2 * grid_sizes) + anchors[:, :2]
    actn_hw = (actn_bbs[:, 2:] / 2 + 1) * anchors[:, 2:]
    return hw2corners(actn_centers, actn_hw)


def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


try:

    @torch.jit.script
    def nms_jit(
        boxes, scores, overlap: float = 0.5, top_k: int = 100
    ) -> Tuple[Tensor, Tensor]:
        keep = torch.zeros(scores.size(0), dtype=torch.long).to(scores.device)

        count = 0
        if boxes.numel() == 0:
            return keep, torch.tensor([count]).to(scores.device).int()
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        limiter = 0 if top_k == 0 else len(idx) - top_k  # TODO:here
        idx = idx[limiter:]  # indices of the top-k largest vals
        xx1 = torch.empty((0,), dtype=boxes.dtype).to(scores.device)
        yy1 = torch.empty((0,), dtype=boxes.dtype).to(scores.device)
        xx2 = torch.empty((0,), dtype=boxes.dtype).to(scores.device)
        yy2 = torch.empty((0,), dtype=boxes.dtype).to(scores.device)
        w = torch.empty((0,), dtype=boxes.dtype).to(scores.device)
        h = torch.empty((0,), dtype=boxes.dtype).to(scores.device)

        while idx.numel() > 0:
            i = idx[len(idx) - 1]  # index of current largest val
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = (
                idx[: len(idx) - 1].clone().to(scores.device)
            )  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]

        return keep, torch.tensor([count]).to(scores.device).int()

except:
    print("\ntorch not available\n")


def get_nms_preds(b_clas, b_bb, idx, anchors, grid_sizes, classes, nms_overlap, thres):
    a_ic = actn_to_bb(b_bb[idx], anchors, grid_sizes)
    clas_pr, clas_ids = b_clas[idx].max(1)
    clas_pr = clas_pr.sigmoid()

    conf_scores = b_clas[idx].sigmoid().t().data

    out1, out2, cc = [], [], []
    for cl in range(1, len(conf_scores)):
        c_mask = conf_scores[cl] > thres
        if c_mask.sum() == 0:
            continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1).expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4)
        ids, count = nms(
            boxes.data, scores, nms_overlap, 50
        )  # FIX- NMS overlap hardcoded
        ids = ids[:count]
        out1.append(scores[ids])
        out2.append(boxes.data[ids])
        cc.append([cl] * count)
    if cc == []:
        cc = [[0]]
    cc = tensor(np.concatenate(cc))
    if out1 == []:
        out1 = [torch.Tensor()]
    out1 = torch.cat(out1)
    if out2 == []:
        out2 = [torch.Tensor()]
    out2 = torch.cat(out2)
    bbox, clas, prs, thresh = out2, cc, out1, thres  # FIX- hardcoded threshold
    return predictions(
        bbox, to_np(clas), to_np(prs) if prs is not None else None, thresh, classes
    )


def predictions(
    bbox, clas=None, prs=None, thresh=0.3, classes=None
):  # FIX- take threshold from user
    # bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
    if len(classes) is None:
        raise Exception("Classes are None")
    else:
        classes = ["bg"] + classes
    bb = bbox
    if prs is None:
        prs = [None] * len(bb)
    if clas is None:
        clas = [None] * len(bb)
    predictions = []
    for i, (b, c, pr) in enumerate(zip(bb, clas, prs)):
        if (b[2] > 0) and (pr is None or pr > thresh):
            cat_str = classes[c]
            score = pr * 100
            bb_np = to_np(b).astype("float64")
            predictions.append(pred2dict(bb_np, score, cat_str, c))
    return predictions


def detect_objects_image_space(
    model, tiles, anchors, grid_sizes, device, classes, nms_overlap, thres, model_info
):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    if "NormalizationStats" in model_info:
        img_normed = normalize_batch(tiles, model_info)
    else:
        img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    clas, bbox = predict_(model, img_normed, device)

    preds = {}

    for batch_idx in range(bbox.size()[0]):
        preds[batch_idx] = get_nms_preds(
            clas, bbox, batch_idx, anchors, grid_sizes, classes, nms_overlap, thres
        )

    batch_size = bbox.size()[0]
    side = math.sqrt(batch_size)

    num_boxes = 0
    for batch_idx in range(batch_size):
        num_boxes = num_boxes + len(preds[batch_idx])

    bounding_boxes = np.empty(shape=(num_boxes, 4), dtype=float)
    scores = np.empty(shape=(num_boxes), dtype=float)
    classes = np.empty(shape=(num_boxes), dtype=np.uint8)

    idx = 0
    for batch_idx in range(batch_size):
        i, j = batch_idx // side, batch_idx % side

        for pred in preds[batch_idx]:
            bounding_boxes[idx, 0] = (pred["y1"] + i) * tile_height
            bounding_boxes[idx, 1] = (pred["x1"] + j) * tile_width
            bounding_boxes[idx, 2] = (pred["y2"] + i) * tile_height
            bounding_boxes[idx, 3] = (pred["x2"] + j) * tile_width
            scores[idx] = pred["score"]
            classes[idx] = pred["class"]
            idx = idx + 1

    return bounding_boxes, scores, classes


def segment_image(model, images, device, predict_bg, model_info):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    output = model(normed_batch_tensor)
    ignore_mapped_class = model_info.get("ignore_mapped_class", [])
    for k in ignore_mapped_class:
        output[:, k] = output.min() - 1
    if predict_bg:
        return output.max(dim=1)[1]
    else:
        output[:, 0] = output.min() - 1
        return output.max(dim=1)[1]


def superres_image(model, images, device):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    output = model(normed_batch_tensor)
    return output


def cyclegan_image(model, images, device, direction):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    if direction == "BtoA":
        output = model.G_A(normed_batch_tensor)
    else:
        output = model.G_B(normed_batch_tensor)
    return output


def pix2pix_image(model, images, device, label_nc=0):
    from .._pix2pix_hd_utils import encode_input

    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    if label_nc:
        normed_batch_tensor, _, _, _ = encode_input(normed_batch_tensor, label_nc)
    output = model.G(normed_batch_tensor)
    return output


def remap(tensor, idx2pixel):
    modified_tensor = torch.zeros_like(tensor)
    for id, pixel in idx2pixel.items():
        modified_tensor[tensor == id] = pixel
    return modified_tensor


def wnet_cgan_image(model, images_a, images_b, device):
    model = model.to(device)
    normed_batch_tensor_a, normed_batch_tensor_b = (
        tensor(images_a).to(device).float(),
        tensor(images_b).to(device).float(),
    )
    output = model.G(normed_batch_tensor_a, normed_batch_tensor_b)
    return output


def pixel_classify_image(model, tiles, device, classes, predict_bg, model_info):
    class_values = [clas["Value"] for clas in model_info["Classes"]]
    is_contiguous = is_cont([0] + class_values)

    if not is_contiguous:
        pixel_mapping = [0] + class_values
        idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}

    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    if "NormalizationStats" in model_info:
        img_normed = normalize_batch(tiles, model_info)
    else:
        img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    semantic_predictions = segment_image(
        model, img_normed, device, predict_bg, model_info
    )
    if not is_contiguous:
        semantic_predictions = remap(semantic_predictions, idx2pixel)
    return semantic_predictions


def pixel_classify_superres_image(
    model, tiles, device, model_info, sampling, n_timestep
):
    import numpy as np

    tiles, is_multispec = tensor(tiles), model_info.get("is_multispec")
    modarch = model_info.get("ModelArch", "UNet")
    if modarch == "SR3":
        norm_stats_a = model_info.get("image_stats", None)
        norm_stats_b = model_info.get("image_stats2", None)
        model_info["ExtractBands"] = list(range(tiles.shape[1]))

        if model_info.get("is_multispec"):
            img_scaled = scale_batch(tiles, model_info, norm_stats_a)
        else:
            img_scaled = tiles / 255
        img_normed = -1 + 2 * img_scaled

        kwargs = model_info.get("Kwargs")

        if sampling == "ddim":
            nstp = {"n_timestep": kwargs.get("n_timestep", 1000)}
        else:
            nstp = {"n_timestep": n_timestep}
        combkwargs = {**kwargs, **nstp}
        model.set_new_noise_schedule(device, **combkwargs)

        model = model.to(device)
        normed_batch_tensor = tensor(img_normed).to(device).float()

        if sampling == "ddim":
            superres_predictions = model.super_resolution(
                x_in=normed_batch_tensor,
                continous=False,
                sampling_timesteps=n_timestep,
                ddim_sampling_eta=1,
                sampling="ddim",
            )
        else:
            superres_predictions = (
                model.super_resolution(x_in=normed_batch_tensor, continous=True)
            )[-normed_batch_tensor.shape[0] :]

        min_values = (
            torch.tensor(norm_stats_b["band_min_values"], device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        max_values = (
            torch.tensor(norm_stats_b["band_max_values"], device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        superres_predictions = (
            superres_predictions.add_(1)
            .div_(2)
            .mul_(max_values - min_values)
            .add_(min_values)
        )
    else:
        if is_multispec:
            n_mean, n_std = np.array(
                model_info.get("image_stats")[0], dtype=np.float32
            ), np.array(model_info.get("image_stats")[1], dtype=np.float32)
            dn_mean, dn_std = (
                model_info.get("image_stats2")[0],
                model_info.get("image_stats2")[1],
            )
            nrm = lambda x, m, s: (x - m) / s
            img_normed = nrm(tiles.permute(0, 2, 3, 1), n_mean, n_std).permute(
                0, 3, 1, 2
            )
        else:
            img_normed = norm(tiles.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            dn_mean, dn_std = imagenet_stats[0], imagenet_stats[1]
        superres_predictions = superres_image(model, img_normed, device)
        superres_predictions = (
            superres_predictions
            * torch.tensor(dn_std).view(1, -1, 1, 1).to(superres_predictions)
        ) + torch.tensor(dn_mean).view(1, -1, 1, 1).to(superres_predictions)
    return superres_predictions


def pixel_classify_cyclegan_image(model, tiles, device, direction, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    num_channel = model_info.get("n_intput_channel", None)
    if num_channel == None:
        num_channel = model_info.get("n_channel", None)
    num_channel_tar = model_info.get("n_channel", None)
    if direction == "BtoA":
        norm_stats_a = model_info.get("NormalizationStats_b", None)
    else:
        norm_stats_a = model_info.get("NormalizationStats", None)
    img_scaled = scale_batch(tiles, model_info, norm_stats_a)
    img_normed = -1 + 2 * img_scaled
    if img_normed.shape[1] < num_channel:
        cont = []
        for j in range(img_normed.shape[0]):
            tile = img_normed[j, :, :, :]
            last_tile = np.expand_dims(tile[tile.shape[0] - 1, :, :], 0)
            res = abs(num_channel - tile.shape[0])
            for i in range(res):
                tile = np.concatenate((tile, last_tile), axis=0)
            cont.append(tile)
        img_normed = np.stack(cont, axis=0)

    cyclegan_predictions = cyclegan_image(model, img_normed, device, direction)
    if direction == "BtoA":
        norm_stats_b = model_info.get("NormalizationStats", None)
    else:
        norm_stats_b = model_info.get("NormalizationStats_b", None)
    cyclegan_predictions = cyclegan_predictions.detach().cpu().numpy()
    if num_channel_tar == None:
        cyclegan_predictions = torch.tensor(
            rescale_batch(cyclegan_predictions, model_info, norm_stats_b)
        )
    else:
        if direction == "BtoA":
            num_channel_tar_rev = model_info.get("n_channel_rev", None)
            cyclegan_predictions = torch.tensor(
                rescale_batch(cyclegan_predictions, model_info, norm_stats_b)
            )[:, :num_channel_tar_rev, :, :]
        else:
            cyclegan_predictions = torch.tensor(
                rescale_batch(cyclegan_predictions, model_info, norm_stats_b)
            )[:, :num_channel_tar, :, :]
    return cyclegan_predictions


def pixel_classify_ts_image(model, tiles, device, model_info):
    tiles = torch.tensor(tiles)
    tile_height, tile_width, nch = tiles.shape[2], tiles.shape[3], tiles.shape[1]
    means = np.array((model_info.get("mean_norm_stats", None))["mean_stats"])
    stds = np.array((model_info.get("std_norm_stats", None))["std_stats"])
    tile_stack = lambda lst: torch.cat(lst, axis=0)
    img_arr = tile_stack([tile.permute(1, 2, 0)[None, :, :, :] for tile in tiles])
    timeseries_arr = tile_stack(
        [
            torch.reshape(tile_arr, (1, tile_height * tile_width, nch))
            for tile_arr in img_arr
        ]
    )
    ntemp = int(model_info.get("n_temporal", None))
    nchannel = int(model_info.get("n_channel", None))
    ntemp_infer = model_info.get("timestep_infer", None)
    nchannel_infer = model_info.get("channels_infer", None)
    class_dict = model_info.get("Class_mapping", None)
    num_class_dict = model_info.get("Num_class_mapping", None)
    convertmap = model_info.get("convertmap", None)
    bandidx = model_info.get("bandindex", None)
    timeidx = model_info.get("timeindex", None)

    if num_class_dict:
        pixel_num_class_mapping = num_class_dict
    else:
        pixel_num_class_mapping = class_dict

    if bandidx or timeidx:
        final = tile_stack(
            [
                torch.reshape(
                    time_arr, (1, time_arr.shape[0], ntemp_infer, nchannel_infer)
                )
                for time_arr in timeseries_arr
            ]
        )
        if bandidx:
            final = final[:, :, :, np.array(bandidx) - 1]
        if timeidx:
            final = final[:, :, np.array(timeidx) - 1, :]

    else:
        final = tile_stack(
            [
                torch.reshape(time_arr, (1, time_arr.shape[0], ntemp, nchannel))
                for time_arr in timeseries_arr
            ]
        )

    def ts_normalization(x, m, s):
        x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
        x = (x - m) / s
        return torch.tensor(x[None, :, :, :, None])

    normalized_ts = tile_stack(
        [
            ts_normalization(np.rollaxis(tile.numpy(), 0, 3), means, stds)
            for tile in final
        ]
    )
    model = model.to(device)
    pred_list = []
    for i in normalized_ts:
        sim = torch.ones(i.shape[0], i.shape[1], 1).to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(i.float().to(device), sim)

        pred_out = prediction.argmax(dim=1).cpu()
        pred_list.append(pred_out)

    if convertmap:
        convmap = {int(value): int(key) for key, value in convertmap.items()}
        pixel_num_class_mapping = {
            convmap.get(int(key)): int(value)
            for key, value in pixel_num_class_mapping.items()
        }

    remap_pred_list = [
        torch.tensor(
            [pixel_num_class_mapping.get(int(item.numpy())) for item in tile_pred_list]
        )
        for tile_pred_list in pred_list
    ]
    tile_rshp = tile_stack(
        [torch.reshape(i, (1, 1, tile_height, tile_width)) for i in remap_pred_list]
    )
    return tile_rshp


def pixel_classify_pix2pix_image(model, tiles, device, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]

    num_channel = model_info.get("n_intput_channel", None)
    if num_channel == None:
        num_channel = model_info.get("n_channel", None)
    num_channel_tar = model_info.get("n_channel", None)

    norm_stats_a = model_info.get("NormalizationStats", None)
    model_info["ExtractBands"] = list(range(tiles.shape[1]))
    img_scaled = scale_batch(tiles, model_info, norm_stats_a)
    img_normed = -1 + 2 * img_scaled

    if img_normed.shape[1] < num_channel:
        cont = []
        for j in range(img_normed.shape[0]):
            tile = img_normed[j, :, :, :]
            last_tile = np.expand_dims(tile[tile.shape[0] - 1, :, :], 0)
            res = abs(num_channel - tile.shape[0])
            for i in range(res):
                tile = np.concatenate((tile, last_tile), axis=0)
            cont.append(tile)
        img_normed = np.stack(cont, axis=0)

    pix2pix_predictions = pix2pix_image(model, img_normed, device)
    norm_stats_b = model_info.get("NormalizationStats_b", None)
    pix2pix_predictions = pix2pix_predictions.detach().cpu().numpy()
    if num_channel_tar == None:
        pix2pix_predictions = torch.tensor(
            rescale_batch(pix2pix_predictions, model_info, norm_stats_b)
        )
    else:
        pix2pix_predictions = torch.tensor(
            rescale_batch(pix2pix_predictions, model_info, norm_stats_b)
        )[:, :num_channel_tar, :, :]

    return pix2pix_predictions


def pixel_classify_wnet_image(model, tiles, device, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]

    num_band_a = model_info.get("n_band_a", None)
    num_band_b = model_info.get("n_band_b", None)
    num_band_tar = model_info.get("n_band_c", None)

    if num_band_a == 1 and num_band_b == 1:
        tile_a = tiles[:, num_band_a, :, :][:, None, :, :]
        tile_b = tiles[:, num_band_b, :, :][:, None, :, :]
    else:
        tile_a = tiles[:, :num_band_a, :, :]
        tile_b = tiles[:, num_band_a:, :, :]

    num_chanel = model_info.get("n_channel", None)

    norm_stats = model_info.get("NormalizationStats", None)
    norm_stats_b = model_info.get("NormalizationStats_b", None)

    img_scaled_a = scale_batch(tile_a, model_info, norm_stats)
    img_normed_a = -1 + 2 * img_scaled_a

    img_scaled_b = scale_batch(tile_b, model_info, norm_stats_b)
    img_normed_b = -1 + 2 * img_scaled_b

    if img_normed_a.shape[1] < num_chanel:
        cont = []
        for j in range(img_normed_a.shape[0]):
            tile = img_normed_a[j, :, :, :]
            last_tile = np.expand_dims(tile[tile.shape[0] - 1, :, :], 0)
            res = abs(num_chanel - tile.shape[0])
            for i in range(res):
                tile = np.concatenate((tile, last_tile), axis=0)
            cont.append(tile)
        img_normed_a = np.stack(cont, axis=0)

    if img_normed_b.shape[1] < num_chanel:
        cont = []
        for j in range(img_normed_b.shape[0]):
            tile = img_normed_b[j, :, :, :]
            last_tile = np.expand_dims(tile[tile.shape[0] - 1, :, :], 0)
            res = abs(num_chanel - tile.shape[0])
            for i in range(res):
                tile = np.concatenate((tile, last_tile), axis=0)
            cont.append(tile)
        img_normed_b = np.stack(cont, axis=0)

    wnet_predictions = wnet_cgan_image(model, img_normed_a, img_normed_b, device)
    norm_stats_c = model_info.get("NormalizationStats_c", None)
    wnet_predictions = wnet_predictions.detach().cpu().numpy()
    wnet_predictions = torch.tensor(
        rescale_batch(wnet_predictions, model_info, norm_stats_c)
    )[:, :num_band_tar, :, :]
    return wnet_predictions


def variable_tile_size_check(json_info, parameters):
    if json_info.get("SupportsVariableTileSize", False):
        parameters.extend(
            [
                {
                    "name": "tile_size",
                    "dataType": "numeric",
                    "value": int(json_info["ImageHeight"]),
                    "required": False,
                    "displayName": "Tile Size",
                    "description": "Tile size used for inferencing",
                }
            ]
        )
    return parameters


def detect_change(model, batch, device, model_info):
    mean = 255 * np.array(
        [0.5] * (len(model_info["ExtractBands"]) // 2), dtype=np.float32
    )
    std = 255 * np.array(
        [0.5] * (len(model_info["ExtractBands"]) // 2), dtype=np.float32
    )
    norm = lambda x: (x - mean) / std
    B, C, H, W = batch.shape
    batch_before = batch[:, : C // 2]
    batch_after = batch[:, C // 2 :]

    if "NormalizationStats" in model_info:
        mean = np.array(
            [0.5] * (len(model_info["ExtractBands"]) // 2), dtype=np.float32
        )
        std = np.array([0.5] * (len(model_info["ExtractBands"]) // 2), dtype=np.float32)
        batch_before = scale_batch(batch_before, model_info, break_extract_bands=True)
        batch_after = scale_batch(batch_after, model_info, break_extract_bands=True)

    batch_before = norm(batch_before.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    batch_after = norm(batch_after.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    batch_before = torch.tensor(batch_before, device=device).float()
    batch_after = torch.tensor(batch_after, device=device).float()
    from ..._utils.change_detection_data import post_process

    with torch.no_grad():
        predictions = post_process(model(batch_before, batch_after))
    # find the non zero class of the two classes.
    change_class = [c["Value"] for c in model_info["Classes"] if c["Value"] != 0][0]
    predictions[predictions != 0] = change_class
    return predictions[:, 0]


def pixel_classify_pix2pix_hd_image(model, tiles, device, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    num_channel = model_info.get("n_intput_channel", None)
    if num_channel == None:
        num_channel = model_info.get("n_channel", None)
    num_channel_tar = model_info.get("n_channel", None)
    label_nc = model_info.get("label_nc", 0)
    if label_nc == 0:
        norm_stats = model_info.get("NormalizationStats", None)
        img_scaled = scale_batch(tiles, model_info, norm_stats)
        img_normed = -1 + 2 * img_scaled
        if img_normed.shape[1] < num_channel:
            cont = []
            for j in range(img_normed.shape[0]):
                tile = img_normed[j, :, :, :]
                last_tile = np.expand_dims(tile[tile.shape[0] - 1, :, :], 0)
                res = abs(num_channel - tile.shape[0])
                for i in range(res):
                    tile = np.concatenate((tile, last_tile), axis=0)
                cont.append(tile)
            img_normed = np.stack(cont, axis=0)

        pix2pix_predictions = pix2pix_image(model, img_normed, device, label_nc)
        norm_stats_b = model_info.get("NormalizationStats_b", None)
        pix2pix_predictions = pix2pix_predictions.detach().cpu().numpy()
        if num_channel_tar == None:
            pix2pix_predictions = torch.tensor(
                rescale_batch(pix2pix_predictions, model_info, norm_stats_b)
            )
        else:
            pix2pix_predictions = torch.tensor(
                rescale_batch(pix2pix_predictions, model_info, norm_stats_b)
            )[:, :num_channel_tar, :, :]
        return pix2pix_predictions
    else:
        mask_map = model_info.get("mask_map", None)
        for i, j in enumerate(mask_map):
            tiles[tiles == j] = i
        img_normed = tiles

        pix2pix_predictions = pix2pix_image(model, img_normed, device, label_nc)
        pix2pix_predictions = pix2pix_predictions / 2 + 0.5
        return pix2pix_predictions


# functions for test time augmentation and smooth blending for pixel classification


def dihedral_transform(x, k):  # expects [C, H, W]
    flips = []
    if k & 1:
        flips.append(1)
    if k & 2:
        flips.append(2)
    if flips:
        x = torch.flip(x, flips)
    if k & 4:
        x = x.transpose(1, 2)
    return x.contiguous()


def create_interpolation_mask(side, border, device, window_fn="bartlett"):
    if window_fn == "bartlett":
        window = torch.bartlett_window(side, device=device).unsqueeze(0)
        interpolation_mask = window * window.T
    elif window_fn == "hann":
        window = torch.hann_window(side, device=device).unsqueeze(0)
        interpolation_mask = window * window.T
    else:
        linear_mask = torch.linspace(0, 1, border, device=device).repeat(side, 1)
        remainder_tile = torch.ones((side, side - border), device=device)
        interp_tile = torch.cat([linear_mask, remainder_tile], dim=1)

        interpolation_mask = torch.ones((side, side), device=device)
        for i in range(4):
            interpolation_mask = interpolation_mask * interp_tile.rot90(i)
    return interpolation_mask


def unfold_tensor(tensor, tile_size, stride):  # expects tensor  [1, C, H, W]
    mask = torch.ones_like(tensor[0][0].unsqueeze(0).unsqueeze(0))

    unfold = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p = unfold(mask)
    patches = unfold(tensor)

    patches = patches.reshape(tensor.size(1), tile_size, tile_size, -1).permute(
        3, 0, 1, 2
    )
    masks = mask_p.reshape(1, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    return masks, (tensor.size(2), tensor.size(3)), patches


def fold_tensor(input_tensor, masks, t_size, tile_size, stride):
    input_tensor_permuted = (
        input_tensor.permute(1, 2, 3, 0).reshape(-1, input_tensor.size(0)).unsqueeze(0)
    )
    mask_tt = masks.permute(1, 2, 3, 0).reshape(-1, masks.size(0)).unsqueeze(0)

    fold = nn.Fold(
        output_size=(t_size[0], t_size[1]),
        kernel_size=(tile_size, tile_size),
        stride=stride,
    )
    output_tensor = fold(input_tensor_permuted) / fold(mask_tt)
    return output_tensor


def split_predict_interpolate(child_image_classifier, normalized_image_tensor):
    kernel_size = child_image_classifier.tytx
    stride = kernel_size - (2 * child_image_classifier.padding)

    # Split image into overlapping tiles
    masks, t_size, patches = unfold_tensor(normalized_image_tensor, kernel_size, stride)

    with torch.no_grad():
        output = child_image_classifier.model(patches)

    interpolation_mask = create_interpolation_mask(
        kernel_size, 0, child_image_classifier.device, "hann"
    )
    output = output * interpolation_mask
    masks = masks * interpolation_mask

    # merge predictions from overlapping chips
    int_surface = fold_tensor(output, masks, t_size, kernel_size, stride)

    return int_surface


def tta_predict(child_image_classifier, normalized_image_tensor, test_time_aug=True):
    all_activations = []

    transforms = [0]
    if test_time_aug:
        if child_image_classifier.json_info["ImageSpaceUsed"] == "MAP_SPACE":
            transforms = list(range(8))
        else:
            transforms = [
                0,
                2,
            ]  # no vertical flips for pixel space (oriented imagery)

    for k in transforms:
        flipped_image_tensor = dihedral_transform(normalized_image_tensor[0], k)
        int_surface = split_predict_interpolate(
            child_image_classifier, flipped_image_tensor.unsqueeze(0)
        )
        corrected_activation = dihedral_transform(int_surface[0], k)

        if k in [5, 6]:
            corrected_activation = dihedral_transform(int_surface[0], k).rot90(
                2, [1, 2]
            )

        all_activations.append(corrected_activation)

    all_activations = torch.stack(all_activations)

    return all_activations


def update_pixels_tta(
    child_image_classifier, tlc, shape, props, **pixelBlocks
):  # 8 x 224 x 224 x 3
    model_info = child_image_classifier.json_info

    class_values = [clas["Value"] for clas in model_info["Classes"]]
    is_contiguous = is_cont([0] + class_values)

    if not is_contiguous:
        pixel_mapping = [0] + class_values
        idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}

    input_image = pixelBlocks["raster_pixels"].astype(np.float32)
    input_image_tensor = (
        torch.tensor(input_image).to(child_image_classifier.device).float()
    )

    if "NormalizationStats" in model_info:
        normalized_image_tensor = normalize_batch(input_image_tensor.cpu(), model_info)
        normalized_image_tensor = normalized_image_tensor.float().to(
            input_image_tensor.device
        )
    else:
        from torchvision import transforms

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        normalized_image_tensor = normalize(input_image_tensor / 255.0).unsqueeze(0)

    all_activations = tta_predict(
        child_image_classifier,
        normalized_image_tensor,
        test_time_aug=child_image_classifier.use_tta,
    )
    # probability of each class in 2nd dimension
    all_activations = all_activations.mean(dim=0, keepdim=True)
    softmax_surface = all_activations.softmax(dim=1)

    ignore_mapped_class = model_info.get("ignore_mapped_class", [])

    for k in ignore_mapped_class:
        softmax_surface[:, k] = -1

    if not child_image_classifier.predict_background:
        softmax_surface[:, 0] = -1

    result = softmax_surface.max(dim=1)[1]

    if not is_contiguous:
        result = remap(result, idx2pixel)

    pad = child_image_classifier.padding

    return result.cpu().numpy().astype("i4")[:, pad : -pad or None, pad : -pad or None]


def update_pixels_img_trans(self, tlc, shape, props, **pixelBlocks):
    kernel_size = self.json_info["ImageHeight"]
    stride = kernel_size - (2 * self.padding)

    model_name = self.json_info.get("ModelName")

    pixelblock = pixelBlocks["raster_pixels"].astype(np.float32)
    pixelblock_image_tensor = tensor(pixelblock).float().unsqueeze(0).cpu()

    masks, t_size, patches = unfold_tensor(pixelblock_image_tensor, kernel_size, stride)

    if model_name == "Pix2PixHD":
        prediction = pixel_classify_pix2pix_hd_image(
            self.model, patches, self.device, model_info=self.json_info
        )
    elif model_name == "CycleGAN":
        prediction = pixel_classify_cyclegan_image(
            self.model, patches, self.device, self.direction, model_info=self.json_info
        )
    elif model_name == "Pix2Pix":
        prediction = pixel_classify_pix2pix_image(
            self.model, patches, self.device, model_info=self.json_info
        )
    elif model_name == "SuperResolution":
        prediction = pixel_classify_superres_image(
            self.model,
            patches,
            self.device,
            self.json_info,
            getattr(self, "sampling", None),
            getattr(self, "n_timestep", None),
        )
    elif model_name == "WNetcGAN":
        prediction = pixel_classify_wnet_image(
            self.model, patches, self.device, model_info=self.json_info
        )

    interpolation_mask = create_interpolation_mask(kernel_size, 0, self.device, "hann")

    output = (tensor(prediction).to(self.device)) * interpolation_mask
    masks_inp = (tensor(masks).to(self.device)) * interpolation_mask

    merged_preds = fold_tensor(output, masks_inp, t_size, kernel_size, stride)

    return merged_preds.cpu().numpy()[
        0,
        :,
        self.padding : -self.padding or None,
        self.padding : -self.padding or None,
    ]
