import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

    tf.enable_eager_execution()
    import onnx
    from onnx_tf.backend import prepare
import numpy as np
import math


def A(*a):
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

mean = 255 * np.array(imagenet_stats[0], dtype=np.float32)
std = 255 * np.array(imagenet_stats[1], dtype=np.float32)

norm = lambda x: (x - mean) / std
denorm = lambda x: x * std + mean


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
    return np.array(x)


def actn_to_bb(actn, anchors, grid_sizes):
    actn_bbs = tf.tanh(actn).numpy()
    actn_centers = (actn_bbs[:, :2] / 2 * grid_sizes) + anchors[:, :2]
    actn_hw = (actn_bbs[:, 2:] / 2 + 1) * anchors[:, 2:]
    return hw2corners(actn_centers, actn_hw)


def hw2corners(ctr, hw):
    return tf.concat([ctr - hw / 2, ctr + hw / 2], axis=1).numpy()


def nms(boxes, scores, overlap=0.3, top_k=100):
    keep = []
    if len(boxes) == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()
    order = order[-top_k:]

    while order.size > 0:
        i = order[-1]
        keep.append(i)
        order = order[:-1]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        w = xx2 - xx1
        h = yy2 - yy1
        inter = w * h
        rem_areas = areas[order]
        union = (rem_areas - inter) + areas[i]
        IoU = inter / union
        order = order[IoU <= overlap]

    return keep, len(keep)


def get_nms_preds(
    bb_class, bb_act, idx, anchors, grid_sizes, classes, nms_overlap, thres
):
    a_ic = actn_to_bb(bb_act[idx], anchors, grid_sizes)
    box_clas = tf.math.sigmoid((bb_class[idx]).max(1)).numpy()
    conf_scores = tf.transpose(tf.math.sigmoid(bb_class[idx])).numpy()

    out_scores, out_box, out_clas = [], [], []

    for i in range(1, len(conf_scores)):
        mask = conf_scores[i] > thres
        if mask.sum() == 0:
            continue
        scores = conf_scores[i][mask]
        mask1 = np.expand_dims(mask, 1) * np.ones(a_ic.shape)
        mask1 = mask1.astype("bool")
        boxes = a_ic[mask1].reshape(-1, 4)
        idx, cnt = nms(boxes, scores, nms_overlap, 50)
        idx = idx[:cnt]
        out_scores.append(scores[idx])
        out_box.append(boxes[idx])
        out_clas.append([i] * cnt)

    if out_clas == []:
        out_clas = [[0]]

    if out_box == []:
        out_box = [[]]

    if out_scores == []:
        out_scores = [[]]

    return predictions(
        out_box[0],
        out_clas[0],
        out_scores[0] if out_scores[0] is not None else None,
        thres,
        classes,
    )


def predictions(bbox, clas=None, prs=None, thresh=0.3, classes=None):
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
            bb_np = b.astype("float64")
            predictions.append(pred2dict(bb_np, score, cat_str, c))
    return predictions


def detect_objects_image_space(
    graph, tiles, anchors, grid_sizes, classes, nms_overlap, thres
):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]

    tiles = tiles.astype("float32")
    img_t = norm(tiles.transpose(0, 2, 3, 1))
    img_t = img_t.transpose(0, 3, 1, 2)

    output_dict = graph.run(img_t)

    clas, bbox = output_dict._0, output_dict._1

    preds = {}

    for batch_idx in range(bbox.shape[0]):
        preds[batch_idx] = get_nms_preds(
            clas, bbox, batch_idx, anchors, grid_sizes, classes, nms_overlap, thres
        )

    batch_size = bbox.shape[0]
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
