import numpy as np

from arcgis.auth.tools import LazyLoader

cv2 = LazyLoader("cv2")


def _pad_image(image, stride):
    img_h = image.shape[0] + stride * 2
    img_w = image.shape[1] + stride * 2

    tmp = np.zeros((img_h, img_w, image.shape[2]))
    tmp[stride : stride + image.shape[0], stride : stride + image.shape[1], :] = image[
        :, :, :
    ]
    img = tmp.astype(np.uint8)

    return img


def _get_image_chips(image, chip_dim):
    """Function to take an image and
    return sequentially cropped and padded chips of size chip_dim."""

    img_h, img_w, _ = image.shape
    chips_data = []
    stride = chip_dim // 2

    start_x = 0
    start_y = 0

    if chip_dim >= img_w and chip_dim >= img_h:
        return [
            {
                "height": img_h,
                "width": img_w,
                "chip": image,
                "xmin": start_x,
                "ymin": start_y,
                "predictions": [],
            }
        ]

    # Add zero-padding to enable prediction on all parts of the image
    padded_image = _pad_image(image, stride)
    img_h, img_w, _ = padded_image.shape

    while start_x + chip_dim <= img_w:
        start_y = 0
        while start_y + chip_dim <= img_h:
            chip = padded_image[
                start_y : start_y + chip_dim, start_x : start_x + chip_dim, :
            ]

            if (chip.shape[0] != chip_dim) or (chip.shape[1] != chip_dim):
                tmp = np.zeros((chip_dim, chip_dim, 3))
                tmp[0 : chip.shape[0], 0 : chip.shape[1], :] = chip[:, :, :]
                chip = tmp.astype(np.uint8)

            # To translate the bbox coordinates according to unpadded image
            xmin = start_x - stride
            ymin = start_y - stride

            chips_data.append(
                {
                    "height": chip_dim,
                    "width": chip_dim,
                    "chip": chip,
                    "xmin": xmin,
                    "ymin": ymin,
                    "predictions": [],
                }
            )
            start_y = start_y + stride

        start_x = start_x + stride

    return chips_data


def _get_transformed_predictions(chips_data):
    predictions = []
    labels = []
    scores = []
    for chip_data in chips_data:
        for prediction in chip_data["predictions"]:
            prediction["xmin"] = prediction["xmin"] + chip_data["xmin"]
            prediction["ymin"] = prediction["ymin"] + chip_data["ymin"]

            predictions.append(
                [
                    prediction["xmin"],
                    prediction["ymin"],
                    prediction["width"],
                    prediction["height"],
                ]
            )
            labels.append(prediction["label"].obj)
            scores.append(prediction["score"])

    return predictions, labels, scores


def _draw_predictions(
    frame,
    predictions,
    labels,
    scores=None,
    show_scores=True,
    show_labels=True,
    color=(255, 255, 255),
    fontface=0,
    thickness=2,
):
    for index, data in enumerate(predictions):
        frame = cv2.rectangle(
            frame,
            (int(data[0]), int(data[1])),
            (int(data[0] + data[2]), int(data[1] + data[3])),
            color,
            thickness,
        )

        text_to_display = None
        if show_labels:
            text_to_display = str(labels[index])

        if show_scores and scores is not None:
            if text_to_display:
                scores[index] = float("{0:.2f}".format(scores[index]))
                text_to_display = text_to_display + ": " + str(scores[index])
            else:
                text_to_display = str(scores[index])

        if text_to_display:
            frame = cv2.putText(
                frame,
                text_to_display,
                (int(data[0]), int(data[1]) - 10),
                fontface,
                0.7,
                color,
                thickness,
            )

    return frame


def _exclude_detection(data, chip_width, chip_height):
    if chip_height < chip_width:
        padding = chip_height // 4
    else:
        padding = chip_width // 4

    center_coord_x = data[0] + data[2] / 2
    center_coord_y = data[1] + data[3] / 2

    if (
        center_coord_x < padding
        or center_coord_y < padding
        or center_coord_x > (chip_width - padding)
        or center_coord_y > (chip_height - padding)
    ):
        return True

    return False
