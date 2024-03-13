import os, csv
from warnings import warn

HAS_OPENCV = True
HAS_FASTAI = True
HAS_ARCPY = True

try:
    from .models import _tracker_util
    from ._image_utils import _draw_predictions
    from fastai.vision.image import bb2hw, Image, pil2tensor
    from fastprogress.fastprogress import progress_bar
    import arcpy
except Exception:
    HAS_FASTAI = False

try:
    import cv2
except Exception:
    HAS_OPENCV = False


class VideoUtils:
    @staticmethod
    def predict_video(
        model,
        input_video_path,
        metadata_file,
        threshold=0.5,
        nms_overlap=0.1,
        track=False,
        visualize=False,
        output_file_path=None,
        multiplex=False,
        multiplex_file_path=None,
        tracker_options={
            "assignment_iou_thrd": 0.3,
            "vanish_frames": 40,
            "detect_frames": 10,
        },
        visual_options={
            "show_scores": True,
            "thickness": 2,
            "fontface": 0,
            "show_labels": True,
            "color": (255, 255, 255),
        },
        resize=False,
    ):
        if not HAS_OPENCV:
            raise Exception(
                "This function requires opencv 4.0.1.24. Install it using pip install opencv-python==4.0.1.24"
            )

        if not os.path.exists(input_video_path):
            raise Exception("The input video path doesn't exist.")

        video_read = cv2.VideoCapture(input_video_path)
        fps = video_read.get(cv2.CAP_PROP_FPS)
        video_obj = None
        success = True
        total_frames = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        vmtis = ["vmtilocaldataset"]

        object_id_mapping = {}
        object_id = 1
        tracker_list = []
        tracker_ind = 0

        thickness = visual_options.get("thickness", 2)
        fontface = visual_options.get("fontface", 0)
        show_labels = visual_options.get("show_labels", True)
        color = visual_options.get("color", (255, 255, 255))
        show_scores = visual_options.get("show_scores", True)

        for pb in progress_bar(range(total_frames)):
            success, frame = video_read.read()
            frame_number = frame_number + 1

            if not success and frame_number < total_frames:
                continue
            elif not success:
                break

            height, width, _ = frame.shape
            if visualize and not video_obj:
                if not output_file_path:
                    output_file_path = os.path.join(
                        os.path.dirname(input_video_path),
                        os.path.basename(input_video_path).split(".")[0]
                        + "_predictions.avi",
                    )
                video_obj = cv2.VideoWriter(
                    output_file_path,
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps,
                    (width, height),
                )
                if not video_obj.isOpened():
                    raise Exception("Unable to write to output file path.")

            predictions, labels, scores = model.predict(
                frame,
                threshold=threshold,
                nms_overlap=nms_overlap,
                return_scores=True,
                resize=resize,
            )
            vmti_detections = "\n"

            if predictions:
                if track:
                    bboxes = []
                    for prediction in predictions:
                        bboxes.append(
                            [
                                prediction[1],
                                prediction[0],
                                prediction[1] + prediction[3],
                                prediction[0] + prediction[2],
                            ]
                        )

                    (
                        predictions,
                        labels,
                        scores,
                        tracker_list,
                        tracker_ind,
                    ) = _tracker_util.main_tracker(
                        frame,
                        bboxes,
                        scores,
                        tracker_options["assignment_iou_thrd"],
                        tracker_options["vanish_frames"],
                        tracker_options["detect_frames"],
                        tracker_list,
                        tracker_ind,
                    )

                    for index, data in enumerate(predictions):
                        top_left = max(0, (int(data[1]) - 1)) * width + int(data[0])
                        bottom_right = max(
                            0, (int(data[1] + data[3]) - 1)
                        ) * width + int(data[0] + data[2])
                        center_pixel = (int(data[1]) + int((data[3]) / 2)) * width + (
                            int(data[0]) + int((data[2]) / 2)
                        )

                        vmti_detections = (
                            f"{labels[index]} {scores[index] * 100} {top_left} {bottom_right} {center_pixel};"
                            + vmti_detections
                        )
                else:
                    for index, data in enumerate(predictions):
                        top_left = max(0, (int(data[1]) - 1)) * width + int(data[0])
                        bottom_right = max(
                            0, (int(data[1] + data[3]) - 1)
                        ) * width + int(data[0] + data[2])
                        center_pixel = (int(data[1]) + int((data[3]) / 2)) * width + (
                            int(data[0]) + int((data[2]) / 2)
                        )

                        if not object_id_mapping.get(labels[index]):
                            object_id_mapping[labels[index]] = object_id
                            object_id = object_id + 1

                        vmti_detections = (
                            f"{object_id_mapping[labels[index]]} {scores[index] * 100} {top_left} {bottom_right} {center_pixel};"
                            + vmti_detections
                        )

                image = _draw_predictions(
                    frame,
                    predictions,
                    labels,
                    scores=scores,
                    show_scores=show_scores,
                    thickness=thickness,
                    fontface=fontface,
                    color=color,
                    show_labels=show_labels,
                )
            else:
                image = frame

            vmtis.append(vmti_detections)

            if visualize:
                video_obj.write(image)

        if video_obj:
            video_obj.release()

        video_read.release()

        data = []
        index = 0

        file_exists = True
        fields = []

        if not os.path.exists(metadata_file):
            file_exists = False
            for vmti in vmtis:
                data.append([vmti])
        else:
            with open(metadata_file, "r") as csvinput:
                for row in csv.reader(csvinput):
                    if index == 0:
                        fields = row
                    if len(vmtis) <= index:
                        data.append(row + [""])
                    else:
                        data.append(row + [vmtis[index]])
                    index = index + 1

        if "vmtilocaldataset" in fields:
            warn(
                "Field 'vmtilocaldataset' already exists in the file, appending column at the end."
            )

        if len(data) < len(vmtis):
            warn(f"Writing {len(data)} rows only!")

        with open(metadata_file, "w", newline="") as csvoutput:
            writer = csv.writer(csvoutput)
            for row in data:
                writer.writerow(row)

        if not multiplex:
            return

        if not HAS_ARCPY:
            warn("Arcpy doesn't exist, multiplexing skipped.")
            return

        if not file_exists:
            warn("Metadata file doesn't exist, multiplexing skipped.")
            return

        if not multiplex_file_path:
            multiplex_file_path = os.path.join(
                os.path.dirname(input_video_path),
                os.path.basename(input_video_path).split(".")[0] + "_multiplex.MOV",
            )

        arcpy.ia.VideoMultiplexer(input_video_path, metadata_file, multiplex_file_path)
