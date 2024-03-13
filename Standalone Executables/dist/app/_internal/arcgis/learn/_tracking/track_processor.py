try:
    from .._data import _raise_conda_import_error
    import traceback
    from . import _track_processor

    HAS_CONDA = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_CONDA = False


def get_default_tracker_options():
    tracker_options = {
        "enable_post_processing": True,
        "detection_interval": 1,
        "detection_threshold": 0.0,
        "detect_track_failure": False,
        "recover_track": False,
        "stab_period": 6,
        "detect_fail_interval": 0,
        "min_obj_size": 10,
        "template_history": 25,
        "status_history": 60,
        "status_fail_threshold": 0.6,
        "search_period": 60,
        "knn_distance_ratio": 0.75,
        "recover_conf_threshold": 0.1,
        "recover_iou_threshold": 0.1,
    }

    return tracker_options


class TrackProcessor:

    """
    Creates TrackProcessor Object.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    processor_options       Optional dictionary. A dictionary with
                            keys as parameter names and values as
                            parameter values.

                            "detection_interval" refers to
                            the interval in frames at which the detector
                            is invoked.

                            "detection_threshold" refers to
                            the lower threshold for selecting the
                            detections.

                            "detect_track_failure" refers to
                            the flag which enables/disables the logic
                            to detect whether the object appearance
                            has changed detection.

                            "recover_track" refers to the flag which
                            enables/disables track recovery post failure.

                            "stab_period" refers to the number of frames
                            after which post processing starts.

                            "detect_fail_interval" refers to the number
                            of frames after which to detect track failure.

                            "min_obj_size" refers to the size in pixels
                            below which tracking is assumed to have
                            failed.

                            "template_history" refers to the number of
                            frames before the current frame at which
                            template image is fetched.

                            "status_history" refers to the
                            number of frames over which status of the
                            track is used to detect track failure.

                            "status_fail_threshold" refers to the
                            threshold for the ratio between number
                            of frames for which object is searched
                            for and the total number of frames which
                            needs to be crossed for track failure
                            detection.

                            "search_period" refers to the
                            number of frames for which object is
                            searched for before declaring object is
                            lost.

                            "knn_distance_ratio" refers to the
                            threshold for ratio of the distances between
                            template descriptor and the two best matched
                            detection descriptor, used for filtering
                            best matches.

                            "recover_conf_threshold" refers
                            to the minimum confidence value over which
                            recovery logic is enabled.

                            "recover_iou_threshold" refers to the minimum
                            overlap between template and detection for
                            successful recovery.
    =====================   ===========================================

    :return: `TrackProcessor` Object
    """

    def __init__(
        self,
        processor_options={
            "detect_track_failure": True,
            "recover_track": True,
            "stab_period": 6,
            "detect_fail_interval": 5,
            "min_obj_size": 10,
            "template_history": 25,
            "status_history": 60,
            "status_fail_threshold": 0.6,
            "search_period": 60,
            "knn_distance_ratio": 0.75,
            "recover_conf_threshold": 0.1,
            "recover_iou_threshold": 0.1,
        },
    ):
        if not HAS_CONDA:
            _raise_conda_import_error(import_exception=import_exception)

        self.tracks = []
        params = []
        params.append(processor_options.get("detect_track_failure", True))
        params.append(processor_options.get("recover_track", True))
        params.append(processor_options.get("stab_period", 6))
        params.append(processor_options.get("detect_fail_interval", 5))
        params.append(processor_options.get("min_obj_size", 10))
        params.append(processor_options.get("template_history", 25))
        params.append(processor_options.get("status_history", 60))
        params.append(processor_options.get("status_fail_threshold", 0.6))
        params.append(processor_options.get("search_period", 60))
        params.append(processor_options.get("knn_distance_ratio", 0.75))
        params.append(processor_options.get("recover_conf_threshold", 0.1))
        params.append(processor_options.get("recover_iou_threshold", 0.1))
        _track_processor.create(params)

    def init(self, frame, dets, reset=True):
        """
        Initializes tracks based on the detections returned by detector/
        manually fed to the function.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is used to initialize
                                the objects to track.
        ---------------------   -------------------------------------------
        dets                    Required list. A 1D list with bounding boxes
                                to intialize the tracks.
        ---------------------   -------------------------------------------
        reset                   Optional flag. Indicates whether to reset
                                the tracker and remove all existing tracks
                                before initialization.
        =====================   ===========================================

        :return: None
        """

        # reset = True

        if reset:
            _track_processor.init(frame, dets)
        else:
            _track_processor.add(frame, dets)
        # TODO: remove return - no need
        return None

    def update(self, frame, dets):
        """
        Post-processes the tracks, detects track failure and recovers the
        lost objects if possible.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is the current
                                frame to be used to track the objects.
        ---------------------   -------------------------------------------
        dets                    Required list. A 1D list with state of
                                tracks needed for post processing.
        =====================   ===========================================

        :return: 1D list with updated state of track objects
        """

        dets = _track_processor.update(frame, dets)
        return dets

    def remove(self, tracks_ids):
        """
        Removes the tracks corresponding to track_ids parameter

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        tracks_ids              Required list. List of track ids to be
                                removed.
        =====================   ===========================================

        """
        _track_processor.purge(tracks_ids)
        return None
