# This code is based on https://github.com/kcg2015/Vehicle-Detection-and-Tracking by kyle Guan
try:
    import numpy as np
    import cv2
    from numpy import dot
    from scipy.linalg import inv, block_diag
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    from collections import deque
    import enum
except Exception as e:
    pass


class TrackStatus(enum.Enum):
    lost = 0
    searching = 8
    tracking = 16


class Track:
    """
    Creates a Track object, used to maintain the state of a track.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    id                      Required int. ID for each track initialized
    ---------------------   -------------------------------------------
    label                   Required String. label/class name of the track
    ---------------------   -------------------------------------------
    bbox                    Required list. Bounding box of the track
    ---------------------   -------------------------------------------
    mask                    Required numpy array. Mask for the tack
    =====================   ===========================================

    :return: :class:`~arcgis.learn.Track` Object
    """

    def __init__(self, id, label, bbox, mask):
        self.id = id
        self.label = label
        self.bbox = bbox
        self.score = 1
        self.status = 16
        self.mask = mask
        self.location = None
        self.age = 0


class Tracker:  # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.trackid = 0  # tracker's id
        self.class_name = ""
        self.score = 0.0
        self.box = []  # list to store the coordinates for a bounding box
        self.hits = 0  # number of detection matches
        self.lost_tracks = 0  # number of unmatched tracks (track loss)

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state = []
        self.dt = 1.0  # time interval

        # Process matrix, assuming constant velocity model
        self.A = np.array(
            [
                [1, self.dt, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, self.dt, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, self.dt, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, self.dt],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement matrix, assuming we can only measure the coordinates

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        # Initialize the state covariance
        self.state_variance = 10.0
        self.P = np.diag(self.state_variance * np.ones(8))

        # Initialize the process covariance
        self.Q_comp_mat = np.array(
            [
                [self.dt**4 / 4.0, self.dt**3 / 2.0],
                [self.dt**3 / 2.0, self.dt**2],
            ]
        )
        self.Q = block_diag(
            self.Q_comp_mat, self.Q_comp_mat, self.Q_comp_mat, self.Q_comp_mat
        )

        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array(
            [
                self.state_variance,
                self.state_variance,
                self.state_variance,
                self.state_variance,
            ]
        )
        self.R = np.diag(self.R_diag_array)

    def kalman_filter(self, z):
        """
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        """
        x = self.x_state
        # Predict
        x = dot(self.A, x)
        self.P = dot(self.A, self.P).dot(self.A.T) + self.Q

        # Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S))  # Kalman gain
        y = z - dot(self.H, x)  # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int)  # convert to integer coordinates
        # (pixel values)

    def predict(self):
        """
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        """
        x = self.x_state
        x = dot(self.A, x)
        self.P = dot(self.A, self.P).dot(self.A.T) + self.Q
        self.x_state = x.astype(int)


def delete_trackers(deleted_tracks, tracker_list):
    """
    Delete unused tracks from memory.

    """
    for trk in deleted_tracks:
        tracker_list.remove(trk)


def box_iou(a, b):
    """
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    """
    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)


def munkres_assignment(trackers, detections, iou_thrd):
    """
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    """
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)

    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = box_iou(trk, det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx_row, matched_idx_col = linear_sum_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if t not in matched_idx_row:
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if d not in matched_idx_col:
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object
    matched_idx = np.array([matched_idx_row, matched_idx_col]).T

    for m in matched_idx:
        if IOU_mat[m[0], m[1]] < iou_thrd:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def main_tracker(
    img,
    detections,
    scores,
    assignment_iou_thrd,
    vanish_frames,
    detect_frames,
    tracker_list,
    tracker_ind,
):
    """
    main_tracker function for detection and tracking
    """
    vanish_frames = vanish_frames  # no.of consecutive unmatched detection before

    detect_frames = (
        detect_frames  # no. of consecutive matches needed to establish a track
    )

    x_box = []
    z_box = detections  # measurement

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    # Hungarian/Munkres Assignment
    matched, unmatched_dets, unmatched_trks = munkres_assignment(
        x_box, z_box, iou_thrd=assignment_iou_thrd
    )

    # Deal with matched detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.score = scores[det_idx]
            tmp_trk.hits += 1
            tmp_trk.lost_tracks = 0

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.score = scores[idx]
            tmp_trk.trackid = tracker_ind  # assign an ID for the tracker
            tracker_ind += 1
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.lost_tracks += 1
            tmp_trk.predict()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # The list of tracks to be annotated
    good_tracker_list = []
    obj_info = {}
    predictions = []
    scores = []
    labels = []
    for trk in tracker_list:
        if (trk.hits >= detect_frames) and (trk.lost_tracks <= vanish_frames):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            obj_info[trk.trackid] = (x_cv2, trk.score)

            predictions.append(
                [x_cv2[1], x_cv2[0], x_cv2[3] - x_cv2[1], x_cv2[2] - x_cv2[0]]
            )
            scores.append(trk.score)
            labels.append(trk.trackid)

    # Book keeping
    deleted_tracks = filter(lambda x: x.lost_tracks > vanish_frames, tracker_list)

    delete_trackers(deleted_tracks, tracker_list)

    return predictions, labels, scores, tracker_list, tracker_ind
