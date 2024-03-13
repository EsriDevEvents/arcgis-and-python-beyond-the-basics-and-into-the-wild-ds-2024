import traceback

HAS_DEPS = True
try:
    from arcgis.auth.tools import LazyLoader
    from pandas import DataFrame

    cv2 = LazyLoader("cv2")
    import numpy as np

    plt = LazyLoader("matplotlib.pyplot")
    import os
    import unittest

    h5py = LazyLoader("h5py")
    import math
    import sys
    import arcgis
    import xml.etree.ElementTree as ET
    from datetime import datetime
    from arcgis.geometry import Polygon

    ipd = LazyLoader("IPython.display")
except Exception as e:
    import_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_DEPS = False


def generate_kernel(k_size, k_type):
    """
    Generates kernel for morphological operations.
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    k_size                  List that determines the size of the kernel.

    ---------------------   -------------------------------------------
    k_type                  String that determines type of the kernel.
                            values can be "rect", "elliptical", "cross"
    =====================   ===========================================

    return:
    closing_kernel: Closing kernel object
    opening_kernel: Opening kernel object

    """
    closing_kernel, opening_kernel, kernel_object = None, None, None
    if k_type == "rect":
        kernel_object = cv2.MORPH_RECT
    if k_type == "elliptical":
        kernel_object = cv2.MORPH_ELLIPSE
    if k_type == "cross":
        kernel_object = cv2.MORPH_CROSS

    if k_size[0] is not None:
        closing_kernel = cv2.getStructuringElement(
            kernel_object, (k_size[0], k_size[0])
        )

    if k_size[1] is not None:
        opening_kernel = cv2.getStructuringElement(
            kernel_object, (k_size[1], k_size[1])
        )

    return closing_kernel, opening_kernel


def fill_mask_pixels(image, masked_path, land_color):
    """
    Fill the extracted mask region with the color of land region.
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    image                   Input image.

    ---------------------   -------------------------------------------
    masked_path             Path of extracted masked region image

    ---------------------   -------------------------------------------
    land_color              Land color to be filled at masked region

    =====================   ===========================================
    return:
    image: Image with colored masked replaced by land color
    """

    masked_image = cv2.imread(os.path.join(masked_path))
    mask = cv2.inRange(masked_image, np.array([0, 0, 0]), np.array([50, 50, 50]))
    h = image.shape[0]
    w = image.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if mask[y][x] == 0:
                image[y, x] = land_color

    return image


def detect_template(template, search_img, start_scale, end_scale, num_scales):
    """
    Performs multi-scale template matching to locate the template on the
    bigger search region
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    template                Template image which needs to be matched on
                            the bigger search region image

    ---------------------   -------------------------------------------
    search_img              Search region image in which template
                            needs to be searched

    ---------------------   -------------------------------------------
    start_scale             Starting value of scale interval

    ---------------------   -------------------------------------------
    end_scale               Ending value of scale interval

    ---------------------   -------------------------------------------
    num_scales              Number of scales within the scale interval


    =====================   ===========================================

    returns:
        start_x: Starting X co-ordinate of detected bounding box
        start_y: Starting Y co-ordinate of detected bounding box
        end_x: Ending X co-ordinate of detected bounding box
        end_y: Ending Y co-ordinate of detected bounding box
        max_val: Value of template match score for the detected bounding box
        scale_factor: Scale at which the match was found for the template
    """

    template_height = template.shape[0]
    template_width = template.shape[1]

    unittest.TestCase().assertGreaterEqual(
        search_img.shape[0],
        template_height,
        """Height of the search image should be \
                                           greater then the height of template image""",
    )
    unittest.TestCase().assertGreaterEqual(
        search_img.shape[1],
        template_width,
        """Width of the search image should be \
                                           greater then the width of template image""",
    )

    start_x = 0
    start_y = 0
    end_x = 0
    end_y = 0
    scale_factor = -1
    found = None
    for scale in np.linspace(start_scale, end_scale, num_scales)[::-1]:
        resize_width, resize_height = (
            search_img.shape[1] * scale,
            search_img.shape[0] * scale,
        )
        resized = cv2.resize(search_img, (int(resize_width), int(resize_height)))
        scale_factor = search_img.shape[1] / float(resized.shape[1])
        if resized.shape[0] < template_height or resized.shape[1] < template_width:
            break
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
        if found is None or max_val > found[0]:
            found = (max_val, max_loc, scale_factor)

    if found is not None:
        (max_val, max_loc, scale_factor) = found
        (start_x, start_y) = (
            int(max_loc[0] * scale_factor),
            int(max_loc[1] * scale_factor),
        )
        (end_x, end_y) = (
            int((max_loc[0] + template_width) * scale_factor),
            int((max_loc[1] + template_height) * scale_factor),
        )
    else:
        max_val = -1

    return start_x, start_y, end_x, end_y, max_val, scale_factor


def compute_reference_homography(
    bb_start_x,
    bb_start_y,
    bb_end_x,
    bb_end_y,
    search_region_height,
    search_region_width,
    template_height,
    template_width,
):
    """
    Computes the homography transform using matched location of scanned map on
    search region
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        bb_start_x              Bounding box left x co-ordinate

        ---------------------   -------------------------------------------
        bb_start_y              Bounding box top y co-ordinate

        ---------------------   -------------------------------------------
        bb_end_x                Bounding box right x co-ordinate

        ---------------------   -------------------------------------------
        template_height         Height of template image

        ---------------------   -------------------------------------------
        template_width          Width of template image


        =====================   ===========================================

    returns:
        homography_matrix : Computed transformation matrix
    """

    src_points = np.array(
        [
            [0, 0],
            [template_width - 1, 0],
            [template_width - 1, template_height - 1],
            [0, template_height - 1],
        ],
        dtype=np.float32,
    )

    dst_points = np.array(
        [
            [bb_start_x, bb_start_y],
            [bb_end_x, bb_start_y],
            [bb_end_x, bb_end_y],
            [bb_start_x, bb_end_y],
        ],
        dtype=np.float32,
    )
    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return homography_matrix


def apply_homography(query_img, search_region, homography_matrix):
    """
    Creates warped scanned map on search region using estimated homography
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    query_img               Image to be processed

    ---------------------   -------------------------------------------
    search_region           Image for search region

    ---------------------   -------------------------------------------
    homography_matrix       Transformed matrix to be used
    =====================   ===========================================

    returns:
        warped_query_img: Modified world search region image
    """

    search_region_height = search_region.shape[0]
    search_region_width = search_region.shape[1]

    warped_query_img = cv2.warpPerspective(
        query_img, homography_matrix, (search_region_width, search_region_height)
    )

    return warped_query_img


def check_bb_boundary(bb_start_x, bb_start_y, bb_end_x, bb_end_y, height, width):
    """
    Checks the boundary and modifies the values if necessary
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        bb_start_x              X co-ordinate of left top corner of bounding
                                box
        ---------------------   -------------------------------------------
        bb_start_y              Y co-ordinate of left top corner of bounding
                                box
        ---------------------   -------------------------------------------
        bb_end_x                X co-ordinate of right bottom corner of
                                bounding box
        ---------------------   -------------------------------------------
        bb_end_y                Y co-ordinate of right bottom corner of
                                bounding box
        ---------------------   -------------------------------------------
        height                  Height of the image

        ---------------------   -------------------------------------------
        width                   Width of the image

        =====================   ===========================================

    returns:
        bb_start_x: Modified X co-ordinate of left top corner of bounding box
        bb_start_y: Modified Y co-ordinate of left top corner of bounding box
        bb_end_x: Modified X co-ordinate of right bottom corner of bounding box
        bb_end_y: Modified Y co-ordinate of right bottom corner of bounding box
    """

    if bb_start_x < 0:
        bb_start_x = 0

    if bb_start_y < 0:
        bb_start_y = 0

    if bb_end_x >= width:
        bb_end_x = width - 1

    if bb_end_y >= height:
        bb_end_y = height - 1
    return bb_start_x, bb_start_y, bb_end_x, bb_end_y


def detect_contours(input_img):
    """
    Detects contours in the input image
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    input_img               Input image for which contours need to be
                            computed

    =====================   ===========================================

    returns:
        contours: List of computed contours for the input image
    """

    if input_img.ndim != 2:
        input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        input_img_gray = input_img

    ret, input_img_gray_thresh = cv2.threshold(
        input_img_gray, 127, 255, cv2.THRESH_BINARY
    )

    contours, hierarchy = cv2.findContours(
        input_img_gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def calculate_diagonal_length(length, height):
    """
    Find the length of diagonal of a rectangle
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    length                  Length of rectangle
    ---------------------   -------------------------------------------
    height                  Height of rectangle
    =====================   ===========================================

    returns:
        length_diag: Length of the diagonal
    """

    length_diag = math.sqrt(length**2 + height**2)

    return length_diag


def calculate_nonzero_ratio(input_img):
    """
    Calculates the ratio of non-zero pixels area w.r.t entire area of image
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    input_img               Image for which ratio needs to be computed

    =====================   ===========================================

    returns:
        ratio: Non-zero area ratio
    """

    non_zero_area = cv2.countNonZero(input_img)
    total_area = input_img.shape[0] * input_img.shape[1]

    ratio = non_zero_area / total_area

    return ratio


def calculate_distance_ratios(
    bb_start_x, bb_start_y, bb_end_x, bb_end_y, point_x, point_y
):
    """
    Calculates the distance ratio w.r.t X and Y co-ordinate of a point
    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    bb_start_x              X co-ordinate of left top corner of bounding
                            box
    ---------------------   -------------------------------------------
    bb_start_y              Y co-ordinate of left top corner of bounding
                            box
    ---------------------   -------------------------------------------
    bb_end_x                X co-ordinate of right bottom corner of
                            bounding box
    ---------------------   -------------------------------------------
    bb_end_y                Y co-ordinate of right bottom corner of
                            bounding box
    ---------------------   -------------------------------------------
    point_x                 X co-ordinate of point

    ---------------------   -------------------------------------------
    point_y                 Y co-ordinate of point
    =====================   ===========================================

    returns:
        ratio_x : X co-ordinate ratio
        ratio_y : Y co-ordinate ratio
    """

    epsilon = 0.00001

    distance_point_x_start_x = point_x - bb_start_x
    distance_point_x_end_x = bb_end_x - point_x

    distance_point_y_start_y = point_y - bb_start_y
    distance_point_y_end_y = bb_end_y - point_y

    ratio_x = distance_point_x_start_x / (distance_point_x_end_x + epsilon)

    ratio_y = distance_point_y_start_y / (distance_point_y_end_y + epsilon)

    return ratio_x, ratio_y


def find_distance(point1_x, point1_y, point2_x, point2_y):
    """
    Find distance between two points
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        point1_x                X co-ordinate of first point
        ---------------------   -------------------------------------------
        point1_y                Y co-ordinate of first point
        ---------------------   -------------------------------------------
        point2_x                X co-ordinate of second point
        ---------------------   -------------------------------------------
        point2_y                y co-ordinate of second point

        =====================   ===========================================

    return:
        distance: distance between two points
    """

    distance = math.sqrt((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2)
    return distance


def find_nearest_contour_point(input_contour, pt_x, pt_y):
    """
    Find point on contour which is nearest to the given point
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        input_contour            Contour on which points need to be
                                 evaluated for nearness
        ---------------------   -------------------------------------------
        pt_x                     X co-ordinate of point
        ---------------------   -------------------------------------------
        pt_y                     Y co-ordinate of point

        =====================   ===========================================

    returns:
        min_x: X co-ordinate of contour point nearest to input point
        min_y: Y co-ordinate of contour point nearest to input point
        min_index: Index of contour point nearest to input point
        min_distance: Distance of contour point nearest to input point
    """
    min_distance = float("inf")
    min_index = -1
    min_x = -1
    min_y = -1

    for i, contour_pt in enumerate(input_contour):
        distance = find_distance(contour_pt[0][0], contour_pt[0][1], pt_x, pt_y)

        if distance < min_distance:
            min_distance = distance
            min_index = i
            min_x = contour_pt[0][0]
            min_y = contour_pt[0][1]

    return min_x, min_y, min_index, min_distance


def create_random_color_list(size):
    """
    Generate a list of specified size with random color values
        =====================   ==================================
        **Parameter**            **Description**
        ---------------------   ----------------------------------
         size                    Size of the list which needs to
                                 be returned
        =====================   ==================================

    returns:
       color_list: List of random color values
    """

    color_list = []

    for i in range(0, size):
        color_tuple = tuple(np.random.choice(range(256), size=3))
        color_tuple = tuple([int(x) for x in color_tuple])
        color_list.append(color_tuple)

    return color_list


def annotate_control_pts(query_image, search_region, control_pts):
    """
    Annotate control points on search region and scanned map
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        query_image             Input image
        ---------------------   -------------------------------------------
        search_region           Color image of search region
        ---------------------   -------------------------------------------
        control_pts             Control points to be annotated

        =====================   ===========================================

    returns:
        search_region_img_updated: Annotated search region image
        query_image_updated: Annotated scanned map
    """
    search_region_img_copy = search_region.copy()
    query_image_copy = query_image.copy()
    colors = create_random_color_list(len(control_pts))

    for i, control_pt in enumerate(control_pts):
        color = colors[i]
        color = tuple([int(x) for x in color])

        query_image_copy = cv2.drawMarker(
            query_image_copy, (control_pt[0], -1 * control_pt[1]), color
        )

        search_region_img_copy = cv2.drawMarker(
            search_region_img_copy, (int(control_pt[5]), int(-1 * control_pt[6])), color
        )

    return search_region_img_copy, query_image_copy


def compute_refined_homography_outlier_mask(control_pts):
    """
    Computes refined homography and outlier mask by applying RANSAC
     on mapped control points
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        control_pts             List of control points in both scanned map
                                and search region

        =====================   ===========================================

    returns:
        homography_matrix: Computed transformed matrix
        outlier_mask: Mask array with flags indicating whether control points
                      are outliers (denoted by 0) or inliers(denoted by 1)
    """

    if len(control_pts) < 4:
        return None, None

    src_points = np.zeros((len(control_pts), 2), dtype=np.float32)
    dst_points = np.zeros((len(control_pts), 2), dtype=np.float32)

    for i, control_pt in enumerate(control_pts):
        src_points[i, 0] = control_pt[0]
        src_points[i, 1] = -1 * control_pt[1]
        dst_points[i, 0] = control_pt[5]
        dst_points[i, 1] = -1 * control_pt[6]

    homography_matrix, outlier_mask = cv2.findHomography(
        src_points, dst_points, cv2.RANSAC, 3
    )

    return homography_matrix, outlier_mask


def filter_control_pts(control_pts, outlier_mask):
    """
    Filter outliers from control points using outlier mask
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        control_pts             List of control points in scanned map and
                                corresponding control points in search
                                region image
        ---------------------   -------------------------------------------
        outlier_mask            Outlier mask with flags indicating whether
                                outlier or not

        =====================   ===========================================

    returns:
        filtered_control_pts: Control Points without outliers
    """
    filtered_control_pts = []

    for i, control_pt in enumerate(control_pts):
        if outlier_mask[i] == 1:
            filtered_control_pts.append(control_pt)

    return filtered_control_pts


def estimate_control_points(
    search_reg_roi,
    template,
    search_reg_roi_offset_x,
    search_reg_roi_offset_y,
    search_reg_roi_start_x,
    search_reg_roi_start_y,
    search_region_img,
    extent,
):
    """
    Detects contours in template image and finds correspondences
    in processed search region image ROI

        =========================   ====================================
        **Parameter**                **Description**
        -------------------------   -----------------------------------
        search_reg_roi               Transformed search region image ROI
        -------------------------    -----------------------------------
        template                     Template image
        -------------------------    ------------------------------------
        search_reg_roi_offset_x      X margin for search region ROI w.r.t
                                     warped template
        -------------------------    ------------------------------------
        search_reg_roi_offset_y      Y margin for search region ROI w.r.t
                                     warped template
        -------------------------    -------------------------------------
        search_reg_roi_start_x       X co-ordinate of left top corner of
                                     search region roi
        -------------------------    -------------------------------------
        search_reg_roi_offset_y      Y co-ordinate of left top corner of
                                     search region roi
        -------------------------    --------------------------------------
        search_region_img            Transformed search region image ROI
        -------------------------    --------------------------------------
        extent                       Search region extent
        -------------------------    --------------------------------------

        =========================    ===========================

    returns:
        mapped_control_pts: List of control points in template and
        corresponding points in search image
    """

    degree_per_pixel_x = (extent["xmax"] - extent["xmin"]) / search_region_img.shape[1]

    degree_per_pixel_y = (extent["ymin"] - extent["ymax"]) / search_region_img.shape[0]

    template_contours = detect_contours(template)
    biggest_template_contour = max(template_contours, key=cv2.contourArea)

    contour_pts = []

    bb_cnt_xmin, bb_cnt_ymin, bb_cnt_width, bb_cnt_height = cv2.boundingRect(
        biggest_template_contour
    )

    bb_cnt_area = bb_cnt_width * bb_cnt_height
    template_area = template.shape[0] * template.shape[1]

    area_ratio = bb_cnt_area / template_area

    if area_ratio < 0.7:
        for contour in template_contours:
            for contour_pt in contour:
                contour_pts.append(contour_pt)

    else:
        contour_pts = biggest_template_contour

    length_diagonal_template_img = calculate_diagonal_length(
        template.shape[1], template.shape[0]
    )
    contour_patch_win = int(length_diagonal_template_img / 8)

    delta_contour_patch_win = contour_patch_win / 2

    search_reg_patch_win = int(contour_patch_win * 1.5)
    delta_search_patch_win = search_reg_patch_win / 2

    mapped_control_pts = []

    for contour_pt in contour_pts:
        contour_pt_x = contour_pt[0][0]
        contour_pt_y = contour_pt[0][1]
        cont_patch_start_y = contour_pt_y - delta_contour_patch_win
        cont_patch_end_y = contour_pt_y + delta_contour_patch_win
        cont_patch_start_x = contour_pt_x - delta_contour_patch_win
        cont_patch_end_x = contour_pt_x + delta_contour_patch_win

        (
            cont_patch_start_x,
            cont_patch_start_y,
            cont_patch_end_x,
            cont_patch_end_y,
        ) = check_bb_boundary(
            cont_patch_start_x,
            cont_patch_start_y,
            cont_patch_end_x,
            cont_patch_end_y,
            template.shape[0],
            template.shape[1],
        )

        cont_patch = template[
            int(cont_patch_start_y) : int(cont_patch_end_y),
            int(cont_patch_start_x) : int(cont_patch_end_x),
        ]

        search_reg_start_y = (
            contour_pt_y - delta_search_patch_win + search_reg_roi_offset_y
        )
        search_reg_end_y = (
            contour_pt_y + delta_search_patch_win + search_reg_roi_offset_y
        )
        search_reg_start_x = (
            contour_pt_x - delta_search_patch_win + search_reg_roi_offset_x
        )
        search_reg_end_x = (
            contour_pt_x + delta_search_patch_win + search_reg_roi_offset_x
        )

        (
            search_reg_start_x,
            search_reg_start_y,
            search_reg_end_x,
            search_reg_end_y,
        ) = check_bb_boundary(
            search_reg_start_x,
            search_reg_start_y,
            search_reg_end_x,
            search_reg_end_y,
            search_reg_roi.shape[0],
            search_reg_roi.shape[1],
        )

        search_reg = search_reg_roi[
            int(search_reg_start_y) : int(search_reg_end_y),
            int(search_reg_start_x) : int(search_reg_end_x),
        ]

        (
            detection_start_x,
            detection_start_y,
            detection_end_x,
            detection_end_y,
            detection_val,
            scale_factor,
        ) = detect_template(cont_patch, search_reg, 1.0, 1.0, 1)
        search_reg_contours = detect_contours(search_reg)

        if detection_val > 0:
            ratio_x, ratio_y = calculate_distance_ratios(
                cont_patch_start_x,
                cont_patch_start_y,
                cont_patch_end_x,
                cont_patch_end_y,
                contour_pt_x,
                contour_pt_y,
            )

            search_reg_centroid_x = (
                (ratio_x * detection_end_x) + detection_start_x
            ) / (ratio_x + 1)
            search_reg_centroid_y = (
                (ratio_y * detection_end_y) + detection_start_y
            ) / (ratio_y + 1)

            search_reg_contour_min_x = float("inf")
            search_reg_contour_min_y = float("inf")
            search_reg_min_distance = float("inf")

            for search_reg_contour in search_reg_contours:
                (min_x, min_y, min_index, min_distance) = find_nearest_contour_point(
                    search_reg_contour, search_reg_centroid_x, search_reg_centroid_y
                )

                if min_distance < search_reg_min_distance:
                    search_reg_min_distance = min_distance
                    search_reg_contour_min_x = min_x
                    search_reg_contour_min_y = min_y

            search_reg_centroid_x = search_reg_contour_min_x
            search_reg_centroid_y = search_reg_contour_min_y

            centroid_search_reg_detection_x = (
                search_reg_centroid_x + search_reg_start_x + search_reg_roi_start_x
            )
            centroid_search_reg_detection_y = (
                search_reg_centroid_y + search_reg_start_y + search_reg_roi_start_y
            )

            centroid_search_reg_detection_x_long = extent["xmin"] + (
                degree_per_pixel_x * centroid_search_reg_detection_x
            )
            centroid_search_reg_detection_y_lat = extent["ymax"] + (
                degree_per_pixel_y * centroid_search_reg_detection_y
            )

            mapped_control_pts.append(
                [
                    contour_pt_x,
                    -1 * contour_pt_y,
                    centroid_search_reg_detection_x_long,
                    centroid_search_reg_detection_y_lat,
                    detection_val,
                    centroid_search_reg_detection_x,
                    -1 * centroid_search_reg_detection_y,
                ]
            )

    return mapped_control_pts


def calculate_iou_binary_images(image1, image2):
    """
    Calculate iou using two binary images
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        image1                  First binary image
        ---------------------   -------------------------------------------
        image2                  Second binary image

        =====================   ===========================================

    returns:
        iou: Computed IOU value
    """
    intersection_img = cv2.bitwise_and(image1, image2)
    intersection_pixels = cv2.countNonZero(intersection_img)

    union_img = cv2.bitwise_or(image1, image2)
    union_pixels = cv2.countNonZero(union_img)

    iou = intersection_pixels / union_pixels
    return iou


def compare_homography_matrices(reference_img, refined_img, template, control_pts):
    """
    Compares and selects the best homography transform between reference homography
    and refined homography
        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        reference_img           Image for computed reference homography
        ---------------------   -------------------------------------------
        refined_img             Image for computed refined homography
        ---------------------   -------------------------------------------
        template                Template image
        ---------------------   -------------------------------------------
        control_pts             Control points

        =====================   ===========================================

    returns:
        index: Index of selected transformed matrix
    """

    control_pts_to_use = control_pts
    src_points = np.zeros((len(control_pts_to_use), 2), dtype=np.float32)
    min_x = float("inf")
    max_x = -1
    min_y = float("inf")
    max_y = -1

    for i, control_pt in enumerate(control_pts_to_use):
        src_points[i, 0] = control_pt[0]
        src_points[i, 1] = -1 * control_pt[1]

        if src_points[i, 0] > max_x:
            max_x = src_points[i, 0]
        if src_points[i, 0] < min_x:
            min_x = src_points[i, 0]

        if src_points[i, 1] > max_y:
            max_y = src_points[i, 1]
        if src_points[i, 1] < min_y:
            min_y = src_points[i, 1]

    area = (max_x - min_x) * (max_y - min_y)
    area_non_zero = cv2.countNonZero(template)

    area_ratio = area / area_non_zero

    reference_img_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    refined_img_gray = cv2.cvtColor(refined_img, cv2.COLOR_BGR2GRAY)

    ret, reference_img_mask = cv2.threshold(
        reference_img_gray, 0, 255, cv2.THRESH_BINARY
    )
    ret, refined_img_mask = cv2.threshold(refined_img_gray, 0, 255, cv2.THRESH_BINARY)

    iou = calculate_iou_binary_images(reference_img_mask, refined_img_mask)

    if area_ratio > 0.35 and iou > 0.70:
        return 1
    else:
        return 0


def map_species_region(
    region_mask_img,
    search_region,
    reference_homography,
    refined_homography,
    selected_index,
    detection_start_x,
    detection_start_y,
    detection_end_x,
    detection_end_y,
    color,
    output_path,
    image_name,
    extent,
    idx,
    region,
):
    """
    Maps the species region masks on the search region using the computed
    homography matrices
    =====================   ===================================================
    **Parameter**            **Description**
    ---------------------   ----------------------------------------------------
    region_mask_img          Species region mask depicting species distribution
    ---------------------   ----------------------------------------------------
    search_region            Color image of search region
    ---------------------   ----------------------------------------------------
    reference_homography     Computed reference homography
    ---------------------   ----------------------------------------------------
    refined_homography       Computed refined homography
    ---------------------   ----------------------------------------------------
    selected_index           Index indicating which transformed matrix is the
                             final computed homography
    ---------------------   ----------------------------------------------------
    detection_start_x        X co-ordinate of the top left corner of detected
                             bounding box
    ---------------------   ----------------------------------------------------
    detection_start_y        Y co-ordinate of the top left corner of detected
                             bounding box
    ---------------------   ----------------------------------------------------
    detection_end_x          X co-ordinate of the bottom right corner of
                             detected bounding box
    ---------------------   ----------------------------------------------------
    detection_end_y          Y co-ordinate of the bottom right corner of
                             detected bounding box
    ---------------------   ----------------------------------------------------
    color                    Color of the current masked region to be mapped
    ---------------------   ----------------------------------------------------
    output_path              Path to generate shapefile output
    ---------------------   ----------------------------------------------------
    image_name               Name of the image
    ---------------------   ----------------------------------------------------
    extent                   Selected extent for search region
    ---------------------   ----------------------------------------------------
    idx                      Index of the current scanned map in the folder
    ---------------------   ----------------------------------------------------
    region                   Index of the current masked region

    =====================   ====================================================

    returns:
        mapped_species_region: Search region with species region mapped
    """

    search_region_height = search_region.shape[0]
    search_region_width = search_region.shape[1]

    warped_img = cv2.warpPerspective(
        region_mask_img,
        reference_homography,
        (search_region_width, search_region_height),
    )

    if selected_index == 1:
        img_to_warp = warped_img[
            detection_start_y:detection_end_y, detection_start_x:detection_end_x
        ]
        warped_img = cv2.warpPerspective(
            img_to_warp, refined_homography, (search_region_width, search_region_height)
        )

    warped_img_contours = detect_contours(warped_img)
    data = write_shapefile(
        warped_img_contours,
        output_path,
        image_name,
        extent,
        search_region_height,
        search_region_width,
        idx,
        region,
    )

    modified_search_region = search_region.copy()
    modified_search_region = cv2.drawContours(
        modified_search_region,
        warped_img_contours,
        -1,
        (int(color[2]), int(color[1]), int(color[0])),
        cv2.FILLED,
    )

    return modified_search_region, data


def process_contour(input_contour):
    """
         Adds end points to contours such that the endpoint is same as
         the start point and reverses the direction of
         the contour
        =====================   ================================
        **Parameter**            **Description**
        ---------------------   --------------------------------
        input_contour            contour to be processed

        =====================   ================================

    returns:
        final_contour: Processed contour
    """

    if len(input_contour) == 0:
        return

    processed_contour = np.flip(input_contour, 0)
    final_contour = []
    for point in processed_contour:
        final_contour.append(point)

    final_contour.append(processed_contour[0])
    return final_contour


def find_current_timestamp():
    """
    Creates a string representing the current timestamp

    :return:
        timestamp_str: Current timestamp string in the format "%H_%M_%S_%f"
    """
    datetime_obj = datetime.now()
    timestamp_str = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S_%f")

    return timestamp_str


def combine_shapefiles(output_dir, file_name, data, extent):
    """
    Combines shapefile outputs in a particular folder
    =====================   ==================================================
    **Parameter**            **Description**
    ---------------------   --------------------------------------------------
    output_dir               Output directory where combined shapefile needs
                             to be dumped
    ---------------------   --------------------------------------------------
    file_name                Filename of the combined shapefile
    ---------------------   --------------------------------------------------
    data                     Data to generate the shapefile
    ---------------------   --------------------------------------------------
    extent                   Selected extent for search region

    =====================   ==================================================

    returns:
        None
    """
    data_frame = DataFrame(data, columns=["FID", "SHAPE", "SPECIES", "SCANNED"])

    data_frame["SHAPE"] = data_frame["SHAPE"].apply(Polygon)
    data_frame.spatial.set_geometry("SHAPE", extent["spatialReference"]["wkid"])

    file_path = os.path.join(output_dir, file_name + ".shp")

    data_frame.spatial.to_featureclass(file_path)


def write_shapefile(
    input_contours,
    output_dir,
    file_name,
    extent,
    search_region_height,
    search_region_width,
    idx,
    region,
):
    """
    Creates data frame using input contours for feature layer creation
    =====================   ===================================================
    **Parameter**            **Description**
    ---------------------   ----------------------------------------------------
    input_contours           Contours which need to be transformed to lat-long
                             co-ordinates
    ---------------------   ----------------------------------------------------
    output_dir               Output directory where combined shapefile needs
                             to be dumped
    ---------------------   ----------------------------------------------------
    file_name                Filename of the combined shapefile
    ---------------------   ----------------------------------------------------
    extent                   Selected extent for search region
    ---------------------   ----------------------------------------------------
    search_region_height     Height of search region image
    ---------------------   ----------------------------------------------------
    search_region_width      Width of search region image
    ---------------------   ----------------------------------------------------
    idx                      Index of the current scanned map in the folder
    ---------------------   ----------------------------------------------------
    region                   Index of the current masked region

    =====================   ====================================================

    returns:
        None
    """

    if len(input_contours) == 0:
        return
    shape = dict()
    shape["rings"] = []
    data_temp = []

    for index, contour in enumerate(input_contours):
        contour = process_contour(contour)
        list_long_lat = calculate_contour_lat_long(
            contour, extent, search_region_height, search_region_width
        )
        shape["rings"].append(list_long_lat)

    shape["spatialReference"] = {"wkid": extent["spatialReference"]["wkid"]}
    shape["hasZ"] = False
    shape["hasM"] = False

    species_name = file_name
    scanned_img = file_name + ".jpg"
    data_temp.append([idx, shape, species_name, scanned_img])

    data_frame = DataFrame(data_temp, columns=["FID", "SHAPE", "SPECIES", "SCANNED"])

    data_frame["SHAPE"] = data_frame["SHAPE"].apply(Polygon)
    data_frame.spatial.set_geometry("SHAPE", extent["spatialReference"]["wkid"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "region" + str(region))):
        os.makedirs(os.path.join(output_dir, "region" + str(region)))

    file_path = os.path.join(output_dir, "region" + str(region), file_name + ".shp")

    data_frame.spatial.to_featureclass(file_path)
    return [idx, shape, species_name, scanned_img]


def calculate_contour_lat_long(
    input_contour, extent, search_region_height, search_region_width
):
    """
    Calculates latitude and longitude co-ordinates for x and y image
    co-ordinates of contour points
    =====================   =============================================
    **Parameter**            **Description**
    ---------------------   ---------------------------------------------
    input_contours           Contours which need to be transformed
                             to lat-long co-ordinates
    ---------------------   ---------------------------------------------
    extent                   Selected extent for search region
    ---------------------   ---------------------------------------------
    search_region_height     Height of search region image
    ---------------------   ---------------------------------------------
    search_region_width      Width of search region image

    =====================   =============================================

    returns:
        list_long_lat: List of latitude and longitude co-ordinates
         corresponding to x and y image co-ordinates of contour points

    """
    degree_per_pixel_x = (extent["xmax"] - extent["xmin"]) / search_region_width

    degree_per_pixel_y = (extent["ymin"] - extent["ymax"]) / search_region_height
    list_long_lat = []

    for point in input_contour:
        long_val = extent["xmin"] + (degree_per_pixel_x * point[0][0])
        lat_val = extent["ymax"] + (degree_per_pixel_y * point[0][1])
        long_lat = [long_val, lat_val]
        list_long_lat.append(long_lat)

    return list_long_lat


def write_georeference_xml_file(control_point, path, image_name, extent):
    """
    Writes XML file used to geo-reference the image on the search image
    =====================   ===================================================
    **Parameter**            **Description**
    ---------------------   ----------------------------------------------------
    control_point            Calculated control points for corresponding scanned
                             map
    ---------------------   ----------------------------------------------------
    path                     Path to dump XML file
    ---------------------   ----------------------------------------------------
    image_name               Name of scanned map

    =====================   ====================================================

    returns:
        None
    """

    cps = control_point
    aux_template = (
        """
    <PAMDataset>
      <Metadata domain="IMAGE_STRUCTURE">
        <MDI key="INTERLEAVE">PIXEL</MDI>
      </Metadata>
      <Metadata domain="xml:ESRI" format="xml">
        <GeodataXform xsi:type="typens:PolynomialXform" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xs="
        http://www.w3.org/2001/XMLSchema" xmlns:typens="http://www.esri.com/schemas/ArcGIS/2.5.0">
          <PolynomialOrder>1</PolynomialOrder>
          <SpatialReference xsi:type="typens:GeographicCoordinateSystem">
            <WKT>GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]</WKT>
            <XOrigin>-400</XOrigin>
            <YOrigin>-400</YOrigin>
            <XYScale>999999999.99999988</XYScale>
            <ZOrigin>-100000</ZOrigin>
            <ZScale>10000</ZScale>
            <MOrigin>-100000</MOrigin>
            <MScale>10000</MScale>
            <XYTolerance>8.983152841195215e-09</XYTolerance>
            <ZTolerance>0.001</ZTolerance>
            <MTolerance>0.001</MTolerance>
            <HighPrecision>true</HighPrecision>
            <LeftLongitude>"""
        + str(extent["xmin"])
        + """</LeftLongitude>
            <WKID>"""
        + str(extent["spatialReference"]["wkid"])
        + """</WKID>
            <LatestWKID>"""
        + str(extent["spatialReference"]["wkid"])
        + """</LatestWKID>
          </SpatialReference>
          <SourceGCPs xsi:type="typens:ArrayOfDouble">
            <Double>201.77370325693613</Double>
            <Double>-74.731001206272538</Double>
            <Double>389.95419492620601</Double>
            <Double>-1362.5314876417463</Double>
            <Double>1216.3773578347073</Double>
            <Double>-428.69359007958542</Double>
            <Double>101.70061599187169</Double>
            <Double>-594.21366542213968</Double>
          </SourceGCPs>
          <TargetGCPs xsi:type="typens:ArrayOfDouble">
            <Double>72.481240538209477</Double>
            <Double>35.906754227992636</Double>
            <Double>77.413842048833715</Double>
            <Double>8.1204837096300242</Double>
            <Double>97.145971270897093</Double>
            <Double>27.129739103324169</Double>
            <Double>70.554967027409319</Double>
            <Double>24.491757967748939</Double>
          </TargetGCPs>
          <Name />
        </GeodataXform>
      </Metadata>
    </PAMDataset>
    """.strip()
    )

    aux = ET.ElementTree(ET.fromstring(aux_template))
    gf = aux.findall("Metadata")[1].find("GeodataXform")
    source_gcps = gf.find("SourceGCPs")
    target_gcps = gf.find("TargetGCPs")

    for child in source_gcps.findall("Double"):
        source_gcps.remove(child)

    for child in target_gcps.findall("Double"):
        target_gcps.remove(child)

    for cp in cps:
        image_x = ET.SubElement(source_gcps, "Double")
        image_x.text = str(cp[0])
        image_y = ET.SubElement(source_gcps, "Double")
        image_y.text = str(cp[1])
        map_x = ET.SubElement(target_gcps, "Double")
        map_x.text = str(cp[2])
        map_y = ET.SubElement(target_gcps, "Double")
        map_y.text = str(cp[3])

    gf.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    gf.set("xmlns:xs", "http://www.w3.org/2001/XMLSchema")
    gf.set("xmlns:typens", "http://www.esri.com/schemas/ArcGIS/2.5.0")

    aux_file_out = os.path.join(path, os.path.basename(image_name) + ".jpg.aux.xml")
    aux.write(
        aux_file_out,
        encoding="utf-8",
    )


def convert_to_xy(lat_val, long_val, extent, degree_per_pixel_x, degree_per_pixel_y):
    """
    Converts latitude and longitude to image x and y co-ordinates
        =====================   ====================================
        **Parameter**            **Description**
        ---------------------   ------------------------------------
        lat_val                  Latitude value of the search
                                 region image point
        ---------------------   ------------------------------------
        long_val                 Longitude value of the search
                                 region image point
        ---------------------   ------------------------------------
        degree_per_pixel_x       Calculated degree w.r.t X axis
        ---------------------   ------------------------------------
        degree_per_pixel_y       Calculated degree w.r.t Y axis

        =====================   ====================================

    returns:
        [x_val, y_val]: Tuple with image x and y co-ordinates corresponding to
        input latitude and longitude co-ordinates
    """
    extreme_left = extent["xmin"]
    extreme_top = extent["ymax"]

    x_val = (long_val - extreme_left) / degree_per_pixel_x

    y_val = (lat_val - extreme_top) / degree_per_pixel_y
    return [int(x_val), int(y_val)]


def calculate_contour_xy(list_lat_long, extent, degree_per_pixel_x, degree_per_pixel_y):
    """
        Calculates image x and y co-ordinates for all points corresponding to
        latitude and longitude co-ordinates
        =====================   =======================================
        **Parameter**            **Description**
        ---------------------   ---------------------------------------
        list_lat_long            List of all lat-long pairs
                                 corresponding to points
        ---------------------   ---------------------------------------
        extent                   Selected extent for search region
        ---------------------   ---------------------------------------
        degree_per_pixel_x       Calculated degree w.r.t X axis
        ---------------------   ---------------------------------------
        degree_per_pixel_y       Calculated degree w.r.t Y axis
        =====================   =======================================

    returns:
        contour_xy: List of x-y pairs corresponding to points
    """
    contour_xy = []
    for point in list_lat_long:
        xy = convert_to_xy(
            point[1], point[0], extent, degree_per_pixel_x, degree_per_pixel_y
        )
        contour_xy.append([xy])

    return contour_xy


def make_world_map_from_shapefile(
    image_path,
    image_height,
    image_width,
    extent,
    degree_per_pixel_x,
    degree_per_pixel_y,
    color,
):
    """
        Creates search image using shapefile
        =====================   =========================================
        **Parameter**            **Description**
        ---------------------   -----------------------------------------
        image_path               Path of the scanned map
        ---------------------   -----------------------------------------
        image_height             Height of scanned map
        ---------------------   -----------------------------------------
        image_width              Height of scanned map
        ---------------------   -----------------------------------------
        extent                   Selected extent for search region
        ---------------------   -----------------------------------------
        degree_per_pixel_x       Calculated degree w.r.t X axis
        ---------------------   -----------------------------------------
        degree_per_pixel_y       Calculated degree w.r.t Y axis
        ---------------------   -----------------------------------------
        color                    r, g, b value for water color
        =====================   =========================================

    returns:
        world_map_img: Color image of the search image
    """

    data_frame = arcgis.features.GeoAccessor.from_featureclass(os.path.join(image_path))

    contours = []
    max_length = -1
    for index in data_frame.index:
        contour = calculate_contour_xy(
            data_frame["SHAPE"][index]["rings"][0],
            extent,
            degree_per_pixel_x,
            degree_per_pixel_y,
        )
        length = len(contour)
        if length > max_length:
            max_length = length

        contours.append(np.array(contour))

    world_map_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    world_map_img[:, :, 0] = color[2]
    world_map_img[:, :, 1] = color[1]
    world_map_img[:, :, 2] = color[0]
    world_map_img = cv2.drawContours(
        world_map_img, contours, -1, (255, 255, 255), cv2.FILLED
    )

    return world_map_img


def generate_search_template(search_image, process_folder, x1, y1, x2, y2):
    """
        Crops search template from the bigger image using a given extent and converts to
        binary image
        =====================   ==========================================
        **Parameter**            **Description**
        ---------------------   ------------------------------------------
        search_image             Search Image
        ---------------------   ------------------------------------------
        process_folder           Path of the intermediate result folder
        ---------------------   ------------------------------------------
        x1                       Start index of column
        ---------------------   ------------------------------------------
        y1                       Start index of row
        ---------------------   ------------------------------------------
        x2                       End index of column
        ---------------------   ------------------------------------------
        y2                       End index of row
        =====================   ==========================================

    returns:
        transformed_img: transformed image of the search region
    """
    all_images = os.listdir(os.path.join(process_folder))
    display_img = search_image
    image = search_image
    image = np.zeros(search_image.shape, np.uint8)
    image[y1:y2, x1:x2] = search_image[y1:y2, x1:x2]

    if image.ndim != 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    transformed_img = np.zeros(
        (image_gray.shape[0], image_gray.shape[1]), dtype=np.uint8
    )

    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]

    for y in range(0, image_height):
        for x in range(0, image_width):
            if image_gray[y][x] >= 245:
                transformed_img[y][x] = 255
            else:
                transformed_img[y][x] = 0

    for image_path in all_images:
        path = os.path.join(process_folder, image_path, "search_region")
        if not os.path.isdir(path):
            os.mkdir(path)
        cv2.imwrite(os.path.join(path, "search_region_rgb.jpg"), display_img)
        cv2.imwrite(os.path.join(path, "search_region_template.jpg"), transformed_img)

    return transformed_img


def generate_search_coordinate(
    extent,
    bigger_extreme_left,
    bigger_extreme_top,
    bigger_extreme_right,
    bigger_extreme_bottom,
    degree_per_pixel_x,
    degree_per_pixel_y,
):
    """
        Generates co-ordinate for search region
        =====================   ============================================
        **Parameter**            **Description**
        ---------------------   --------------------------------------------
        extent                   Selected extent for search region
        ---------------------   --------------------------------------------
        bigger_extreme_left      Extreme left co-ordinate of world image
        ---------------------   --------------------------------------------
        bigger_extreme_top       Extreme top co-ordinate of world image
        ---------------------   --------------------------------------------
        bigger_extreme_right     Extreme right co-ordinate of world image
        ---------------------   --------------------------------------------
        bigger_extreme_bottom    Extreme bottom co-ordinate of world image
        ---------------------   --------------------------------------------
        degree_per_pixel_x       Calculated degree w.r.t X axis
        ---------------------   --------------------------------------------
        degree_per_pixel_y       Calculated degree w.r.t Y axis

        =====================   ============================================

    returns:
    x1, y1, x2, y2 to crop extent from search region
    """

    extreme_left = extent["xmin"]
    extreme_top = extent["ymax"]
    extreme_right = extent["xmax"]
    extreme_bottom = extent["ymin"]

    x1 = int(((abs(bigger_extreme_left) - abs(extreme_left)) / abs(degree_per_pixel_x)))
    y1 = int(((abs(bigger_extreme_top) - abs(extreme_top)) / abs(degree_per_pixel_y)))
    y2 = int(
        ((abs(bigger_extreme_bottom) + abs(extreme_bottom)) / abs(degree_per_pixel_y))
    )
    x2 = int(
        ((abs(bigger_extreme_right) + abs(extreme_right)) / abs(degree_per_pixel_x))
    )
    return x1, y1, x2, y2


def plot_image(*images):
    """
        Displays output
        =====================   ============================================
        **Parameter**            **Description**
        ---------------------   --------------------------------------------
        images                   List of images and their descriptions
        =====================   ============================================
    returns:
        None
    """
    all_images = images[0]
    all_names = images[1]
    size = images[2]
    fig, ax = plt.subplots(1, len(all_images), figsize=(size, size))

    for i, img in enumerate(all_images):
        ax[i].axis("off")
        font_dict = {"fontsize": 10}
        ax[i].set_aspect(aspect=0.5)
        ax[i].title.set_position([0.5, -0.07 * len(all_images)])
        ax[i].set_title(all_names[i], fontdict=font_dict)
        ax[i].imshow(img)


def display_progress_bar(i, max_images, postText):
    """
        Displays progress bar
        =====================   =======================================
        **Parameter**            **Description**
        ---------------------   ---------------------------------------
        i                        Index of the image inside the loop
        ---------------------   ---------------------------------------
        max_images               Maximum number of images
        ---------------------   ---------------------------------------
        postText                 Text to be added at the end of the
                                 progress bar
        =====================   =======================================
    returns:
        None
    """
    n_bar = 50
    j = i / max_images
    sys.stdout.write("\r Status: ")
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


class ScannedMapDigitizer:
    """
    Creates the object for :class:`~arcgis.learn.ScannedMapDigitizer` class

    =====================   ============================================
    **Parameter**            **Description**
    ---------------------   --------------------------------------------
    input_folder             Path to the folder that contains extracted
                             maps
    ---------------------   --------------------------------------------
    output_folder            Path to the folder where intermediate
                             results should get generated

    =====================   ============================================

    """

    input_path = None
    process_path = None
    all_images_path = []
    extent_data = {}

    def __init__(self, input_folder, output_folder):
        if not HAS_DEPS:
            print("**Environment fails**")
            raise Exception(
                f"""{import_exception} \n\nThis module requires opencv, pandas,
                                        numpy and matplotlib as its dependencies."""
            )
        else:
            self.initialize_variables(input_folder, output_folder)

    @classmethod
    def get_search_region_extent(cls):
        """
        Getter function for search region extent
        """
        try:
            extent = {
                "spatialReference": {"wkid": cls.wkid},
                "xmin": cls.logitude_extreme_left,
                "ymax": cls.latitude_extreme_top,
                "xmax": cls.logitude_extreme_right,
                "ymin": cls.latitude_extreme_bottom,
            }
        except AttributeError:
            print(
                "Search region extent is not set, please use set_search_region_extent() to set the search extent. "
            )
            extent = {}
        return extent

    @classmethod
    def set_search_region_extent(cls, extent):
        """
        Creates the object for :class:`~arcgis.learn.ScannedMapDigitizer` class

        =====================   ============================================
        **Parameter**            **Description**
        ---------------------   --------------------------------------------
        extent                  Extent defines the extreme longitude/latitude
                                of the search region.

        =====================   ============================================

        """

        cls.wkid = extent["spatialReference"]["wkid"]
        cls.logitude_extreme_left = extent["xmin"]
        cls.latitude_extreme_top = extent["ymax"]
        cls.logitude_extreme_right = extent["xmax"]
        cls.latitude_extreme_bottom = extent["ymin"]
        return "Search region extent updated successfully !!"

    @classmethod
    def initialize_variables(cls, input_folder, output_folder):
        cls.input_path = input_folder
        cls.process_path = output_folder
        cls.all_images_path = []

    @classmethod
    def create_mask(
        cls,
        color_list,
        color_delta=60,
        kernel_size=None,
        kernel_type="rect",
        show_result=True,
    ):
        """
        Generates the binary masked images

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        color_list              A list containing different color inputs
                                in list/tuple format [(r, g, b)].
                                For eg: [[110,10,200], [210,108,11]].
        ---------------------   -------------------------------------------
        color_delta             A value which defines the range around the
                                threshold value for a specific color used
                                for creating the mask images.
                                Default value is 60.
        ---------------------   -------------------------------------------
        kernel_size             A list of 2 integers corresponding to size
                                of the morphological filter operations
                                closing and opening respectively.
        ---------------------   -------------------------------------------
        kernel_type             A string value defining the type/shape of
                                the kernel. kernel type can be "rect",
                                "elliptical" or "cross".
                                Default value is "rect".
        ---------------------   -------------------------------------------
        show_result             A boolean value. Set to "True" to visualize
                                results and set to "False" otherwise.
        =====================   ===========================================
        """

        input_folder = cls.input_path
        output_folder = cls.process_path
        cls.all_images_path = []
        all_images = os.listdir(os.path.join(input_folder))

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        if kernel_size is not None:
            unittest.TestCase().assertEqual(
                len(kernel_size),
                2,
                """Expected length of \
                                                                    kernel_size is 2""",
            )
            closing, opening = generate_kernel(kernel_size, kernel_type)
        else:
            closing, opening = [None, None]

        for idx, image_path in enumerate(all_images):
            image_name = image_path.split(".")[0]
            cls.all_images_path.append(image_name)
            image_folder = os.path.join(output_folder, image_name)
            if not os.path.isdir(image_folder):
                os.mkdir(image_folder)
            image_folder = os.path.join(image_folder, "mask")
            if not os.path.isdir(image_folder):
                os.mkdir(image_folder)

            image = cv2.imread(os.path.join(input_folder, image_path))
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hf = h5py.File(
                os.path.join(
                    output_folder, image_name, "mask", "mask_color_details.h5"
                ),
                "w",
            )
            img_masks = [rgb_img]
            img_name = ["Scanned Map"]
            for i, color in enumerate(color_list):
                clr = np.array([[color]])
                mask = cv2.inRange(rgb_img, clr - color_delta, clr + color_delta)
                background = np.full(rgb_img.shape, 255, dtype=np.uint8)
                image_output = cv2.bitwise_or(background, background, mask=mask)
                if closing is not None:
                    image_output = cv2.morphologyEx(
                        image_output, cv2.MORPH_CLOSE, closing
                    )
                if opening is not None:
                    image_output = cv2.morphologyEx(
                        image_output, cv2.MORPH_OPEN, opening
                    )

                cv2.imwrite(os.path.join(output_folder, image_name, "input.jpg"), image)
                cv2.imwrite(
                    os.path.join(image_folder, "region_mask_" + str(i) + ".jpg"),
                    image_output,
                )
                hf.create_dataset("region_mask_" + str(i) + ".jpg", data=color)
                img_masks.append(image_output)
                img_name.append("Region Binary Mask")

            hf.close()
            ipd.clear_output(wait=True)
            display_progress_bar(idx + 1, len(all_images), "Completed \n\n\r")
            if show_result:
                plot_image(img_masks, img_name, 10)

        print("Output generated at ", output_folder)

    @classmethod
    def create_template_image(
        cls, color, color_delta=10, kernel_size=2, show_result=True
    ):
        """
        This method generates templates and color masks from scanned maps which
        are used in the subsequent step of template matching.

        =====================   ===============================================
        **Parameter**            **Description**
        ---------------------   -----------------------------------------------
        color                   A list containing r, g, b value representing land color.
                                The color parameter is required for extracting
                                the land region and generating the binary mask.
        ---------------------   -----------------------------------------------
        color_delta             A value which defines the range around the
                                threshold value for a specific color used for
                                creating the mask images.
                                Default value is 60.
        ---------------------   -----------------------------------------------
        kernel_size             An integer corresponding to size of kernel
                                used for dilation(morphological operation).
        ---------------------   -----------------------------------------------
        show_result             A Boolean value. Set to "True" to visualize
                                results and set to "False" otherwise.
        =====================   ===============================================

        """
        process_folder = cls.process_path
        all_images = cls.all_images_path
        clr = np.array(color)
        for idx, image_path in enumerate(all_images):
            image = cv2.imread(os.path.join(process_folder, image_path, "input.jpg"))
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            display_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
            image_name = ["Scanned Map"]
            all_masked_images = os.listdir(
                os.path.join(process_folder, image_path, "mask")
            )
            all_masked_images = [
                img_name for img_name in all_masked_images if "region_mask_" in img_name
            ]
            for masked_image in all_masked_images:
                masked_path = os.path.join(
                    process_folder, image_path, "mask", masked_image
                )
                mask_filled_image = fill_mask_pixels(rgb_img, masked_path, color)
                if kernel_size is not None:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    rgb_img = cv2.dilate(mask_filled_image, kernel, iterations=1)
                else:
                    rgb_img = mask_filled_image

            black_mask = cv2.inRange(rgb_img, clr - color_delta, clr + color_delta)
            background = np.full(rgb_img.shape, 255, dtype=np.uint8)
            image_output = cv2.bitwise_or(background, background, mask=black_mask)
            template_image_path = os.path.join(
                process_folder, image_path, "template_image"
            )
            if not os.path.isdir(template_image_path):
                os.mkdir(template_image_path)
            cv2.imwrite(
                os.path.join(template_image_path, "masked_template.jpg"), image_output
            )
            display_images.append(image_output)
            image_name.append("Binary Template")
            ipd.clear_output(wait=True)
            display_progress_bar(idx + 1, len(all_images), "Completed \n\n\r")
            if show_result:
                plot_image(display_images, image_name, 12)

        print("Output generated at:", process_folder)

    @classmethod
    def prepare_search_region(
        cls, search_image, color, extent, image_height, image_width, show_result=True
    ):
        """
        This method prepares the search region in which the prepared templates are
        to be searched.

        =====================   ==================================================
        **Parameter**            **Description**
        ---------------------   --------------------------------------------------
        search_image            Path to the bigger image/shapefile.
        ---------------------   --------------------------------------------------
        color                   A list containing r, g, b value representing water color.
                                For Eg: [173, 217, 219].
        ---------------------   --------------------------------------------------
        extent                  Extent defines the extreme longitude/latitude
                                of the search region.
        ---------------------   --------------------------------------------------
        image_height            Height of the search region.
        ---------------------   --------------------------------------------------
        image_width             Width of the search region.
        ---------------------   --------------------------------------------------
        show_result             A boolean value. Set to "True" to visualize
                                results and set to "False" otherwise.
        =====================   ==================================================
        """

        cls.extent_data = extent
        try:
            bigger_extreme_left = cls.logitude_extreme_left
            bigger_extreme_top = cls.latitude_extreme_top
            bigger_extreme_right = cls.logitude_extreme_right
            bigger_extreme_bottom = cls.latitude_extreme_bottom
        except AttributeError:
            cls.set_search_region_extent(extent)
            bigger_extreme_left = cls.logitude_extreme_left
            bigger_extreme_top = cls.latitude_extreme_top
            bigger_extreme_right = cls.logitude_extreme_right
            bigger_extreme_bottom = cls.latitude_extreme_bottom

        bigger_extent = {
            "xmin": bigger_extreme_left,
            "ymax": bigger_extreme_top,
            "xmax": bigger_extreme_right,
            "ymin": bigger_extreme_bottom,
        }

        degree_per_pixel_x = (bigger_extreme_right - bigger_extreme_left) / image_width
        degree_per_pixel_y = (bigger_extreme_bottom - bigger_extreme_top) / image_height

        if search_image.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif")
        ):
            image = cv2.imread(os.path.join(search_image))
            image = cv2.resize(image, (image_width, image_height))

        elif search_image.lower().endswith(".shp"):
            print("Processing Shape file...")
            image = make_world_map_from_shapefile(
                search_image,
                image_height,
                image_width,
                bigger_extent,
                degree_per_pixel_x,
                degree_per_pixel_y,
                color,
            )
            print("Done.")
        else:
            raise NameError("Invalid input image or file extention!!")

        print("Extracting given extent from the world imagery...")
        process_folder = cls.process_path
        x1, y1, x2, y2 = generate_search_coordinate(
            extent,
            bigger_extreme_left,
            bigger_extreme_top,
            bigger_extreme_right,
            bigger_extreme_bottom,
            degree_per_pixel_x,
            degree_per_pixel_y,
        )
        processed_image = generate_search_template(
            image, process_folder, x1, y1, x2, y2
        )

        path = os.path.join(process_folder, "search_region")
        if not os.path.isdir(path):
            os.mkdir(path)
        print("Done.")
        cv2.imwrite(os.path.join(path, "search_template.jpg"), processed_image)
        cv2.imwrite(os.path.join(path, "search_rgb.jpg"), image)
        title = ["Search Region", "Binary Search Region"]
        if show_result:
            plot_image(
                [
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB),
                ],
                title,
                13,
            )

    @classmethod
    def match_template_multiscale(
        cls, min_scale, max_scale, num_scales, show_result=True
    ):
        """
        This method finds the location of the best match of a smaller image
        (template) in a larger image(search image) assuming it exists in the
        larger image.

        =====================   ============================================
        **Parameter**            **Description**
        ---------------------   --------------------------------------------
        min_scale               An integer representing the minimum scale
                                at which template matching is performed.
        ---------------------   --------------------------------------------
        max_scale               An integer representing maximum scale at
                                which template matching is performed.
        ---------------------   --------------------------------------------
        num_scales              An integer representing the number
                                of scales at which template matching is
                                performed.
        ---------------------   --------------------------------------------
        show_result             A Boolean value. Set to "True" to visualize
                                results and set to "False" otherwise.
        =====================   ============================================

        """
        process_folder = cls.process_path

        all_images = cls.all_images_path

        region_name = []
        region_accuracy = []

        for idx, image_path in enumerate(all_images):
            image = cv2.imread(os.path.join(process_folder, image_path, "input.jpg"))
            display_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
            image_template = cv2.imread(
                os.path.join(
                    process_folder, image_path, "template_image", "masked_template.jpg"
                ),
                0,
            )
            search_image = cv2.imread(
                os.path.join(
                    process_folder, image_path, "search_region", "search_region_rgb.jpg"
                )
            )
            search_image_template = cv2.imread(
                os.path.join(
                    process_folder,
                    image_path,
                    "search_region",
                    "search_region_template.jpg",
                ),
                0,
            )
            (
                detection_start_x,
                detection_start_y,
                detection_end_x,
                detection_end_y,
                max_val,
                scale_factor,
            ) = detect_template(
                image_template, search_image_template, min_scale, max_scale, num_scales
            )

            region_name.append(image_path)
            region_accuracy.append(round(max_val, 2))
            reference_homography = compute_reference_homography(
                detection_start_x,
                detection_start_y,
                detection_end_x,
                detection_end_y,
                search_image_template.shape[0],
                search_image_template.shape[1],
                image_template.shape[0],
                image_template.shape[1],
            )
            path = os.path.join(process_folder, image_path, "template_match")
            if not os.path.isdir(path):
                os.mkdir(path)

            hf = h5py.File(os.path.join(path, "reference_homography.h5"), "w")
            hf.create_dataset(
                "detection_points",
                data=(
                    detection_start_x,
                    detection_start_y,
                    detection_end_x,
                    detection_end_y,
                ),
            )
            hf.create_dataset("reference_homography", data=reference_homography)
            hf.close()
            reference_image = apply_homography(
                image, search_image, reference_homography
            )

            cv2.imwrite(os.path.join(path, "reference_image.jpg"), reference_image)

            reference_world_image = cv2.addWeighted(
                search_image, 0.5, reference_image, 0.5, 0
            )
            cv2.imwrite(
                os.path.join(path, "reference_world_image.jpg"), reference_world_image
            )
            display_images.append(
                cv2.cvtColor(reference_world_image, cv2.COLOR_BGR2RGB)
            )
            title = [
                "Scanned Map",
                "Match Accuracy: " + str(round(max_val, 3) * 100) + " %",
            ]
            ipd.clear_output(wait=True)
            display_progress_bar(idx + 1, len(all_images), "Completed \n\n\r")
            if show_result:
                plot_image(display_images, title, 12)

        print("Output generated at: ", process_folder)

    @classmethod
    def georeference_image(cls, padding_param, show_result=True):
        """
        This method estimates the control point pairs by traversing the
        contours of template image and finding the corresponding matches
        on the search region ROI image

        =====================   ========================================
        **Parameter**            **Description**
        ---------------------   ----------------------------------------
        padding_param           A tuple that contains x-padding
                                and y-padding at 0th and 1st index
                                respectively.
        ---------------------   ----------------------------------------
        show_result             A Boolean value. Set to "True" to
                                visualize results and set to "False"
                                otherwise.
        =====================   ========================================
        """

        process_folder = cls.process_path

        all_images = cls.all_images_path
        extent = cls.get_search_region_extent()

        for idx, image_path in enumerate(all_images):
            image = cv2.imread(os.path.join(process_folder, image_path, "input.jpg"))
            image_template = cv2.imread(
                os.path.join(
                    process_folder, image_path, "template_image", "masked_template.jpg"
                ),
                0,
            )
            search_image = cv2.imread(
                os.path.join(
                    process_folder, image_path, "search_region", "search_region_rgb.jpg"
                )
            )
            search_image_template = cv2.imread(
                os.path.join(
                    process_folder,
                    image_path,
                    "search_region",
                    "search_region_template.jpg",
                ),
                0,
            )
            reference_image = cv2.imread(
                os.path.join(
                    process_folder, image_path, "template_match", "reference_image.jpg"
                )
            )

            hf = h5py.File(
                os.path.join(
                    process_folder,
                    image_path,
                    "template_match",
                    "reference_homography.h5",
                ),
                "r",
            )
            detection = hf.get("detection_points")
            (
                detection_start_x,
                detection_start_y,
                detection_end_x,
                detection_end_y,
            ) = np.array(detection)
            homography = hf.get("reference_homography")
            reference_homography = np.array(homography)
            hf.close()

            search_image_height = search_image.shape[0]
            search_image_width = search_image.shape[1]
            search_image_roi_margin_x = padding_param[0]
            search_image_roi_margin_y = padding_param[1]

            warped_processed_image = cv2.warpPerspective(
                image_template,
                reference_homography,
                (search_image_width, search_image_height),
            )
            warped_original_image = cv2.warpPerspective(
                image, reference_homography, (search_image_width, search_image_height)
            )

            warped_image = warped_original_image[
                detection_start_y:detection_end_y, detection_start_x:detection_end_x
            ]
            warped_template = warped_processed_image[
                detection_start_y:detection_end_y, detection_start_x:detection_end_x
            ]

            search_image_roi_start_x = detection_start_x - search_image_roi_margin_x
            search_image_roi_end_x = detection_end_x + search_image_roi_margin_x
            search_image_roi_start_y = detection_start_y - search_image_roi_margin_y
            search_image_roi_end_y = detection_end_y + search_image_roi_margin_y

            (
                search_image_roi_start_x,
                search_image_roi_start_y,
                search_image_roi_end_x,
                search_image_roi_end_y,
            ) = check_bb_boundary(
                search_image_roi_start_x,
                search_image_roi_start_y,
                search_image_roi_end_x,
                search_image_roi_end_y,
                search_image_template.shape[0],
                search_image_template.shape[1],
            )

            search_image_offset_start_x = detection_start_x - search_image_roi_start_x
            search_image_offset_start_y = detection_start_y - search_image_roi_start_y

            search_image_roi_processed = search_image_template[
                int(search_image_roi_start_y) : int(search_image_roi_end_y),
                int(search_image_roi_start_x) : int(search_image_roi_end_x),
            ]

            control_pts = estimate_control_points(
                search_image_roi_processed,
                warped_template,
                search_image_offset_start_x,
                search_image_offset_start_y,
                search_image_roi_start_x,
                search_image_roi_start_y,
                search_image,
                extent,
            )

            path = os.path.join(process_folder, image_path, "georeference_image")

            refined_homography, outlier_mask = compute_refined_homography_outlier_mask(
                control_pts
            )

            filtered_control_pts = filter_control_pts(control_pts, outlier_mask)
            (
                search_image_filtered_control_pts,
                input_image_filtered_control_pts,
            ) = annotate_control_pts(warped_image, search_image, filtered_control_pts)

            refined_image = apply_homography(
                warped_image, search_image, refined_homography
            )
            refined_search_image = cv2.addWeighted(
                search_image, 0.5, refined_image, 0.5, 0
            )
            selected_index = compare_homography_matrices(
                reference_image, refined_image, warped_template, control_pts
            )

            if not os.path.isdir(path):
                os.mkdir(path)
            write_georeference_xml_file(filtered_control_pts, path, image_path, extent)
            hf = h5py.File(os.path.join(path, "refined_homography.h5"), "w")
            hf.create_dataset("refined_homography", data=refined_homography)
            hf.create_dataset("selected_index", data=[selected_index])
            hf.close()
            cv2.imwrite(
                os.path.join(path, "filtered_control_pts.jpg"),
                input_image_filtered_control_pts,
            )
            cv2.imwrite(os.path.join(path, "warped_image.jpg"), warped_image)
            cv2.imwrite(os.path.join(path, image_path + ".jpg"), warped_image)
            title = [
                "Control Points on Scanned Map",
                "Control Points on Search Region",
                "Transformed Image",
            ]
            ipd.clear_output(wait=True)
            display_progress_bar(idx + 1, len(all_images), "Completed \n\n\r")
            if show_result:
                plot_image(
                    [
                        cv2.cvtColor(
                            input_image_filtered_control_pts, cv2.COLOR_BGR2RGB
                        ),
                        cv2.cvtColor(
                            search_image_filtered_control_pts, cv2.COLOR_BGR2RGB
                        ),
                        cv2.cvtColor(refined_search_image, cv2.COLOR_BGR2RGB),
                    ],
                    title,
                    15,
                )

        print("Output generated at: ", process_folder)

    @classmethod
    def digitize_image(cls, show_result=True):
        """
        This method is the final step in the pipeline that maps the
        species regions on the search image using the computed
        transformations.
        Also, it generates the shapefiles for the species region that can be
        visualized using ArcGIS Pro and further edited.

        =====================   =============================================
        **Parameter**            **Description**
        ---------------------   ---------------------------------------------
        show_result             A Boolean value. Set to "True" to visualize
                                results and set to "False" otherwise.
        =====================   =============================================

        """
        process_folder = cls.process_path
        extent = cls.get_search_region_extent()

        all_images = cls.all_images_path
        all_combined = {}
        for idx, image_path in enumerate(all_images):
            data = []
            color_counter = 0
            image = cv2.imread(os.path.join(process_folder, image_path, "input.jpg"))
            search_image = cv2.imread(
                os.path.join(
                    process_folder, image_path, "search_region", "search_region_rgb.jpg"
                )
            )

            hf = h5py.File(
                os.path.join(
                    process_folder,
                    image_path,
                    "template_match",
                    "reference_homography.h5",
                ),
                "r",
            )
            detection = hf.get("detection_points")
            (
                detection_start_x,
                detection_start_y,
                detection_end_x,
                detection_end_y,
            ) = np.array(detection)
            homography = hf.get("reference_homography")
            reference_homography = np.array(homography)
            hf.close()

            hf = h5py.File(
                os.path.join(
                    process_folder,
                    image_path,
                    "georeference_image",
                    "refined_homography.h5",
                ),
                "r",
            )
            homography = hf.get("refined_homography")
            refined_homography = np.array(homography)
            index = hf.get("selected_index")
            selected_index = int(np.array(index)[0])
            hf.close()

            path = os.path.join(process_folder, image_path, "digitize_map")
            if not os.path.isdir(path):
                os.mkdir(path)

            mapped_species_region_img = search_image
            hf = h5py.File(
                os.path.join(
                    process_folder, image_path, "mask", "mask_color_details.h5"
                ),
                "r",
            )
            all_masked_images = os.listdir(
                os.path.join(process_folder, image_path, "mask")
            )
            all_masked_images = [
                img_name for img_name in all_masked_images if "region_mask_" in img_name
            ]

            for region, masked_image in enumerate(all_masked_images):
                color_counter += 1
                masked_path = os.path.join(
                    process_folder, image_path, "mask", masked_image
                )
                region_mask_img = cv2.imread(masked_path)
                color = np.array(hf.get(masked_image), dtype="int64")
                (mapped_species_region_img, shape_file_details) = map_species_region(
                    region_mask_img,
                    mapped_species_region_img,
                    reference_homography,
                    refined_homography,
                    selected_index,
                    detection_start_x,
                    detection_start_y,
                    detection_end_x,
                    detection_end_y,
                    color,
                    path,
                    image_path,
                    extent,
                    idx,
                    region,
                )
                data.append(shape_file_details)
                if idx == 0:
                    all_combined[str(region)] = [shape_file_details]
                else:
                    all_combined[str(region)].append(shape_file_details)

            hf.close()

            cv2.imwrite(
                os.path.join(path, "map_with_specie_region.jpg"),
                mapped_species_region_img,
            )
            title = ["Scanned Map", "Transformed Region Mask"]
            ipd.clear_output(wait=True)
            display_progress_bar(idx + 1, len(all_images), "Completed \n\n\r")
            if show_result:
                plot_image(
                    [
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(mapped_species_region_img, cv2.COLOR_BGR2RGB),
                    ],
                    title,
                    12,
                )

        combined_folder = os.path.join(process_folder, "combined_shapefiles")
        if not os.path.isdir(combined_folder):
            os.mkdir(combined_folder)
        for key, val in all_combined.items():
            if not os.path.isdir(
                os.path.join(combined_folder, "combined_region_" + str(key))
            ):
                os.mkdir(os.path.join(combined_folder, "combined_region_" + str(key)))
            combine_shapefiles(
                os.path.join(combined_folder, "combined_region_" + str(key)),
                "combined_region",
                val,
                extent,
            )
