"""
Affine transformation functions
"""
from __future__ import division
import math
from . import Point, Polygon, Polyline
from . import MultiPoint, Geometry

GEOM_TYPES = (Point, Polygon, Polyline, MultiPoint)
__all__ = ["scale", "rotate", "skew", "translate"]


# -------------------------------------------------------------------------
def scale(geom: Geometry, *scale_factor: int):
    """
    Create a scaling transform from a scalar value (float)

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    geom                Required class:`~arcgis.geometry.Geometry object or dictionary.
    ---------------     --------------------------------------------------------------------
    scale_factor        Required int. A scalar value will scale in both dimensions equally
    ===============     ====================================================================

    :return:
        Dictionary or class:`~arcgis.geometry.Geometry` object
    """
    import numpy as np

    if len(scale_factor) == 1:
        sy = sx = scale_factor[0]
    elif len(scale_factor) > 1:
        sx = scale_factor[0]
        sy = scale_factor[1]
    else:
        return
    A = np.matrix([[sx, 0.0], [0.0, sy]])
    if isinstance(geom, dict) or isinstance(geom, GEOM_TYPES):
        if "x" in geom and "y" in geom:  # translates point
            matrix = np.matrix([[geom["x"]], [geom["y"]]])
            val = (A * matrix).tolist()
            geom["x"] = val[0][0]
            geom["y"] = val[1][0]
            return geom
        elif "rings" in geom or "paths" in geom:  # translates lines/polygons
            rings = []
            if "paths" in geom:
                coords = geom["paths"]
            else:
                coords = geom["rings"]
            for part in coords:
                x_coords = [pt[0] for pt in part]
                y_coords = [pt[1] for pt in part]
                matrix = np.matrix([x_coords, y_coords])
                vals = (A * matrix).tolist()
                rings.append(list(zip(vals[0], vals[1])))
            if "paths" in geom:
                geom["paths"] = rings
            else:
                geom["rings"] = rings
            return geom
        elif "points" in geom:  # translates Multipoint
            for pt in geom["points"]:
                matrix = np.matrix([[pt[0]], [pt[1]]])
                val = (A * matrix).tolist()
                pt[0] = val[0][0]
                pt[1] = val[1][0]
            return geom
        else:
            return None
    return None


# -------------------------------------------------------------------------
def rotate(geom: Geometry, theta: float):
    """
    Rotates a geometry counter-clockwise by some degree theta

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    geom                Required class:`~arcgis.geometry.Geometry object or dictionary.
    ---------------     --------------------------------------------------------------------
    theta               Required angle of rotation
    ===============     ====================================================================

    :return:
        Dictionary or class:`~arcgis.geometry.Geometry` object
    """
    import numpy as np

    A = np.matrix(
        [[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    if isinstance(geom, dict) or isinstance(geom, GEOM_TYPES):
        if "x" in geom and "y" in geom:  # translates point
            matrix = np.matrix([[geom["x"]], [geom["y"]]])
            val = (A * matrix).tolist()
            geom["x"] = val[0][0]
            geom["y"] = val[1][0]
            return geom
        elif "rings" in geom or "paths" in geom:  # translates lines/polygons
            rings = []
            if "paths" in geom:
                coords = geom["paths"]
            else:
                coords = geom["rings"]
            for part in coords:
                x_coords = [pt[0] for pt in part]
                y_coords = [pt[1] for pt in part]
                matrix = np.matrix([x_coords, y_coords])
                vals = (A * matrix).tolist()
                rings.append(list(zip(vals[0], vals[1])))
            if "paths" in geom:
                geom["paths"] = rings
            else:
                geom["rings"] = rings
            return geom
        elif "points" in geom:  # translates Multipoint
            for pt in geom["points"]:
                matrix = np.matrix([[pt[0]], [pt[1]]])
                val = (A * matrix).tolist()
                pt[0] = val[0][0]
                pt[1] = val[1][0]
            return geom
        else:
            return None
    return None


# -------------------------------------------------------------------------
def skew(geom: Geometry, x_angle: float = 0, y_angle: float = 0):
    """
    Create a skew transform along one or both axes.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    geom                Required class:`~arcgis.geometry.Geometry object or dictionary.
    ---------------     --------------------------------------------------------------------
    x_angle             Required angle to skew in the x-coordinates
    ---------------     --------------------------------------------------------------------
    y_angle             Required angle to skew in the y-coordinates
    ===============     ====================================================================

    :return:
        Dictionary or class:`~arcgis.geometry.Geometry` object
    """
    import numpy as np

    A = np.matrix([[1, math.tan(x_angle)], [math.tan(y_angle), 1]])
    if isinstance(geom, dict) or isinstance(geom, GEOM_TYPES):
        if "x" in geom and "y" in geom:  # translates point
            matrix = np.matrix(
                [
                    [geom["x"]],
                    [geom["y"]],
                ]
            )
            val = (A * matrix).tolist()
            geom["x"] = val[0][0]
            geom["y"] = val[1][0]
            return geom
        elif "rings" in geom or "paths" in geom:  # translates lines/polygons
            rings = []
            if "paths" in geom:
                coords = geom["paths"]
            else:
                coords = geom["rings"]
            for part in coords:
                x_coords = [pt[0] for pt in part]
                y_coords = [pt[1] for pt in part]
                matrix = np.matrix([x_coords, y_coords])
                vals = (A * matrix).tolist()
                rings.append(list(zip(vals[0], vals[1])))
            if "paths" in geom:
                geom["paths"] = rings
            else:
                geom["rings"] = rings
            return geom
        elif "points" in geom:  # translates Multipoint
            for pt in geom["points"]:
                matrix = np.matrix(
                    [
                        [pt[0]],
                        [pt[1]],
                    ]
                )
                val = (A * matrix).tolist()
                pt[0] = val[0][0]
                pt[1] = val[1][0]
            return geom
        else:
            return None


# -------------------------------------------------------------------------
def translate(geom: Geometry, x_offset: float, y_offset: float):
    """
    Moves a geometry by some distance

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    geom                Required class:`~arcgis.geometry.Geometry object or dictionary.
    ---------------     --------------------------------------------------------------------
    x_offset            Required int. Distance to move coordinates in the x-direction
    ---------------     --------------------------------------------------------------------
    y_offset            Required int. Distance to move coordinates in the y-direction
    ===============     ====================================================================

    :return:
        Dictionary or class:`~arcgis.geometry.Geometry` object
    """
    import numpy as np

    A = np.matrix([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
    if isinstance(geom, dict) or isinstance(geom, GEOM_TYPES):
        if "x" in geom and "y" in geom:  # translates point
            matrix = np.matrix([[geom["x"]], [geom["y"]], [1]])
            val = (A * matrix).tolist()
            geom["x"] = val[0][0]
            geom["y"] = val[1][0]
            return geom
        elif "rings" in geom or "paths" in geom:  # translates lines/polygons
            rings = []
            if "paths" in geom:
                coords = geom["paths"]
            else:
                coords = geom["rings"]
            for part in coords:
                x_coords = [pt[0] for pt in part]
                y_coords = [pt[1] for pt in part]
                matrix = np.matrix([x_coords, y_coords, [1] * len(y_coords)])
                vals = (A * matrix).tolist()
                rings.append(list(zip(vals[0], vals[1])))
            if "paths" in geom:
                geom["paths"] = rings
            else:
                geom["rings"] = rings
            return geom
        elif "points" in geom:  # translates Multipoint
            for pt in geom["points"]:
                matrix = np.matrix([[pt[0]], [pt[1]], [1]])
                val = (A * matrix).tolist()
                pt[0] = val[0][0]
                pt[1] = val[1][0]
            return geom
        else:
            return None


if __name__ == "__main__":
    shapes = [
        {
            "rings": [
                [
                    [-97.06138, 32.837],
                    [-97.06133, 32.836],
                    [-97.06124, 32.834],
                    [-97.06127, 32.832],
                    [-97.06138, 32.837],
                ],
                [
                    [-97.06326, 32.759],
                    [-97.06298, 32.755],
                    [-97.06153, 32.749],
                    [-97.06326, 32.759],
                ],
            ],
            "spatialReference": {"wkid": 4326},
        },
        {"x": 50, "y": 60},
        {
            "paths": [
                [
                    [-97.06138, 32.837],
                    [-97.06133, 32.836],
                    [-97.06124, 32.834],
                    [-97.06127, 32.832],
                ],
                [[-97.06326, 32.759], [-97.06298, 32.755]],
            ],
            "spatialReference": {"wkid": 4326},
        },
        {
            "points": [
                [-97.06138, 32.837],
                [-97.06133, 32.836],
                [-97.06124, 32.834],
                [-97.06127, 32.832],
            ],
            "spatialReference": {"wkid": 4326},
        },
    ]
    # translate Test
    print("translate test")
    for shape in shapes:
        # val_dict = translate(shape,-10, 10)
        val_geom = translate(Geometry(shape), -10, 10)
        print(val_geom)
        print()
    print("rotate shapes")
    for shape in shapes:
        val_geom = rotate(geom=Geometry(shape), theta=20)
        print(val_geom, val_geom.type)
        del shape
    print("scale factor shapes")
    for shape in shapes:
        val_geom = scale(Geometry(shape), 10)
        val_geom = scale(Geometry(shape), 10, 20)
    print("skew test")
    for shape in shapes:  # Skew test
        val_geom = skew(geom=Geometry(shape), x_angle=45, y_angle=-20)
    print()
