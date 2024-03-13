"""
Wrapper for implementing Spatial Indexing for DataFrames
"""
from .quadtree import Index as QIndex

try:
    from rtree.index import Index as RIndex

    HASRTREE = True
except:
    HASRTREE = False
    RIndex = None


class SpatialIndex:
    """

    A spatial index is a type of extended index that allows you to index a
    spatial column. A spatial column is a table column that contains data of a
    spatial data type.

    Spatial indexes help to improve spatial query performance on a dataframe.
    Identifying a feature, selecting features, and joining data all have better
    performace when using spatial indexing.


    ====================     ==================================================
    Arguement                Description
    --------------------     --------------------------------------------------
    stype                    Required String. This sets the type of spatial
                             index being used by the user. The current types of
                             spatial indexes are: custom, rtree and quadtree.
    --------------------     --------------------------------------------------
    bbox                     Optional Tuple. The extent of the spatial data as:
                             (xmin, ymin, xmax, ymax). This parameter is required
                             if a QuadTree Spatial Index is being used.

                             Example:
                             bbox=(-100, -50, 100, 50)
    --------------------     --------------------------------------------------
    filename                 Optional String. The name of the spatial index
                             file. This is only supported by rtree spatial
                             indexes. For large datasets an rtree index can be
                             saved to disk and used at a later time. If this is
                             not provided the r-tree index will be in-memory.
    --------------------     --------------------------------------------------
    custom_index             Optional Object. Sometimes QuadTree and Rtree
                             indexing is not enough. A custom spatial index class
                             can be giving to the SpatialIndex class and used
                             using encapsulation.  The custom index must have two
                             methods: `intersect` that excepts a tuple, and
                             `insert` which must accept an oid and a bounding
                             box. This object is required when `stype` of
                             'custom' is specified.
    ====================     ==================================================


    """

    _stype = None
    _bbox = None
    _index = None
    _df = None

    # ----------------------------------------------------------------------
    def __init__(self, stype, bbox=None, **kwargs):
        """initializer"""
        ci = kwargs.pop("custom_index", None)
        self._filename = kwargs.pop("filename", None)
        self._bbox = bbox
        self._stype = stype.lower()
        self._df = None
        if ci and stype.lower() == "custom":
            self._index = ci
        elif stype.lower() == "quadtree" and bbox:
            self._index = QIndex(bbox=bbox)
        elif RIndex and stype.lower() == "rtree":
            self._index = RIndex(self._filename)
        else:
            raise ValueError("Could not create the spatial index.")

    # ----------------------------------------------------------------------
    def intersect(self, bbox):
        """
        Returns the spatial features that intersect the bbox

        :bbox: tuple - (xmin,ymin,xmax,ymax)

        :return: List of the intersecting features
        """
        if self._stype.lower() in ["rtree"]:
            return list(self._index.intersection(bbox))
        elif self._stype.lower() in ["quadtree"]:
            return list(self._index.intersect(bbox=bbox))
        else:
            return list(self._index.intersect(bbox))

    # ----------------------------------------------------------------------
    def insert(self, oid, bbox):
        """
        Inserts the entry into the spatial index

        :oid: unique id
        :bbox: tuple - (xmin,ymin,xmax,ymax)
        """
        if self._index is None:
            raise Exception(
                ("Could not insert into a spatial index because " "it does not exist.")
            )
        if self._stype == "rtree" and HASRTREE and isinstance(self._index, RIndex):
            r = self._index.insert(id=oid, coordinates=bbox, obj=None)
            self.flush()
            return r
        elif self._stype.lower() == "quadtree":
            return self._index.insert(item=oid, bbox=bbox)
        elif self._stype.lower() == "custom":
            r = self._index.intersect(oid, bbox)
            self.flush()
            return r

    # ----------------------------------------------------------------------
    def flush(self):
        """
        Saves the index to disk if a filename is given for an R-Tree Spatial Index.

        **This applies only to the R-Tree implementation of the spatial index.**

        :return: Boolean

        """
        if hasattr(self._index, "flush"):
            getattr(self._index, "flush")()
        elif self._stype == "rtree" and self._filename:
            self._index.close()
            self._index = RIndex(self._filename)
        else:
            return False
        return True
