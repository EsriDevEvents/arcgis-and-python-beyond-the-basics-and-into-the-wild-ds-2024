class DaskSpatialIndex:
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
    indexes                  A collection of spatial index objects.
    ====================     ==================================================


    """

    _indexes = None

    # ----------------------------------------------------------------------
    def __init__(self, indexes: tuple):
        """initializer"""
        self._indexes = indexes

    def __str__(self):
        return "<Dask QuadTree Index>"

    # ----------------------------------------------------------------------
    def intersect(self, bbox: tuple) -> list:
        """
        Returns the spatial features that intersect the bbox

        :bbox: tuple - (xmin,ymin,xmax,ymax)

        :returns: list
        """
        res = []
        [res.extend(index.intersect(bbox)) for index in self._indexes]
        return res
