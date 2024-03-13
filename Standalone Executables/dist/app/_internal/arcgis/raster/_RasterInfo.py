import json


class RasterInfo(object):
    """
    The ``RasterInfo`` class allows for creation of a  ``RasterInfo`` object that describes a set of raster properties to
    facilitate the creation of local raster dataset using the :class:`~arcgis.raster.Raster` class

    .. note::
        The ``RasterInfo`` class requires ArcPy

    A ``RasterInfo`` object can be created by instantiating it from a dictionary,
    or by calling an :class:`~arcgis.raster.ImageryLayer` or :class:`~arcgis.raster.Raster` object's
    ``raster_info`` property.

    Information about the raster can also be set through the following properties available on the ``RasterInfo``
    object: ``band_count``, ``extent``, ``pixel_size_x``, ``pixel_size_y``, ``pixel_type``, ``block_height``,
    ``block_width``, ``no_data_values``, ``spatial_reference``

    To construct a ``RasterInfo`` object from a dictionary, use the ``from_dict`` method on this class.

    .. code-block:: python

        # Usage Example 1: This example creates a new Raster object from the raster_info of another Raster object. (requires arcpy)
        raster_obj = Raster(<raster dataset path>)
        ras_info = RasterInfo(raster_obj.raster_info)
        rinfo_based_ras = Raster(rasInfo2)

        #To write pixel values to this temporary Raster object:
        rinfo_based_ras.write(<numpy_array>)

        #To save this temporary raster locally:
        rinfo_based_ras.save(r"C:\data\persisted_raster.crf")

    RasterInfo object can also be used in raster functions that take in raster info as a parameter. (does not require arcpy)
    example: As value to the raster_info parameter for :meth:`arcgis.raster.functions.constant_raster` and :meth:`arcgis.raster.functions.random_raster`

    """

    def __init__(self, raster_info_dict=None):
        self._band_count = None
        self._extent = None
        self._pixel_size_x = None
        self._pixel_size_y = None
        self._pixel_type = None
        self._block_height = None
        self._block_width = None
        self._spatial_reference = None
        self._no_data_values = None
        self._dict = raster_info_dict

    def __repr__(self):
        """returns object as string"""
        return json.dumps(self.to_dict())

    __str__ = __repr__

    @property
    def band_count(self):
        """
        Get/Set information about the band count of a raster.
        """
        return self._band_count

    @band_count.setter
    def band_count(self, value):
        self._band_count = value

    @property
    def extent(self):
        """
        Get/Set information about the extent of a raster.
        """
        return self._extent

    @extent.setter
    def extent(self, value):
        self._extent = value

    @property
    def pixel_type(self):
        """
        Get/Set information about the pixel type of a raster.
        """
        return self._pixel_type

    @pixel_type.setter
    def pixel_type(self, value):
        self._pixel_type = value

    @property
    def pixel_size_x(self):
        """
        Get/Set information about the pixel size of a raster in
        x direction.
        """
        return self._pixel_size_x

    @pixel_size_x.setter
    def pixel_size_x(self, value):
        self._pixel_size_x = value

    @property
    def pixel_size_y(self):
        """
        Get/Set information about the pixel size of a raster in
        the y direction.
        """
        return self._pixel_size_y

    @pixel_size_y.setter
    def pixel_size_y(self, value):
        self._pixel_size_y = value

    @property
    def block_height(self):
        """
        Get/Set information about the block height of a raster.
        """
        return self._block_height

    @block_height.setter
    def block_height(self, value):
        self._block_height = value

    @property
    def block_width(self):
        """
        Get/Set information about the block width of a raster.
        """
        return self._block_width

    @block_width.setter
    def block_width(self, value):
        self._block_width = value

    @property
    def no_data_values(self):
        """
        Get/Set information about the ``no_data_values`` of a
        raster.
        """
        return self._no_data_values

    @no_data_values.setter
    def no_data_values(self, value):
        self._no_data_values = value

    @property
    def spatial_reference(self):
        """
        The ``spatial_reference`` property retrieves information about the spatial reference of a
        raster.
        """
        return self._spatial_reference

    @extent.setter
    def spatial_reference(self, value):
        self._spatial_reference = value

    def to_dict(self):
        """
        The ``to_dict`` method is used to return Raster Info in dictionary format.
        """
        # rinfo_dict = self.__dict__
        new_rinfo_dict = {}
        if self._dict is None:
            if self.band_count is not None:
                new_rinfo_dict.update({"bandCount": self.band_count})
            if self.extent is not None:
                new_rinfo_dict.update({"extent": self.extent})
            if self.pixel_size_x is not None:
                new_rinfo_dict.update({"pixelSizeX": self.pixel_size_x})
            if self.pixel_size_y is not None:
                new_rinfo_dict.update({"pixelSizeY": self.pixel_size_y})
            if self.pixel_type is not None:
                new_rinfo_dict.update({"pixelType": self.pixel_type})
            if self.block_height is not None:
                new_rinfo_dict.update({"blockHeight": self.block_height})
            if self.block_width is not None:
                new_rinfo_dict.update({"blockWidth": self.block_width})
            if self.spatial_reference is not None:
                new_rinfo_dict.update({"spatialReference": self.spatial_reference})
            if self.no_data_values is not None:
                new_rinfo_dict.update({"noDataValues": self.no_data_values})
        else:
            new_rinfo_dict = self._dict
        return new_rinfo_dict

    def from_dict(self, raster_info_dict):
        """
        The ``from_dict`` method can be used to initialise a :class:`~arcgis.raster.RasterInfo` object from a raster info
        dictionary.

        .. code-block:: python

            # Usage Example :
            rinfo = RasterInfo()
            rinfo.from_dict({'bandCount': 3,
                             'extent': {"xmin": 4488761.95,
                                         "ymin": 5478609.805,
                                         "xmax": 4489727.05,
                                         "ymax": 5479555.305,
                                         "spatialReference": {
                                           "wkt": "PROJCS[\"Deutsches_Hauptdreiecksnetz_Transverse_Mercator\",
                                           GEOGCS[\"GCS_Deutsches_Hauptdreiecksnetz\",DATUM[\"D_Deutsches_Hauptdreiecksnetz\",
                                           SPHEROID[\"Bessel_1841\",6377397.155,299.1528128]],PRIMEM[\"Greenwich\",0.0],
                                           UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],
                                           PARAMETER[\"false_easting\",4500000.0],PARAMETER[\"false_northing\",0.0],
                                           PARAMETER[\"central_meridian\",12.0],PARAMETER[\"scale_factor\",1.0],
                                           PARAMETER[\"latitude_of_origin\",0.0],UNIT[\"Meter\",1.0]]"
                                         }},
                             'pixelSizeX': 0.0999999999999614,
                             'pixelSizeY': 0.1,
                             'pixelType': 'U8'})
        """
        if raster_info_dict is not None and isinstance(raster_info_dict, dict):
            if "extent" in raster_info_dict:
                self.extent = raster_info_dict["extent"]

            if "bandCount" in raster_info_dict:
                self.band_count = raster_info_dict["bandCount"]

            if "pixelType" in raster_info_dict:
                self.pixel_type = raster_info_dict["pixelType"]

            if "pixelSizeX" in raster_info_dict:
                self.pixel_size_x = raster_info_dict["pixelSizeX"]

            if "pixelSizeY" in raster_info_dict:
                self.pixel_size_y = raster_info_dict["pixelSizeY"]

            self._dict = raster_info_dict
