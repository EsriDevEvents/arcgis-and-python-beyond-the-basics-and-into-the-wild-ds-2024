import json
import tempfile

import arcgis


class LinearUnit(object):
    """
    A data object containing a linear distance, used as input to some Geoprocessing tools

    ================  ========================================================
    **Parameter**      **Description**
    ----------------  --------------------------------------------------------
    distance          required number, the value of the linear distance.

    ----------------  --------------------------------------------------------
    units             required string,  unit type of the linear distance,
                      such as "Meters", "Miles", "Kilometers", "Inches",
                      "Points", "Feet", "Yards", "NauticalMiles",
                      "Millimeters", "Centimeters", "DecimalDegrees",
                      "Decimeters"
    ================  ========================================================
    """

    def __init__(self, distance, units):
        self.distance = distance
        if units.startswith("esri"):
            self.units = units.title()
        else:
            self.units = "esri" + units.title()

    def to_dict(self):
        """Converts an instance of this class to its dict representation."""
        return {"distance": self.distance, "units": self.units}

    def __repr__(self):
        """returns object as string"""
        return json.dumps(self.to_dict())

    __str__ = __repr__

    @classmethod
    def from_dict(cls, datadict):
        """Creates an instance of this class from its dict representation."""
        distance = datadict.get("distance", None)
        units = datadict.get("units", None)

        return cls(distance, units)

    @classmethod
    def from_str(cls, how_far):
        """Creates a linear unit from a string like '5 miles'."""
        dist, units = how_far.split()
        return cls(float(dist), units.title())


class DataFile(object):
    """
    A data object containing a data source, used as input/output by some Geoprocessing tools

    ================  ========================================================
    **Parameter**      **Description**
    ----------------  --------------------------------------------------------
    url               optional string, URL to the location of the data file.

    ----------------  --------------------------------------------------------
    item_id           optional string,  The id of the uploaded file returned
                      as a result of the upload operation.

    ----------------  --------------------------------------------------------
    portal_item       optional :class:`~arcgis.gis.Item`. A data type item used for GP tool.
    ================  ========================================================
    """

    def __init__(self, url=None, item_id=None, portal_item=None):
        self.url = url
        self.item_id = item_id
        self.portal_item = portal_item

    def to_dict(self):
        """Converts an instance of this class to its dict representation."""
        datafile = {}
        if self.url is not None:
            datafile["url"] = self.url
        if self.item_id is not None:
            datafile["itemID"] = self.item_id
        if self.portal_item is not None and isinstance(self.portal_item, str):
            datafile["portalItemID"] = self.portal_item
        elif self.portal_item is not None and hasattr(self.portal_item, "itemid"):
            datafile["portalItemID"] = self.portal_item.itemid

        return datafile

    def __repr__(self):
        """returns object as string"""
        return json.dumps(self.to_dict())

    __str__ = __repr__

    @classmethod
    def from_dict(cls, datadict):
        """Creates an instance of this class from its dict representation."""
        url = datadict.get("url", None)
        item_id = datadict.get("item_id", None)

        return cls(url, item_id)

    @classmethod
    def from_str(cls, url):
        """Creates a data file from a url."""
        return cls(url, None)

    def download(self, save_path=None):
        """Downloads the data to the specified folder or a temporary folder if a folder isn't provided"""
        data_path = self.url
        if not save_path:
            save_path = tempfile.gettempdir()
        if data_path:
            gis = arcgis.env.active_gis
            if gis._con.product == "AGOL":
                return gis._con.get(
                    path=data_path,
                    out_folder=save_path,
                    try_json=False,
                    add_token=False,
                    token=gis._con.token,
                )
            else:
                return gis._con.get(
                    path=data_path,
                    out_folder=save_path,
                    try_json=False,
                    token=gis._con.token,
                )


class RasterData(object):
    """
    A data object containing a raster data source,
    used as input/output by some Geoprocessing tools

    ================  ========================================================
    **Parameter**      **Description**
    ----------------  --------------------------------------------------------
    url               optional string, URL to the location of the raster data
                      file.
    ----------------  --------------------------------------------------------
    item_id           optional string,  The id of the uploaded file returned
                      as a result of the upload operation.
    ----------------  --------------------------------------------------------
    format            optional string, Specifies the format of the raster
                      data, such as "jpg", "tif", etc.
    ================  ========================================================
    """

    def __init__(self, url=None, format=None, item_id=None):
        self.url = url
        self.format = format
        self.item_id = item_id

    def to_dict(self):
        """Converts an instance of this class to its dict representation."""
        rasterdata = {}
        if self.url is not None:
            rasterdata["url"] = self.url
        if self.item_id is not None:
            rasterdata["itemID"] = self.item_id
        if self.format is not None:
            rasterdata["format"] = self.format

        return rasterdata

    def __repr__(self):
        """returns object as string"""
        return json.dumps(self.to_dict())

    __str__ = __repr__

    @classmethod
    def from_dict(cls, datadict):
        """Creates an instance of this class from its dict representation."""
        url = datadict.get("url", None)
        item_id = datadict.get("item_id", None)
        format = datadict.get("format", None)

        return cls(url, format, item_id)
