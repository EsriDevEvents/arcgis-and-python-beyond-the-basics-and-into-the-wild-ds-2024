"""
Contains tools to manage a GIS' metadata properties
"""
from arcgis._impl.common._mixins import PropertyMap
from .. import GIS


########################################################################
class MetadataManager(object):
    """
    Provides Administrators an Easy value to enable, update and disable
    metadata settings on a Web GIS Site (Enterprise or ArcGIS Online)
    """

    _gis = None
    _portal = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Constructor"""
        self._gis = gis
        self._portal = gis._portal
        self._con = gis._portal.con

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s for %s >" % (type(self).__name__, self._gis._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s at %s >" % (type(self).__name__, self._gis._url)

    # ----------------------------------------------------------------------
    def enable(self, metadata_format: str = "arcgis"):
        """
        This operation turns on metadata for items and allows the
        administrator to set the default metadata scheme.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        metadata_format                 Required string. Sets the default metadata format. The allowed
                                        values are: inspire,iso19139-3.2,fgdc,iso19139,arcgis, or iso19115
        ===========================     ====================================================================

        :return: boolean

        """
        lookup = {
            "fgdc": "fgdc",
            "inspire": "inspire",
            "iso19139": "iso19139",
            "iso19139-3.2": "iso19139-3.2",
            "iso19115": "iso19115",
            "arcgis": "arcgis",
        }
        if not metadata_format.lower() in lookup.keys():
            raise ValueError("Invalid metadata_format")
        params = {
            "metadataEditable": True,
            "metadataFormats": lookup[metadata_format.lower()],
        }
        return self._gis.update_properties(properties_dict=params)

    # ----------------------------------------------------------------------
    def disable(self):
        """
        This operation turns off metadata for items.

        :return: boolean
        """

        params = {"metadataEditable": False, "metadataFormats": ""}
        return self._gis.update_properties(properties_dict=params)

    # ----------------------------------------------------------------------
    def update(self, metadata_format: str = "arcgis"):
        """
        This operation allows administrators to update the current metdata
        properties.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        metadata_format                 Required string. Sets the default metadata format. The allowed
                                        values are: inspire,iso19139-3.2,fgdc,iso19139,arcgis, or iso19115
        ===========================     ====================================================================

        :return: boolean

        """
        lookup = {
            "fgdc": "fgdc",
            "inspire": "inspire",
            "iso19139": "iso19139",
            "iso19139-3.2": "iso19139-3.2",
            "iso19115": "iso19115",
            "arcgis": "arcgis",
        }
        if not metadata_format.lower() in lookup.keys():
            raise ValueError("Invalid metadata_format")
        params = {
            "metadataEditable": True,
            "metadataFormats": lookup[metadata_format.lower()],
        }
        return self._gis.update_properties(properties_dict=params)

    # ----------------------------------------------------------------------
    @property
    def is_enabled(self):
        """returns boolean to show if metadata is enable on a GIS"""
        try:
            return self._gis.properties.metadataEditable
        except:
            return False
