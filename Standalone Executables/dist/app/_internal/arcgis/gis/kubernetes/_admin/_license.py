from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS


###########################################################################
class LicenseManager(_BaseKube):
    """The license manager for the Kubernetes deployment."""

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    def load(self, license_file: str, overwrite: bool) -> bool:
        """
        `load` imports and applies an ArcGIS Server authorization file. By
        default, this operation will append authorizations from the
        imported license file to the current authorizations. Optionally,
        you can select the overwrite option to fully replace the current
        authorizations with those from the imported license file.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        license_file           Required string. The ArcGIS Server authorization file (either in
                               .epc or .prvc file format).
        ------------------     --------------------------------------------------------------------
        overwrite              Required bool. Specifies whether the authorizations in the imported
                               license file will fully replace or be appended to the current
                               authorizations. If true, the authorizations from the imported
                               license file will replace the current authorizations. If false, the
                               authorizations from the imported license file will be appended to
                               the current authorizations.
        ==================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        url = f"{self._url}/validateLicense"
        params = {
            "f": "json",
            "overwrite": overwrite,
        }
        files = file = {"licenseFile": license_file}
        res = self._con.post(url, params, files=file)
        return res.get("status", "failed") == "success"
