from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS
from typing import Dict, Any, Optional, List


class UpgradeManager(_BaseKube):
    _url = None
    _gis = None
    _con = None
    _properties = None

    def __init__(self, url: str, gis: GIS):
        self._gis = gis
        self._con = gis._con
        self._url = url

    @property
    def version(self) -> Dict[str, Any]:
        """
        Returns the current version for a deployment. When an patch or
        release is installed, this resource will update to reflect the
        information included in the update's version object as well as
        the job messages recorded during the upgrade process.
        """
        url = f"{self._url}/currentVersion"
        params = {
            "f": "json",
        }
        return self._con.get(url, params)

    @property
    def history(self) -> Dict[str, Any]:
        """
        Returns the transaction history for all upgrade and rollback jobs.

        :return: Dict[str, Any]
        """
        url = f"{self._url}/history"
        params = {
            "f": "json",
        }
        return self._con.get(url, params)

    @property
    def installed_updates(self) -> List[Dict[str, Any]]:
        """
        Returns a cumulative list of patches and releases that are installed in the deployment

        :return: List[Dict[str, Any]]
        """
        url = f"{self._url}/installed"
        params = {
            "f": "json",
        }
        return self._con.get(url, params).get("updates", [])

    @property
    def rollback_options(self) -> List[Dict[str, Any]]:
        """
        Returns a list of possible rollback options for the site, depending
        on the patch that is installed. The ID for the specific rollback
        version is passed as input for the `rollback` operation.

        :return: List[Dict[str, Any]]
        """
        url = f"{self._url}/checkRollback"
        params = {
            "f": "json",
        }
        return self._con.post(url, params).get("updates", [])

    def rollback(
        self,
        version: Dict[str, Any],
        settings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        This operation uninstalls a patch, removing the updates and fixes
        that had been applied to specific containers, and restoring the
        deployment to a previous, user-specified version of the software.
        The rollback operation cannot be performed for release-based
        updates.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        version                Required Dict[str, str]. The version of the deployment the operation
                               will rollback towards. This value can be retrieved from the
                               `rollback_options`.
        ------------------     --------------------------------------------------------------------
        settings               Optional Dict[str, str]. A configuration for patch settings.
                               This is only available at 10.9.1+.
        ==================     ====================================================================

        :return: Dict[str, Any]


        """
        url = f"{self._url}/rollback"
        params = {
            "f": "json",
            "versionManifest": version,
            "rollbackSettings": settings or {},
        }
        return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def upgrades(
        self,
        version_manifest: str,
        settings: dict,
        manifest_file: str | None = None,
    ):
        """
        The upgrade operation upgrades, through either a patch or a
        release, an ArcGIS Enterprise on Kubernetes deployment to the
        current version.

        Before performing an upgrade, the unique ID associated with the
        patch or release must be retrieved from the version manifest using
        the available operation. The version manifest is a JSON array of
        version objects that contain update-specific information, including
        a JSON array of container objects that specify affected containers
        and include their name, checksum, and image values.

        Once the ID has been retrieved, you must also retrieve the required
        upgrade settings that will be passed through as part of the upgrade
        operation. Some settings will require user input before they can be
        used during an upgrade. For more information about current upgrade
        settings, see the Upgrade settings section below.

        Once the upgrade job request has been submitted, the deployment
        will either install a new patch on the base version or upgrade the
        entire deployment to the latest release. While the job is running,
        the upgrades resource will return detailed, real-time job messages
        and status information. The upgrades resource's child operations
        and resources will remain inaccessible for the duration of the
        upgrade.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        version_manifest       Optional Dict[str, str]. The unique ID associated with a patch or
                               release. You can get the version manifest ID for a patch or release
                               from the JSON view of the available operation.
        ------------------     --------------------------------------------------------------------
        settings               Optional Dict[str, str]. A JSON object containing details for
                               release upgrade settings. These settings, retrieved from the
                               getUpgradeSettings operation, must be included in the request for
                               the upgrade to be successful. Currently, the object supports the
                               following three upgrade settings: updateToLatestPatch,
                               licenseUpload, and volumesConfig. These settings are applicable to
                               ArcGIS Enterprise on Kubernetes versions 11.0 and later.
        ------------------     --------------------------------------------------------------------
        manifest_file          Optional String. The file containing the version manifest.
        ==================     ====================================================================

        :return: Dict[str, Any]

        """
        url = f"{self._url}/upgrade"
        params = {
            "f": "json",
            "versionManifest": version_manifest,
            "upgradeSettings": settings,
        }
        files = {"licenseUpload": manifest_file}
        if manifest_file:
            return self._con.post(url, params, files=files)
        else:
            return self._con.post(url, params)

    # ---------------------------------------------------------------------
    def import_manifest(self, manifest: str) -> dict:
        """
        The importManifest operation allows organization administrators to
        import the version manifest into a disconnected environment that
        can be used to discover available updates and releases and upgrade
        an ArcGIS Enterprise on Kubernetes deployment. The version manifest
        must be downloaded from My Esri, which requires an initial internet
        connection.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        manifest               Required String. The file containing the version manifest (.dat
                               file), used to discover available updates or releases for an ArcGIS
                               Enterprise on Kubernetes deployment.
        ==================     ====================================================================

        :returns: dict
        """
        url = f"{self._url}/importManifest"
        params = {
            "f": "json",
        }
        files = {"manifestFile": manifest}
        return self._con.post_multipart(url, params, files=files)

    # ---------------------------------------------------------------------

    def upgrade_settings(self, upgrade_id: str) -> dict:
        """
        The getUpgradeSettings operation returns the required upgrade
        settings, and their expected formats, needed for a specific
        release, applicable to ArcGIS Enterprise on Kubernetes versions
        11.0 and later. These settings must be passed through as values for
        the upgradeSettings parameter to successfully upgrade an ArcGIS
        Enterprise on Kubernetes deployment. Some upgrade settings may
        require their value property to be modified before being submitted
        as part of the upgrade operation. For example, when upgrading an
        ArcGIS Enterprise on Kubernetes deployment from version 10.9.1 to
        11.0, you will need to modify the value property for the
        licenseUpload JSON object.
        """
        url = f"{self._url}/getUpgradeSettings"
        params = {
            "f": "json",
            "upgradeId": upgrade_id,
        }
        return self._con.post(url, params)

    def available(self) -> Dict[str, List]:
        """
        This operation returns the version manifest, a cumulative list of
        release and patch versions that have been made available to an
        ArcGIS Enterprise organization.

        :return: Dict[str, List]
        """
        url = f"{self._url}/available"
        params = {
            "f": "json",
        }
        return self._con.post(url, params)
