import os
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


class SiteManager(object):
    """
    Provides the ability to update and restore notebook sites. An object of this
    class can be created using :attr:`~arcgis.gis.nb.NotebookServer.site` property of the
    :class:`~arcgis.gis.nb.NotebookServer` class

    """

    _nb = None
    _url = None
    _gis = None
    _con = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, notebook, gis):
        """Constructor"""
        self._url = url
        self._nb = notebook
        gis = notebook._gis
        if isinstance(gis, GIS):
            self._gis = gis
            self._con = self._gis._con
        else:
            raise ValueError("Invalid GIS object")

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            params = {"f": "json"}
            res = self._gis._con.get(self._url, params)
            self._properties = PropertyMap(res)
        except:
            self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< SiteManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< SiteManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def export_site(self, location: str):
        """
        ArcGIS Notebook Server provides this operation to back up the site's
        configuration store, along with the importSite operation to restore
        a site configuration from a backup. The configuration store hosts
        essential information about the ArcGIS Notebook Server site and its
        machines.

        The output of this operation is a ZIP file with the .agssite file
        extension.

        There are many items and directories that are not backed up by this
        operation. Among them:

        - Notebooks
        - Container settings
        - Jobs directory

        If desired, you can create your own file system backups for these items.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Required String. The folder to save the site to.
        ==================     ====================================================================

        :return: string

        """
        url = "{base}/exportSite".format(base=self._url)
        params = {"f": "json"}
        return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def import_site(self, location: str):
        """
        ArcGIS Notebook Server provides this operation to restore a site
        configuration from a backup. The backup will have been created and
        exported by the exportSite operation as a ZIP file with the
        .agssite file extension.

        Performing this operation will overwrite the current contents of
        your ArcGIS Notebook Server site's configuration store with the
        contents from the backup. You can use it to restore a site
        configuration in the event of machine failure or human error.

        The import operation may take a while to complete. When you execute
        the operation, keep the tab open on your browser until the
        operation completes, as a report will be delivered to the page.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        location               Required String. Path to the backup site file.
        ==================     ====================================================================

        :return: Boolean

        """
        url = "{base}/importSite".format(base=self._url)
        params = {"f": "json"}
        params["EXPORT_LOCATION"] = location
        res = self._con.post(url, params)
        if "status" in res:
            return res["status"]
        return res
