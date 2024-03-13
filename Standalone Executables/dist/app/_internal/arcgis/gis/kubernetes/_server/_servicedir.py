import logging
from arcgis.gis import GIS

from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis.server._service import Service
from typing import Optional

_log = logging.getLogger()


class KubeServiceDirectory(_BaseKube):
    """
    A representation of the Kubernetes Hosting Service Directory.

    This is a private method and should not be created by a user.
    """

    _con = None
    _gis = None
    _url = None
    _folders = None
    _folder = None
    _services = None

    def __init__(self, url: str, gis: GIS) -> None:
        """initializer"""
        super()
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def report(self, as_html: bool = True, folder: Optional[str] = None):
        """
        Generates a table of Services in the given folder, as a Pandas dataframe.


        """
        import pandas as pd

        pd.set_option("display.max_colwidth", None)
        data = []
        a_template = """<a href="%s?token=%s">URL Link</a>"""
        columns = ["Service Name", "Service URL"]
        if folder is None:
            url = self._url
            res = self._con.get(url, {"f": "json"})
        elif folder.lower() in [f.lower() for f in self.folders]:
            url = "%s/%s" % (self._url, folder)
            res = self._con.get(url, {"f": "json"})
        if "services" in res:
            for s in res["services"]:
                # if s['name'].split('/')[-1].lower() == name.lower():
                url = "%s/%s/%s" % (url, s["name"], s["type"])
                data.append(
                    [
                        s["name"].split("/")[-1],
                        """<a href="%s">Service</a>""" % url,
                    ]
                )

        df = pd.DataFrame(data=data, columns=columns)
        if as_html:
            table = (
                """<div class="9item_container" style="height: auto; overflow: hidden; """
                + """border: 1px solid #cfcfcf; border-radius: 2px; background: #f6fafa; """
                + """line-height: 1.21429em; padding: 10px;">%s</div>"""
                % df.to_html(escape=False, index=False)
            )
            return table.replace("\n", "")
        else:
            return df

    # ----------------------------------------------------------------------
    def get(self, name: str, folder: Optional[str] = None):
        """returns a single service in a folder"""
        if folder is None:
            url = self._url
            res = self._con.get(self._url, {"f": "json"})
        elif folder.lower() in [f.lower() for f in self.folders]:
            url = "%s/%s" % (self._url, folder)
            res = self._con.get("%s/%s" % (self._url, folder), {"f": "json"})
        if "services" in res:
            for s in res["services"]:
                if s["name"].split("/")[-1].lower() == name.lower():
                    return Service(
                        url="%s/%s/%s" % (url, s["name"], s["type"]),
                        server=self._con,
                    )
                del s
        return None

    # ----------------------------------------------------------------------
    def list(self, folder: Optional[str] = None):
        """
        returns a list of services at the given folder
        """
        services = []
        if folder:
            url = "%s/%s" % (self._url, folder)
        else:
            url = self._url
        if folder is None:
            res = self._con.get(url, {"f": "json"})
        elif folder.lower() in [f.lower() for f in self.folders]:
            res = self._con.get(url, {"f": "json"})
        if "services" in res:
            for s in res["services"]:
                try:
                    services.append(
                        Service(
                            url="%s/%s/%s" % (url, s["name"], s["type"]),
                            server=self._con,
                        )
                    )

                except:
                    url = "%s/%s/%s" % (url, s["name"], s["type"])
                    _log.warning("Could not load service: %s" % url)
        return services

    # ----------------------------------------------------------------------
    def find(self, service_name: str, folder: Optional[str] = None):
        """
        finds a service based on it's name in a given folder
        """
        return self.get(name=service_name, folder=folder)

    # ----------------------------------------------------------------------
    @property
    def folders(self):
        """
        returns a list of server folders
        """
        self._properties = None
        if self._is_agol:
            return ["/"]
        else:
            return self.properties["folders"]
        return []

    # ----------------------------------------------------------------------
    def publish_sd(
        self,
        sd_file: str,
        folder: Optional[str] = None,
        service_config: Optional[dict] = None,
    ):
        """
        Publishes a service definition file to ArcGIS Server.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        sd_file                Required string. The service definition file to be uploaded and published.
        ------------------     --------------------------------------------------------------------
        folder                 Optional string. The folder in which to publish the service definition
                               file to.  If this folder is not present, it will be created.  The
                               default is None in which case the service definition will be published
                               to the System folder.
        ------------------     --------------------------------------------------------------------
        service_config         Optional Dict[str, Any]. A set of configuration overwrites that overrides the service definitions defaults.
        ==================     ====================================================================

        :return:
           A boolean indicating success (True) or failure (False).
        """
        import json

        sm = self._gis.admin.services
        catalog = self._gis.admin.services_catalog
        uploads = self._gis.admin.uploads
        if sd_file.lower().endswith(".sd") == False:
            return False
        # catalog = self.content
        if "System" not in catalog.folders:
            return False
        if folder and folder.lower() not in [f.lower() for f in catalog.folders]:
            sm.create_folder(folder)
        service = catalog.get(name="PublishingTools", folder="System")
        if service is None:
            service = catalog.get(name="PublishingToolsEx", folder="System")
        if service is None:
            return False
        status, res = uploads.upload(path=sd_file, description="sd file")
        if status:
            uid = res["item"]["itemID"]
            config = uploads._service_configuration(uid).get("service", {})
            if service_config or folder:
                if service_config:
                    config.update(service_config)
                if folder:
                    if "folderName" in config:
                        config["folderName"] = folder
                res = service.publish_service_definition(
                    in_sdp_id=uid, in_config_overwrite=json.dumps(config)
                )
            else:
                uid = res["item"]["itemID"]
                res = service.publish_service_definition(in_sdp_id=uid)
            return True
        return False
