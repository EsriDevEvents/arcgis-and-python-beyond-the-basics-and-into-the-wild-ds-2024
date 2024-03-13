from __future__ import annotations
from arcgis.gis._impl._con import Connection
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import Item
from arcgis.gis.server._service import Service
from arcgis._impl.backport import cached_property
from functools import lru_cache
from typing import Optional, Union


###########################################################################
class MissionJob(object):
    """
    Represents a Single `Job` operation for Mission Server
    """

    _properties = None
    _url = None
    _con = None
    _gis = None

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: "GIS", **kwargs) -> "MissionJob":
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    @cached_property
    def properties(self):
        """returns the properties of the resource"""

        if self._properties is None:
            try:
                self._properties = PropertyMap(self._con.get(self._url, {"f": "json"}))
            except:
                self._properties = PropertyMap(self._con.post(self._url, {"f": "json"}))
        return self._properties

    # ---------------------------------------------------------------------
    @property
    def status(self) -> object:
        """
        Returns the status

        :return: string
        """
        resp = self._con.get(self._url, {"f": "json"})
        try:
            return resp["status"]
        except:
            return "UNKNOWN"

    # ---------------------------------------------------------------------
    def result(self) -> "Mission":
        """Returns the results of the process"""
        if self.status.upper() == "COMPLETED":
            resp = self._con.get(self._url, {"f": "json"})
            if self.properties["type"] == "createNewMissionService":
                url = f"{self._url.split('/jobs/')[0]}/missions/{resp['customAttributes']['missionId']}"
                return Mission(url=url, gis=self._gis)
            else:
                return self.properties
        return None


###########################################################################
class Mission(object):
    """
    A single registered `Mission` on the Enterprise.
    """

    _properties = None
    _url = None
    _con = None
    _gis = None

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: "GIS", **kwargs) -> "Mission":
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ---------------------------------------------------------------------
    def _dictionary_check(self, i):
        """
        First prints the final entry in the dictionary (most nested) and its key
        Then prints the keys leading into this
        * could be reversed to be more useful, I guess
        """
        for key, value in i.items():
            if isinstance(value, dict):
                self._dictionary_check(value)
            else:
                if (
                    isinstance(value, str)
                    and value.lower().find("/featureserver/") > -1
                ):
                    i[key] = Service(url=value, server=self._gis)

    # ---------------------------------------------------------------------
    @cached_property
    def properties(self):
        """returns the properties of the resource"""
        if self._properties is None:
            try:
                r = self._con.get(self._url, {"f": "json"})
                self._dictionary_check(r)
                self._properties = r
            except:
                r = self._con.post(self._url, {"f": "json"})
                self._dictionary_check(r)
                self._properties = r
        return self._properties

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Deletes a :class:`~arcgis.gis.mission.api.Mission` from the server.

        :return: Boolean

        """
        params = {"f": "json"}
        url = f"{self._url}/delete"
        res = self._con.post(url, params)
        if res.get("status") or res.get("success"):
            return res.get("status") or res.get("success")
        return res

    # ---------------------------------------------------------------------
    def add_report(
        self,
        title: str,
        service_name: str,
        questions: list[dict],
        source: str = "user",
        description: Optional[str] = None,
        tags: Optional[Union[list, str]] = None,
        display_field: Optional[str] = None,
        drawing_info: Optional[dict] = None,
        locale: str = "en",
        share_as_template: bool = False,
    ) -> dict:
        """

        This method adds a report to the associated mission.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title	               Required String. The name of the report.
        ------------------     --------------------------------------------------------------------
        service_name           Required String. The desired service name for the report.
                               This value cannot be longer than 87 characters and cannot contain the following values:
                               #,$, %,^,*,=,`,{,},[,],,,{space},:,*, ?, ",<,>,|,\, /, +
        ------------------     --------------------------------------------------------------------
        questions              Required String. A string formatted as a JSON Array containing questions and their fields. If an empty array is passed, the request is rejected. Used to represent the desired questions and their fields. Question types are based on Survey123 question type fields.
                               Available question types: Single Line Text, Single Choice, Number,
                               Image, Multiline Text, Dropdown, Multiple Choice, and Date/Time.
                               See https://doc.arcgis.com/en/survey123/browser/create-surveys/quickreferencecreatesurveys.htm#GUID-2D96112F-85B1-4C41-9C6F-A85BB6026A51 for details.
        ------------------     --------------------------------------------------------------------
        description	           Optional String. A description of the report.
        ------------------     --------------------------------------------------------------------
        tags	               Optional. A comma-separated list of strings. Used to add tags to the report.
        ------------------     --------------------------------------------------------------------
        display_field          Optional String. The name of the report feature layer's display field. Normally this is the field name of one of the report questions. If absent, the report feature layer's display field will be the first question's field name.
        ------------------     --------------------------------------------------------------------
        drawing_info           Optional. JSON Object. Defines the report feature layer's drawing info, including a feature renderer.
                               See: https://developers.arcgis.com/documentation/common-data-types/drawinginfo.htm
        ------------------     --------------------------------------------------------------------
        locale                 Optional. String. The locale used to generate the report. Must be a valid IETF BCP 47 language tag. Defaults to en
        ------------------     --------------------------------------------------------------------
        share_as_template      Optional Boolean. Shares the report as a template.
        ==================     ====================================================================

        :return: Dict
        """
        params = {
            "title": title,
            "serviceName": service_name,
            "source": source,
            "questions": questions,
            "description": description or "",
            "tags": tags or "report",
            "displayField": display_field or "",
            "drawingInfo": drawing_info or "",
            "shareAsTemplate": share_as_template,
            "locale": locale,
            "f": "json",
        }
        url = f"{self._url}/addReport"
        return self._con.post(url, params)


###########################################################################
class MissionCatalog:
    """
    The ArcGIS Mission Server catalog.
    """

    _con = None
    _gis = None
    _url = None
    _admin = None
    _properties = None

    # ---------------------------------------------------------------------
    def __init__(self, gis: "GIS") -> "MissionCatalog":
        url = None
        urls = {"admin": None, "url": None}
        for rs in gis._registered_servers()["servers"]:
            if rs["serverType"] == "ARCGIS_MISSION_SERVER":
                url = f"{rs['url']}/rest"
                urls["url"] = url
                urls["admin"] = f"{rs['adminUrl']}/admin"

        if url is None:
            raise Exception("No registered mission server found.")

        self._url = url
        self._gis = gis
        self._con = gis._con
        self._urls = urls

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"< {self.__class__.__name__} @ {self._url} >"

    # ---------------------------------------------------------------------
    @cached_property
    def properties(self):
        """returns the properties of the resource"""

        if self._properties is None:
            try:
                self._properties = PropertyMap(self._con.get(self._url, {"f": "json"}))
            except:
                self._properties = PropertyMap(self._con.post(self._url, {"f": "json"}))
        return self._properties

    def admin(self):
        """provides a connection to the administrative API of Mission Server Produce"""
        if self._admin is None:
            from arcgis.gis.mission import MissionServer

            self._admin = MissionServer(url=self._urls["admin"], gis=self._gis)
        return self._admin

    # ---------------------------------------------------------------------
    def create_mission(
        self,
        title: str,
        snippet: Optional[str] = None,
        description: Optional[str] = None,
        license_info: Optional[str] = None,
        tags: Optional[Union[list, str]] = None,
        capabilities: Optional[Union[list, str]] = None,
        extent: Optional[str] = None,
        template_item: Optional[Item] = None,
        locale: str = "en",
        base_map: Optional[dict] = None,
        wm_description: Optional[str] = None,
        webmap_id: Optional[Union[str, Item]] = None,
        app_id: Optional[str] = None,
    ) -> MissionJob:
        """

        Creates a new `Mission` on the enterprise.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title	               Required. String. The title of the mission. This is the human readable title that is displayed to users.
        ------------------     --------------------------------------------------------------------
        snippet	               Optional String. A short summary description of the item.
        ------------------     --------------------------------------------------------------------
        description	           Optional String. Mission description.
        ------------------     --------------------------------------------------------------------
        capabilities	       Optional String. The comma-separated list of desired capabilites for the mission.
                               Valid values are CHAT, TASK, REPORT, PRESENCE.
        ------------------     --------------------------------------------------------------------
        license_info	       Optional String. Any license information or restrictions.
        ------------------     --------------------------------------------------------------------
        tags	               Optional. Comma-separated list of strings used to tag the mission.
                               Format: tag1,tag2,...,tagN
        ------------------     --------------------------------------------------------------------
        extent	               Optional String. Comma-separated list that defines the bounding
                               rectangle of the mission. Should always be in WGS84. The
                               default is -180, -90, 180, 90.

                               **Format: <xmin>, <ymin>, <xmax>, <ymax>**
        ------------------     --------------------------------------------------------------------
        webmap_id	       Optional String. The portal item id of the web map to use as a template for the mission.
        ------------------     --------------------------------------------------------------------
        locale	               Optional String. The locale in which to generate Mission assets with. Must be a valid IETF BCP 47 language tag. Defaults to en
        ------------------     --------------------------------------------------------------------
        base_map               Optional JSON Object. The basemap to add to the mission.
                               See: https://developers.arcgis.com/documentation/common-data-types/basemap.htm
        ------------------     --------------------------------------------------------------------
        wm_description         Optional string. The description of the web map added to the mission.
        ------------------     --------------------------------------------------------------------
        app_id                 Optional string. The ID of the app that created the mission.
        ==================     ====================================================================


        :return:
            :class:`~arcgis.gis.mission.api.MissionJob`


        """
        if isinstance(webmap_id, Item):
            webmap_id = webmap_id.itemid
        url = f"{self._url}/services/createMission"
        params = {
            "title": title,
            "snippet": snippet or "",
            "description": description or "",
            "licenseInfo": license_info or "",
            "capabilities": capabilities or "",
            "async": True,
            "tags": tags or "",
            "extent": extent or "-180,-90,180,90",
            "locale": locale or "en",
            "baseMap": base_map or "",
            "appId": app_id or "",
            "webMapDescription": wm_description,
            "templateWebMapId": webmap_id,
            "f": "json",
        }
        resp = self._con.post(url, params)
        return MissionJob(url=f"{self._url}/jobs/{resp.get('jobId')}", gis=self._gis)

    # ---------------------------------------------------------------------
    @property
    def jobs(self) -> list:
        """returns a list of :class:`~arcgis.gis.mission.api.MissionJob` on the server"""
        url = f"{self._url}/jobs"
        params = {"f": "json"}
        resp = self._con.get(url, params)
        return [MissionJob(url=f"{url}/{j}", gis=self._gis) for j in resp["asyncJobs"]]

    # ---------------------------------------------------------------------
    @property
    def missions(self) -> list:
        """
        returns a list of :class:`~arcgis.gis.mission.api.Mission` on the server

        :return: List
        """
        url = f"{self._url}/services"
        params = {"f": "json"}
        resp = self._con.get(url, params)
        return [
            Mission(url=f"{url}/{j['name']}/MissionServer", gis=self._gis)
            for j in resp["services"]
            if j["type"] == "MissionServer"
        ]
