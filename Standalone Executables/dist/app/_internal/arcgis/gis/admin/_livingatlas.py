"""
class to work with the living atlas
"""
from typing import Optional
from .._impl._con import Connection
from ..._impl.common._mixins import PropertyMap
from ...gis import GIS
from ...gis import Group, GroupManager
from ._base import BasePortalAdmin


########################################################################
class LivingAtlas(BasePortalAdmin):
    """
    Living Atlas of the World content is a collection of authoritative,
    ready-to-use, global geographic content available from ArcGIS Online.
    The content includes valuable maps, data layers, tools, services and
    apps for geographic analysis.
    When you make Living Atlas content available to your portal members,
    you're providing them with ready-made content that they can use
    alone or in combination with their own content to create maps,
    scenes, and apps and perform analysis in the portal Map Viewer or
    Insights for ArcGIS.

    :Note:
       Your portal must have access to the Internet to use Living Atlas
       content from ArcGIS Online

    Types of content available
    All the Living Atlas content you access from Portal for ArcGIS was
    created by Esri. If your portal can connect to the Internet, the
    following three levels of Living Atlas content are available to you from
    ArcGIS Online:

    ================     ====================================================
    **Content Type**        **Description**
    ----------------     ----------------------------------------------------
    Default              Content that does not require you to sign in to an
                         ArcGIS Online account. Available by default in ArcGIS
                         Enterprise.
    ----------------     ----------------------------------------------------
    Subscriber           Subscriber content is the collection of ready-to-use
                         map layers, analytic tools, and services published
                         by Esri that requires an ArcGIS Online organizational
                         subscription account to access. This includes layers
                         from Esri such as Landsat 8 imagery, NAIP imagery,
                         landscape analysis layers, and historical maps.
                         Subscriber content is provided as part of your
                         organizational subscription and does not consume
                         any credits.
    ----------------     ----------------------------------------------------
    Premium              Premium content is a type of subscriber content that
                         requires an ArcGIS Online organizational subscription
                         account to access and consumes credits. Access and
                         credit information is listed in the  description details
                         for each item. Premium content provides portal members
                         with access to ready-to-use  content such as demographic
                         and lifestyle maps as well as tools for  geocoding,
                         geoenrichment, network analysis, elevation analysis, and
                         spatial analysis.
    ================     ====================================================

    See `Configure Living Atlas content: Types of Content Available <https://enterprise.arcgis.com/en/portal/latest/administer/windows/configure-living-atlas-content.htm#ESRI_SECTION1_7F44ACDF4DFE408A8430BD29C9DDFC67>`_
    for complete details.

    Portal administrators do not need to create this class directly in most
    circumstances. Instead, first access the :class:`PortalAdminManager<arcgis.gis.admin.PortalAdminManager>`
    using the `admin` property of the :class:`GIS<arcgis.gis.GIS>`. Then use
    the `living_atlas` property to return a :class:`LivingAtlas` object.

    .. code-block:: python

        ent_living_atlas = gis.admin.living_atlas

    To create an instance directly:

    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    url                 required string, the web address of the site to
                        manage licenses.
    ---------------     ----------------------------------------------------
    gis                 required :class:`GIS<arcgis.gis.GIS>` object.
    ===============     ====================================================

    .. code-block:: python

        ent_living_atlas = LivingAtlas(url="https://portal_url/web_adaptor/portaladmin/system/content/livingatlas"
                                       gis=gis)

    """

    _groupquery = None
    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None
    _living_atlas_group = None
    _living_atlas_content_group = None
    _groups = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""

        super(LivingAtlas, self).__init__(url=url, gis=gis)
        self._url = url.replace("http://", "https://")
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")

        self._init()

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        """initializer"""
        try:
            self._groupquery = self._gis.properties["livingAtlasGroupQuery"]
        except:
            self._groupquery = 'title:"Living Atlas" AND owner:esri_livingatlas'
        groups = self._gis.groups
        self._groups = []
        for group in groups.search(query=self._groupquery, outside_org=True):
            if group.title.lower() == "living atlas".lower():
                self._living_atlas_group = group
            elif group.title.lower() == "Living Atlas Analysis Layers".lower():
                self._living_atlas_content_group = group
            self._groups.append(group)
            del group
        del groups
        self._properties = PropertyMap({})

    # ----------------------------------------------------------------------
    def enable_public_access(self):
        """
        Enables the Public Living Atlas content.

        Living Atlas of the World content is a collection of authoritative,
        ready-to-use, global geographic content available from ArcGIS Online.
        The content includes valuable maps, data layers, tools, services and
        apps for geographic analysis.

        :return:
           Boolean. `True` if enabled. `False` if failed to enable.

        """
        url = self._url + "/share"
        results = []
        for g in self.groups:
            params = {"f": "json", "groupId": g.id, "type": "Public"}
            res = self._con.post(path=url, postdata=params)
            results.append(res["status"] == "success")
        return all(results)

    # ----------------------------------------------------------------------
    def disable_public_access(self):
        """
        Disables the Public Living Atlas content.

        :return:
           Boolean. True means disabled, False means failure to disable.

        """
        url = self._url + "/unshare"
        results = []
        for g in self.groups:
            params = {"f": "json", "groupId": g.id, "type": "Public"}
            res = self._con.post(path=url, postdata=params)
            results.append(res["status"] == "success")
        return all(results)

    # ----------------------------------------------------------------------
    def status(self, group: str):
        """
        Returns information about the sharing status of the Living
        Atlas with the group.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        group               required string or Group object
        ===============     ====================================================

        .. code-block:: python

            >>> ent_living_atlas = gis.admin.living_atlas

            >>> liv_atl_groups = ent_living_atlas.groups
            >>> liv_atl_groups

                [<Group title:"Living Atlas" owner:esri_livingatlas>,
                 <Group title:"Living Atlas Analysis Layers" owner:esri_livingatlas>]

            >>> liv_atl_group = liv_atl_groups[0]

            >>> living_atlas.status(liv_atl_group)

                 {'publicContentEnabled': True,
                  'subscriberContentEnabled': True,
                  'premiumContentEnabled': False,
                  'publicContentShared': True,
                  'subscriberContentShared': True,
                  'premiumContentShared': False,
                  'subscriberContentUsername': 'demos_deldev',
                  'subscriberUserValid': 'Valid',
                  'premiumContentUsername': None,
                  'premiumUserValid': 'UnKnown',
                  'upgraded': True}

        """
        url = "%s/status" % self._url
        params = {"f": "json"}
        if isinstance(group, str):
            params["groupId"] = group
        elif isinstance(group, Group):
            params["groupId"] = group.id
        return self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def upgrade(self):
        """
        Upgrades the Living Atlas Group to the latest version of the Living Atlas
        data. See `Living Atlas content life cycles and updates <https://enterprise.arcgis.com/en/portal/latest/use/living-atlas-content-life-cycles.htm>`_
        for details.

        :return: Boolean
        """
        url = "%s/upgrade"
        params = {"f": "json"}
        try:
            for g in self.groups:
                params["groupId"] = g.id
                self._con.post(url, params)
            return True
        except:
            return False

    # ----------------------------------------------------------------------
    def update_subscriber_account(self, username: str, password: str):
        """
        Updates the Username/Password for the Living Atlas Subscriber User.
        The account must be an ArcGIS Online account.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            Required string. The user who will be used for
                            to access the subscriber Living Atlas content.
        ---------------     ----------------------------------------------------
        password            Required string. The credentials for the user above.
        ===============     ====================================================

        :return: Boolean. True if successful else False.

        """
        url = "%s/update" % self._url
        r = []
        for g in self.groups:
            params = {
                "f": "json",
                "groupId": g.id,
                "type": "Premium",
                "username": username,
                "password": password,
            }
            res = self._con.post(url, params)
            if "success" in res:
                r.append(True)
            else:
                r.append(False)
        return all(r)

    # ----------------------------------------------------------------------
    def update_premium_account(self, username: str, password: str):
        """
        Updates the Username/Password for the Living Atlas Premium User.
        The account must be an ArcGIS Online account.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            Required string. The user who will be used for
                            to access the subscriber Living Atlas content.
        ---------------     ----------------------------------------------------
        password            Required string. The credentials for the user above.
        ===============     ====================================================

        :return: Boolean. True if successful else False.

        """
        url = "%s/update" % self._url
        r = []
        for g in self.groups:
            params = {
                "f": "json",
                "groupId": g.id,
                "type": "Premium",
                "username": username,
                "password": password,
            }
            res = self._con.post(url, params)
            if "success" in res:
                r.append(True)
            else:
                r.append(False)
        return all(r)

    # ----------------------------------------------------------------------
    @property
    def groups(self):
        """returns a list of all living atlas groups"""
        if not self._groups:
            self._init()
        return self._groups

    # ----------------------------------------------------------------------
    def validate_credentials(
        self, username: str, password: str, online_url: Optional[str] = None
    ):
        """
        Ensures the arguments contain valid credentials to access an active
        ArcGIS Online Organization.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, username for ArcGIS Online
        ---------------     ----------------------------------------------------
        password            required string, login password for ArcGIS Online account
        ---------------     ----------------------------------------------------
        online_url          optional string, Url to ArcGIS Online site.
                            default is https://www.arcgis.com
        ===============     ====================================================

        :return:
          Boolean. True if successful else False.

        """
        if online_url is None:
            online_url = "https://www.arcgis.com"
        url = "%s/validate" % self._url
        params = {
            "username": username,
            "password": password,
            "onlineUrl": online_url,
            "f": "json",
        }
        res = self._con.post(path=url, postdata=params)
        return res["status"] == "success"

    # ----------------------------------------------------------------------
    def enable_premium_atlas(self, username: str, password: str):
        """
        Enables the Premium Living Atlas Content for a local portal.

        Premium content is a type of subscriber content that requires an
        ArcGIS Online organizational subscription account to access and
        consumes credits. Access and credit information is listed in the
        description details for each item.
        Premium content provides portal members with access to ready-to-use
        content such as demographic and lifestyle maps as well as tools for
        geocoding, geoenrichment, network analysis, elevation analysis, and
        spatial analysis.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, username for ArcGIS Online
        ---------------     ----------------------------------------------------
        password            required string, login password for ArcGIS Online account
        ===============     ====================================================

        :Note:
          This will cost you credits.

        .. code-block:: python

            >>> ent_living_atlas = gis.admin.living_atlas

            >>> liv_atl_groups = ent_living_atlas.groups
            >>> liv_atl_groups

                [<Group title:"Living Atlas" owner:esri_livingatlas>,
                 <Group title:"Living Atlas Analysis Layers" owner:esri_livingatlas>]

            >>> liv_atl_group = liv_atl_groups[0]

            >>> living_atlas.status(liv_atl_group)

                 {'publicContentEnabled': True,
                  'subscriberContentEnabled': True,
                  'premiumContentEnabled': False,
                  'publicContentShared': True,
                  'subscriberContentShared': True,
                  'premiumContentShared': False,
                  'subscriberContentUsername': 'demos_deldev',
                  'subscriberUserValid': 'Valid',
                  'premiumContentUsername': None,
                  'premiumUserValid': 'UnKnown',
                  'upgraded': True}

            >>> living_atlas.enable_premium_atlas("org_admin",
                                                  "org_admin_password")

                   True

            >>> living_atlas.status(liv_atl_group)

                 {'publicContentEnabled': True,
                  'subscriberContentEnabled': True,
                  'premiumContentEnabled': True,
                  'publicContentShared': True,
                  'subscriberContentShared': True,
                  'premiumContentShared': True,
                  'subscriberContentUsername': 'demos_deldev',
                  'subscriberUserValid': 'Valid',
                  'premiumContentUsername': 'arcgispyapibot',
                  'premiumUserValid': 'InValid',
                  'upgraded': True}

        """
        group_id = None
        for g in self.groups:
            if g.title.lower() == "living atlas":
                group_id = g.id
                break
        params = {
            "f": "json",
            "username": username,
            "password": password,
            "type": "Premium",
            "groupId": group_id,
        }
        url = "%s/enable" % self._url
        res = self._con.post(path=url, postdata=params)
        if "status" in res and res["status"] == "success" and group_id:
            url = "%s/share" % self._url
            params = {"f": "json", "groupId": group_id, "type": "Premium"}
            res = self._con.post(path=url, postdata=params)
            return res["status"] == "success"
        else:
            return False
        return

    # ----------------------------------------------------------------------
    def enable_subscriber_atlas(self, username: str, password: str):
        """
        Enables the Subscriber level Living Atlas Content for an ArcGIS Enterprise portal.

        Subscriber content is the collection of ready-to-use map layers,
        analytic tools, and services published by Esri that requires an
        ArcGIS Online organizational subscription account to access.
        This includes layers from Esri such as Landsat 8 imagery,
        NAIP imagery, landscape analysis layers, and historical maps.
        Subscriber content is provided as part of your
        organizational subscription and does not consume any credits.
        Layers included in the Living Atlas subscriber content are suitable
        for use with analysis tools.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            required string, username for ArcGIS Online
        ---------------     ----------------------------------------------------
        password            required string, login password for the specific ArcGIS Online account
        ===============     ====================================================

        :Note:
          Use of these layers will **not** incur a credit cost for your organization.


        """
        group_ids = []
        for g in self.groups:
            group_ids.append(g.id)
        params = {
            "f": "json",
            "username": username,
            "password": password,
            "type": "Subscriber",
        }
        try:
            for ids in group_ids:
                params["groupId"] = ids
                url = "%s/enable" % self._url
                res = self._con.post(path=url, postdata=params)
            for ids in group_ids:
                url = "%s/share" % self._url
                params = {"f": "json", "groupId": ids, "type": "Subscriber"}
                res = self._con.post(path=url, postdata=params)
            return True
        except:
            return False

    # ----------------------------------------------------------------------
    def disable_subscriber_atlas(self):
        """
        Disables the Subscriber level Living Atlas Content for a local portal.

        """
        group_ids = []
        for g in self.groups:
            group_ids.append(g.id)
        params = {"f": "json", "type": "Subscriber"}
        try:
            for ids in group_ids:
                params["groupId"] = ids
                url = "%s/disable" % self._url
                res = self._con.post(path=url, postdata=params)
            for ids in group_ids:
                url = "%s/unshare" % self._url
                params = {"f": "json", "groupId": ids, "type": "Subscriber"}
                res = self._con.post(path=url, postdata=params)
            return True
        except:
            return False

    # ----------------------------------------------------------------------
    def disable_premium_atlas(self):
        """
        Disables the Premium Living Atlas Content for a local portal.

        .. code-block:: python

            >>> living_atlas = gis.admin.living_atlas

            >>> living_atlas.disable_premium_atlas()

                True

            >>> living_atlas.status(liv_atl_group)

                {'publicContentEnabled': True,
                 'subscriberContentEnabled': True,
                 'premiumContentEnabled': False,
                 'publicContentShared': True,
                 'subscriberContentShared': True,
                 'premiumContentShared': False,
                 'subscriberContentUsername': 'demos_deldev',
                 'subscriberUserValid': 'Valid',
                 'premiumContentUsername': None,
                 'premiumUserValid': 'UnKnown',
                 'upgraded': True}

        """
        group_id = None
        for g in self.groups:
            if g.title.lower() == "living atlas":
                group_id = g.id
                break
        params = {"f": "json", "groupId": group_id, "type": "Premium"}
        url = "%s/disable" % self._url
        res = self._con.post(path=url, postdata=params)
        if "status" in res and res["status"] == "success" and group_id:
            url = "%s/unshare" % self._url
            params = {"f": "json", "groupId": group_id, "type": "Premium"}
            res = self._con.post(path=url, postdata=params)
            return res["status"] == "success"
        else:
            return False
        return False
