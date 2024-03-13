from __future__ import annotations

import concurrent.futures
import uuid
import tempfile
from enum import Enum
import os
import json
from typing import Any
from arcgis._impl.common._deprecate import deprecated
from arcgis.auth.tools import LazyLoader
from arcgis.gis import Group, User
from arcgis.gis.clone._ux import UXCloner

_basemap_definitions = LazyLoader("arcgis.mapping._basemap_definitions")
_arcgis_gis = LazyLoader("arcgis.gis")


class StockImage(Enum):
    RIVERVIOLET = "1"
    RIVEREMERALD = "2"
    GEOLOGYAMBER = "3"
    GEOLOGYLILAC = "4"
    SOILGOLD = "5"
    SOILSILVER = "6"


###########################################################################
class UX(object):
    """Helper class for modifying common org settings. This class is not created by users directly. An instance of
    the class, called 'ux', is available as a property of the GIS object. Users call methods on this 'ux' object
    to set informational banner, background, logo, name etc. There are also other helper classes to call from this.
    By calling the 'org_map_editor' or 'homepage_editor' more methods can be found to change org settings specific
    to those categories."""

    _cloner: UXCloner | None = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
        self._portal_resources = gis.admin.resources

        # Determine if using old or new homepage
        if "homePage" in gis.properties["portalProperties"]:
            self._new_hp = (
                True
                if gis.properties["portalProperties"]["homePage"] == "modernOnly"
                or gis.properties["portalProperties"]["homePage"] == "modern"
                else False
            )
        else:
            self._new_hp = False

    # ----------------------------------------------------------------------
    def clone(
        self,
        *,
        targets: list[_arcgis_gis.GIS | None] | None = None,
        workspace_folder: str | None = None,
        package_save_folder: str | None = None,
        package_name: str | None = None,
    ) -> list[concurrent.futures.Future]:
        """
        Copies the UX settings from the source WebGIS site to the destination WebGIS site or to
        a `.uxpk` offline file.
        When directly connected to two WebGIS', this method performs the clone operation immediately.
        When cloning in an offline situation, a `.uxpk` is created and stored on the user's local
        hard drive.

        ====================  ===============================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------------
        targets               list[GIS | None]. The sites to clone to. If None is given, then
                              a local file is returned.
        --------------------  ---------------------------------------------------------------
        workspace_folder      Optional String. The workspace where the temporary files are
                              processed.
        --------------------  ---------------------------------------------------------------
        package_save_folder   Optional String. The output folder where the offline package is
                              saved.
        --------------------  ---------------------------------------------------------------
        package_name          Optional String. The saved package name minus the extension.
        ====================  ===============================================================

        :returns: list[concurrent.futures.Future]

        """
        if self._cloner is None:
            self._cloner = UXCloner(gis=self._gis)
        return self._cloner.clone(
            targets=targets,
            workspace_folder=workspace_folder,
            package_save_folder=package_save_folder,
            package_name=package_name,
        )

    # ----------------------------------------------------------------------
    def load_offline_configuration(self, package: str) -> concurrent.futures.Future:
        """
        Loads the UX configuration file into the current active portal.
        """
        if self._cloner is None:
            self._cloner = UXCloner(gis=self._gis)
        return self._cloner.load_offline_configuration(package)

    # ----------------------------------------------------------------------
    @property
    def name(self):
        """
        Get/Set the site's name.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        name              required string. Name of the site.
        ================  ===============================================================

        :return: string of the name of the site

        """
        return self._gis.properties["name"]

    # ----------------------------------------------------------------------
    @name.setter
    def name(self, name: str):
        """
        See main ``name`` property docstring
        """
        import json

        if self._gis.properties.name != name:
            rps = [dict(r) for r in self._gis.properties.rotatorPanels]
            for r in rps:
                r["innerHTML"] = r["innerHTML"].replace(self._gis.properties.name, name)

            res = self._gis.update_properties(
                {"name": name, "rotatorPanels": json.dumps(rps)}
            )
            params = {
                "key": "localizedOrgProperties",
                "text": json.dumps({"default": {"name": name, "description": None}}),
                "f": "json",
            }
            url = f"{self._gis._portal.resturl}portals/self/addResource"
            res = self._gis._con.post(url, params)
            return res

    # ----------------------------------------------------------------------
    @property
    def summary(self):
        """
        Allows the get/setting of a brief summary to describe your organization on the sign in page
        associated with its custom apps and sites. This summary has a maximum of 310 characters.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        text              Required string. The brief description of the organization.
        ================  ===============================================================

        :return: string
        """
        try:
            with open(
                self._portal_resources.get("localizedOrgProperties"), "r"
            ) as reader:
                res = json.loads(reader.read())

        except:
            # if summary has never been set for org then need to create the resource
            self.summary = ""
            return self.summary
        return res["default"]["description"]

    # ----------------------------------------------------------------------

    @summary.setter
    def summary(self, text: str):
        """
        See main ``summary`` property docstring
        """
        if text == "":
            text = None
        params = {
            "key": "localizedOrgProperties",
            "text": {"default": {"name": self.name, "description": text}},
            "f": "json",
        }
        self._portal_resources.add(
            key="localizedOrgProperties", text=json.dumps(params["text"])
        )

    # ----------------------------------------------------------------------
    def set_org_language(self, language: str, format: str | None = None):
        """
        Choose the default language for members of your organization. This
        choice affects the user interface as well as the way time,
        date, and numerical values appear. Individual members can customize
        this choice on their settings page.

        ================        ========================================================
        **Parameter**            **Description**
        ----------------        --------------------------------------------------------
        language                Required string. To see all available languages, use
                                the `languages` property in the GIS class.
        ----------------        --------------------------------------------------------
        format                  Optional string. Determine the culture format to be
                                used depending on the language. To see the culture formats
                                available, use the `languages` property in the GIS class
                                and look at the 'cultureFormats' key for each language.
        ================        ========================================================

        :return: True | False
        """
        languages = self._gis.languages
        for lng in languages:
            if lng["language"].lower() == language.lower():
                culture = lng["culture"]
                if "cultureFormats" in lng:
                    if format:
                        for clt_format in lng["cultureFormats"]:
                            if clt_format["name"].lower() == format.lower():
                                culture_format = clt_format["format"]
                    else:
                        culture_format = lng["cultureFormats"][0]["format"]
                    # culture name includes format if available
                    culture = culture + "-" + culture_format
                    break
                else:
                    # default set when no other choices
                    culture_format = "en"
        return self._gis.update_properties(
            {
                "culture": culture,
                "cultureFormat": culture_format,
                "clearEmptyFields": True,
            }
        )

    # ----------------------------------------------------------------------
    @property
    def contact_link(self):
        """
        Get and set the contact link for the site.
        """
        props: dict = self._gis._get_properties()
        if "links" in props["portalProperties"]:
            return props["portalProperties"]["links"]
        else:
            return None

    # ----------------------------------------------------------------------
    @contact_link.setter
    def contact_link(self, url: str):
        portal_properties = self._gis.properties["portalProperties"]
        if url:
            portal_properties["links"] = {
                "contactUs": {"url": f"{url}", "visible": True}
            }
        else:
            portal_properties["links"] = {"contactUs": {"url": "", "visible": False}}

        self._gis.update_properties({"portalProperties": portal_properties})

    # ----------------------------------------------------------------------
    @property
    def admin_contacts(self):
        """
        An array of chosen administrators listed as points of contact whose
        email addresses will be listed as points of contact in the automatic
        email notifications sent to org members when they request password resets,
        help with their user names, modifications to their accounts, or any issues
        related to the allocation of credits to their accounts.
        """
        return self._gis.properties["contacts"]

    # ----------------------------------------------------------------------
    @admin_contacts.setter
    def admin_contacts(self, users: list[User, str]):
        admins = []
        if users is None:
            raise ValueError(
                "Cannot set empty list as Administrative contacts. You must have at least one administrator in the list."
            )
        for user in users:
            if isinstance(user, User):
                user = user.username
            role = self._gis.users.search(user)[0].role
            if role == "org_admin":
                admins.append(user)
        if len(admins) == 0:
            raise ValueError(
                "None of the usernames provided are org admins. Please provide org admins."
            )
        else:
            self._gis.update_properties({"contacts": admins})

    # ----------------------------------------------------------------------
    @property
    def help_source(self):
        """
        Toggle if the help source is turned on (True) or off (False).
        It provides the base URL for your organization's help documentation.
        """
        if "helpBase" in self._gis.properties:
            return self._gis.properties["helpBase"]
        else:
            return None

    # ----------------------------------------------------------------------
    @help_source.setter
    def help_source(self, enabled: bool):
        if enabled is False:
            # this will reset it to default based on language
            self._gis.update_properties({"helpBase": ""})
        else:
            if self._gis._is_agol:
                raise ValueError(
                    "This parameter can only be set for Enterprise 10.8.1+."
                )
            else:
                culture = self._gis.properties["culture"]
                if culture not in [
                    "ar",
                    "pt-BR",
                    "fr",
                    "de",
                    "it",
                    "ja",
                    "ko",
                    "pl",
                    "ru",
                    "zh-CN",
                    "es-es",
                ]:
                    culture = "en"
                self._gis.update_properties(
                    {"helpBase": "https://enterprise.arcgis.com/{culture}"}
                )

    # ----------------------------------------------------------------------
    def set_logo(self, logo_file: str | None = None, show_logo: bool | None = None):
        """
        Configure your home page by setting the organization's logo image. For best results the logo file should be
        65 x 65 pixels in dimension.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-general.htm

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        logo_file           Optional string. Specify path to image file. If None, existing thumbnail is removed.
        ----------------    ---------------------------------------------------------------
        show_logo           Optional bool. Specify whether the logo is visible on the homepage or not.
        ================    ===============================================================

        :return: True | False
        """
        return self.homepage_settings.set_logo(logo_file=logo_file, show_logo=show_logo)

    # ----------------------------------------------------------------------
    def get_logo(self, download_path: str):
        """
        Get your organization's logo/thumbnail. You can use the `set_logo()` method to set an image as your logo.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the logo file.
        ================  ===============================================================

        :return: Path to downloaded logo file. If None, then logo is not set and nothing was downloaded.

        """
        return self.homepage_settings.get_logo(download_path=download_path)

    # ----------------------------------------------------------------------
    @property
    def description_visibility(self):
        """
        Get/Set the site's description visibility

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        visiblity         Required boolean. If True, the desciptive text will show on the
                          home page. If False, the descriptive text will not be displayed
        ================  ===============================================================

        :return: boolean or error

        """
        try:
            return self._gis.properties["showHomePageDescription"]
        except:
            return "This property no longer exists on your org"

    # ----------------------------------------------------------------------
    @description_visibility.setter
    def description_visibility(self, visiblity: bool):
        """
        See main ``description_visibility`` property docstring
        """
        return self._gis.update_properties({"showHomePageDescription": visiblity})

    # ----------------------------------------------------------------------
    @property
    def description(self):
        """
        Get/Set the site's description.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        description       Required string. Descriptive text of the site. If None, the
                          value is reset to default.
        ================  ===============================================================

        :return: dictionary
        """
        return self._gis.properties["description"]

    # ----------------------------------------------------------------------
    @description.setter
    def description(self, description: str | None = None):
        """
        See main ``description`` property docstring
        """
        if description is None:
            description = "<br/>"
        return self._gis.update_properties({"description": description})

    # ----------------------------------------------------------------------
    @property
    def featured_content(self) -> dict:
        """
        Gets/Sets the featured content group information.

        If you set the featured content, reinstantiate to update the gis properties and see the updated
        list of featured_content.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        content           Required list or dictionary, defines the group(s) of the feature
                          content area on an organizational site.  A value of None will
                          reset the value back to no featured groups.

                          To add a new group to the list you must pass in the list of current
                          featured content and include the new group in the list. "id" key must be in
                          the dictionary, extra keys are ok.

                          It is also acceptable to pass a list of Group class instances.

                          Example:
                          [{"id": <group_id>}, {"id": <group_id>}, etc.]
        ================  ===============================================================

        :return: dictionary


        .. code-block:: python
            *Usage Example*
            >>> data = ux.featured_content
            >>> ux.featured_content = data
            True

        """
        return self._gis.properties["featuredGroups"]

    # ----------------------------------------------------------------------
    @featured_content.setter
    def featured_content(self, contents: dict):
        """
        See main ``featured_content`` property docstring
        """
        from .. import Group

        featured_groups = []
        if isinstance(contents, list):
            for content in contents:
                if isinstance(content, Group):
                    group_id = content.id
                    group_title = content.title
                    group_owner = content.owner
                    featured_groups.append(
                        {
                            "id": group_id,
                            "title": group_title,
                            "owner": group_owner,
                        }
                    )
                elif (
                    isinstance(contents, dict)
                    and "id" in contents
                    and isinstance(contents["id"], str)
                ) or isinstance(contents, str):
                    if isinstance(contents, dict):
                        contents = contents["id"]
                    group = Group(self._gis, contents)
                    group_id = group.id
                    group_title = group.title
                    group_owner = group.owner
                    featured_groups.append(
                        {
                            "id": group_id,
                            "title": group_title,
                            "owner": group_owner,
                        }
                    )
        elif (
            isinstance(contents, dict)
            and "id" in contents
            and isinstance(contents["id"], str)
        ) or isinstance(contents, str):
            if isinstance(contents, dict):
                contents = contents["id"]
            group = Group(self._gis, contents)
            group_id = group.id
            group_title = group.title
            group_owner = group.owner
            featured_groups.append(
                {"id": group_id, "title": group_title, "owner": group_owner}
            )

        self._gis.update_properties({"featuredGroups": featured_groups})

    # ----------------------------------------------------------------------
    def navigation_bar(
        self,
        gallery: str | None = None,
        map: str | None = None,
        scene: str | None = None,
        groups: str | None = None,
        search: str | None = None,
    ):
        """
        Set the visibility of the content in the navigation bar. To get the current navigation
        bar settings do not pass in any values for the parameters.

        .. note::
            The Home link is always visible to everyone. The Content link is always visible to members.
            Member roles determine Organization link visibility.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        gallery             Optional string.
                            Values: "all" | "members" | "noOne"
        ----------------    ---------------------------------------------------------------
        map                 Optional string.
                            Values: "all" | "members" | "mapCreators"
        ----------------    ---------------------------------------------------------------
        scene               Optional string.
                            Values: "all" | "members" | "sceneCreators"
        ----------------    ---------------------------------------------------------------
        groups              Optional string.
                            Values: "all" | "members"
        ----------------    ---------------------------------------------------------------
        search              Optional string.
                            Values: "all" | "members"
        ================    ===============================================================

        :return: Dictionary of the navigation bar and it's settings.
        """
        portal_properties = self._gis.properties["portalProperties"]
        top_nav = {
            "gallery": "all",
            "map": "all",
            "scene": "all",
            "groups": "all",
            "search": "all",
        }
        if "topNav" in portal_properties:
            # get existing top nav settings
            top_nav = portal_properties["topNav"]

        # only change what is necessary
        if gallery:
            top_nav["gallery"] = gallery
        if map:
            top_nav["map"] = map
        if scene:
            top_nav["scene"] = scene
        if groups:
            top_nav["groups"] = groups
        if search:
            top_nav["search"] = search

        portal_properties["topNav"] = top_nav
        self._gis.update_properties({"portalProperties": portal_properties})
        return top_nav

    # ----------------------------------------------------------------------
    def shared_theme(
        self,
        header: dict[str:str] | None = None,
        button: dict[str:str] | None = None,
        body: dict[str:str] | None = None,
        logo: str | None = None,
    ):
        """
        Use the shared theme to apply your organization's brand colors and
        logo to information products created from ArcGIS Configurable Apps templates,
        Web AppBuilder, and Enterprise Sites. To see the current settings, call the method with
        no parameters passed in.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        header              Optional dict. Composed of two keys: "background" and "text" that
                            determine the shared theme color for each of these keys. Color
                            can be passed in a hexadecimal string.

                            ex: header = {"background" : "#0d7bba", "text" : "#000000"}
        ----------------    ---------------------------------------------------------------
        button              Optional dict. Composed of two keys: "background" and "text" that
                            determine the shared theme color for each of these keys.
        ----------------    ---------------------------------------------------------------
        body                Optional dict. Composed of three keys: "background", "text" and
                            "link" that determine the shared theme color for each of these keys.
        ----------------    ---------------------------------------------------------------
        logo                Optional str. The file path or link to the image that will be uploaded
                            as the shared theme logo.
                            To remove the logo and not replace it then pass in: ""
        ================    ===============================================================

        :return: Dictionary of the shared theme that is set on the org.
        """
        portal_properties = self._gis.properties["portalProperties"]
        shared_theme = {
            "header": {"background": "no-color", "text": "no-color"},
            "button": {"background": "no-color", "text": "no-color"},
            "body": {
                "background": "no-color",
                "text": "no-color",
                "link": "no-color",
            },
            "logo": {"small": ""},
        }
        if "sharedTheme" in portal_properties:
            shared_theme = portal_properties["sharedTheme"]
        if header:
            if "background" in header:
                shared_theme["header"]["background"] = header["background"]
            if "text" in header:
                shared_theme["header"]["text"] = header["text"]
        if button:
            if "background" in button:
                shared_theme["button"]["background"] = button["background"]
            if "text" in button:
                shared_theme["button"]["text"] = button["text"]
        if body:
            if "background" in body:
                shared_theme["body"]["background"] = body["background"]
            if "text" in body:
                shared_theme["body"]["text"] = body["text"]
            if "link" in body:
                shared_theme["body"]["link"] = body["link"]
            # find image extension
        if logo is not None and os.path.isfile(logo):
            # add item
            item_props = {
                "title": "Shared Theme Logo",
                "description": "This image was uploaded for use as your organizations shared theme logo.",
                "tags": ["SharedTheme", "Logo"],
                "type": "Image",
            }
            im_item = self._gis.content.add(item_props, logo)
            # share to everyone
            im_item.share(everyone=True)
            # set in shared_theme dict
            shared_theme["logo"]["small"] = im_item.homepage + "/data"
        elif logo == "":
            shared_theme["logo"] = {"small": "", "link": ""}
        elif logo is not None:
            # case where logo is a url link
            shared_theme["logo"]["link"] = logo

        portal_properties["sharedTheme"] = shared_theme
        self._gis.update_properties(
            {"portalProperties": portal_properties, "clearEmptyFields": True}
        )
        return shared_theme

    # ----------------------------------------------------------------------
    @property
    def gallery_group(self):
        """
        The gallery highlights your organization's content.
        Choose a group whose content will be shown in the gallery.
        To change the group, assign either an instance of Group or the group id.
        Setting to None will revert to default.

        :return: An instance of Group if a group is set, else the default or None
        """
        group = self._gis.properties["featuredItemsGroupQuery"]
        if "id:" in group:
            # must use [3::] to slice string since format of: "id:123abc"
            gallery_grps = self._gis.groups.search(group[3::])
            if len(gallery_grps) > 0:
                return gallery_grps[0]
            else:
                return group
        else:
            return group

    # ----------------------------------------------------------------------
    @gallery_group.setter
    def gallery_group(self, group: Group | str | None):
        if isinstance(group, Group):
            group = "id:" + group.id
        elif isinstance(group, str):
            res = self._gis.groups.search(group)
            if len(res) == 0:
                raise ValueError(
                    "The group id provided could not be found in your org."
                )
            else:
                group = "id:" + group
        self._gis.update_properties(
            {"featuredItemsGroupQuery": group, "clearEmptyFields": True}
        )

    # ----------------------------------------------------------------------
    @property
    def homepage_settings(self):
        """
        Get an instance of the :class:`~arcgis.gis.admin.HomePageSettings` class
        to make edits to the organization's  homepage such as the background,
        title, logo, etc.
        """
        return HomePageSettings(gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def map_settings(self):
        """
        Get an instance of the :class:`~arcgis.gis.admin.MapSettings` class to
        make edits to the org's default map settings such as extent, basemap, etc.
        """
        return MapSettings(gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def item_settings(self):
        """
        Get an instance of the :class:`~arcgis.gis.admin.ItemSettings` class to make edits to the org's default
        map settings such as comments, metadata, etc.

        """
        return ItemSettings(gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    def security_settings(self):
        """
        Get an instance of the :class:`~arcgis.gis.admin.SecuritySettings` class
        to make edits to the organization's default map settings such as
        the informational banner, password policy, etc.
        """
        return SecuritySettings(gis=self._gis)

    # ----------------------------------------------------------------------
    @property
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def enable_comments(self):
        """
        Get/Set item commenting and comments.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            Optional boolean. If True, the comments for the site are turned
                          on.  False will disable comments (default)
        ================  ===============================================================

        :return: True if enabled, False if disabled
        """
        return self.item_settings.enable_comments

    # ----------------------------------------------------------------------
    @enable_comments.setter
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def enable_comments(self, enable: bool = False):
        """
        See main ``enable_comments`` property docstring.
        """
        self.item_settings.enable_comments = enable

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def set_background(
        self, background_file: str | None = None, is_built_in: bool = True
    ):
        """
        Configure your home page by setting the organization's background image. You can choose no image, a built-in image
        or upload your own. If you upload your own image, the image is positioned at the top and center of the page.
        The image repeats horizontally if it is smaller than the browser or device window. For best results, if you want
        a single, nonrepeating background image, the image should be 1,920 pixels wide (or smaller if your users are on
        smaller screens). The website does not resize the image. You can upload a file up to 1 MB in size.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        background_file     Optional string. If using a custom background, specify path to image file.
                            To remove an existing background, specify None for this argument and
                            False for is_built_in argument.
        ----------------    ---------------------------------------------------------------
        is_built_in         Optional bool, default=True. The built-in background is set by default.
                            If uploading a custom image, this parameter is ignored.
        ================    ===============================================================

        :return: True | False
        """
        return self.homepage_settings.set_background(
            background_file=background_file, is_built_in=is_built_in
        )

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def get_background(self, download_path: str):
        """
        Get your organization's home page background image. You can use the `set_background()` method to set an image
        as the home page background image.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the background file.
        ================  ===============================================================

        :return: Path to downloaded background file. If None, then background is not set and nothing was downloaded.
        """
        return self.homepage_settings.get_background(download_path=download_path)

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def set_banner(
        self,
        banner_file: str | None = None,
        is_built_in: bool = False,
        custom_html: str | None = None,
    ):
        """
        Configure your home page by setting the organization's banner. You can choose one of the 5 built-in banners or
        upload your own. For best results the dimensions of the banner image should be 960 x 180 pixels. You can also
        specify a custom html for how the banner space should appear. For more information, refer to
        http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        .. note::
            This has now been replaced by the `set_informational_banner` method

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        banner_file         Optional string. If uploading a custom banner, then path to the
                            banner file. If using a built-in banner, valid values are:

                            * banner-1
                            * banner-2
                            * banner-3
                            * banner-4
                            * banner-5

                            .. note::
                                If `None`, existing banner is removed.
        ----------------    ---------------------------------------------------------------
        is_built_in         Optional bool, default=False. Specify True if using a built-in
                            banner file.
        ----------------    ---------------------------------------------------------------
        custom_html         Optional string. Specify exactly how the banner should appear in
                            html. For help on this, refer to
                            http://server.arcgis.com/en/portal/latest/administer/windows/supported-html.htm
        ================    ===============================================================

        :return: True | False
        """
        # region check if banner has to be removed
        if not banner_file and not custom_html:
            # remove code

            # find existing banner resource file
            resource_list = self._portal_resources.list()
            e_banner = [
                banner for banner in resource_list if banner["key"].startswith("banner")
            ]

            # loop through and remove existing banner resource file
            for banner in e_banner:
                try:
                    self._portal_resources.delete(banner["key"])
                except:
                    continue

            # reset the home page - recurse
            return self.set_banner("banner-2", True)
        # endregion

        # region: Set banner using banner file - built-in or new image
        if banner_file:
            rotator_panel = []
            if not is_built_in:  # adding a new image file
                # find image extension
                from pathlib import Path

                fpath = Path(banner_file)
                f_splits = fpath.name.split(".")
                if len(f_splits) > 1 and f_splits[1] == "png":
                    key_val = "banner.png"
                elif len(f_splits) > 1 and f_splits[1] == "jpg":
                    key_val = "banner.jpg"
                else:
                    raise RuntimeError("Invalid image extension")

                add_result = self._portal_resources.add(key_val, banner_file)

                if add_result and custom_html:
                    rotator_panel = [{"id": "banner-custom", "innerHTML": custom_html}]

                elif add_result and not custom_html:
                    # set rotator_panel_text
                    rotator_panel = [
                        {
                            "id": "banner-custom",
                            "innerHTML": "<img src='{}/portals/self/resources/{}?token=SECURITY_TOKEN' "
                            "style='-webkit-border-radius:0 0 10px 10px; -moz-border-radius:0 0 10px 10px;"
                            " -o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px; margin-top:0; "
                            "width:960px;'/>".format(self._portal.con.baseurl, key_val),
                        }
                    ]
            else:  # using built-in image
                if not custom_html:  # if no custom html is specified for built-in image
                    rotator_panel = [
                        {
                            "id": banner_file,
                            "innerHTML": "<img src='images/{}.jpg' "
                            "style='-webkit-border-radius:0 0 10px 10px; -moz-border-radius:0 0 10px 10px; "
                            "-o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px; margin-top:0; "
                            "width:960px; height:180px;'/><div style='position:absolute; bottom:80px; "
                            "left:80px; max-height:65px; width:660px; margin:0;'>"
                            "<img src='{}/portals/self/resources/thumbnail.png?token=SECURITY_TOKEN' "
                            "class='esriFloatLeading esriTrailingMargin025' style='margin-bottom:0; "
                            "max-height:100px;'/><span style='position:absolute; bottom:0; margin-bottom:0; "
                            "line-height:normal; font-family:HelveticaNeue,Verdana; font-weight:600; "
                            "font-size:32px; color:#369;'>{}</span></div>".format(
                                banner_file,
                                self._portal.con.baseurl,
                                self._gis.properties.name,
                            ),
                        }
                    ]
                else:  # using custom html for built-in image
                    rotator_panel = [{"id": banner_file, "innerHTML": custom_html}]
        # endregion

        # region: Set banner just using a html text
        elif custom_html:
            rotator_panel = [{"id": "banner-html", "innerHTML": custom_html}]
        # endregion

        # Update the portal self with these banner values
        update_result = self._gis.update_properties({"rotatorPanels": rotator_panel})
        return update_result

    # ----------------------------------------------------------------------
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def get_banner(self, download_path: str):
        """
        Get your organization's home page banner image. You can use the `set_banner()` method to set an image or custom HTML
        code as your banner.

        .. note::
            This method has been replaced with the `get_informational_banner` method.

        ================    =================================================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------------------------
        download_path       required string. Folder path to download the banner file.
        ================    =================================================================================

        :return: Path to downloaded banner file. If None, then banner is not set and nothing was downloaded.

        """
        # create a portal resource manager obj

        # find existing banner resource file
        resource_list = self._portal_resources.list()
        e_banner = [
            banner for banner in resource_list if banner["key"].startswith("banner")
        ]

        # loop through and remove existing banner resource file
        banner_path = None
        for banner in e_banner:
            try:
                banner_path = self._portal_resources.get(banner["key"], download_path)

            except:
                continue
        return banner_path

    # ----------------------------------------------------------------------
    @property
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def default_extent(self):
        """
        Get/Set the site's default extent

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        extent            Required dictionary. The default extent defines where a webmap
                          will open.
                          If a value of None is given, the default extent will be provided.
                          Example Extent (default):
                          {"type":"extent","xmin":-17999999.999994524,"ymin":-11999999.999991827,
                          "xmax":17999999.999994524,"ymax":15999999.999982955,
                          "spatialReference":{"wkid":102100}}
        ================  ===============================================================

        :return: dictionary

        """
        return self.map_settings.default_extent

    # ----------------------------------------------------------------------
    @default_extent.setter
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def default_extent(self, extent: dict):
        """
        See main ``default_extent`` property docstring
        """
        self.map_settings.default_extent = extent

    # ----------------------------------------------------------------------
    @property
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def default_basemap(self):
        """
        Get/Set the site's default basemap.

        The Default Basemap opens when users click New Map. Set the group
        in the Basemap Gallery above and choose the map to open. It will
        open at the default extent you set.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           Required string. The new default basemap to set. If None, the
                          default value will be set.
        ================  ===============================================================

        :return: dictionary

        """
        return self.map_settings.default_basemap

    # ----------------------------------------------------------------------
    @default_basemap.setter
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def default_basemap(self, value: str):
        """
        See main ``default_basemap`` property docstring
        """
        self.map_settings.default_basemap = value

    # ----------------------------------------------------------------------
    @property
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def vector_basemap(self):
        """
        Get/Set the default vector basemap

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           required dictionary. The new default vector basemap to set for
                          a given site.
        ================  ===============================================================

        :return: The current default vector basemap
        """
        return self.map_settings.vector_basemap

    # ----------------------------------------------------------------------
    @vector_basemap.setter
    @deprecated(deprecated_in="2.1.0", removed_in="3.0.0", current_version="2.2.0")
    def vector_basemap(self, basemap: dict):
        """
        See main ``vector_basemap`` property docstring
        """
        self.map_settings.vector_basemap = basemap


#############################################################################
class HomePageSettings(object):
    """
    Helper class called from the UX class property: 'homepage_settings'. Make edits to background,
    title, logo, etc.
    """

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
        self._portal_resources = gis.admin.resources

        # Determine if using old or new homepage
        if "homePage" in gis.properties["portalProperties"]:
            self._new_hp = (
                True
                if gis.properties["portalProperties"]["homePage"] == "modernOnly"
                or gis.properties["portalProperties"]["homePage"] == "modern"
                else False
            )
        else:
            self._new_hp = False

    def _reader_hp(self) -> dict[str, Any]:
        """reads the homepage settings as a dictionary."""
        with open(self._portal_resources.get("home.page.json"), "r") as reader:
            return json.loads(reader.read())
        return {}

    # ----------------------------------------------------------------------
    def set_background(
        self,
        background_file: str | None = None,
        is_built_in: bool = True,
        stock_image: StockImage | None = None,
        layout: str | None = None,
        opacity: int | None = None,
    ):
        """
        Configure your home page by setting the organization's background image. You can choose no image, a built-in image
        or upload your own. If you upload your own image, the image is positioned at the top and center of the page.
        The image repeats horizontally if it is smaller than the browser or device window. For best results, if you want
        a single, nonrepeating background image, the image should be 1,920 pixels wide (or smaller if your users are on
        smaller screens). The website does not resize the image. You can upload a file up to 1 MB in size.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        background_file     Optional string. If using a custom background, specify path to image file.
                            To remove an existing background, specify None for this argument and
                            False for is_built_in argument.
        ----------------    ---------------------------------------------------------------
        is_built_in         Optional bool, default=True. The built-in background is set by default.
                            If uploading a custom image, this parameter is ignored.
        ----------------    ---------------------------------------------------------------
        stock_image         Optional instance of StockImage class or string that represents
                            a stock image key.
        ----------------    ---------------------------------------------------------------
        layout              Optional string. The layout height of the cover image. The values are:
                            "Full-height", "Two-thirds-height", or "Half-height".
        ----------------    ---------------------------------------------------------------
        opacity             Optional int that represent the opacity of the header. The value is
                            anything from 0 to 1 included.
        ================    ===============================================================

        :return: True | False
        """
        from pathlib import Path

        if self._new_hp is False:
            # Add resource if using a custom background file.
            background_update_val = None
            if background_file:
                # find image extension
                fpath = Path(background_file)
                f_splits = fpath.name.split(".")
                if len(f_splits) > 1 and f_splits[1] == "png":
                    key_val = "background.png"
                elif len(f_splits) > 1 and f_splits[1] == "jpg":
                    key_val = "background.jpg"
                else:
                    raise RuntimeError("Invalid image extension")

                add_result = self._portal_resources.add(key_val, background_file)
                if not add_result:
                    raise RuntimeError(
                        "Error adding background image as a resource file"
                    )
                background_update_val = key_val

            elif is_built_in:  # using built-in
                background_update_val = "images/arcgis_background.jpg"
            else:
                background_update_val = "none"

            # Update the portal self with these banner values
            return self._gis.update_properties(
                {"backgroundImage": background_update_val}
            )
        elif self._new_hp:
            if background_file:
                fpath = Path(background_file)
                f_splits = fpath.name.split(".")
                if len(f_splits) > 1 and f_splits[1] == "png":
                    key_val = "background.png"
                elif len(f_splits) > 1 and f_splits[1] == "jpg":
                    key_val = "background.jpg"
                else:
                    raise RuntimeError("Invalid image extension")

                add_result = self._portal_resources.add(key_val, background_file)
                if not add_result:
                    raise RuntimeError(
                        "Error adding background image as a resource file"
                    )
                background_update_val = key_val
                cover_type = "custom"
                cover_image_stock = ""
            elif is_built_in and stock_image is not None:
                cover_type = "stock"
                background_update_val = ""
                if isinstance(stock_image, str):
                    # User passed in string of the stock image key
                    stock_image = StockImage[stock_image]
                cover_image_stock = stock_image.value

            hp = self._reader_hp()
            hp["header"]["coverImg"] = background_update_val
            hp["header"]["coverType"] = cover_type
            hp["header"]["coverImgStock"] = cover_image_stock
            if layout and layout in [
                "Full-height",
                "Two-thirds-height",
                "Half-height",
            ]:
                hp["header"]["coverImgLayout"] = layout
            if opacity is not None and (opacity >= 0 or opacity <= 1):
                hp["header"]["opacity"] = opacity

            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )

    # ----------------------------------------------------------------------
    def get_background(self, download_path: str | None = None):
        """
        Get your organization's home page background image. You can use the `set_background()` method to set an image
        as the home page background image.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-home.htm

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     Required string. Folder path to download the background file.
                          If the background image is a stock image then an instance of the
                          StockImage class will be returned and the download_path will be ignored.
                          If you know the background is a stock image then don't provide a download path.
        ================  ===============================================================

        :return: Path to downloaded background file. If the cover image is a stock image then the stock image's name from the StockImage class will be returned.
        """

        # create a portal resource manager obj
        # find existing banner resource file
        bckgrnd_path = None
        if self._new_hp:
            # Need to read the homepage settings
            hp = json.loads(
                open(
                    self._portal_resources.get("home.page.json"),
                    "r",
                ).read()
            )
            # Have to check the cover type first
            cover_type = hp["header"]["coverType"]
            if cover_type == "stock":
                return StockImage(hp["header"]["coverImgStock"]).name
            else:
                if download_path is None:
                    raise ValueError(
                        "A download path is needed since the background is not a stock image."
                    )
                # If not stock then image is set
                resource_list = self._portal_resources.list()
                e_background = [
                    banner
                    for banner in resource_list
                    if banner["key"].startswith("background")
                ]
                for background in e_background:
                    try:
                        bckgrnd_path = self._portal_resources.get(
                            background["key"], download_path
                        )

                    except:
                        if self._new_hp:
                            # see if named something else
                            hp = json.loads(
                                open(
                                    self._portal_resources.get("home.page.json"),
                                    "r",
                                ).read()
                            )
                            background = hp["header"]["coverImg"]
                            if background:
                                bckgrnd_path = self._portal_resources.get(
                                    background["key"], download_path
                                )
                        else:
                            continue
        return bckgrnd_path

    # ----------------------------------------------------------------------
    def set_logo(
        self,
        logo_file: str | None = None,
        show_logo: bool | None = None,
        alignment: str | None = None,
        position: str | None = None,
    ):
        """
        Configure your home page by setting the organization's logo image. For best results the logo file should be
        65 x 65 pixels in dimension.

        For more information, refer to http://server.arcgis.com/en/portal/latest/administer/windows/configure-general.htm

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        logo_file           Optional string. Specify path to image file. If None, existing thumbnail is removed.
        ----------------    ---------------------------------------------------------------
        show_logo           Optional bool. Specify whether the logo is visible on the homepage or not.
        ----------------    ---------------------------------------------------------------
        alignment           Optional string. Values can either be "center" or "left". This will set
                            the alignment for the title and logo on the header.
        ----------------    ---------------------------------------------------------------
        position            Optional string. Set the title and logo position on the homepage.
                            Values: "middle" | "above" | "below" | "top3rd" | "bottom3rd"
        ================    ===============================================================

        :return: True | False
        """

        # Add resource file

        from pathlib import Path

        key_val = ""
        # find image extension
        if logo_file is not None and os.path.isfile(logo_file):
            fpath = Path(logo_file)
            f_splits = fpath.name.split(".")
            if len(f_splits) > 1 and f_splits[-1] == "png":
                key_val = "thumbnail.png"
            elif len(f_splits) > 1 and f_splits[-1] == "jpg":
                key_val = "thumbnail.jpg"
            elif len(f_splits) > 1 and f_splits[-1] == "gif":
                key_val = "thumbnail.gif"

            self._portal_resources.add(key_val, logo_file)
        elif logo_file is None:
            if "thumbnail" in dict(self._gis.properties):
                resource = self._gis.properties["thumbnail"]
                if resource and len(resource) > 0:
                    self._portal_resources.delete(resource)
                key_val = ""
        else:
            for ext in [".png", ".jpg", ".gif"]:
                try:
                    self._portal_resources.delete("thumbnail" + ext)
                except:
                    continue
            key_val = None

        # extra step for new homepage editor
        if self._new_hp:
            hp = self._reader_hp()
            if show_logo is not None:
                hp["header"]["showLogo"] = show_logo
            hp["header"]["logo"] = key_val

            if alignment and alignment in ["center", "left"]:
                hp["header"]["titleLogoAlign"] = alignment
            if position and position in [
                "middle",
                "above",
                "below",
                "top3rd",
                "bottom3rd",
            ]:
                hp["header"]["titleLogoPos"] = position

            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )

        # Update the portal self with these banner values (need to run for Enterprise < 11)
        if logo_file is not None:
            return self._gis.update_properties({"thumbnail": key_val})
        else:
            rp = self._gis.properties["rotatorPanels"]
            for idx, r in enumerate(rp):
                if r["id"].lower() == "banner-2":
                    r["innerHTML"] = (
                        "<img src='images/banner-2.jpg' style='-webkit-border-radius:0 0 10px 10px;"
                        + " -moz-border-radius:0 0 10px 10px; -o-border-radius:0 0 10px 10px; border-radius:0 0 10px 10px;"
                        + " margin-top:0; width:960px; height:180px;'/><div style='position:absolute; bottom:80px; left:80px;"
                        + " max-height:65px; width:660px; margin:0;'><span style='position:absolute; bottom:0; "
                        "margin-bottom:0; line-height:normal; "
                        + "font-family:HelveticaNeue,Verdana; font-weight:600; font-size:32px; "
                        "color:#369;'>{}</span></div>".format(self._gis.properties.name)
                    )
            return self._gis.update_properties(
                {
                    "clearEmptyFields": True,
                    "thumbnail": "",
                    "rotatorPanels": rp,
                }
            )

    # ----------------------------------------------------------------------
    def get_logo(self, download_path: str):
        """
        Get your organization's logo/thumbnail. You can use the `set_logo()` method to set an image as your logo.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        download_path     required string. Folder path to download the logo file.
        ================  ===============================================================

        :return: Path to downloaded logo file. If None, then logo is not set and nothing was downloaded.

        """
        if self._new_hp is False:
            props = self._gis.properties
            if "thumbnail" in props:
                resource = props["thumbnail"]
        else:
            hp = self._reader_hp()
            resource = hp["header"]["logo"]
        if resource is not None and len(str(resource)) > 0:
            output = self._portal_resources.get(
                resource_name=resource, download_path=download_path
            )
            return output
        return None

    # ----------------------------------------------------------------------
    def set_title(
        self,
        title: str | None = None,
        show_title: bool | None = None,
        color: str | None = None,
        alignment: str | None = None,
        position: str | None = None,
    ):
        """
        Set the homepage title and it's visibility

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        title               Optional string. The title to show on the homepage.
        ----------------    ---------------------------------------------------------------
        show_title          Optional boolean. Determine if title is shown (True) or hidden (False).
        ----------------    ---------------------------------------------------------------
        color               Optional string. Specifies the font color for the for the
                            title. This property recognizes common color
                            names (such as red or blue) and hexadecimal color values.
        ----------------    ---------------------------------------------------------------
        alignment           Optional string. Values can either be "center" or "left". This will set
                            the alignment for the title and logo on the header.
        ----------------    ---------------------------------------------------------------
        position            Optional string. Set the title and logo position on the homepage.
                            Values: "middle" | "above" | "below" | "top3rd" | "bottom3rd"
        ================    ===============================================================

        :return: True | False
        """
        if self._new_hp:
            hp = self._reader_hp()
            if title:
                hp["header"]["title"] = title
            if show_title:
                hp["header"]["showTitle"] = show_title
            if color:
                hp["header"]["titleColor"] = color
            if alignment and alignment in ["center", "left"]:
                hp["header"]["titleLogoAlign"] = alignment
            if position and position in [
                "middle",
                "above",
                "below",
                "top3rd",
                "bottom3rd",
            ]:
                hp["header"]["titleLogoPos"] = position
            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )
        else:
            return False

    # ----------------------------------------------------------------------
    def get_title(self):
        """
        Get the title displayed on the homepage if show title is set to True.

        :return: Dict or None if using old homepage
        """
        if self._new_hp:
            hp = self._reader_hp()
            title = {
                "title": hp["header"]["title"],
                "show_title": hp["header"]["showTitle"],
                "color": hp["header"]["titleColor"],
                "position": hp["header"]["titleLogoPos"],
                "alignment": hp["header"]["titleLogoAlign"],
            }
            return title
        else:
            return None

    # ----------------------------------------------------------------------
    def set_contact_email(
        self, email: str | None = None, show_email: bool | None = None
    ):
        """Set the email shown in the footer of the homepage and whether it is visible."""
        if self._new_hp:
            hp = self._reader_hp()
            if email:
                hp["footer"]["contact"] = email
            if show_email:
                hp["footer"]["showContact"] = show_email
            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )
        else:
            return None

    # ----------------------------------------------------------------------
    def get_contact_email(self):
        """Get the email and whether it is shown from the footer of the homepage."""
        if self._new_hp:
            hp = self._reader_hp()
            contact = {
                "email": hp["footer"]["contact"],
                "show_email": hp["footer"]["showContact"],
            }
            return contact

    # ----------------------------------------------------------------------
    def get_footer(self):
        """Get the footer of the homepage"""
        if self._new_hp:
            hp = self._reader_hp()
            footer = {
                "contact": self.get_contact_email(),
                "text": hp["footer"]["copy"] if "copy" in hp["footer"] else "",
                "show_text": hp["footer"]["showCopy"],
                "color": hp["footer"]["bgColor"] if "bgColor" in hp["footer"] else "",
                "custom_color": hp["footer"]["bgCustom"]
                if "bgCustom" in hp["footer"]
                else "",
            }
            return footer

    # ----------------------------------------------------------------------
    def set_footer(self, text: str, show_text: bool | None = None):
        """Set the text and the visibility of the text in the footer"""
        if self._new_hp:
            hp = self._reader_hp()
            if text:
                hp["footer"]["copy"] = text
            if show_text:
                hp["footer"]["showCopy"] = show_text
            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )
        else:
            return None

    # ----------------------------------------------------------------------
    def get_typography(self):
        """Get the footer of the homepage"""
        if self._new_hp:
            hp = self._reader_hp()
            if hp["useCustomTypography"] == True:
                return hp["customTypography"]
            else:
                return hp["typography"]

    # ----------------------------------------------------------------------
    def set_typography(self, font_family: list, custom: bool = False):
        """
        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        font_family         Optional list. The combination of fonts to use for typography
                            of the homepage. The first string represents the title font and
                            the second string represents the body font. If a custom typography
                            will be set, specify custom = True parameter

                            Values:
                            ["Avenir Next", "Avenir Next"]
                            ["Avenir Next", "Noto Serif"]
                            ["Noto Serif", "Avenir Next"]
                            ["Noto Serif", "Noto Serif"]
        ----------------    ---------------------------------------------------------------
        custom              Optional bool. If True, a custom list of typography was passed in
                            as the value for the font_family parameter.
        ================    ===============================================================

        """
        if self._new_hp:
            hp = self._reader_hp()
            if custom:
                hp["useCustomTypography"] = True
                hp["customTypography"] = font_family
            else:
                hp["useCustomTypography"] = False
                hp["typography"] = font_family
            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )
        else:
            return None

    # ----------------------------------------------------------------------
    def set_base_color(self, color: str):
        """
        Set the base color with a string color name.
        """
        if self._new_hp:
            hp = self._reader_hp()
            hp["baseColor"] = color
            params = {
                "key": "home.page.json",
                "text": hp,
                "f": "json",
            }
            return self._portal_resources.add(
                key="home.page.json", text=json.dumps(params["text"])
            )
        else:
            return None

    # ----------------------------------------------------------------------
    def get_base_color(self):
        """Gets the base color of the home page."""
        if self._new_hp:
            hp = self._reader_hp()
            return hp["baseColor"]


##############################################################################
class MapSettings(object):
    """Helper class that can be called off of UX class using the 'map_settings' property.
    Edit org map settings such as the default extent, default basemap, etc.
    """

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
        self._portal_resources = gis.admin.resources

    # ----------------------------------------------------------------------
    @property
    def default_extent(self):
        """
        Get/Set the site's default extent

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        extent            Required dictionary. The default extent defines where a webmap
                          will open.
                          If a value of None is given, the default extent will be provided.
                          Example Extent (default):
                          {"type":"extent","xmin":-17999999.999994524,"ymin":-11999999.999991827,
                          "xmax":17999999.999994524,"ymax":15999999.999982955,
                          "spatialReference":{"wkid":102100}}
        ================  ===============================================================

        :return: dictionary

        """
        return self._gis.properties["defaultExtent"]

    # ----------------------------------------------------------------------
    @default_extent.setter
    def default_extent(self, extent: dict):
        """
        See main ``default_extent`` property docstring
        """
        if extent is None:
            extent = {
                "type": "extent",
                "xmin": -17999999.999994524,
                "ymin": -11999999.999991827,
                "xmax": 17999999.999994524,
                "ymax": 15999999.999982955,
                "spatialReference": {"wkid": 102100},
            }
        return self._gis.update_properties({"defaultExtent": extent})

    # ----------------------------------------------------------------------
    @property
    def default_basemap(self):
        """
        Get/Set the site's default basemap.

        The Default Basemap opens when users click New Map. Set the group
        in the Basemap Gallery above and choose the map to open. It will
        open at the default extent you set.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           Required string. The new default basemap to set. If None, the
                          default value will be set.
        ================  ===============================================================

        :return: dictionary

        """
        return self._gis.properties["defaultBasemap"]

    # ----------------------------------------------------------------------
    @default_basemap.setter
    def default_basemap(self, value: str):
        """
        See main ``default_basemap`` property docstring
        """
        try:
            basemap = {
                "baseMapLayers": _basemap_definitions.basemap_dict[value],
                "title": value.replace("-", " ").title(),
            }
            return self._gis.update_properties({"defaultBasemap": basemap})
        except:
            raise ValueError(
                "Valid Basemaps: 'dark-gray-vector', 'gray-vector', 'hybrid', 'oceans', 'osm', 'satellite', 'streets-navigation-vector', 'streets-night-vector', 'streets-relief-vector', 'streets-vector', 'terrain', 'topo-vector'"
            )

    # ----------------------------------------------------------------------
    @property
    def use_vector_basemap(self):
        """
        If true, the organization uses the Esri vector basemaps in supported
        ArcGIS apps and basemapGalleryGroupQuery will not be editable
        and will be set to the default query.
        """
        return self._gis.properties["useVectorBasemaps"]

    # ----------------------------------------------------------------------
    @use_vector_basemap.setter
    def use_vector_basemap(self, value: bool):
        if value in [True, False]:
            self._gis.update_properties({"useVectorBasemaps": value})

    # ----------------------------------------------------------------------
    @property
    def vector_basemap(self):
        """
        Get/Set the default vector basemap

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        basemap           required dictionary. The new default vector basemap to set for
                          a given site.
        ================  ===============================================================

        :return: The current default vector basemap
        """
        return self._gis.properties["defaultVectorBasemap"]

    # ----------------------------------------------------------------------
    @vector_basemap.setter
    def vector_basemap(self, basemap: dict):
        """
        See main ``vector_basemap`` property docstring
        """
        value = {}
        if "title" in basemap:
            value["title"] = basemap["title"].replace("-", " ").title()
        if "layers" in basemap:
            value["baseMapLayers"] = basemap["layers"]
        elif "baseMapLayers" in basemap:
            value["baseMapLayers"] = basemap["baseMapLayers"]
        return self._gis.update_properties({"defaultVectorBasemap": value})

    # ----------------------------------------------------------------------
    @property
    def basemap_gallery_group(self):
        """
        Select the group whose web maps will be shown in the basemap gallery.
        To change the group, assign either an instance of Group or the group id.
        Setting to None will revert to default.

        :return: An instance of Group if a group is set, else the default or None
        """
        group = self._gis.properties["basemapGalleryGroupQuery"]
        if "id:" in group:
            # must use [3::] to slice string since format of: "id:123abc"
            groups = self._gis.groups.search(group[3::])
            if groups:
                return self._gis.groups.search(group[3::])[0]
            else:
                return None
        else:
            return group

    # ----------------------------------------------------------------------
    @basemap_gallery_group.setter
    def basemap_gallery_group(self, group: Group | str | None):
        if isinstance(group, Group):
            group = "id:" + group.id
            value = False
        elif isinstance(group, str):
            res = self._gis.groups.search(group)
            if len(res) == 0:
                raise ValueError(
                    "The group id provided could not be found in your org."
                )
            else:
                group = "id:" + group
                value = False
        elif group is None:
            value = True
        self._gis.update_properties(
            {"basemapGalleryGroupQuery": group, "useVectorBasemaps": value}
        )

    # ----------------------------------------------------------------------
    def update_basemap_gallery(self):
        """
        Update the basemap gallery group by getting rid of deprecated maps.
        Returns the updated group.
        """
        if self.use_vector_basemap:
            return self.basemap_gallery_group
        basemap_group = self.basemap_gallery_group
        for item in basemap_group.content():
            if item.content_status == "deprecated" and item.type == "Web Map":
                item.unshare([basemap_group])
        return basemap_group

    # ----------------------------------------------------------------------
    @property
    def default_mapviewer(self):
        """
        Get/Set whether the org's default Map Viewer is MapViewerClassic or the
        modern Map Viewer.

        Values: "modern" | "classic"
        """
        if "mapViewer" in self._gis.properties["portalProperties"]:
            return self._gis.properties["portalProperties"]["mapViewer"]

    # ----------------------------------------------------------------------
    @default_mapviewer.setter
    def default_mapviewer(self, value: str):
        if value not in ["modern", "classic"]:
            raise ValueError("The two accepted values are 'modern' or 'classic'")

        portal_properties = self._gis.properties["portalProperties"]
        portal_properties["mapViewer"] = value
        self._gis.update_properties({"portalProperties": portal_properties})

    # ----------------------------------------------------------------------
    @property
    def units(self):
        """
        Get/Set the default map units. Either 'english' or 'metric'.
        """
        if "units" in self._gis.properties:
            return self._gis.properties["units"]

    # ----------------------------------------------------------------------
    @units.setter
    def units(self, value: str):
        if value not in ["english", "classic", "metric"]:
            raise ValueError("The two accepted values are 'english' and 'metric'")

        self._gis.update_properties({"units": value})

    # ----------------------------------------------------------------------
    def bing_map(self, bing_key: str | None = None, share_public: bool | None = None):
        """
        Provide a Microsoft-supplied Bing Maps key to use Bing Maps in your portal's web maps.

        Bing Map Key: https://www.bingmapsportal.com/

        ======================      ==============================================
        **Parameter**                    **Description**
        ----------------------      ----------------------------------------------
        bing_key                    Optional str. The bing key to pass in. To remove
                                    pass in "".
        ----------------------      ----------------------------------------------
        share_public                Optional bool. If True, allows this Bing Maps
                                    key to be used in maps shared publicly by
                                    organization members.
                                    This requires the access of the portal to be set as public.
        ======================      ==============================================

        :return: Dictionary containing the bing key and whether is is publicly shared
        """
        if bing_key is not None:
            if bing_key == "":
                bing_key = None
            self._gis.update_properties({"bingKey": bing_key})
        if share_public:
            self._gis.update_properties({"canShareBingPublic": share_public})
        bing_dict = {
            "key": self._gis.properties["bingKey"]
            if "bingKey" in self._gis.properties
            else None,
            "public": self._gis.properties["canShareBingPublic"],
        }
        return bing_dict

    # ----------------------------------------------------------------------
    @property
    def config_apps_group(self):
        """
        ArcGIS Configurable Apps contain various settings users can configure
        to create web apps. Map-based apps display one or more maps.
        Choose which group contains the apps you want to use in the configurable apps
        gallery.

        Assign either an instance of Group class, a group id, or None to reset
        to default.

        :return: An instance of group if a group is set, else the default or None
        """
        if "templatesGroupQuery" in self._gis.properties:
            group = self._gis.properties["templatesGroupQuery"]
            if "id:" in group:
                # must use [3::] to slice string since format of: "id:123abc"
                groups = self._gis.groups.search(group[3::])
                if groups:
                    return groups[0]
                else:
                    return None
            else:
                return group
        else:
            return "Default"

    # ----------------------------------------------------------------------
    @config_apps_group.setter
    def config_apps_group(self, group: Group | str | None):
        if isinstance(group, Group):
            group = "id:" + group.id
        elif isinstance(group, str):
            res = self._gis.groups.search(group)
            if len(res) == 0:
                raise ValueError(
                    "The group id provided could not be found in your org."
                )
            else:
                group = "id:" + group

        self._gis.update_properties(
            {"templatesGroupQuery": group, "clearEmptyFields": True}
        )

    # ----------------------------------------------------------------------
    def web_styles(
        self,
        group: Group | str | None = None,
        two_dimensional_map: bool = False,
        three_dimensional_map: bool = False,
    ):
        """
        Web styles are collections of symbols stored in an item. Apps can
        use web styles to symbolize point features with 2D or 3D symbols.
        Select a group to be used in symbol galleries.

        ======================      ==============================================
        **Parameter**                **Description**
        ----------------------      ----------------------------------------------
        group                       Optional str or Group. either an instance of Group class,
                                    a group id, or None to reset to default.
        ----------------------      ----------------------------------------------
        two_dimensional_map         Optional bool. If True, the group will be assigned
                                    to 2D Web Style.
        ----------------------      ----------------------------------------------
        three_dimensional_map       Optional bool. If True, the group will be assigned
                                    to 3D Web Style.
        ======================      ==============================================

        :return: dict indicating the group(s) set for the 2D and 3D styles
        """
        if isinstance(group, Group):
            group = "id:" + group.id
        elif isinstance(group, str):
            res = self._gis.groups.search(group)
            if len(res) == 0:
                raise ValueError(
                    "The group id provided could not be found in your org."
                )
            else:
                group = "id:" + group
        if two_dimensional_map:
            res = self._gis.update_properties({"2DStylesGroupQuery": group})
        if three_dimensional_map:
            res = self._gis.update_properties({"stylesGroupQuery": group})
        return {
            "2DStyles": self._gis.properties[
                "2DStylesGroupQuery",
                "3DStyles" : self._gis.properties["stylesGroupQuery"],
            ]
        }

    # ----------------------------------------------------------------------
    @property
    def analysis_layer_group(self):
        """
        Select the group whose layers will be shown in the Analysis Layer
        gallery for the analysis tools. It is best practice to share feature
        items that contain only a single layer with this group.
        If your feature layer item contains multiple layers, save any of the
        layers as an item and share it with the group.

        :return: If set, an instance of Group else the default or None
        """
        if "analysisLayersGroupQuery" in self._gis.properties:
            group = self._gis.properties["analysisLayersGroupQuery"]
            if "id:" in group:
                # must use [3::] to slice string since format of: "id:123abc"
                groups = self._gis.groups.search(group[3::])
                if groups:
                    return groups[0]
                return "Default"
            else:
                return group
        else:
            return "Default"

    # ----------------------------------------------------------------------
    @analysis_layer_group.setter
    def analysis_layer_group(
        self,
        group: Group | str | None = None,
    ):
        if isinstance(group, Group):
            group = "id:" + group.id
        elif isinstance(group, str):
            res = self._gis.groups.search(group)
            if len(res) == 0:
                raise ValueError(
                    "The group id provided could not be found in your org."
                )
            else:
                group = "id:" + group
        self._gis.update_properties(
            {"analysisLayersGroupQuery": group, "clearEmptyFields": True}
        )


#############################################################################
class ItemSettings(object):
    """Helper class that can be called off of UX class using the 'item_settings' property.
    Edit org item settings such as the enabling/disabling comments, metadata info, etc.
    """

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
        self._portal_resources = gis.admin.resources

    # ----------------------------------------------------------------------
    @property
    def enable_comments(self):
        """
        Get/Set item commenting and comments.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            Optional boolean. If True, the comments for the site are turned
                          on.  False will disable comments (default)
        ================  ===============================================================

        :return: True if enabled, False if disabled
        """
        return self._gis.properties["commentsEnabled"]

    # ----------------------------------------------------------------------
    @enable_comments.setter
    def enable_comments(self, enable: bool = False):
        """
        See main ``enable_comments`` property docstring.
        """
        self._gis.update_properties({"commentsEnabled": enable})

    # ----------------------------------------------------------------------
    @property
    def enable_metadata_edit(self):
        """
        Get/Set item metadata editable ability.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        enable            Optional boolean. If True, the editing of metadata on items is turned
                          on (default). False will disable metadata editing on items.
        ================  ===============================================================
        """
        return self._gis.properties["metadataEditable"]

    # ----------------------------------------------------------------------
    @enable_metadata_edit.setter
    def enable_metadata_edit(self, enable: bool):
        self._gis.update_properties({"metadataEditable": enable})

    # ----------------------------------------------------------------------
    @property
    def metadata_format(self):
        """
        Get/Set the metadata format used

        Values: 'arcgis' | 'fgdc' | 'inspire' | 'iso19139' | 'iso19139-3.2' | 'iso19115'
        """
        if (
            "metadataFormats" in self._gis.properties
            and self._gis.properties["metadataFormats"]
        ):
            return self._gis.properties["metadataFormats"][0]
        return "arcgis"

    # ----------------------------------------------------------------------
    @metadata_format.setter
    def metadata_format(self, format):
        if format not in [
            "arcgis",
            "fgdc",
            "inspire",
            "iso19139",
            "iso19139-3.2",
            "iso19115",
        ]:
            raise ValueError("Format assigned is not an acceptable format.")
        if self.enable_metadata_edit is not True:
            self.enable_metadata_editable = True
        self._gis.update_properties({"metadataFormats": format})


#############################################################################
class SecuritySettings(object):
    """Helper class that can be called off of UX class using the 'security_settings' property.
    Edit org item settings such as the informational banner, password policy, etc.
    """

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Creates helper object to manage portal home page, resources, update resources"""
        self._gis = gis
        self._portal = gis._portal
        self._portal_resources = gis.admin.resources

    # ----------------------------------------------------------------------
    def set_informational_banner(
        self,
        text: str | None = None,
        bg_color: str | None = None,
        font_color: str | None = None,
        enabled: bool | None = None,
    ):
        """
        The informational banner that is shown at the top of your organization's page.

        ================    ===============================================================
        **Parameter**        **Description**
        ----------------    ---------------------------------------------------------------
        text                Optional string. The text that the informational banner will display.
                            To set an empty text use: ""
        ----------------    ---------------------------------------------------------------
        bg_color            Optional string. Specifies the background color for the
                            informational banner. This property recognizes common color names
                            (such as red or blue) and hexadecimal color values. While you
                            are able to choose any color for this property, it is recommended
                            that you choose a color that contrasts appropriately with the
                            font_color, as a poor contrast will cause a warning to appear
                            in the Security settings page of yourEnterprise portal
                            alerting you to the insufficient contrast.
        ----------------    ---------------------------------------------------------------
        font_color          Optional string. Specifies the font color for the for the
                            informational banner. This property recognizes common color
                            names (such as red or blue) and hexadecimal color values.
                            While you are able to choose any color for this property,
                            it is recommended that you choose a color that contrasts
                            appropriately with the bg_color, as a poor contrast will
                            cause a warning to appear in the Security settings page of your
                            Enterprise portal alerting you to the insufficient contrast.
        ----------------    ---------------------------------------------------------------
        enabled             Optional bool. Determine whether the informational banner is
                            enabled (True) or disabled (False).
        ================    ===============================================================

        :return: True if succeeded
        """
        # if user wants to change one thing, keep other settings
        if "informationalBanner" in self._gis.org_settings:
            current_info_banner = self._gis.org_settings["informationalBanner"]
        else:
            current_info_banner = {
                "text": "",
                "bgColor": "white",
                "fontColor": "black",
                "enabled": False,
            }
        # set new params if given
        informational_banner = {
            "text": text if text else current_info_banner["text"],
            "bgColor": bg_color if bg_color else current_info_banner["bgColor"],
            "fontColor": font_color if font_color else current_info_banner["fontColor"],
            "enabled": enabled
            if enabled is not None
            else current_info_banner["enabled"],
        }

        # get all the org settings
        org_settings = self._gis.org_settings
        # change the informational banner part
        org_settings["informationalBanner"] = informational_banner
        # reset org settings
        self._gis.org_settings = org_settings
        return True

    # ----------------------------------------------------------------------
    def get_informational_banner(self):
        """
        Get the informational banner dictionary from the org's settings.
        If none set, return None
        """
        if "informationalBanner" in self._gis.org_settings:
            return self._gis.org_settings["informationalBanner"]
        else:
            return None

    # ----------------------------------------------------------------------
    @property
    def enable_https(self):
        """Allow access to the portal through HTTPS only (True)."""
        return self._gis.properties["allSSL"]

    # ----------------------------------------------------------------------
    @enable_https.setter
    def enable_https(self, enable: bool):
        if enable in [True, False]:
            self._gis.update_properties({"allSSL": enable})

    # ----------------------------------------------------------------------
    @property
    def anonymous_access(self):
        """Allow anonymous access to your Portal (True), or make private (False)"""
        return self._gis.properties["access"]

    # ----------------------------------------------------------------------
    @anonymous_access.setter
    def anonymous_access(self, access: bool):
        if access is True or access != "private":
            access = "public"
            if "canShareBingPublic" in self._gis.properties:
                bing = self._gis.properties["canShareBingPublic"]
            else:
                bing = False
        elif access is False or access == "private":
            access = "private"
            bing = False
        self._gis.update_properties({"canShareBingPublic": bing, "access": access})

    # ----------------------------------------------------------------------
    @property
    def enable_update_user_profile(self):
        """Allow members to edit biographical information and who can see their profile."""
        return self._gis.properties["updateUserProfileDisabled"]

    # ----------------------------------------------------------------------
    @enable_update_user_profile.setter
    def enable_update_user_profile(self, enable: bool):
        self._gis.update_properties({"updateUserProfileDisabled": enable})

    # ----------------------------------------------------------------------
    @property
    def share_public(self):
        """Members can share content publicly."""
        return self._gis.properties["canSharePublic"]

    # ----------------------------------------------------------------------
    @share_public.setter
    def share_public(self, enable: bool):
        self._gis.update_properties({"canSharePublic": enable})

    # ----------------------------------------------------------------------
    @property
    def show_social_media(self):
        """
        Show social media links on item and group pages.
        """
        if "showSocialMediaLinks" in self._gis.properties["portalProperties"]:
            return self._gis.properties["portalProperties"]["showSocialMediaLinks"]
        else:
            return False

    # ----------------------------------------------------------------------
    @show_social_media.setter
    def show_social_media(self, enable: bool):
        pp = self._gis.properties["portalProperties"]
        pp["showSocialMediaLinks"] = enable
        self._gis.update_properties({"portalProperties": pp})

    # ----------------------------------------------------------------------
    def get_password_policy(self):
        """
        Get the password policy currently set for your org.
        Returns a dictionary indicating the rules currently set as the policy.
        """
        url = self._portal.resturl + "/portals/self/securitypolicy"
        params = {"f": "json"}
        return self._gis._con.post(url, params)["passwordPolicy"]

    # ----------------------------------------------------------------------
    def update_password_policy(
        self,
        min_length: int | None = None,
        include_uppercase: bool | None = None,
        include_lowercase: bool | None = None,
        include_letter: bool | None = None,
        include_number: bool | None = None,
        include_special_char: bool | None = None,
        expires_in: int | None = None,
        history_number: int | None = None,
    ):
        """
        Set the password policy for members in your organization that have ArcGIS
        accounts. Note that member passwords may not match their username. Weak
        passwords will be rejected. You may set the following rules for
        these passwords by turning them on (True) or off (False) and specifying a number
        where requested.

        =====================       ===============================================================
        **Parameter**                **Description**
        ---------------------       ---------------------------------------------------------------
        min_length                  Optional int. Password must contain at least the following
                                    number of characters. Cannot set this under 8 characters.
        ---------------------       ---------------------------------------------------------------
        include_uppercase           Optional bool. Must contain at least one upper case letter (A-Z).
        ---------------------       ---------------------------------------------------------------
        include_lowercase           Optional bool. Must contain at least one lower case letter (a-z).
        ---------------------       ---------------------------------------------------------------
        include_letter              Optional bool. Must contain at least one letter (A-Z, a-z).
        ---------------------       ---------------------------------------------------------------
        include_number              Optional bool. Must contain at least one number (0-9).
        ---------------------       ---------------------------------------------------------------
        include_special_char        Optional bool. Must contain at least one special
                                    (non-alphanumeric) character
        ---------------------       ---------------------------------------------------------------
        expires_in                  Optional int. Password expires after the specified number of days.
                                    Value between 1 and 1095 days.
                                    If value of 0 is passed in then it will disable this parameter.
        ---------------------       ---------------------------------------------------------------
        history_number              Optional int. Users cannot reuse the specified number of last passwords.
                                    Password history may have a value between 1 and 24 passwords.
                                    If value of 0 is passed in then it will disable this parameter.
        =====================       ===============================================================


        """
        # Value checks before starting
        if min_length and min_length < 8 and min_length > 1000:
            return ValueError("The length of a password must be between 8 and 1000.")
        if expires_in and expires_in > 1095:
            return ValueError(
                "The password expiration value must be between 1 and 1095 days to set, if 0 then will be turned off."
            )
        if history_number and history_number > 24:
            return ValueError(
                "The password history must have a value between 1 and 24 passwords, if 0 then will be turned off."
            )

        # set url
        url = self._portal.resturl + "/portals/self/securitypolicy/update"

        # get current settings, be careful not all settings are present
        current_policy = self.get_password_policy()

        policy = {
            "f": "json",
            "minLength": min_length if min_length else current_policy["minLength"],
        }

        # For all parameters, only need to set if True or value passed in.
        # If current policy has it set and user specifies False, then need to remove.
        # If None and does not exist in the current policy then can ignore.
        # Include Uppercase
        if include_uppercase is True:
            policy["minUpper"] = 1
        elif include_uppercase is False:
            policy["minUpper"] = None
        elif include_uppercase is None and "minUpper" in current_policy:
            policy["minUpper"] = current_policy["minUpper"]
        # Include Lowercase
        if include_lowercase is True:
            policy["minLower"] = 1
        elif include_lowercase is False:
            policy["minLower"] = None
        elif include_lowercase is None and "minLower" in current_policy:
            policy["minLower"] = current_policy["minLower"]
        # Include Letter
        if include_letter is True:
            policy["minLetter"] = 1
        elif include_letter is False:
            policy["minLetter"] = None
        elif include_letter is None and "minLetter" in current_policy:
            policy["minLetter"] = current_policy["minLetter"]
        # Include Number
        if include_number is True:
            policy["minDigit"] = 1
        elif include_number is False:
            policy["minDigit"] = None
        elif include_number is None and "minDigit" in current_policy:
            policy["minDigit"] = current_policy["minDigit"]
        # Include Special Character
        if include_special_char is True:
            policy["minOther"] = 1
        elif include_special_char is False:
            policy["minOther"] = None
        elif include_special_char is None and "minOther" in current_policy:
            policy["minOther"] = current_policy["minOther"]
        # Number Days Password Expires In
        if expires_in:
            if expires_in == 0:
                policy["expirationInDays"] = None
            else:
                policy["expirationInDays"] = expires_in
        elif "expirationInDays" in current_policy:
            policy["expirationInDays"] = current_policy["expirationInDays"]
        # Number Passwords In History
        if history_number:
            if history_number == 0:
                policy["historySize"] = None
            else:
                policy["historySize"] = history_number
        elif "historySize" in current_policy:
            policy["historySize"] = current_policy["historySize"]

        # send request
        return self._gis._con.post(url, policy)

    # ----------------------------------------------------------------------
    def reset_password_policy(self):
        """
        Reset the password policy to base settings.
        """
        url = self._portal.resturl + "/portals/self/securitypolicy/reset"
        params = {"f": "json"}
        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    @property
    def allowed_origins(self):
        """
        Get/Set the list of allowed origins.

        Allowed Origins limit the web application domains that can connect
        via Cross-Origin Resource Sharing (CORS) to the ArcGIS REST API.

        Can set a list of up to 100 web application domains to restrict CORS access to the REST API.
        """
        return self._gis.properties["allowedOrigins"]

    # ----------------------------------------------------------------------
    @allowed_origins.setter
    def allowed_origins(self, allowed_origins: list[str]):
        if isinstance(allowed_origins, (tuple, list)):
            allowed_origins = ",".join(allowed_origins)
            self._gis.update_properties({"allowedOrigins": allowed_origins})
        elif allowed_origins is None or allowed_origins == "":
            allowed_origins = ""
            self._gis.update_properties(
                {
                    "allowedOrigins": allowed_origins,
                    "clearEmptyFields": True,
                }
            )
        else:
            self._gis.update_properties({"allowedOrigins": allowed_origins})

    # ----------------------------------------------------------------------
    @property
    def allowed_redirect_uris(self):
        """
        Configure a list of portals with which you want to share secure content.
        This will allow members of your organization to use their enterprise logins
        to access the secure content when viewing it from these portals.

        Set a list of allowed redirect URIs which represent portal instances that
        you share secure content with. This will allow your organization users
        to be able to use enterprise logins to access the secured content
        through web applications hosted on these portals.
        """
        if "allowedRedirectUris" in self._gis.properties:
            return self._gis.properties["allowedRedirectUris"]
        return None

    # ----------------------------------------------------------------------
    @allowed_redirect_uris.setter
    def allowed_redirect_uris(self, uris: list[str]):
        if isinstance(uris, (tuple, list)):
            uris = ",".join(uris)
            self._gis.update_properties({"allowedRedirectUris": uris})
        elif uris is None or uris == "":
            uris = ""
            self._gis.update_properties(
                {
                    "allowedRedirectUris": uris,
                    "clearEmptyFields": True,
                }
            )
        else:
            self._gis.update_properties({"allowedRedirectUris": uris})

    # ----------------------------------------------------------------------
    @property
    def trusted_servers(self):
        """
        Configure the list of trusted servers you wish your organization to
        send credentials to when working with services secured with web-tier authentication.

        Set a list of trusted servers that clients can send credentials to when
        making Cross-Origin Resource Sharing (CORS) requests to access web-tier secured services.
        """
        if "authorizedCrossOriginDomains" in self._gis.properties:
            return self._gis.properties["authorizedCrossOriginDomains"]
        return None

    # ----------------------------------------------------------------------
    @trusted_servers.setter
    def trusted_servers(self, servers: list):
        self._gis.update_properties({"authorizedCrossOriginDomains": servers})

    # ----------------------------------------------------------------------
    def set_org_access_notice(
        self,
        title: str | None = None,
        text: str | None = None,
        button_type: str | None = None,
    ):
        """
        Provide a notice of terms to be displayed to organization members
        after they have signed in. Members can proceed only if they accept
        the terms of the notice. They will not be prompted again till the
        next time they sign in.

        ======================      ====================================================
        **Parameter**                **Description**
        ----------------------      ----------------------------------------------------
        title                       Optional string. The title to set the for the notice.
                                    If None then notice will be disabled.
        ----------------------      ----------------------------------------------------
        text                        Optional string. The text body to set for the notice.
                                    If None then notice will be disabled.
        ----------------------      ----------------------------------------------------
        button_type                 Optional string. The button types the users will see
                                    for the notice. Default is an accept and decline button.

                                    Values: "acceptAndDecline" | "okOnly"
        ======================      ====================================================

        :return: True | False
        """
        org_settings = self._gis.org_settings
        if button_type is None:
            button_type = "acceptAndDecline"
        notice = {"title": title, "text": text, "buttons": button_type}
        if title and text:
            notice["enabled"] = True
        else:
            notice["enabled"] = False

        org_settings["authenticatedAccessNotice"] = notice
        org_settings["clearEmptyFields"] = True
        self._gis.org_settings = org_settings
        return True

    # ----------------------------------------------------------------------
    def get_org_access_notice(self):
        """
        Get the provided notice of terms to be displayed to organization members
        after they have signed in. Members can proceed only if they accept
        the terms of the notice. They will not be prompted again till the
        next time they sign in.

        :return: The dict representation of the notice or if none set then None.
        """
        org_settings = self._gis.org_settings

        if "authenticatedAccessNotice" in org_settings:
            return org_settings["authenticatedAccessNotice"]
        else:
            return None

    # ----------------------------------------------------------------------
    def set_anonymous_access_notice(
        self,
        title: str | None = None,
        text: str | None = None,
        button_type: str | None = None,
        enabled: bool | None = False,
    ):
        """
        Provide a notice of terms to be displayed to all users who access your
        organization. Users can proceed only if they accept the terms of the
        notice. They will not be prompted again for the remainder of the
        browser session. If you set both types of access notices, an
        organization member will see two notices.

        ======================      ====================================================
        **Parameter**                **Description**
        ----------------------      ----------------------------------------------------
        title                       Optional string. The title to set the for the notice.
                                    If None then notice will be disabled.
        ----------------------      ----------------------------------------------------
        text                        Optional string. The text body to set for the notice.
                                    If None then notice will be disabled.
        ----------------------      ----------------------------------------------------
        button_type                 Optional string. The button types the users will see
                                    for the notice. Default is an accept and decline button.

                                    Values: "acceptAndDecline" | "okOnly"
        ======================      ====================================================

        :return: True | False
        """
        org_settings = self._gis.org_settings
        if button_type is None:
            button_type = "acceptAndDecline"
        notice = {"title": title, "text": text, "buttons": button_type}
        if title and text:
            notice["enabled"] = enabled
        else:
            notice["enabled"] = enabled

        org_settings["anonymousAccessNotice"] = notice
        org_settings["clearEmptyFields"] = True
        self._gis.org_settings = org_settings
        return True

    # ----------------------------------------------------------------------
    def get_anonymous_access_notice(self):
        """
        Get the provided notice of terms to be displayed to all users who
        access your organization. Users can proceed only if they accept the
        terms of the notice. They will not be prompted again for the remainder
        of the browser session. If you set both types of access notices,
        an organization member will see two notices.

        :return: The dict representation of the notice or if none set then None.
        """
        org_settings = self._gis.org_settings

        if "anonymousAccessNotice" in org_settings:
            return org_settings["anonymousAccessNotice"]
        else:
            return None

    # ----------------------------------------------------------------------
    def set_multifactor_authentication(
        self, admins: list[str] | None = None, enabled: bool = False
    ):
        """
        Multifactor authentication provides all members with ArcGIS accounts
        in your organization with an extra level of security by requesting
        an additional verification code at the time of login.

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        admins                          Optional list of strings. Designate at least two
                                        administrators who will receive email requests
                                        to troubleshoot members' multifactor
                                        authentication issues. Provide a list of at least
                                        two admin usernames.
        -------------------------       --------------------------------------------------
        enabled                         Optional bool. Allow members to choose whether to
                                        set up multifactor authentication for
                                        their individual accounts. Default is False.
        =========================       ==================================================

        :return: True | False
        """
        # Run some checks
        if enabled is True:
            admins_ok = []
            if admins is None:
                raise ValueError(
                    "Cannot set empty list as Administrative contacts. You must have at least two administrators in the list."
                )
            for ad in admins:
                role = self._gis.users.get(ad).role
                if role == "org_admin":
                    admins_ok.append(ad)
            if len(admins_ok) < 2:
                raise ValueError(
                    "None of the usernames provided are org admins. Please provide at least 2 org admins."
                )
            else:
                admins = admins_ok
        # perform update

        return self._gis.update_properties(
            {
                "mfaEnabled": enabled,
                "mfaAdmins": admins,
                "clearEmptyFields": True,
            }
        )

    # ----------------------------------------------------------------------
    def get_multifactor_authentication(self):
        """
        See if multifactor authentication is set and who are the admins that
        are set as points of contact.

        :return: dict
        """
        mfa = {}
        if "mfaEnabled" in self._gis.properties:
            mfa["enabled"] = self._gis.properties["mfaEnabled"]
        if "mfaAdmins" in self._gis.properties:
            mfa["admins"] = self._gis.properties["mfaAdmins"]
        return mfa

    # ----------------------------------------------------------------------
    def set_email_settings(
        self,
        smtp_host: str,
        smtp_port: int,
        from_address: str,
        from_address_label: str,
        encryption_method: str = "SSL",
        auth_required: bool = False,
        username: str | None = None,
        password: str | None = None,
    ):
        """
        This operation allows you to configure email settings for your organization.
        These settings will be used to send out email notifications from ArcGIS
        Enterprise portal regarding password policy updates and license expirations.

        .. note::
            Not for ArcGIS Online.

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        smtp_host                       Requried string. The IP address, or the fully
                                        qualified domain name (FDQN), of the SMTP Server.

                                        Example: smtpServer=smtp.myorg.org
        -------------------------       --------------------------------------------------
        smtp_port                       Required integer. The port the SMTP Server will
                                        communicate over. Some of the most common communication
                                        ports are 25, 465, and 587.
        -------------------------       --------------------------------------------------
        from_address                    Required string. The email address that will be
                                        used to send emails from the ArcGIS Enterprise portal.
                                        It is recommended that the user associated with
                                        this email address is listed under the Administrative
                                        Contacts for your organization.
        -------------------------       --------------------------------------------------
        from_address_label              Required string. The label, or person, associated
                                        with the from_email_address. This information will be
                                        displayed as the sender in the From line for
                                        all email notifications.
        -------------------------       --------------------------------------------------
        encryption_method               Optional string. The encryption method for email
                                        messages sent from ArcGIS Enterprise portal.

                                        `Values: 'SSL' | 'TLS' | 'NONE'`
        -------------------------       --------------------------------------------------
        auth_required                   Optional bool. Specifies if authentication is
                                        required (True) to connect with SMTP server specified
                                        above. At 10.8.1, only basic authentication
                                        (username and password) is supported.
                                        The default is False.
        -------------------------       --------------------------------------------------
        username                        Optional string. If auth_required is True, this
                                        specifies the username of a user who is authorized
                                        to access the SMTP server. This field will be unable
                                        to be defined if auth_required is False.
        -------------------------       --------------------------------------------------
        password                        Optional string. If auth_required is True, this
                                        specifies the password associated the authorized
                                        user specified above. This field will be unable to
                                        be defined if auth_required is False.
        =========================       ==================================================

        :return: Dictionary indicating success or failure.
        """
        if self._gis._is_agol is True:
            return None
        url = self._portal.resturl + "portals/self/setEmailSettings"
        params = {
            "f": "json",
            "smtpHost": smtp_host,
            "smtpPort": smtp_port,
            "fromAddress": from_address,
            "fromAddressLabel": from_address_label,
            "authRequired": auth_required,
            "encryptionMethod": encryption_method,
        }
        if auth_required:
            if username:
                params["username"] = username
            if password:
                params["password"] = password

        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def get_email_settings(self):
        """
        This resource returns the email settings that have been configured
        for your organization. These settings can be used to send out email
        notifications from ArcGIS Enterprise portal about password policy
        updates and user type, add-on, or organization capability license expirations.

        :return: Dictionary of email settings, if None set then empty dict is returned.
        """
        if self._gis._is_agol is True:
            return None
        url = self._portal.resturl + "portals/self/emailSettings"
        return self._gis._con.post(url, {"f": "json"})

    # ----------------------------------------------------------------------
    def delete_email_settings(self):
        """
        This operation deletes all previously configured email settings for
        your organization. Once deleted, email notifications about password
        policy changes and license expirations will no longer be received by
        members listed under your Administrative Contacts.
        As well, users will no longer be able to use their email to retrieve forgotten passwords.
        """
        if self._gis._is_agol is True:
            return None
        url = self._portal.resturl + "portals/self/emailSettings/delete"
        return self._gis._con.post(url, {"f": "json"})

    # ----------------------------------------------------------------------
    def test_email_settings(self, mail_to: str):
        """
        This operation can be used once the email settings have been
        configured using the set_email_settings operation to send a test
        email via the SMTP server to ensure that the organization's email
        settings have been properly configured. If successful, an email
        will be sent out to the specified email address (mail_to).

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        mail_to                         Requried string. The email the test message will
                                        be sent to.
        =========================       ==================================================

        :return: Dictionary result stating success and number of seconds to wait before rechecking.
        """
        if self._gis._is_agol is True:
            return None
        url = self._portal.resturl + "portals/self/emailSettings/test"
        return self._gis._con.post(url, {"f": "json", "mailTo": mail_to})

    # ----------------------------------------------------------------------
    @property
    def signin_settings(self):
        """
        Get the signin settings for the org.

        :return: Dictionary response of the settings and their values
        """
        url = self._portal.resturl + "portals/self/signinSettings"
        params = {"f": "json"}

        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def set_approved_apps(self, block_unapproved: bool = False):
        """
        Control which apps members are allowed to access without a
        'Request for Permissions' prompt. Approved web apps can optionally be
        made available to organization members in the App Launcher.

        .. note::
            Only available in ArcGIS Online

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        block_unapproved                Optional bool. Determine whether members can only
                                        sign in to external apps that are approved. Default
                                        is False.
        =========================       ==================================================
        """
        if self._gis._is_agol is True:
            url = self._portal.resturl + "portals/self/setSigninSettings"
            params = {
                "f": "json",
                "blockUnapprovedThirdpartyApps": block_unapproved,
            }

            return self._gis._con.post(url, params)
        else:
            return False

    # ----------------------------------------------------------------------
    def set_blocked_apps(
        self, block_beta_apps: bool = False, apps: list[str] | None = None
    ):
        """
        Control which apps are blocked for all members in your organization
        in order to comply with regulations, standards, and best practices.

        .. note::
            Only available in ArcGIS Online

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        block_beta_apps                 Optional bool. Block Esri apps while they are in beta.
                                        Default is False.
        =========================       ==================================================

        :return: Json dictionary response indicating success or failure.
        """
        if self._gis._is_agol is True:
            url = self._portal.resturl + "portals/self/setSigninSettings"
            params = {
                "f": "json",
                "blockBetaApps": block_beta_apps,
            }

            return self._gis._con.post(url, params)
        else:
            return False

    # ----------------------------------------------------------------------
    def set_social_media_login(
        self,
        social_login: bool,
        social_networks: list[str] | None = None,
        social_network_order: list[str] | None = None,
    ):
        """
        Customize the organization's sign in page so that members can sign in using any
        of the methods below. The order they appear here will determine the order that
        they appear in the sign in page.

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        social_login                    Required bool. Allow members to sign up and sign in
                                        to your organization using their login from the
                                        following social networks: Facebook, Google, Github, Apple
        -------------------------       --------------------------------------------------
        social_networks                 Optional list of strings. The social networks allowed to
                                        use as sign in options for the org.

                                        Options: Facebook, Google, Github, Apple
        -------------------------       --------------------------------------------------
        social_network_order            Optional list of strings. The order in which the
                                        social network login options will appear on the
                                        login page.

                                        If none is set then current settings are used.
        =========================       ==================================================

        """
        if self._gis._is_agol is True:
            # Update if can sign in with social media
            url = self._portal.resturl + "portals/self/update"
            params = {
                "f": "json",
                "canSignInSocial": social_login,
            }
            res = self._gis._con.post(url, params)
            # Update the social networks used
            if social_login and social_networks:
                url = self._portal.resturl + "portals/self/setSigninSettings"
                signin_options = self.signin_settings
                if "signinOptionsOrder" in signin_options:
                    params = signin_options
                    params["f"] = "json"
                else:
                    params = {
                        "f": "json",
                        "signinOptionsOrder": {
                            "logins": ["arcgis", "social"],
                            "social": [
                                "facebook",
                                "google",
                                "github",
                                "apple",
                            ],
                        },
                    }
                # specify order if passed in
                if social_network_order:
                    params["signinOptionsOrder"]["social"] = social_network_order
                res = self._gis._con.post(url, params)
                # configure the social providers
                for network in social_networks:
                    networks = []
                    if network.lower() in [
                        "facebook",
                        "google",
                        "github",
                        "apple",
                    ]:
                        networks.append(network)
                url = self._portal.resturl + "portals/self/socialProviders/configure"
                params = {
                    "f": "json",
                    "providers": ",".join(networks).lower(),
                    "signUpMode": "Invitation",
                }
                res = self._gis._con.post(url, params)
            return res
        else:
            return None

    # ----------------------------------------------------------------------
    def enable_arcgis_online_login(self, enable: bool):
        """
        Allow users to sign in with their ArcGIS login.
        """
        url = self._portal.resturl + "portals/self/update"
        params = {"f": "json", "canSignInArcGIS": enable}

        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def register_idp(
        self,
        name: str,
        metadata_file: str | None = None,
        metadata_url: str | None = None,
        binding_url: str | None = None,
        post_binding_url: str | None = None,
        logout_url: str | None = None,
        entity_id: str | None = None,
        signup_mode: str = "Invitation",
        encryption_supported: bool = False,
        support_signed_request: bool = False,
        use_SHA256: bool = False,
        support_logout_request: bool = False,
        update_profile_at_signin: bool = False,
        update_groups_at_signin: bool = False,
    ):
        """
        Allows organization administrators to configure a new enterprise login.
        Configuring enterprise login allows members of your organization to sign
        in to your organization using the same logins they use to access your
        enterprise information systems without creating additional logins.
        ArcGIS Online and ArcGIS Enterprise are compliant with SAML 2.0 and integrate
        with IDPs that support SAML 2 web single sign-on for securely exchanging
        authentication and authorization data between your organization and ArcGIS Online
        or ArcGIS Enterprise as a service provider (SP). An organization can be set up
        using either a single IDP or a federation, but not both.

        =========================       ==================================================
        **Parameter**                    **Description**
        -------------------------       --------------------------------------------------
        name                            Required string. The identity provider name.
        -------------------------       --------------------------------------------------
        metadata_file                   Optional string. Metadata file that contains information
                                        about the IDP. One can also specify the settings
                                        using metadata_url or binding_url and post_binding_url
                                        parameters alternatively.
        -------------------------       --------------------------------------------------
        metadata_url                    Optional string. Metadata URL that returns information
                                        about information about the IDP.
        -------------------------       --------------------------------------------------
        binding_url                     Optional string. The HTTP redirect binding IDP's
                                        URL that your organization uses to allow a member
                                        to sign in.
        -------------------------       --------------------------------------------------
        post_binding_url                Optional string. The HTTP POST binding IDP's URL
                                        that your organization uses to allow a member to sign in.
        -------------------------       --------------------------------------------------
        logout_url                      Optional string. IDP URL used to sign out a signed-in
                                        user (automatically set if the property is specified
                                        in the IDP metadata file).
        -------------------------       --------------------------------------------------
        entity_id                       Optional string. Entity ID used to identify the
                                        organization in IDP.
        -------------------------       --------------------------------------------------
        signup_mode                     Optional string. Specifies whether enterprise members
                                        join the organization automatically or through an invitation.

                                        `Values: 'Automatic' | 'Invitation'`
        -------------------------       --------------------------------------------------
        encryption_supported            Optional bool. If True, it indicates to the identity
                                        provider that encrypted SAML assertion responses
                                        are supported. The default is False.
        -------------------------       --------------------------------------------------
        support_signed_request          Optional bool. If True, the organization signs the
                                        SAML authentication request sent to the IDP.
                                        The default is False.
        -------------------------       --------------------------------------------------
        use_SHA256                      Optional bool. If True, the organization signs the
                                        request using the SHA-256 hash function. This is
                                        used when support_signed_request is True.
                                        The default is False.
        -------------------------       --------------------------------------------------
        support_logout_request          Optional bool. If True, signing out of the organization
                                        propagates logout of the IDP. The default is False.
        -------------------------       --------------------------------------------------
        update_profile_at_signin        Optional bool. If True, automatically syncs user
                                        account information (that is, full name and email address)
                                        stored in your organization with the information
                                        received from the IDP. The default is False.
        -------------------------       --------------------------------------------------
        update_groups_at_signin         Optional bool. If True, enables SAML-based group
                                        membership that allows organization members to link
                                        specified SAML-based enterprise groups to your
                                        organization's groups during group creation.
                                        The default is False.
        =========================       ==================================================

        :return: Dictionary json response indicating success and IDP Id
        """
        url = self._portal.resturl + "portals/self/idp/register"
        params = {
            "f": "json",
            "name": name,
            "idpMetadataFile": metadata_file,
            "idpMetadataUrl": metadata_url,
            "bindingUrl": binding_url,
            "postBindingUrl": post_binding_url,
            "logoutUrl": logout_url,
            "entityId": entity_id,
            "signUpMode": signup_mode,
            "encryptionSupported": encryption_supported,
            "supportSignedRequest": support_signed_request,
            "useSHA256": use_SHA256,
            "supportsLogoutRequest": support_logout_request,
            "updateProfileAtSignin": update_profile_at_signin,
            "updateGroupsAtSignin": update_groups_at_signin,
        }

        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def unregister_idp(self, idp: str):
        """
        The unregister IDP operation (POST only) allows organization
        administrator to remove the enterprise login set up with a single identity provider.

        ==================      =======================================
        **Parameter**            **Description**
        ------------------      ---------------------------------------
        idp                     The idp id to unregister.
        ==================      =======================================

        :return: Json dictionary response indicating success.
        """
        url = self._portal.resturl + "portals/self/idp/{idp}/register"
        return self._gis._con.post(url, {"f": "json"})

    # ----------------------------------------------------------------------
    def get_idp(self, idp: str | None = None):
        """
        List organization identity federation information configured using a
        single identity provider such as Active Directory Federation
        Services (ADFS) 2.0 and later, Okta, NetIQ Access Manager 3.2
        and later, OpenAM 10.1.0 and later, Shibboleth 3.2 and later, etc.

        ArcGIS Online Only.

        ==================      =======================================
        **Parameter**            **Description**
        ------------------      ---------------------------------------
        idp                     The idp to get. If none provided, all
                                available are returned.
        ==================      =======================================

        :return: Json Dictionary response
        """
        if self._gis._is_agol:
            if idp:
                url = self._portal.resturl + "portals/self/idp/{idp}"
            else:
                url = self._portal.resturl + "portals/self/idp"
            return self._gis._con.post(url, {"f": "json"})
        else:
            return None
