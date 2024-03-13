import json
import datetime
import mimetypes
from urllib.parse import urlparse
from arcgis import env
from arcgis.gis import GIS
from arcgis.gis import Item
from ._ref import reference
from arcgis._impl.common._deprecate import deprecated


@deprecated(deprecated_in="2.0.0", removed_in=None, current_version="2.0.1")
class JournalStoryMap(object):
    """
    Represents a Journal Story Map

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Optional Item. The storymap item.
    ---------------     --------------------------------------------------------------------
    gis                 Optional GIS. The connection to the Enterprise.
    ===============     ====================================================================

    """

    _properties = None
    _gis = None
    _itemid = None
    _item = None

    def __init__(self, item=None, gis=None):
        """initializer"""
        if gis is None:
            self._gis = env.active_gis
        else:
            self._gis = gis
        if item and isinstance(item, str):
            self._item = gis.content.get(item)
            self._itemid = self._item.itemid
            self._properties = self._item.get_data()
        elif item and isinstance(item, Item) and "MapJournal" in item.typeKeywords:
            self._item = item
            self._itemid = self._item.itemid
            self._properties = self._item.get_data()
        elif item and isinstance(item, Item) and "MapJournal" not in item.typeKeywords:
            raise ValueError("Item is not a Journal Story Map")
        else:
            self._properties = reference["journal"]

    # ----------------------------------------------------------------------
    def __str__(self):
        return json.dumps(self._properties)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _refresh(self):
        if self._item:
            self._properties = json.loads(self._item.get_data())

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the storymap's JSON"""
        return self._properties

    # ----------------------------------------------------------------------
    def add(
        self,
        title,
        url_or_item,
        content=None,
        actions=None,
        visible=True,
        alt_text="",
        display="stretch",
        **kwargs,
    ):
        """
        Adds a new section to the StoryMap

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The title of the section.
        ---------------     --------------------------------------------------------------------
        url_or_item         Required string/Item. The web address to the resource or a Web Map
                            item.
        ---------------     --------------------------------------------------------------------
        content             Optional string. The content of the section.
        ---------------     --------------------------------------------------------------------
        actions             Optional list. A collection of actions performed on the section
        ---------------     --------------------------------------------------------------------
        visible             Optional boolean. If True, the section is visible on publish. If
                            False, the section is not displayed.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional string. Specifies an alternate text for an image.
        ---------------     --------------------------------------------------------------------
        display             Optional string. The image display properties.
        ===============     ====================================================================


        **WebMap Options**

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        show_legend              Optional boolean. If True, the legend will be visible.
        --------------------     --------------------------------------------------------------------
        show_default_legend      Optional boolean. Shows the legend on default.
        --------------------     --------------------------------------------------------------------
        extent                   Optional dict/Envelope. The extent of the webmap.
        --------------------     --------------------------------------------------------------------
        layer_visibility         Optional list. The visibility of the layers in a webmap.  This is a
                                 list of dictionaries where the syntax is as follows:

                                 Syntax:

                                 [
                                    {
                                       "id" : "<id>",
                                       "visibility" : "<true/false>"
                                    }
                                 ]

                                 Example:

                                 [
                                 {
                                    "id" : "csv_6005_0",
                                    "visibility" : False,
                                 },
                                 {
                                    "id" : "csv_6006_0",
                                    "visibility" : True,
                                 }
                                 ]
        --------------------     --------------------------------------------------------------------
        popup                    Optional dict. The popup definition for the webmap.
        ====================     ====================================================================



        :return: Boolean


        """
        if isinstance(url_or_item, Item):
            show_legend = kwargs.pop("show_legend", False)
            show_default_legend = kwargs.pop("show_default_legend", False)
            extent = kwargs.pop("extent", None)
            layer_visibility = kwargs.pop("layer_visibility", None)
            popup = kwargs.pop("popup", None)
            if layer_visibility:
                layer_visibility = json.dumps(layer_visibility)
            return self._add_webmap(
                item=url_or_item,
                title=title,
                content=content,
                actions=actions,
                visible=visible,
                alt_text=alt_text,
                display=display,
                show_legend=show_legend,
                show_default_legend=show_default_legend,
                extent=extent,
                layer_visibility=layer_visibility,
                popup=popup,
            )
        elif isinstance(url_or_item, str):
            mt = mimetypes.guess_type(url=url_or_item)
            if mt[0].lower().find("video") > -1:
                return self._add_video(
                    url=url_or_item,
                    title=title,
                    content=content,
                    actions=actions,
                    visible=visible,
                    alt_text=alt_text,
                    display=display,
                )
            elif mt[0].lower().find("image") > -1:
                return self._add_image(
                    title=title,
                    image=url_or_item,
                    content=content,
                    actions=actions,
                    visible=visible,
                    alt_text=alt_text,
                    display=display,
                )
            else:
                return self._add_webpage(
                    title=title,
                    url=url_or_item,
                    content=content,
                    actions=actions,
                    visible=visible,
                    alt_text=alt_text,
                    display=display,
                )
        return False
        # ----------------------------------------------------------------------

    def _add_webpage(
        self,
        title,
        url,
        content=None,
        actions=None,
        visible=True,
        alt_text="",
        display="stretch",
    ):
        """
        Adds a webpage to the storymap

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The title of the section.
        ---------------     --------------------------------------------------------------------
        url                 Required string. The web address of the webpage
        ---------------     --------------------------------------------------------------------
        content             Optional string. The content of the section.
        ---------------     --------------------------------------------------------------------
        actions             Optional list. A collection of actions performed on the section
        ---------------     --------------------------------------------------------------------
        visible             Optional boolean. If True, the section is visible on publish. If
                            False, the section is not displayed.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional string. Specifies an alternate text for an image.
        ---------------     --------------------------------------------------------------------
        display             Optional string. The image display properties.
        ===============     ====================================================================


        :return: Boolean

        """
        if actions is None:
            actions = []
        if visible:
            visible = "PUBLISHED"
        else:
            visible = "HIDDEN"
        self._properties["values"]["story"]["sections"].append(
            {
                "title": title,
                "content": content,
                "contentActions": actions,
                "creaDate": int(datetime.datetime.now().timestamp() * 1000),
                "pubDate": int(datetime.datetime.now().timestamp() * 1000),
                "status": visible,
                "media": {
                    "type": "webpage",
                    "webpage": {
                        "url": url,
                        "type": "webpage",
                        "altText": alt_text,
                        "display": display,
                        "unload": True,
                        "hash": "5",
                    },
                },
            }
        )
        return True

    # ----------------------------------------------------------------------
    def _add_video(
        self,
        url,
        title,
        content,
        actions=None,
        visible=True,
        alt_text="",
        display="stretch",
    ):
        """
        Adds a video section to the StoryMap.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The title of the section.
        ---------------     --------------------------------------------------------------------
        url                 Required string. The web address of the image
        ---------------     --------------------------------------------------------------------
        content             Optional string. The content of the section.
        ---------------     --------------------------------------------------------------------
        actions             Optional list. A collection of actions performed on the section
        ---------------     --------------------------------------------------------------------
        visible             Optional boolean. If True, the section is visible on publish. If
                            False, the section is not displayed.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional string. Specifies an alternate text for an image.
        ---------------     --------------------------------------------------------------------
        display             Optional string. The image display properties.
        ===============     ====================================================================


        :return: Boolean

        """
        if actions is None:
            actions = []
        if visible:
            visible = "PUBLISHED"
        else:
            visible = "HIDDEN"
        video = {
            "title": title,
            "content": content,
            "contentActions": actions,
            "creaDate": 1523450612336,
            "pubDate": 1523450580000,
            "status": visible,
            "media": {
                "type": "video",
                "video": {
                    "url": url,
                    "type": "video",
                    "altText": alt_text,
                    "display": display,
                },
            },
        }
        self._properties["values"]["story"]["sections"].append(video)
        return True

    # ----------------------------------------------------------------------
    def _add_webmap(
        self,
        item,
        title,
        content,
        actions=None,
        visible=True,
        alt_text="",
        display="stretch",
        show_legend=False,
        show_default_legend=False,
        extent=None,
        layer_visibility=None,
        popup=None,
    ):
        """
        Adds a WebMap to the Section.


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        item                Required string/Item. The webmap Item Id or Item of a webmap.
        ---------------     --------------------------------------------------------------------
        title               Required string. The title of the section.
        ---------------     --------------------------------------------------------------------
        url                 Required string. The web address of the image
        ---------------     --------------------------------------------------------------------
        content             Optional string. The content of the section.
        ---------------     --------------------------------------------------------------------
        actions             Optional list. A collection of actions performed on the section
        ---------------     --------------------------------------------------------------------
        visible             Optional boolean. If True, the section is visible on publish. If
                            False, the section is not displayed.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional string. Specifies an alternate text for an image.
        ---------------     --------------------------------------------------------------------
        display             Optional string. The image display properties.
        ===============     ====================================================================


        :return: Boolean

        """
        if isinstance(item, Item):
            item = item.itemid

        if actions is None:
            actions = []
        if visible:
            visible = "PUBLISHED"
        else:
            visible = "HIDDEN"
        wm = {
            "title": title,
            "content": content,
            "contentActions": actions,
            "creaDate": int(datetime.datetime.now().timestamp() * 1000),
            "pubDate": int(datetime.datetime.now().timestamp() * 1000),
            "status": visible,
            "media": {
                "type": "webmap",
                "webmap": {
                    "id": item,
                    "extent": extent,
                    "layers": layer_visibility,
                    "popup": popup,
                    "overview": {"enable": False, "openByDefault": True},
                    "legend": {
                        "enable": show_legend,
                        "openByDefault": show_default_legend,
                    },
                    "geocoder": {"enable": False},
                    "altText": alt_text,
                },
            },
        }
        self._properties["values"]["story"]["sections"].append(wm)
        return True

    # ----------------------------------------------------------------------
    def _add_image(
        self,
        title,
        image,
        content=None,
        actions=None,
        visible=True,
        alt_text=None,
        display="fill",
    ):
        """
        Adds a new image section to the storymap


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Required string. The title of the section.
        ---------------     --------------------------------------------------------------------
        url                 Required string. The web address of the image
        ---------------     --------------------------------------------------------------------
        content             Optional string. The content of the section.
        ---------------     --------------------------------------------------------------------
        actions             Optional list. A collection of actions performed on the section
        ---------------     --------------------------------------------------------------------
        visible             Optional boolean. If True, the section is visible on publish. If
                            False, the section is not displayed.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional string. Specifies an alternate text for an image.
        ---------------     --------------------------------------------------------------------
        display             Optional string. The image display properties.
        ===============     ====================================================================


        :return: Boolean

        """
        if actions is None:
            actions = []
        if visible:
            visible = "PUBLISHED"
        else:
            visible = "HIDDEN"
        self._properties["values"]["story"]["sections"].append(
            {
                "title": title,
                "content": content,
                "contentActions": actions,
                "creaDate": int(datetime.datetime.now().timestamp() * 1000),
                "pubDate": int(datetime.datetime.now().timestamp() * 1000),
                "status": visible,
                "media": {
                    "type": "image",
                    "image": {
                        "url": image,
                        "type": "image",
                        "altText": alt_text,
                        "display": display,
                    },
                },
            }
        )
        return True

    # ----------------------------------------------------------------------
    def remove(self, index):
        """
        Removes a section by index.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required integer. The position of the section to remove.
        ===============     ====================================================================


        :return: Boolean

        """
        try:
            item = self._properties["values"]["story"]["sections"][index]
            self._properties["values"]["story"]["sections"].remove(item)
            return True
        except:
            return False

    # ----------------------------------------------------------------------
    def save(self, title=None, tags=None, description=None):
        """
        Saves an Journal StoryMap to the GIS


        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The title of the StoryMap.
        ---------------     --------------------------------------------------------------------
        tags                Optional string. The tags of the StoryMap.
        ---------------     --------------------------------------------------------------------
        description         Optional string. The description of the StoryMap
        ===============     ====================================================================


        :return: Boolean

        """
        import uuid

        if self._item:
            p = {"text": json.dumps(self._properties)}
            if title:
                p["title"] = title
            if tags:
                p["tags"] = tags
            return self._item.update(item_properties=p)
        else:
            if title is None:
                title = "Map Journal, %s" % uuid.uuid4().hex[:10]
            if tags is None:
                tags = "Story Map,Map Journal"
            typeKeywords = ",".join(
                [
                    "JavaScript",
                    "layout-side",
                    "Map",
                    "MapJournal",
                    "Mapping Site",
                    "Online Map",
                    "Ready To Use",
                    "selfConfigured",
                    "Story Map",
                    "Story Maps",
                    "Web Map",
                ]
            )
            item = self._gis.content.add(
                item_properties={
                    "title": title,
                    "tags": tags,
                    "text": json.dumps(self._properties),
                    "typeKeywords": typeKeywords,
                    "itemType": "text",
                    "type": "Web Mapping Application",
                }
            )
            parse = urlparse(self._gis._con.baseurl)
            isinstance(self._gis, GIS)
            if self._gis._portal.is_arcgisonline:
                url = "%s://%s/apps/MapJournal/index.html?appid=%s" % (
                    parse.scheme,
                    parse.netloc,
                    item.itemid,
                )
            else:
                import os

                wa = os.path.dirname(parse.path[1:])
                url = "%s://%s/%s/sharing/rest/apps/MapJournal/index.html?appid=%s" % (
                    parse.scheme,
                    parse.netloc,
                    wa,
                    item.itemid,
                )
            return item.update(item_properties={"url": url})
        return False

    # ----------------------------------------------------------------------
    def delete(self):
        """Deletes the saved item on ArcGIS Online/Portal"""
        if self._item:
            return self._item.delete()
        return False

    # ----------------------------------------------------------------------
    @property
    def panel(self):
        """
        Gets/Sets the panel state for the Journal Story Map
        """
        return self._properties["values"]["settings"]["layout"]["id"]

    # ----------------------------------------------------------------------
    @panel.setter
    def panel(self, value):
        """
        Gets/Sets the panel state for the Journal Story Map
        """
        if value.lower() == "float":
            self._properties["values"]["settings"]["layout"]["id"] = "float"
        else:
            self._properties["values"]["settings"]["layout"]["id"] = "side"

    # ----------------------------------------------------------------------
    @property
    def header(self):
        """gets/sets the headers for the Journal StoryMap"""
        default = {
            "social": {"bitly": True, "twitter": True, "facebook": True},
            "logoURL": None,
            "linkURL": "https://storymaps.arcgis.com",
            "logoTarget": "",
            "linkText": "A Story Map",
        }
        if "header" in self._properties["values"]["settings"]:
            return self._properties["values"]["settings"]["header"]
        else:
            self._properties["values"]["settings"]["header"] = default
            return default

    # ----------------------------------------------------------------------
    @header.setter
    def header(self, value):
        """ """
        if value is None:
            default = {
                "social": {"bitly": True, "twitter": True, "facebook": True},
                "logoURL": None,
                "linkURL": "https://storymaps.arcgis.com",
                "logoTarget": "",
                "linkText": "A Story Map",
            }
            self._properties["values"]["settings"]["header"] = default
        else:
            self._properties["values"]["settings"]["header"] = value

    # ----------------------------------------------------------------------
    @property
    def theme(self):
        """ """
        default = {
            "colors": {
                "text": "#FFFFFF",
                "name": "float-default-1",
                "softText": "#FFF",
                "media": "#a0a0a0",
                "themeMajor": "black",
                "panel": "#000000",
                "textLink": "#DDD",
                "esriLogo": "white",
                "dotNav": "#000000",
                "softBtn": "#AAA",
            },
            "fonts": {
                "sectionTitle": {
                    "value": "font-family:'open_sansregular', sans-serif;",
                    "id": "default",
                },
                "sectionContent": {
                    "value": "font-family:'open_sansregular', sans-serif;",
                    "id": "default",
                },
            },
        }
        if "theme" in self._properties["values"]["settings"]:
            return self._properties["values"]["settings"]["theme"]
        else:
            self._properties["values"]["settings"]["theme"] = default
            return self._properties["values"]["settings"]["theme"]
        return default

    # ----------------------------------------------------------------------
    @theme.setter
    def theme(self, value):
        """ """
        default = {
            "colors": {
                "text": "#FFFFFF",
                "name": "float-default-1",
                "softText": "#FFF",
                "media": "#a0a0a0",
                "themeMajor": "black",
                "panel": "#000000",
                "textLink": "#DDD",
                "esriLogo": "white",
                "dotNav": "#000000",
                "softBtn": "#AAA",
            },
            "fonts": {
                "sectionTitle": {
                    "value": "font-family:'open_sansregular', sans-serif;",
                    "id": "default",
                },
                "sectionContent": {
                    "value": "font-family:'open_sansregular', sans-serif;",
                    "id": "default",
                },
            },
        }
        if "theme" in self._properties["values"]["settings"]:
            self._properties["values"]["settings"]["theme"] = value
        elif not "theme" in self._properties["values"]["settings"]:
            self._properties["values"]["settings"]["theme"] = value
        elif value is None:
            self._properties["values"]["settings"]["theme"] = default
