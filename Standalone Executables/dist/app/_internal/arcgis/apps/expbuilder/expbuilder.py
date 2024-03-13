from __future__ import annotations
from typing import Optional, Union
import uuid
from enum import Enum
from arcgis.auth.tools import LazyLoader
import copy
from ._ref import templates
from arcgis.gis import GIS
import re
from dataclasses import dataclass
import tempfile

try:
    import ujson as json
except ImportError:
    import json

arcgis = LazyLoader("arcgis")
# json = LazyLoader("json")
time = LazyLoader("time")


class Templates(Enum):
    BLANKFULLSCREEN = "blank fullscreen"
    BLANKSCROLLING = "blank scrolling"
    FOLDABLE = "foldable"
    LAUNCHPAD = "launchpad"
    JEWELERYBOX = "jewelrybox"
    BILLBOARD = "billboard"
    JOURNEY = "journey"
    RIBBON = "ribbon"
    GENERAL = "general"
    INTRODUCTION = "introduction"
    GALLERY = "gallery"
    EPIC = "epic"
    SNAPSHOT = "snapshot"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    SCENIC = "scenic"
    EXHIBITION = "exhibition"
    DART = "dart"
    POCKET = "pocket"
    QUICKNAVIGATION = "quick navigation"
    PARALLAX = "parallax"
    DASH = "dash"
    INDICATOR = "indicator"
    MONITOR = "monitor"
    REVEAL = "reveal"

    def preview(self, width: Optional[int] = 800, height: Optional[int] = 500):
        import threading
        import time

        def thread_delete(item):
            time.sleep(3)
            item.delete()

        try:
            temp = WebExperience(template=self.value)
            temp._item.share(everyone=True)
            temp.save(publish=True)
            from IPython.display import IFrame

            frame = IFrame(
                src=temp._item.url,
                width=width,
                height=height,
            )
            delete = threading.Thread(target=thread_delete, args=([temp._item]))
            delete.start()
            return frame

        except:
            return False


class WebExperience(object):

    """
    A Web Experience is web-based application that provides viewers with an interactive
    interface to maps, data, feature layers, and other components of the creator's design.
    Though these experiences are normally constructed via a GUI found on ArcGIS Online or
    Enterprise, this class provides users a host of supplemental options to manage experiences,
    in addition to basic creation of experiences.

    ===============     ====================================================================
    **Argument**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Optional String or Item. The string for an item id or an item of type
                        'Web Experience'. If no item is passed, a new experience is created
                        and saved to your active portal.
    ---------------     --------------------------------------------------------------------
    gis                 Optional instance of :class:`~arcgis.gis.GIS`. If none provided the active gis is used.
    ---------------     --------------------------------------------------------------------
    template            Optional string. If a new experience is being created, the template
                        used to construct the layout. If necessary and none provided, template
                        will default to `blank fullscreen`.
    ---------------     --------------------------------------------------------------------
    path                Optional string. Used if a WebExperience is being based on a local
                        config json file, as one would with Experiences made via the desktop
                        version of Experience Builder.
    ---------------     --------------------------------------------------------------------
    name                Optional string. If a new experience is being created, the name of the
                        item. Otherwise, will default to "Experience via Python" followed by a
                        random number.
    ===============     ====================================================================
    """

    _draft = {}
    _itemid = None
    _item = None
    _local = False
    _properties = None
    _gis = None
    _resources = None
    _expdict = {}

    def __init__(
        self,
        item: Optional[Union[arcgis.gis.Item, str]] = None,
        path: Optional[str] = None,
        gis: Optional[arcgis.gis.GIS] = None,
        template: Optional[Union[Templates, str]] = None,
        name: Optional[str] = None,
    ):
        if gis is None:
            if item and isinstance(item, arcgis.gis.Item):
                self._gis = item._gis
            else:
                self._gis = arcgis.env.active_gis
        else:
            if item and isinstance(item, arcgis.gis.Item):
                if item._gis != gis:
                    raise ValueError("Provided GIS must match item GIS")
            self._gis = gis

        if self._gis is None or self._gis._portal.is_logged_in is False:
            # check to see if user is authenticated
            raise AttributeError(
                "Must be logged into a Portal Account or provide a logged-in GIS object"
            )
        if item and isinstance(item, str):
            # get item using the item id
            item = self._gis.content.get(item)
            if item is None:
                raise ValueError("Item is not accessible with provided GIS")
        if item and isinstance(item, arcgis.gis.Item) and item.type == "Web Experience":
            # set item properties
            self._item = item
            self._itemid = self._item.itemid
            self._resources = self._item.resources.list()
            self._expdict = self._item.resources.get("config/config.json")
            self._draft = self._item.resources.get("config/config.json")
            self._gis = self._item._gis
        elif path and isinstance(path, str):
            # construct web experience from path's config file
            # note that this will not have an associated portal item/itemid/resources
            self._local = True
            with open(path) as json_file:
                config = json.load(json_file)
                self._draft = config
                self._expdict = config
        elif (
            item and isinstance(item, arcgis.gis.Item) and item.type != "Web Experience"
        ):
            # Throw error if item is not of type Experience
            raise ValueError("Item is not a Web Experience or is inaccesible")
        else:
            self._create_new_experience(template=template, name=name)

    # -----------------------------------------------------------------------------------
    @property
    def item(self):
        """
        Returns the portal item associated with the WebExperience, if possible.
        Experiences made from local json files won't have an `item` property until they
        are added to a portal using `upload()`.
        """
        return self._item

    # -----------------------------------------------------------------------------------
    @property
    def itemid(self):
        """
        Returns the item ID of the associated portal item, if possible. As with `item`,
        Experiences made from local json files won't have an `itemid` property until they
        are added to a portal using `upload()`.
        """
        return self._itemid

    # -----------------------------------------------------------------------------------
    @property
    def datasources(self):
        """
        Shows the data sources dictionary found in the experience's draft, allowing users
        to quickly get info on all of the other items in their experience. Changing this
        dictionary will change the draft, making it convenient for remapping data
        sources.
        """
        return self._draft["dataSources"]

    # -----------------------------------------------------------------------------------
    def _create_new_experience(
        self,
        config=None,
        template="blank fullscreen",
        name=None,
        gis=None,
        item_properties={},
    ):
        """
        If no experience is specified when creating a WebExperience, this helper function
        creates a new experience and saves it as an item to the active GIS. Users can specify
        a template from the experience builder to create their template, in addition to a custom
        item name (done as arguments in the initial creation of the WebExperience).
        """

        if config is None:
            if isinstance(template, Templates):
                template = template.value

            # retrieve template for experience
            if template is None:
                template = "blank fullscreen"

            temp_low = template.lower()
            if temp_low in arcgis.apps.expbuilder._ref.templates:
                temp_dict = copy.deepcopy(
                    arcgis.apps.expbuilder._ref.templates[temp_low]
                )
            else:
                temp_dict = copy.deepcopy(
                    arcgis.apps.expbuilder._ref.templates["blank fullscreen"]
                )

            temp_dict["attributes"]["portalUrl"] = self._gis.url
            # temp_dict["timestamp"]
            # create item and generate basic properties
            if name is None:
                title = "Experience via Python %s" % uuid.uuid4().hex[:10]
            else:
                title = name

        else:
            temp_dict = config
            temp_dict["attributes"]["portalUrl"] = self._gis.url
            if name is None:
                title = "Experience via Python %s" % uuid.uuid4().hex[:10]
            else:
                title = name

        if gis:
            temp_dict["attributes"]["portalUrl"] = gis.url
        else:
            temp_dict["attributes"]["portalUrl"] = self._gis.url

        keywords = ",".join(
            [
                "EXB Experience",
                "JavaScript",
                "Ready To Use",
                "status: Draft",
                "Web Application",
                "Web Experience",
                "Web Page",
                "Web Site",
                "expbuilderapp:python-api-" + arcgis.__version__,
            ]
        )
        props = item_properties
        props["title"] = title
        props["type"] = "Web Experience"
        props["typeKeywords"] = keywords

        """item_properties = {
            "type": "Web Experience",
            "title": title,
            "typeKeywords": keywords,
        }"""

        # add to active gis and set properties
        if gis is None:
            item = self._gis.content.add(item_properties=props)
        else:
            item = gis.content.add(item_properties=props)

        # assign to experience properties
        self._item = item
        self._itemid = item.itemid
        self._item.resources.add(
            folder_name="config", file_name="config.json", text=temp_dict
        )
        self._resources = self._item.resources.list()
        self._expdict = temp_dict
        self._draft = temp_dict

    # ----------------------------------------------------------------------
    def save(
        self,
        title: Optional[str] = None,
        tags: Optional[list] = None,
        access: str = None,
        publish: bool = False,
        duplicate: bool = False,
        include_private: Optional[bool] = None,
        item_properties: Optional[dict] = {},
    ):
        """
        This method will save your Web Experience to your active GIS. The experience will be saved
        with unpublished changes unless the `publish` parameter is set to True. Note that this is
        different from the `publish()` method in that this will save and publish the unsaved draft
        of the WebExperience object, as opposed to the already existing save state.

        The title only needs to be specified if a change is wanted, otherwise the existing title
        is used.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The new title of the WebExperience, if desired.
        ---------------     --------------------------------------------------------------------
        tags                Optional string. Updated tags for the WebExperience, if desired.
        ---------------     --------------------------------------------------------------------
        access              Optional string. The sharing setting of the WebExperience. If none
                            is specified, the current access is kept. This is used when the
                            `publish` parameter is set to True.

                            Values: `private` | `org` | `public`
        ---------------     --------------------------------------------------------------------
        publish             Optional boolean. If True, the experience is saved and also
                            published. Default is False, meaning the experience is saved with
                            unpublished changes.
        ---------------     --------------------------------------------------------------------
        duplicate           Optional boolean. If True, the experience is duplicated and a new
                            WebExperience object is saved with any specified changes to
                            title, tags, access, or publish included. Essentially functions as a
                            "Save As" method. Default is False, meaning changes are saved to the
                            original object.
        ---------------     --------------------------------------------------------------------
        include_private     Optional boolean. Only to be included when duplicate is `True`.
                            If True, the private resources of the original item will be included
                            in the new item.
        ---------------     --------------------------------------------------------------------
        item_properties     Optional dictionary. Contains a variety of properties that can be
                            set when creating a new item, much like `item.update()`. See below
                            for a table containing possible properties.
        ===============     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*

        =================  =====================================================================
        **Key**            **Value**
        -----------------  ---------------------------------------------------------------------
        description        Optional string. Description of the item.
        -----------------  ---------------------------------------------------------------------
        url                Optional string. URL to item that are based on URLs.
        -----------------  ---------------------------------------------------------------------
        snippet            Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        -----------------  ---------------------------------------------------------------------
        accessInformation  Optional string. Information on the source of the content.
        -----------------  ---------------------------------------------------------------------
        licenseInfo        Optional string.  Any license information or restrictions regarding the content.
        -----------------  ---------------------------------------------------------------------
        culture            Optional string. Locale, country and language information.
        =================  =====================================================================


        :return: A boolean indicating the success of the operation.
        """
        if duplicate:
            new_exp = self._duplicate(
                title=title, tags=tags, include_private=include_private
            )
            new_exp.save(access=access, publish=publish)
            return new_exp

        keywords = self._item.typeKeywords
        for i in range(len(keywords)):
            if keywords[i] == "status: Published":
                keywords[i] = "status: Changed"
        props = item_properties
        if title:
            props["title"] = title
        if tags:
            props["tags"] = tags

        self._expdict = self._draft
        # Create a temporary file and write data to it
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tfile:
            json.dump(self._expdict, tfile)
            # Close the file explicitly
            tfile.close()
        self._item.resources.update(
            folder_name="config",
            file_name="config.json",
            file=tfile.name,
        )
        self._resources = self._item.resources.list()
        if publish:
            for i in range(len(keywords)):
                if "status" in keywords[i]:
                    keywords[i] = "status: Published"
            if access:
                props["access"] = access
            props["typeKeywords"] = keywords
            if self._gis._is_agol:
                url = "https://experience.arcgis.com/experience/" + self._item.itemid
            else:
                url = (
                    self._gis.url
                    + "/apps/experiencebuilder/experience/?id="
                    + self._item.itemid
                )
            props["url"] = url
            return self._item.update(item_properties=props, data=self._expdict)
            # self._publish(item_properties = item_properties, data = self._expdict)
        else:
            props["typeKeywords"] = keywords
            return self._item.update(item_properties=props)

    # ----------------------------------------------------------------------
    def reload(self):
        """
        Resets any changes that the user has made to the last saved state. Note that
        this only applies to changes made through a Python API object, and not the GUI.

        :return: A boolean indicating the success of the operation.
        """

        self._draft = self._expdict
        # Create a temporary file and write data to it
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tfile:
            json.dump(self._expdict, tfile)
            # Close the file explicitly
            tfile.close()
        return self._item.resources.update(
            folder_name="config", file_name="config.json", file=tfile.name
        )

    # ----------------------------------------------------------------------
    def view(self, width: Optional[int] = 800, height: Optional[int] = 500):
        """
        Shows the currently published experience, if possible. Default width is 800 and default
        height is 500. Note that this displays the actively published version of the
        WebExperience object; to visualize unsaved changes, `preview()` should be used.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        width               Optional integer. The desired width to show the preview.
        ---------------     --------------------------------------------------------------------
        height              Optional integer. The desired height to show the preview.
        ===============     ====================================================================

        .. note::

            In some cases, a dialogue box may pop up asking for credentials when calling this
            method. If the preview isn't rendering, check if pop-ups are disabled in your browser.


        :return:
            An IFrame display of the Experience if possible, else the item url is returned to be
            clicked on. If the item is unpublished, the function returns False.
        """
        keywords = self._item.typeKeywords
        if "status: Published" in keywords:
            from IPython.display import IFrame

            try:
                frame = IFrame(
                    src=self._item.url,
                    width=width,
                    height=height,
                )
                return frame
            except:
                return self._item.url
        else:
            return False

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Deletes the experience and its associated item from the portal.

        :return:
            A boolean indicating the success of the operation.
        """
        item = self._gis.content.get(self._itemid)
        return item.delete()

    # ----------------------------------------------------------------------
    def _add_resource(self, file=None, resource_name=None, text=None, access="inherit"):
        """
        See :class:`~arcgis.gis.ResourceManager`
        """
        resource_manager = arcgis.gis.ResourceManager(self._item, self._gis)
        is_present = False
        if file:
            for resource in self._resources:
                if resource["resource"] in file:
                    is_present = True
                    resp = True
        properties = {
            "editInfo": {
                "editor": self._gis._username,
                "modified": str(int(time.time() * 1000)),
                "id": uuid.uuid4().hex[0:21],
                "app": "python-api",
            }
        }

        # access is inherited from item upon add, except for json where always private
        if is_present is False:
            resp = resource_manager.add(
                file=file,
                file_name=resource_name,
                text=text,
                access=access,
                properties=properties,
            )

        self._resources = self._item.resources.list()
        return resp

    # ----------------------------------------------------------------------
    def _duplicate(
        self,
        title: Optional[str] = None,
        tags: Optional[Union[list[str], str]] = None,
        include_private: Optional[bool] = None,
    ):
        """
        Creates a copy of the experience within the active GIS. Returns a new
        WebExperience object that retains the unsaved changes from the original.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The name of the new experience. If left blank, the
                            new item copies the name of the original.
        ---------------     --------------------------------------------------------------------
        tags                Optional string. The desired tags for the new item, separated by
                            commas.
        ---------------     --------------------------------------------------------------------
        include_private     Optional boolean. If True, the private resources of the original
                            item will be included in the new item.
        ===============     ====================================================================

        :return:
            The newly created WebExperience object.

        """
        new_item = self._item.copy_item(
            title=title,
            tags=tags,
            include_resources=True,
            include_private=include_private,
        )
        if new_item:
            new_exp = WebExperience(new_item)
            new_exp._draft = self._draft
            return new_exp
        else:
            return False

    # ----------------------------------------------------------------------
    def upload(
        self,
        gis: Optional[GIS] = None,
        publish=False,
        title=None,
        item_mapping: Optional[dict] = None,
        auto_remap: Optional[bool] = False,
        item_properties: Optional[dict] = {},
    ):
        """
        Adds a WebExperience created locally through the Developer Edition to a specified
        portal. After doing this, the WebExperience object will obtain item and itemid
        properties. Gives users options to remap their experience's datasources to items
        present in the portal, both manually or automatically

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        gis                 Optional GIS object. The portal to add the WebExperience to. If none
                            is passed in, will default to the GIS of the WebExperience object.
        ---------------     --------------------------------------------------------------------
        publish             Optional boolean. Publishes the experience when adding it to the
                            portal. Default is `False`.
        ---------------     --------------------------------------------------------------------
        title               Optional string. Allows a user to specify the title of their new
                            experience in the portal.
        ---------------     --------------------------------------------------------------------
        item_mapping        Optional dictionary. Allows users to manually remap the datasources
                            of their experience to datasources present in the portal. See
                            example dictionary below.
        ---------------     --------------------------------------------------------------------
        auto_remap          Optional boolean. Searches the portal for matching datasources and
                            automatically remaps the experience to use those accordingly.
                            Default is `False`.
        ---------------     --------------------------------------------------------------------
        item_properties     Optional dictionary. Contains a variety of properties that can be
                            set when creating a new item, much like `ContentManager.add()`. See
                            below for a table containing possible properties.
        ===============     ====================================================================


        *Key:Value Dictionary Options for Argument item_properties*

        ==========================  =====================================================================
        **Key**                     **Value**
        --------------------------  ---------------------------------------------------------------------
        description                 Optional string. Description of the item.
        --------------------------  ---------------------------------------------------------------------
        url                         Optional string. URL to item that are based on URLs.
        --------------------------  ---------------------------------------------------------------------
        tags                        Optional string. Tags listed as comma-separated values, or a list of strings.
                                    Used for searches on items.
        --------------------------  ---------------------------------------------------------------------
        snippet                     Optional string. Provide a short summary (limit to max 250 characters) of the what the item is.
        --------------------------  ---------------------------------------------------------------------
        accessInformation           Optional string. Information on the source of the content.
        --------------------------  ---------------------------------------------------------------------
        licenseInfo                 Optional string.  Any license information or restrictions regarding the content.
        --------------------------  ---------------------------------------------------------------------
        culture                     Optional string. Locale, country and language information.
        --------------------------  ---------------------------------------------------------------------
        commentsEnabled             Optional boolean. Default is true, controls whether comments are allowed (true)
                                    or not allowed (false).
        --------------------------  ---------------------------------------------------------------------
        access                      Optional string. Valid values are private, org, or public. Defaults to private.
        --------------------------  ---------------------------------------------------------------------
        overwrite                   Optional boolean. Default is `false`. Controls whether item can be overwritten.
        ==========================  =====================================================================

        :return:
            The newly added portal item, if successful. Otherwise, returns False.
        """
        if gis is None:
            gis = self._gis
        # this will be the config file that dictates our new portal experience
        new_config = self._draft
        # automatically remap data to matching sources in target GIS
        if auto_remap:
            sources = new_config["dataSources"]
            for source in sources:
                # first, see if we can already access each one anonymously
                # or through the passed in GIS
                url = sources[source]["portalUrl"]
                test_gis = GIS(url=url)
                try:
                    try:
                        targ_item = test_gis.content.get(sources[source]["itemId"])
                    except:
                        targ_item = gis.content.get(sources[source]["itemId"])
                    assert targ_item
                # this will only get triggered if we can't access and need to remap
                except:
                    # if available, match item in target GIS based on title and type
                    source_title = sources[source]["sourceLabel"]
                    source_type = sources[source]["type"]
                    source_reg = re.sub("[^A-Za-z0-9]+", "", source_type).lower()
                    query = 'title:"' + source_title + '"'

                    for item in gis.content.search(query=query):
                        target_title = item.title
                        target_type = item.type
                        target_reg = re.sub("[^A-Za-z0-9]+", "", target_type).lower()

                        # must be exact match on both
                        if source_title == target_title and source_reg == target_reg:
                            new_config["dataSources"][source]["itemId"] = item.id
                            new_config["dataSources"][source]["portalUrl"] = gis.url
                            break

        # if user passed in their own custom remapping, use that
        if item_mapping is not None:
            for source, v in item_mapping.items():
                for attr, new_value in v.items():
                    new_config["dataSources"][source][attr] = new_value

        # create a new portal experience using the config
        self._create_new_experience(
            config=new_config, name=title, gis=gis, item_properties=item_properties
        )
        self._local = False

        # if wish for item to be published, save/publish
        if publish:
            self.save(publish=True)

        if self._item:
            return self._item
        else:
            return False

    # ----------------------------------------------------------------------
    def preview(self, width: Optional[int] = 800, height: Optional[int] = 500):
        """
        Show a preview of the current experience draft. The default is a width of 800 and height of 500.
        Note that this should be used to visualize unsaved changes to the WebExperience object; to see
        the actively published version of the object, `view()` should be used.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        width               Optional integer. The desired width to show the preview.
        ---------------     --------------------------------------------------------------------
        height              Optional integer. The desired height to show the preview.
        ===============     ====================================================================

        .. note::

            In some cases, a dialogue box may pop up asking for credentials when calling this
            method. If the preview isn't rendering, check if pop-ups are disabled in your browser.


        :return:
            An IFrame display of the Experience if possible, else the item url is returned to be
            clicked on.
        """
        import threading
        import time

        def thread_delete(item):
            time.sleep(3)
            item.delete()

        try:
            dummy_exp = self._duplicate()
            dummy_exp._item.share(everyone=True)
            dummy_exp.save(publish=True)
            from IPython.display import IFrame

            frame = IFrame(
                src=dummy_exp._item.url,
                width=width,
                height=height,
            )
            delete = threading.Thread(target=thread_delete, args=([dummy_exp._item]))
            delete.start()
            return frame

        except:
            return self._item.url

    # ----------------------------------------------------------------------
    def clone(self, target, owner, **kwargs):
        """
        Clones the experience and all of it's data sources to a target GIS. User must
        have admin privileges on the original item's GIS, and provide an authenticated
        instance of a target GIS. Users must also specify the name of an account on the
        target GIS to own the items. Also accepts arguments for :class:`~arcgis.gis.Item.clone_items()`

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        target              Required GIS. An authenticated instance of the GIS that the user
                            wishes to clone the experience to.
        ---------------     --------------------------------------------------------------------
        owner               Required string. The username of the account that will be the owner
                            of the experience and its data source items in the target GIS.
        ---------------     --------------------------------------------------------------------
        **kwargs            Optional additional arguments. See ``Item.clone_items()`` for the full
                            list.
        ===============     ====================================================================

        :return:
            The item corresponding to the cloned experience in the target GIS.
        """

        def _clone_dict(data_dict, source, target, owner, **kwargs):
            """
            Helper function to clone items and update appropriate dict
            """
            new_dict = data_dict
            new_dict["attributes"]["portalUrl"] = target.url
            for k, v in new_dict["dataSources"].items():
                if "itemId" not in v:
                    continue
                v["portalUrl"] = target.url
                item = source.content.get(v["itemId"])
                clone_result = target.content.clone_items([item], owner=owner, **kwargs)
                if clone_result:
                    v["itemId"] = clone_result[0].itemid
                    if "url" in v:
                        v["url"] = clone_result[0].url
                else:
                    targ_item = target.content.search(item.title)[0]
                    v["itemId"] = targ_item.itemid
                    if "url" in v:
                        v["url"] = targ_item.url

            return new_dict

        exp_clone = target.content.clone_items([self._item], owner=owner, **kwargs)
        if exp_clone:
            new_dict = _clone_dict(self._expdict, self._gis, target, owner, **kwargs)
            target_exp = WebExperience(exp_clone[0], gis=target)
            target_exp._expdict = new_dict
            # Create a temporary file and write data to it
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as tfile:
                json.dump(self._expdict, tfile)
                # Close the file explicitly
                tfile.close()
            target_exp._item.resources.update(
                folder_name="config", file_name="config.json", file=tfile.name
            )
            keywords = target_exp._item.typeKeywords
            for word in keywords:
                if "status" in word:
                    if "Published" in word or "Changed" in word:
                        new_data = _clone_dict(
                            self._item.get_data(), self._gis, target, owner, **kwargs
                        )
                        target_exp._item.update(item_properties={}, data=new_data)
                    else:
                        target_exp._item.update(
                            item_properties={}, data={"__not_publish": True}
                        )
                    break
            return target_exp._item
        else:
            return False
