from __future__ import annotations
import os
from typing import Optional, Union
import uuid
from enum import Enum
from arcgis.auth.tools import LazyLoader
import re
import copy

arcgis = LazyLoader("arcgis")
Content = LazyLoader("arcgis.apps.storymap.story_content")
json = LazyLoader("json")
time = LazyLoader("time")


class Themes(Enum):
    """
    Represents the Supported Theme Type Enumerations.
    Example: story_map.theme(Theme.Slate)
    """

    SUMMIT = "summit"
    OBSIDIAN = "obsidian"
    RIDGELINE = "ridgeline"
    MESA = "mesa"
    TIDAL = "tidal"
    SLATE = "slate"


###############################################################################################################
class StoryMap(object):
    """
    A Story Map is a web map that has been thoughtfully created, given context, and provided
    with supporting information so it becomes a stand-alone resource. It integrates maps, legends,
    text, photos, and video and provides functionality, such as swipe, pop-ups, and time sliders,
    that helps users explore this content.

    ArcGIS StoryMaps is the next-generation storytelling tool in ArcGIS, and story authors are
    encouraged to use this tool to create stories. The Python API can help you create and edit
    your stories.

    Create a Story Map object to make edits to a story. Can be created from an item of type 'Story Map',
    an item id for that type of item, or if .nothing is passed, a new story is created from a generic draft.

    If an Item or item_id is passed in, only published changes or new drafts are taken from the Story Map.
    If you have a story with unpublished changes, they will not appear when you construct your story with the API.
    If you start to work on your Story that has unpublished changes and save from the Python API, your
    unpublished changes on the GUI will be overwritten with your work from the API.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    item                Optional String or Item. The string for an item id or an item of type
                        'Story Map'. If no item is passed, a new story is created and saved to
                        your active portal.
    ---------------     --------------------------------------------------------------------
    gis                 Optional instance of :class:`~arcgis.gis.GIS` . If none provided the active gis is used.
    ===============     ====================================================================
    """

    _properties = None
    _gis = None
    _itemid = None
    _item = None
    _resources = None

    def __init__(
        self,
        item: Optional[Union[arcgis.gis.Item, str]] = None,
        gis: Optional[arcgis.gis.GIS] = None,
    ):
        # Section: Set up gis
        if gis is None:
            # If no gis, find active env
            gis = arcgis.env.active_gis
            self._gis = gis
        else:
            self._gis = gis
        if gis is None or gis._portal.is_logged_in is False:
            # Check to see if user is authenticated
            raise Exception("Must be logged into a Portal Account")

        # Section: Set up existing story
        if item and isinstance(item, str):
            # Get item using the item id
            item = gis.content.get(item)
            if item is None:
                # Error with storymap in current gis
                raise ValueError(
                    "Cannot find storymap associated with this item id in your portal. Please check it is correct."
                )
        if item and isinstance(item, arcgis.gis.Item) and item.type == "StoryMap":
            # Set item properties from existing item
            self._item = item
            self._itemid = self._item.itemid
            self._resources = self._item.resources.list()
            # Create existing story
            self._create_existing_storymap()
        elif (
            item
            and isinstance(item, arcgis.gis.Item)
            and "StoryMap" not in item.keywords
        ):
            # Throw error if item is not of type Story Map
            raise ValueError("Item is not a Story Map")
        else:
            # If no item was provided create a new story map
            self._create_new_storymap()
        # Get the story url
        self._url = self._get_url()

    # ----------------------------------------------------------------------
    def _create_existing_storymap(self):
        # Get properties from most recent resource file.
        # Can have multiple drafts so need to account for this.
        # Draft file will be of form: draft_{13 digit timestamp}.json or draft.json
        saved_drafts = []
        for resource in self._resources:
            for key, val in resource.items():
                # Find all drafts in the resources and add to a list
                if key == "resource" and (
                    re.match("draft_[0-9]{13}.json", val) or re.match("draft.json", val)
                ):
                    saved_drafts.append(val)
        # Find the correct draft to use
        if len(saved_drafts) == 1:
            # Only one draft saved
            # Open JSON draft file for properties
            data = self._item.resources.get(saved_drafts[0], try_json=True)
            self._properties = data
        elif len(saved_drafts) > 1:
            # Multiple drafts saved
            # Remove draft.json because oldest one
            if "draft.json" in saved_drafts:
                idx = saved_drafts.index("draft.json")
                del saved_drafts[idx]
            # check remaining to find most recent
            start = saved_drafts[0][6:19]  # get only timestamp
            current = saved_drafts[0]
            for draft in saved_drafts:
                compare = draft[6:19]
                if start < compare:
                    start = compare
                    current = draft
            # Open most recent JSON draft file for properties
            data = self._item.resources.get(current, try_json=True)
            self._properties = data
        else:
            # Storymap has no draft json so look for published json
            data = self._item.resources.get("published_data.json", try_json=True)
            self._properties = data

    # ----------------------------------------------------------------------
    def _create_new_storymap(self):
        # Get template from _ref folder
        template = copy.deepcopy(arcgis.apps.storymap._ref.storymap_2)
        # Add correct by-line and locale
        template["nodes"]["n-aTn8ak"]["data"]["byline"] = self._gis._username
        template["nodes"]["n-4xkUEe"]["config"]["storyLocale"] = (
            self._gis.users.me.culture if self._gis.users.me.culture else "en-US"
        )

        # Create unique story node id
        story_node = "n-" + uuid.uuid4().hex[0:6]
        template["root"] = story_node
        template["nodes"][story_node] = template["nodes"]["n-4xkUEe"]
        del template["nodes"]["n-4xkUEe"]
        # Set properties for the story
        self._properties = template
        # Create text for resource call
        text = json.dumps(template)
        # Create a temporary title
        title = "StoryMap via Python %s" % uuid.uuid4().hex[:10]
        # Create draft resource name
        draft = "draft_" + str(int(time.time() * 1000)) + ".json"
        # Will be posted as a draft
        sm_version = self._gis._con.get("https://storymaps.arcgis.com/version")[
            "version"
        ]
        keywords = ",".join(
            [
                "arcgis-storymaps",
                "StoryMap",
                "Web Application",
                "smstatusdraft",
                "smversiondraft:" + sm_version,
                "python-api",
                "smeditorapp:python-api-" + arcgis.__version__,
                "smdraftresourceid:" + draft,
            ]
        )
        # Get default thumbnail for a new item
        thumbnail = self._get_thumbnail()
        # Set the item properties dict to add new item to active gis
        item_properties = {
            "title": title,
            "keywords": keywords,
            "type": "StoryMap",
        }
        # Add item to active gis and set properties
        item = self._gis.content.add(
            item_properties=item_properties, thumbnail=thumbnail
        )
        # Assign to story properties
        self._item = item
        self._itemid = item.itemid
        # Make a resource call with the template to create json draft needed
        self._add_resource(resource_name=draft, text=text, access="private")
        # Assign resources to item
        self._resources = self._item.resources.list()

    # ----------------------------------------------------------------------
    def _repr_html_(self):
        """
        HTML Representation for IPython Notebook
        """
        return self._item._repr_html_()

    # ----------------------------------------------------------------------
    def __str__(self):
        """Return the url of the storymap"""
        return self._url

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _refresh(self):
        """Load the latest data from the item"""
        if self._item:
            self._properties = json.loads(self._item.get_data())

    # ----------------------------------------------------------------------
    def _get_url(self) -> str:
        """
        Private method to determine what the story url is. This is used to publish
        and have the correct path set.
        """
        if self._gis._is_agol:
            # Online
            self._url = "https://storymaps.arcgis.com/stories/{storyid}".format(
                storyid=self._itemid
            )
        else:
            # Enterprise
            self._url = "https://{portal}/apps/storymaps/stories/{storyid}".format(
                portal=self._gis.url, storyid=self._itemid
            )
        return self._url

    # ----------------------------------------------------------------------
    def _get_thumbnail(self) -> str:
        """
        Private method to get the default thumbnail path dependent on whether the
        user is Online or on Enterprise.
        """
        if self._gis._is_agol:
            thumbnail = "https://storymaps.arcgis.com/static/images/item-default-thumbnails/item.jpg"
        else:
            thumbnail = (
                self._gis._url
                + "/apps/storymaps/static/images/item-default-thumbnails/item.jpg"
            )
        return thumbnail

    # ----------------------------------------------------------------------
    def show(self, width: Optional[int] = None, height: Optional[int] = None):
        """
        Show a preview of the story. The default is a width of 700 and height of 300.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        width               Optional integer. The desired width to show the preview.
        ---------------     --------------------------------------------------------------------
        height              Optional integer. The desired height to show the preview.
        ===============     ====================================================================

        :return:
            An Iframe display of the story map if possible, else the item url is returned to be
            clicked on.
        """
        try:
            if self._item:
                width = 700 if width is None else width
                height = 350 if height is None else height
                from IPython.display import IFrame

                return IFrame(
                    src=self._item.url,
                    width=width,
                    height=height,
                    params="title=" + self._item.title,
                )
        except:
            return self._item.url

    # ----------------------------------------------------------------------
    @property
    def cover_date(self):
        """
        Get/Set the date type shown on the story cover.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        date_type           Optional String. Set the desired date type for the story cover.

                            ``Values: "first-published" | "last-published" | "none"``
        ===============     ====================================================================

        """
        root = self._properties["root"]
        return self._properties["nodes"][root]["config"]["coverDate"]

    # ----------------------------------------------------------------------
    @cover_date.setter
    def cover_date(self, date_type):
        """
        See cover_date property doc
        """
        # cover date is found in story node (i.e. root node id)
        root = self._properties["root"]
        self._properties["nodes"][root]["config"]["coverDate"] = date_type
        return self.cover_date

    # ----------------------------------------------------------------------
    @property
    def story_locale(self):
        """
        Get/Set the locale and language of the story.

        If your story was created with the Python API then the default is "en-US"
        """
        # story_locale is found in story node (i.e. root node id)
        root = self._properties["root"]
        return self._properties["nodes"][root]["config"]["storyLocale"]

    # ----------------------------------------------------------------------
    @story_locale.setter
    def story_locale(self, locale):
        """
        See story_locale property above
        """
        # cover date is found in story node (i.e. root node id)
        root = self._properties["root"]
        self._properties["nodes"][root]["config"]["storyLocale"] = locale
        return self.story_locale

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """This property returns the storymap's JSON."""
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def nodes(self):
        """
        Get main nodes in order of appearance in the story. This will return a list
        of dictionaries specifying the node ids and the class content they correspond to.
        If there is no class for the content, a string is returned with the content type.
        """
        # node_dict contains key-value pairs where the value is the class instance
        node_dict = self._create_node_dict()
        # make the value the string representation of the class
        nodes = []
        for node in node_dict:
            nodes.append({k: node[k] for k in node})
        return nodes

    # ----------------------------------------------------------------------
    @property
    def content_list(self):
        """
        Get a list of all the content instances in order of appearance in the story.
        This returns a list of class instances for the content in the story.
        """
        contents = []
        # get the values from the nodes list and return only these
        nodes = self._create_node_dict()
        for node in nodes:
            content = list(node.values())[0]
            contents.append(content)
        return contents

    # ----------------------------------------------------------------------
    def _create_node_dict(self):
        """
        Method called by the nodes property and the get method. However, the nodes
        property will transform the keys whereas the get method needs they keys
        to be class instances.
        """
        # get rood node id since it is story node id
        root_id = self._properties["root"]
        # get list of children from story node
        children = self._properties["nodes"][root_id]["children"]
        nodes = self._properties["nodes"]

        node_order = []
        # for each node assign correct class type to be accessed if needed by user
        for child in children:
            # get only the main nodes and not the subnodes to be returned
            if child in nodes:
                node = self._assign_node_class(child)
                node_order.append({child: node})
        return node_order

    # ----------------------------------------------------------------------
    @property
    def actions(self):
        """
        Get list of action nodes.
        """
        actions = []
        if "actions" in self._properties:
            for action in self._properties["actions"]:
                node = self._assign_node_class(action["origin"])
                actions.append(node)
        return actions

    # ----------------------------------------------------------------------
    @property
    def navigation_list(self):
        """
        Get a list of the nodes that are linked in the navigation.
        """
        # navigation item has list of links corresponding to the text nodes in the navigation
        nav = self.get(type="navigation")[0]
        for key, value in nav.items():
            node_id = key
        try:
            links = self._properties["nodes"][node_id]["data"]["links"]
            node_ids = []
            for link in links:
                for key, value in link.items():
                    # Only return list of node_ids this way easier for navigation method
                    if key == "nodeId":
                        node_ids.append(value)
            return node_ids
        except:
            return None

    # ----------------------------------------------------------------------
    def get(self, node: Optional[str] = None, type: Optional[str] = None):
        """
        Get node(s) by type or by their id. Using this function will help grab a specific node
        from the story if a node id is provided. Set this to a variable and this way edits can be
        made on the node in the story.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        node                Optional string. The node id for the node that should be returned.
                            This will return the class of the node if of type story content.
        ---------------     --------------------------------------------------------------------
        type                Optional string. The type of nodes that user wants returned.
                            If none specified, list of all nodes returned.


                            Values: `image` | `video` | `audio` | `embed` | `webmap` | `text` |
                            `button` | `separator` | `expressmap` | `webscene` | `immersive`
        ===============     ====================================================================

        :return:
            If type specified: List of node ids and their types in order of appearance in the story map.

            If node_id specified: The node itself.


        .. code-block:: python

            >>> story = StoryMap(<story item>)

            # Example get by type
            >>> story.get(type = "text")
            Returns a list of all nodes of type text

            # Example by id
            >>> text = story.get(node= "<id for text node>")
            >>> text.properties
            Returns a specific node of type text

        """
        spec_type = []
        node_id = node
        if node_id and node_id not in self.properties["nodes"]:
            raise ValueError(
                "This node value is not in the story. "
                + "Please check that you have entered the correct node id. "
                + "To see all main nodes and their ids use the nodes property."
            )
        if type is None and node_id is None:
            # return all nodes in order
            return self.nodes
        elif node_id is not None:
            # check first if it's an action
            all_actions = self.actions
            for action in all_actions:
                id = list(action.keys())[0]
                if node_id == id:
                    return list(action.values())[0]
            # return a specific node
            all_nodes = self._create_node_dict()
            # find the node in the list and return it
            for node in all_nodes:
                id = list(node.keys())[0]
                if node_id == id:
                    return list(node.values())[0]
        else:
            # return all nodes of a certain type
            all_nodes = self._create_node_dict()
            for node in all_nodes:
                keyword = str(list(node.values())[0]).lower()
                if isinstance(keyword, str):
                    # Not a type of story content (i.e. navigation)
                    if type.lower() in keyword:
                        spec_type.append(node)
                else:
                    # Find all story content instances (i.e. Text)
                    # Map types are upercase and have spaces so handle
                    if type.lower() in keyword._type.lower().replace(" ", ""):
                        spec_type.append(node)
            return spec_type

    # ----------------------------------------------------------------------
    def cover(
        self,
        title: Optional[str] = None,
        type: str = None,
        summary: Optional[str] = None,
        by_line: Optional[str] = None,
        image: Optional[Content.Image] = None,
    ):
        """
        A story's cover is at the top of the story and always the first node.
        This method allows the cover to be edited by updating the title, byline, image, and more.
        Changing one part of the story cover will not change the rest of the story cover. If just the
        image is passed in then only the image will change.

        .. note::
            To change the date seen on the story cover, use the ``cover_date`` property.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The title of the StoryMap cover.
        ---------------     --------------------------------------------------------------------
        type                Optional string. The type of story cover to be used in the story.

                            ``Values: "full" | "sidebyside" | "minimal"``
        ---------------     --------------------------------------------------------------------
        summary             Optional string. The description of the story.
        ---------------     --------------------------------------------------------------------
        by_line             Optional string. Crediting the author(s).
        ---------------     --------------------------------------------------------------------
        image               Optional url or file path or :class:`~arcgis.apps.storymap.story_content.Image`
                            object. The cover image for the story cover.
        ===============     ====================================================================

        :return: Dictionary representation of the story cover node.

        .. code-block:: python

            story = StoryMap(<story item>)
            story.cover(title="My Story Title", type="minimal", summary="My little summary", by_line="python_dev")
            story.save()

        """
        # story cover is always first node
        dict_node = self.nodes[0]

        # get the node id
        for key, _ in dict_node.items():
            story_cover_node = key

        # get original data of story cover
        orig_data = self._properties["nodes"][story_cover_node]["data"]

        # set the new values, if any
        self._properties["nodes"][story_cover_node] = {
            "type": "storycover",
            "data": {
                "type": orig_data["type"] if type is None else type,
                "title": orig_data["title"] if title is None else title,
                "summary": orig_data["summary"] if summary is None else summary,
                "byline": orig_data["byline"] if by_line is None else by_line,
                "titlePanelPosition": orig_data["titlePanelPosition"]
                if by_line is None
                else "start",
            },
        }

        # set the cover image
        if image is not None:
            if not isinstance(image, Content.Image):
                image = Content.Image(image)
            if image.node not in self._properties["nodes"]:
                # must be added to story resources
                image._add_image(story=self)
            self._properties["nodes"][story_cover_node]["children"] = [image.node]
        else:
            # get original image
            if "children" in self._properties["nodes"][story_cover_node]:
                media = self._properties["nodes"][story_cover_node]["children"][0]
                self._properties["nodes"][story_cover_node]["children"] = [media]

        return self._properties["nodes"][story_cover_node]

    # ----------------------------------------------------------------------
    def navigation(
        self, nodes: Optional[list[str]] = None, hidden: Optional[bool] = None
    ):
        """
        Story navigation is a way for authors to add headings as
        links to allow readers to navigate between different sections
        of a story. The story navigation node takes ``TextStyle.HEADING`` text styles
        as its only allowed children.
        You can only have 30 :class:`~arcgis.apps.storymap.story_content.Text` child nodes
        as visible and act as links within a story.

        The text nodes must already exist in the story. Pass the list of node ids for the heading
        text nodes to assign them to the navigation.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        nodes               Optional list of nodes to include in the navigation. These nodes can
                            only be of style heading ("h2").
                            Include in order. This will override current list and order.

                            To see current list use ``navigation_list`` property.
        ---------------     --------------------------------------------------------------------
        hidden              Optional boolean. If True, the navigation is hidden.
        ===============     ====================================================================

        :return:
            List of nodes in the navigation.

        .. code-block:: python

            #Example
            >>> story = StoryMap("<existing story id>")
            >>> story.navigation_list

            >>> story.navigation(["<header node id>", "<header node id>"], False)
        """

        # Check if navigation node already exists
        for node, node_info in self._properties["nodes"].items():
            for key, val in node_info.items():
                if key == "type" and val == "navigation":
                    node_id = node

        links = []
        # If none is provided, set to what is already there
        if nodes is not None:
            # check nodes are correct and add in order with linkType
            for node in nodes:
                if self._properties["nodes"][node]["data"]["type"] == "h2":
                    links.append({"nodeId": node, "linkType": "story-heading"})
                elif self._properties["nodes"][node]["data"]["type"] == "h4":
                    links.append({"nodeId": node, "linkType": "credits-heading"})
        else:
            links = self._properties["nodes"][node_id]["data"]["links"]
        if hidden is None:
            hidden = self._properties["nodes"][node_id]["config"]["isHidden"]

        # Update navigation
        self._properties["nodes"][node_id] = {
            "type": "navigation",
            "data": {"links": links},
            "config": {"isHidden": hidden},
        }

        return self.navigation_list

    # ----------------------------------------------------------------------
    def theme(self, theme: Union[Themes, str] = Themes.SUMMIT):
        """
        Each story has a theme node in its resources. This method can be used to change the theme.
        To add a custom theme to your story, pass in the item_id for the item of type Story Map Theme.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        theme               Required Themes Style or custom theme item id.
                            The theme to set the story to.

                            Values: `SUMMIT` | `TIDAL` | `MESA` | `RIDGELINE` | `SLATE` | `OBSIDIAN` | `<item_id>`
        ===============     ====================================================================

        .. code-block:: python

            >>> from arcgis.apps.storymap import Themes

            >>> story = StoryMap()
            >>> story.theme(Themes.TIDAL)
        """
        # find the node corresponding to the story theme in resources
        for node, node_info in self._properties["resources"].items():
            for key, val in node_info.items():
                if key == "type" and val == "story-theme":
                    if isinstance(theme, Themes):
                        # theme comes from Themes class
                        self._properties["resources"][node]["data"][
                            "themeId"
                        ] = theme.value
                    if isinstance(theme, str):
                        # theme is an item of type Story Theme
                        self._properties["resources"][node]["data"][
                            "themeItemId"
                        ] = theme

    # ----------------------------------------------------------------------
    def credits(
        self,
        content: Optional[str] = None,
        attribution: Optional[str] = None,
        heading: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Credits are found at the end of the story and thus are always the last node.

        To create a credit, add the text that should be shown on each side of the divider.
        Content represents the text seen on the left side and attribution is in line with content
        on the right side of the divider. (i.e. 'content' | 'attribution')

        Adding ``content`` and ``attribution`` will add a new line to the credits and will not change previous
        credits.

        Adding ``heading`` and ``description`` will change what is currently in place.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        content             Optional String. The content to be added. (Seen on the left side of
                            the credits.)

                            Make sure text has '<strong> </strong>' tags.
                            Adds to the existing credits.
        ---------------     --------------------------------------------------------------------
        attribution         Optional String. The attribution to be added. (Seen on right side of
                            the credits.)
                            Adds to the existing credits.
        ---------------     --------------------------------------------------------------------
        heading             Optional String. Replace current heading for credits.
        ---------------     --------------------------------------------------------------------
        description         Optional String. Replace current description for credits.
        ===============     ====================================================================

        :return:
            A list of strings that are the node ids for the text nodes that belong to credits.

        .. code-block:: python

            #Example
            >>> story = StoryMap()
            >>> story.credits("Python Dev" , "Python API Team", "Thank You", "A big thank you to those who contributed")
        """
        # Find credit node
        dict_node = self.get(type="credits")[0]
        # Get credit node id
        for key, value in dict_node.items():
            credits_node = key
        if "children" in self._properties["nodes"][credits_node]:
            children = self._properties["nodes"][credits_node]["children"]
        else:
            children = []

        nodes = []
        if content or attribution:
            # Create new content node
            node = "n-" + uuid.uuid4().hex[0:6]
            self._properties["nodes"][node] = {
                "type": "attribution",
                "data": {"content": content, "attribution": attribution},
            }
            nodes.append(node)

        # Create new heading and remove old one
        if heading:
            # Create new content node
            node = "n-" + uuid.uuid4().hex[0:6]
            self._properties["nodes"][node] = {
                "type": "text",
                "data": {"text": heading, "type": "h4"},
            }
            nodes.append(node)
            # Find and remove node corresponding to current heading
            for child in children:
                if (
                    self._properties["nodes"][child]["type"] == "text"
                    and self._properties["nodes"][child]["data"]["type"] == "h4"
                ):
                    del self._properties["nodes"][child]
                    self._properties["nodes"][credits_node]["children"].remove(child)

        if description:
            # Create new content node
            node = "n-" + uuid.uuid4().hex[0:6]
            self._properties["nodes"][node] = {
                "type": "text",
                "data": {"text": description, "type": "paragraph"},
            }
            nodes.append(node)
            # Find and remove node corresponding to current heading
            for child in children:
                if (
                    self._properties["nodes"][child]["type"] == "text"
                    and self._properties["nodes"][child]["data"]["type"] == "paragraph"
                ):
                    del self._properties["nodes"][child]
                    self._properties["nodes"][credits_node]["children"].remove(child)

        # Add to children of credits
        for node_id in nodes:
            self._properties["nodes"][credits_node]["children"].append(node_id)
        return self._properties["nodes"][credits_node]["children"]

    # ----------------------------------------------------------------------
    def add(
        self,
        content: Optional[
            Union[
                Content.Image,
                Content.Video,
                Content.Audio,
                Content.Embed,
                Content.Map,
                Content.Button,
                Content.Text,
                Content.Gallery,
                Content.Timeline,
                Content.Sidecar,
            ]
        ] = None,
        caption: Optional[str] = None,
        alt_text: Optional[str] = None,
        display: str = None,
        position: Optional[int] = None,
    ):
        """
        Use this method to add content to your StoryMap. Content can be of various class types and when
        you add this content you can specify a caption, alt_text, display style, and the position
        at which it will be in your story.
        Not passing in any content means a separator will be added.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        content             Optional content of type:
                            :class:`~arcgis.apps.storymap.story_content.Image`,
                            :class:`~arcgis.apps.storymap.story_content.Gallery`,
                            :class:`~arcgis.apps.storymap.story_content.Video`,
                            :class:`~arcgis.apps.storymap.story_content.Audio`,
                            :class:`~arcgis.apps.storymap.story_content.Embed`,
                            :class:`~arcgis.apps.storymap.story_content.Map`,
                            :class:`~arcgis.apps.storymap.story_content.Text`,
                            :class:`~arcgis.apps.storymap.story_content.Button`,
                            :class:`~arcgis.apps.storymap.story_content.Timeline`,
                            :class:`~arcgis.apps.storymap.story_content.Sidecar`
                            :class:`~arcgis.apps.storymap.story_content.Swipe`,
                            :class:`~arcgis.apps.storymap.story_content.Separator`


                            If none is provided, a separator is added.
        ---------------     --------------------------------------------------------------------
        caption             Optional String. Custom text to caption the webmap.
        ---------------     --------------------------------------------------------------------
        alt_text            Optional String. Custom text to be used for screen readers.
        ---------------     --------------------------------------------------------------------
        display             Optional String. How the item will be displayed in the story map.

                            For Image, Video, Audio, or Map object.
                            Values: "standard" | "wide" | "full" | "float"

                            For Gallery:
                            Values: "jigsaw" | "square-dynamic"

                            For Embed:
                            Values: "card" | "inline"

                            For Swipe:
                            Values: "small" | "medium" | "large"
        ---------------     --------------------------------------------------------------------
        position            Optional Integer. Indicates the position in which the content will be
                            added. To see all node positions use the ``node`` property.
        ===============     ====================================================================

        :return: A String depicting the node id for the content that was added.

        .. code-block:: python

            new_story = StoryMap()

            # Example with Image
            >>> image1 = Image("<image-path>.jpg/jpeg/png/gif ")
            >>> new_node = new_story.add(image1, position = 2)

            # Example with Map
            >>> my_map = Map(<item-id of type webmap>)
            >>> new_node = new_story.add(my_map, "A map caption", "A new map alt-text")

            # Example to add a Separator
            >>> new_node = new_story.add()

            >>> print(new_story.nodes)

        """
        if content and content.node in self._properties["nodes"]:
            content.node = "n-" + uuid.uuid4().hex[0:6]

        # Node id included in all content except separator so create node id for that
        node_id = content.node if content is not None else "n-" + uuid.uuid4().hex[0:6]

        # Find instance of content and call correct method
        if isinstance(content, Content.Image):
            content._add_image(caption, alt_text, display, self)
        elif isinstance(content, Content.Gallery):
            content._add_gallery(caption, alt_text, display, self)
        elif isinstance(content, Content.Video):
            content._add_video(caption, alt_text, display, self)
        elif isinstance(content, Content.Audio):
            content._add_audio(caption, alt_text, display, self)
        elif isinstance(content, Content.Map):
            content._add_map(caption, alt_text, display, story=self)
        elif isinstance(content, Content.Embed):
            content._add_link(caption, alt_text, display, self)
        elif isinstance(content, Content.Button):
            content._add_button(self)
        elif isinstance(content, Content.Text):
            content._add_text(self)
        elif isinstance(content, Content.Timeline):
            content._add_timeline(self)
        elif isinstance(content, Content.Sidecar):
            content._add_sidecar(self)
        elif isinstance(content, Content.Swipe):
            content._add_swipe(caption, alt_text, display, self)
        else:
            content = Content.Separator(story=self, node_id=node_id)
            content._add_separator(story=self)

        # Add to story children
        self._add_child(node_id=node_id, position=position)
        return node_id

    # ----------------------------------------------------------------------
    def move(
        self, node_id: str, position: Optional[int] = None, delete_current: bool = False
    ):
        """
        Move a node to another position. The node currently at that position will
        be moved down one space. The node at the current position can be deleted
        instead of moved if `delete_current` is set to True.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        node_id             Required String. The node id for the content that will be moved. Find a
                            list of node order by using the ``nodes`` property.
        ---------------     --------------------------------------------------------------------
        position            Optional Integer. Indicates the position in which the content will be
                            added. If no position is provided, the node will be placed at the end.
        ---------------     --------------------------------------------------------------------
        delete_current      Optional Boolean. If set to True, the node at the current position will
                            be deleted instead of moved down one space. Default is False.
        ===============     ====================================================================

        .. code-block:: python

            new_story = StoryMap()

            # Example with Image
            >>> image1 = Image("<image-path>.jpg/jpeg/png/gif")
            >>> image2 = Image("<image-path>.jpg/jpeg/png/gif")
            >>> new_node = new_story.add(image1, "my caption", "my alt-text", "float", 2)
            >>> new_story.add(image2)
            >>> new_story.move(new_node, 3, False)

        """
        # Get list of story children
        root_id = self._properties["root"]
        children = self._properties["nodes"][root_id]["children"]

        # Remove node id from list since it will be added again at another position
        self._properties["nodes"][root_id]["children"].remove(node_id)

        # If delete_current is True then remove the node currently at this position
        if delete_current:
            if position == 0 or position == len(children):
                raise Exception(
                    "First and last nodes are reserved for Story Cover and Credits"
                )
            self._properties["nodes"][root_id]["children"].pop(position)

        # Add node to new position
        self._add_child(node_id, position)

    # ----------------------------------------------------------------------
    def save(
        self,
        title: Optional[str] = None,
        tags: Optional[list] = None,
        access: str = None,
        publish: bool = False,
    ):
        """
        This method will save your Story Map to your active GIS. The story will be saved
        with unpublished changes unless `publish` parameter is specified to True.

        The title only needs to be specified if a change is wanted, otherwise exisiting title
        is used.

        .. warning::
            Publishing your story through the Python API means it will not go through the Story Map
            issue checker. It is recommended to publish through the Story Maps builder if you
            want your story to go through the issue checker.

        .. warning::
            Changes to the published story may not be visible for up to one hour. You can open
            the story in the story builder to force changes to appear immediately and perform
            other optimizations, such as updating the story's social/SEO metadata.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The title of the StoryMap.
        ---------------     --------------------------------------------------------------------
        tags                Optional string. The tags of the StoryMap.
        ---------------     --------------------------------------------------------------------
        access              Optional string. The access of the StoryMap. If none is specified, the
                            current access is kept. This is used when `publish` parameter is set
                            to True.

                            Values: `private` | `org` | `public`
        ---------------     --------------------------------------------------------------------
        publish             Optional boolean. If True, the story is saved and also published.
                            Default is false so story is saved with unpublished changes.
        ===============     ====================================================================


        :return: The Item that was saved to your active GIS.

        """
        # Remove old draft item
        for resource in self._resources:
            if re.match("draft_[0-9]{13}.json", resource["resource"]) or re.match(
                "draft.json", resource["resource"]
            ):
                self._remove_resource(file=resource["resource"])

        # Add meta settings and change push meta so title doesn't get overwritten on publish at any point.
        if title:
            root = self._properties["root"]
            if "metaSettings" not in self._properties["nodes"][root]["data"]:
                self._properties["nodes"][root]["data"]["metaSettings"] = {
                    "title": None
                }
            self._properties["nodes"][root]["data"]["metaSettings"]["title"] = title
            self._properties["nodes"][root]["config"][
                "shouldPushMetaToAGOItemDetails"
            ] = False

        # Add new draft with time in milliseconds
        draft = "draft_" + str(int(time.time() * 1000)) + ".json"
        json_str = json.dumps(self._properties, ensure_ascii=False)
        self._add_resource(resource_name=draft, text=json_str, access="private")
        # get the story map version from endpoint
        sm_version = self._gis._con.get("https://storymaps.arcgis.com/version")[
            "version"
        ]
        # Find type keywords to use based on whether to publish or not
        # PUBLISH MODE
        if publish is True:
            # Remove old publish item
            for resource in self._resources:
                if (
                    "publish_data" in resource["resource"]
                    or "published_data" in resource["resource"]
                    or "publish" in resource["resource"]
                ):
                    self._remove_resource(file=resource["resource"])
            # Add new publish
            self._add_resource(
                resource_name="published_data.json", text=json.dumps(self._properties)
            )
            # Set the keywords
            # Start by getting the existing keywords and remove what will be replaced
            keywords = self._item.typeKeywords
            if "smstatusunpublishedchanges" in keywords:
                # changing to publish after
                idx = keywords.index("smstatusunpublishedchanges")
                del keywords[idx]
            if "smstatusdraft" in keywords:
                idx = keywords.index("smstatusdraft")
                del keywords[idx]
            for keyword in keywords:
                # iterate through since only know part of keyword we want to remove
                if (
                    "smdraftresourceid"
                    or "smpublisheddate"
                    or "smstatusdraft"
                    or "smpublisherapp"
                ) in keyword:
                    keywords.remove(keyword)
            new_keywords = [
                "smstatuspublished",
                "smversiondraft:" + sm_version,
                "smversionpublished:" + sm_version,
                "python-api",
                "smpublisherapp:python-api-" + arcgis.__version__,
                "smdraftresourceid:" + draft,
                "smpublisheddate:" + str(int(time.time() * 1000)),
            ]
            # Setting the keywords in a set will remove duplicates
            p = {
                "typeKeywords": list(set(keywords + new_keywords)),
                "text": json.dumps(self._properties),
                "url": self._url,
            }
            if title:
                p["title"] = title
            if tags:
                p["tags"] = tags

            # Find and set access
            sharing = access if access is not None else self._item.access
            p["access"] = sharing

            # Update the item and invoke share to have correct access
            self._item.update(item_properties=p)

            if sharing == "private":
                self._item.share(everyone=False, org=False, groups=None)
            elif sharing == "org":
                self._item.share(org=True)
            elif sharing == "public":
                self._item.share(everyone=True)

            if (
                self._gis._con._session.auth
                and self._gis._con._session.auth.token is not None
            ):
                # Make a call to the StoryMaps publish endpoint
                self._gis._con.post(
                    path=self._url + "/publish",
                    params={"f": "json", "token": self._gis._con._session.auth.token},
                )
        else:
            # Set the type keywords
            keywords = self._item.typeKeywords
            previously_published = False
            for keyword in keywords:
                if "smpublisheddate" in keyword:
                    # Update the date in new keywords
                    previously_published = True
                    keywords.remove(keyword)
                elif (
                    "smstatuspublished" in keyword
                    or "smstatusdraft" in keyword
                    or "smdraftresourceid" in keyword
                    or "smeditorapp" in keyword
                    or "Copy Item" in keyword
                ):
                    # Remove old keywords and will be replaced in new keywords
                    keywords.remove(keyword)
            if previously_published is True:
                # Unpublished changes mode
                new_keywords = [
                    "smstatusunpublishedchanges",
                    "smversiondraft:" + sm_version,
                    "python-api",
                    "smeditorapp:python-api-" + arcgis.__version__,
                    "smdraftresourceid:" + draft,
                    "smversionpublished:" + sm_version,
                    "smpublisheddate:" + str(int(time.time() * 1000)),
                ]
            if previously_published is False:
                # Draft mode
                new_keywords = [
                    "smstatusdraft",
                    "smversiondraft:" + sm_version,
                    "python-api",
                    "smeditorapp:python-api-" + arcgis.__version__,
                    "smdraftresourceid:" + draft,
                ]
            # Pass through set first to remove duplicates
            p = {"typeKeywords": list(set(keywords + new_keywords))}
            if title:
                p["title"] = title
            if tags:
                p["tags"] = tags
            # access does not change when only saving
            p["access"] = self._item.access
            self._item.update(item_properties=p)

        self._item = self._gis.content.get(self._itemid)
        return self._item

    # ----------------------------------------------------------------------
    def delete_story(self):
        """
        Deletes the story item.
        """
        # Check if item id exists
        item = self._gis.content.get(self._itemid)
        return item.delete()

    # ----------------------------------------------------------------------
    def duplicate(self, title: Optional[str] = None):
        """
        Duplicate the story. All items will be duplicated as they are. This allows you to create
        a story template and duplicate it when you want to work with it.

        It is highly recommended that once the duplicate is created, open it in Story Maps
        builder to ensure the issue checker finds any issues before editing.

        .. note::
            Can be used with ArcGIS Online or with ArcGIS Enterprise starting 10.8.1

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. The title of the duplicated story. Only available for
                            ArcGISOnline.
        ===============     ====================================================================

        :return:
            The Item that was created.

        .. code-block:: python

            # Example for ArcGIS Online
            >>> story = StoryMap(<story item>)
            >>> story.duplicate("A Story Copy")

            # Example for ArcGIS Enterprise
            >>> story = StoryMap(<story item>)
            >>> story.duplicate()
        """
        # get the item to copy
        item = self._gis.content.get(self._itemid)

        # enterprise copy_item starting at 10.8.1
        if item._portal.is_arcgisonline is False and self._gis.version < [8, 2]:
            clone = self._gis.content.clone_items(items=[item])
        else:
            clone = item.copy_item(
                title="(Copy) " + self._item.title if title is None else title,
                include_resources=True,
                include_private=True,
            )
        # save to update keywords
        clone_story = StoryMap(clone.id)
        return clone_story.save()

    # ----------------------------------------------------------------------
    def copy_content(self, target_story: StoryMap, node_list: list):
        """
        Copy the content from one story to another. This will copy the content
        indicated to the target story in the order they are provided. To change the
        order once the nodes are copied, use the `move()` method on the target story.

        .. note::
            Do not forget to save the target story once you are done copying and making
            any further edits.

        .. note::
            This method can take time depending on the number of resources. Each resource coming
            from a file must be copied over and heavy files, such as videos or audio, can be time
            consuming.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        target_story        Required StoryMap instance. The target story that the content will be
                            copied to.
        ---------------     --------------------------------------------------------------------
        node_list           Required list of strings. The list of node ids indicating the content
                            that will be copied to the target story.
        ===============     ====================================================================

        :return:
            True if all nodes have been successfully copied over.

        """
        # Step 1: Do Checks
        # Check that nodes exist in original story (children of source story contain all of node_list)
        story_children = self._properties["nodes"][self._properties["root"]]["children"]
        check = all(node in story_children for node in node_list)
        # Return an error if not all nodes are in the source story.
        if check is False:
            not_in_story = []
            for node in node_list:
                if node not in story_children:
                    not_in_story.append(node)
            raise ValueError(
                "These nodes are not in the story: "
                + str(not_in_story)
                + ". Please check that the correct node ids are provided."
            )

        # Step 2: Create dictionaries for copying

        # Create node dict of all nodes to add, resource dict, and complete node list
        # Depending on node type, need to take different route to find all children
        original_nodes = node_list
        complete_node_list = []
        complete_node_dict = {}
        complete_resource_dict = {}
        resource_files = {}
        has_children = True

        # internal method to add to correct places
        def _add_to_dicts(node_add, comp_list, comp_node_dict, comp_res_dict):
            # add to complete list of nodes
            comp_list.append(node_add)
            # get the dictionary
            node_dict = self._properties["nodes"][node_add]
            comp_node_dict[node_add] = node_dict

            # find the resource node to add associated with node
            if "data" in node_dict:
                # iterate through values of dict to find any resources
                for _, value in node_dict["data"].items():
                    if isinstance(value, list):
                        for im in value:
                            # express maps keep their images in a list
                            _add_to_resources(im, comp_res_dict)
                    else:
                        _add_to_resources(value, comp_res_dict)

        def _add_to_resources(value, comp_res_dict):
            if isinstance(value, str):
                # check if value is a resource
                if "r-" in value:
                    resource_node = value
                    # get the resource dict
                    resource_dict = self._properties["resources"][resource_node]
                    comp_res_dict[resource_node] = resource_dict
                    if "resourceId" in resource_dict["data"]:
                        # some nodes keep the resource under resourceId key
                        name = resource_dict["data"]["resourceId"]
                        # get the resource file to add to new story
                        resource_file = self._item.resources.get(name)
                        resource_files[name] = resource_file
                    elif "itemId" in resource_dict["data"]:
                        name = resource_dict["data"]["itemId"]
                        # express map keeps resource under itemId key
                        if name.endswith(".json"):
                            # need to add draft_ in front to be one-to-one with builder
                            name = "draft_" + resource_dict["data"]["itemId"]
                            # get the json file draft
                            resource_file = self._item.resources.get(name)
                            resource_files[name] = resource_file

        # Begin populating dicts and list, assume there are children to begin with.
        while has_children is True:
            # new list of nodes to check at next iteration
            new_nodes = []
            for node in node_list:
                # add node info for copying
                _add_to_dicts(
                    node, complete_node_list, complete_node_dict, complete_resource_dict
                )
                # check type of node to see if need to find children
                node_children = self._has_children(node)
                # populate new list with next nodes to add
                if node_children:
                    for child in node_children:
                        new_nodes.append(child)
            # if list is not empty, keep going
            if new_nodes:
                has_children = True
                node_list = new_nodes
            # once list is empty, all children have been accounted for
            else:
                has_children = False

        # Step 3: Make any changes before copying over
        # existing target story node ids
        target_story_nodes = list(target_story._properties["nodes"].keys())

        if any(node in target_story_nodes for node in complete_node_list):
            # find the node and change it everywhere
            for node in complete_node_list:
                if node in target_story_nodes:
                    new_node = "n-" + uuid.uuid4().hex[0:6]
                    # replace node with new node in all places
                    # in the list passed in, if present
                    original_nodes = [s.replace(node, new_node) for s in original_nodes]
                    # in the dictionary of all nodes to copy
                    for key, value in complete_node_dict.items():
                        if key == node:
                            # replace old node id with new node id in keys
                            complete_node_dict[new_node] = complete_node_dict.pop(key)
                        if "children" in value:
                            # replace old node id with new node id if child of another node
                            if node in value["children"]:
                                complete_node_dict[key]["children"] = [
                                    s.replace(node, new_node) for s in value["children"]
                                ]

        # Step 4: Copy nodes to target story
        for key, value in complete_node_dict.items():
            target_story._properties["nodes"][key] = value
        for key, value in complete_resource_dict.items():
            target_story._properties["resources"][key] = value
        for key, value in resource_files.items():
            try:
                target_story._add_resource(file=value, resource_name=key)
            except:
                # express map, image editor, other created files will be here
                text = json.dumps(value)
                target_story._add_resource(resource_name=key, text=text)

        # Step 5: Add the node list to the story children
        for main_node in original_nodes:
            target_story._add_child(main_node)
        return True

    # ----------------------------------------------------------------------
    def _has_children(self, node):
        """
        Check if node has children and return list of children else None.
        """
        node_class = self._assign_node_class(node)
        if (
            isinstance(node_class, Content.Sidecar)
            or isinstance(node_class, Content.Gallery)
            or isinstance(node_class, Content.Timeline)
        ):
            return self._properties["nodes"][node]["children"]
        elif isinstance(node_class, Content.Swipe):
            return list(self._properties["nodes"][node]["data"]["contents"].values())
        elif isinstance(node_class, Content.MapTour):
            mt = self.get(node)
            return mt._children
        elif isinstance(node_class, str):
            if (
                "immersive" in node_class.lower()
                or "credits" in node_class.lower()
                or "event" in node_class.lower()
                or "carousel" in node_class.lower()
            ):
                return (
                    self._properties["nodes"][node]["children"]
                    if "children" in self._properties["nodes"][node]
                    else None
                )
        else:
            return None

    # ----------------------------------------------------------------------
    def _delete(self, node_id):
        # Check if node is in story
        if node_id not in self._properties["nodes"]:
            return False

        # Get list of nodes in the story
        root_id = self._properties["root"]
        children = self._properties["nodes"][root_id]["children"]

        # Remove from children of story
        if node_id in children:
            self._properties["nodes"][root_id]["children"].remove(node_id)
        # Remove from nodes dictionary
        del self._properties["nodes"][node_id]
        # Remove node from any immersive nodes.
        # A node can belong to an immersive narrative panel or an immersive slide
        for node in self._properties["nodes"]:
            if (
                "immersive" in self._properties["nodes"][node]["type"]
                and "children" in self._properties["nodes"][node]
            ):
                for child in self._properties["nodes"][node]["children"]:
                    # iterate through children to see if node is part of it
                    if child == node_id:
                        self._properties["nodes"][node]["children"].remove(node_id)

        return True

    # ----------------------------------------------------------------------
    def _add_child(self, node_id, position=None):
        """
        A story node has children. Children is a list of item nodes that are in
        the story. The order of the list determines the order that the nodes
        appear in the story. First and last nodes are reserved for story_cover
        and credits. The second node is always navigation. If visible is not set
        to True is simply won't be seen but stays in position 2.
        """
        # Get list of children in story
        root_id = self._properties["root"]
        last = len(self._properties["nodes"][root_id]["children"]) - 1

        if position and position < last and position != 0 and position != 1:
            # If the position adheres to rules then add node
            self._properties["nodes"][root_id]["children"].insert(position, node_id)
        elif position and (position == 0 or position == 1):
            # First and second node reserved for story cover and navigation
            # Add as third node if user specified position 0 or 1
            self._properties["nodes"][root_id]["children"].insert(2, node_id)
        else:
            # Last node is reserved for credits so add before this if user wanted last position
            self._properties["nodes"][root_id]["children"].insert(last, node_id)

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
    def _remove_resource(self, file=None):
        """
        See :class:`~arcgis.gis.ResourceManager`
        """
        try:
            resource_manager = arcgis.gis.ResourceManager(self._item, self._gis)
            resp = resource_manager.remove(file=file)
            self._resources = self._item.resources.list()
            return resp
        except:
            # Resource cannot be found. Should not throw error
            return True

    # ----------------------------------------------------------------------
    def _assign_node_class(self, node_id):
        # Find the node type to assign to correct class
        node_type = self._properties["nodes"][node_id]["type"]
        # Create an instance of this class using existing node properties
        if node_type == "separator":
            node = Content.Separator(story=self, node_id=node_id)
        elif node_type == "image":
            node = Content.Image(story=self, node_id=node_id)
        elif node_type == "video":
            node = Content.Video(story=self, node_id=node_id)
        elif node_type == "audio":
            node = Content.Audio(story=self, node_id=node_id)
        elif node_type == "embed":
            # embed has subtype: video or link
            subtype = self._properties["nodes"][node_id]["data"]["embedType"]
            if subtype == "video":
                node = Content.Video(story=self, node_id=node_id)
            else:
                node = Content.Embed(story=self, node_id=node_id)
        elif node_type == "webmap":
            node = Content.Map(story=self, node_id=node_id)
        elif node_type == "text":
            node = Content.Text(story=self, node_id=node_id)
        elif node_type == "button":
            node = Content.Button(story=self, node_id=node_id)
        elif node_type == "swipe":
            node = Content.Swipe(story=self, node_id=node_id)
        elif node_type == "gallery":
            node = Content.Gallery(story=self, node_id=node_id)
        elif node_type == "timeline":
            node = Content.Timeline(story=self, node_id=node_id)
        elif node_type == "tour":
            node = Content.MapTour(story=self, node_id=node_id)
        elif node_type == "immersive":
            # immersive has subtype sidecar (more to add later)
            subtype = self._properties["nodes"][node_id]["data"]["type"]
            if subtype == "sidecar":
                node = Content.Sidecar(story=self, node_id=node_id)
            else:
                node = subtype
        elif node_type == "action-button":
            node = Content.MapAction(story=self, node_id=node_id)
        else:
            # if not of type story content then just return name of type
            node = node_type.capitalize()
        return node
