from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
from arcgis.apps.hub.sites import SiteManager, Site
from collections import OrderedDict
from datetime import datetime
import json


def _lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Initiative(OrderedDict):
    """
    Represents an initiative within a Hub. An Initiative supports
    policy- or activity-oriented goals through workflows, tools and team collaboration.
    """

    def __init__(self, gis, initiativeItem):
        """
        Constructs an empty Initiative object
        """
        self.item = initiativeItem
        self._gis = gis
        self._hub = gis.hub
        try:
            self._initiativedict = self.item.get_data()
            pmap = PropertyMap(self._initiativedict)
            self.definition = pmap
        except:
            self.definition = None

    def __repr__(self):
        return '<%s title:"%s" owner:%s>' % (
            type(self).__name__,
            self.title,
            self.owner,
        )

    @property
    def itemid(self) -> str:
        """
        Returns the item id of the initiative item
        """
        return self.item.id

    @property
    def title(self) -> str:
        """
        Returns the title of the initiative item
        """
        return self.item.title

    @property
    def description(self) -> str:
        """
        Returns the initiative description
        """
        return self.item.description

    @property
    def snippet(self) -> str:
        """
        Getter/Setter for the initiative snippet
        """
        return self.item.snippet

    @snippet.setter
    def snippet(self, value):
        self.item.snippet = value

    @property
    def owner(self) -> str:
        """
        Returns the owner of the initiative item
        """
        return self.item.owner

    @property
    def tags(self) -> str:
        """
        Returns the tags of the initiative item
        """
        return self.item.tags

    @property
    def url(self) -> str:
        """
        Returns the url of the initiative site
        """
        try:
            return self.item.properties["url"]
        except:
            return self.item.url

    @property
    def site_id(self) -> str:
        """
        Returns the item id of the initiative site
        """
        try:
            return self.item.properties["siteId"]
        except:
            return self._initiativedict["steps"][0]["itemIds"][0]

    @property
    def site_url(self) -> str:
        """
        Getter/Setter for the url of the initiative site
        """
        try:
            return self.item.url
        except:
            return self.sites.get(self.site_id).url

    @site_url.setter
    def site_url(self, value):
        self.item.url = value

    @property
    def content_group_id(self) -> str:
        """
        Returns the group id for the content group
        """
        return self.item.properties["contentGroupId"]

    @property
    def collab_group_id(self) -> str:
        """
        Getter/Setter for the group id for the collaboration group
        """
        try:
            return self.item.properties["collaborationGroupId"]
        except:
            return None

    @collab_group_id.setter
    def collab_group_id(self, value):
        self.item.properties["collaborationGroupId"] = value

    @property
    def followers_group_id(self) -> str:
        """
        Returns the group id for the followers group
        """
        return self.item.properties["followersGroupId"]

    @_lazy_property
    def sites(self) -> SiteManager:
        """
        The resource manager for an Initiative's sites.
        See :class:`~hub.sites.SiteManager`.
        """
        return SiteManager(self._hub, self)

    @_lazy_property
    def all_events(self):
        """
        Fetches all events (past or future) pertaining to an initiative
        """
        return self._gis.hub.events.search(initiative_id=self.item.id)

    @_lazy_property
    def followers(self) -> list:
        """
        Fetches the list of followers for initiative.
        """
        # Fetch followers group
        _followers_group = self._gis.groups.get(self.followers_group_id)
        return _followers_group.get_members()

    def add_content(self, items_list: list):
        """
        Adds a batch of items to the initiative content library.

        =====================    ====================================================================
        **Parameter**             **Description**
        ---------------------    --------------------------------------------------------------------
        items_list               Required list. A list of Item or item ids to add to the initiative.
        =====================    ====================================================================

        """
        # If input list is of item_ids, generate a list of corresponding items
        if type(items_list[0]) == str:
            items = [self._gis.content.get(item_id) for item_id in items_list]
        else:
            items = items_list
        # Fetch existing sharing privileges for each item, to retain them after adding to content library
        for item in items:
            sharing = item.shared_with
            everyone = sharing["everyone"]
            org = sharing["org"]
            groups = sharing["groups"]
            # add current initiative's content group to list of groups to share to
            groups.append(self.content_group_id)
            # share item to this group
            status = item.share(everyone=everyone, org=org, groups=groups)
            if status["results"][0]["success"] == False:
                return status
        return status

    def delete(self) -> bool:
        """
        Deletes the initiative, its site and associated groups.
        If unable to delete, raises a RuntimeException.

        :return:
            A bool containing True (for success) or False (for failure).

        .. code-block:: python

            USAGE EXAMPLE: Delete an initiative successfully

            initiative1 = myHub.initiatives.get('itemId12345')
            initiative1.delete()

            >> True
        """
        if self.item is not None:
            # Fetch initiative site
            _site = self._gis.hub.sites.get(self.site_id)
            Site.delete(_site)
            # Fetch and delete Initiative Collaboration group if exists
            try:
                _collab_group = self._gis.groups.get(self.collab_group_id)
                _collab_group.protected = False
                _collab_group.delete()
            except:
                pass
            # Fetch Content Group and delete
            try:
                _content_group = self._gis.groups.get(self.content_group_id)
                _content_group.protected = False
                _content_group.delete()
            except:
                pass
            # Fetch Followers Group and delete
            try:
                _followers_group = self._gis.groups.get(self.followers_group_id)
                _followers_group.protected = False
                _followers_group.delete()
            except:
                pass
            # Delete initiative
            return self.item.delete()

    def reassign_to(self, target_owner: str):
        """
        Allows the administrator to reassign the initiative object from one
        user to another.

        .. note::
            This will transfer ownership of all items (site, pages, content) and groups that
            belong to this initiative to the new target_owner.

        =====================     ====================================================================
        **Parameter**              **Description**
        ---------------------     --------------------------------------------------------------------
        target_owner              Required string. The new desired owner of the initiative.
        =====================     ====================================================================
        """
        # check if admin user is performing this action
        if "admin" not in self._gis.users.me.role:
            return Exception(
                "You do not have the administrator privileges to perform this action."
            )
        # check if core team is needed by checking the role of the target_owner
        if self._gis.users.get(target_owner).role == "org_admin":
            # check if the initiative comes with core team by checking owner's role
            if self._gis.users.get(self.owner).role == "org_admin":
                # fetch the core team for the initative
                core_team = self._gis.groups.get(self.collab_group_id)
                # fetch the contents shared with this team
                core_team_content = core_team.content()
                # check if target_owner is part of core team, else add them to core team
                members = core_team.get_members()
                if (
                    target_owner not in members["admins"]
                    or target_owner not in members["users"]
                ):
                    core_team.add_users(target_owner)
                # remove items from core team
                self._gis.content.unshare_items(core_team_content, groups=[core_team])
                # reassign to target_owner
                for item in core_team_content:
                    item.reassign_to(target_owner)
                # fetch the items again since they have been reassigned
                new_content_list = []
                for item in core_team_content:
                    item_temp = self._gis.content.get(item.id)
                    new_content_list.append(item_temp)
                # share item back to the content group
                self._gis.content.share_items(
                    new_content_list, groups=[core_team], allow_members_to_edit=True
                )
                # reassign core team to target owner
                core_team.reassign_to(target_owner)
            else:
                # create core team necessary for the initiative
                _collab_group_title = title + " Core Team"
                _collab_group_dict = {
                    "title": _collab_group_title,
                    "tags": [
                        "Hub Group",
                        "Hub Initiative Group",
                        "Hub Site Group",
                        "Hub Core Team Group",
                        "Hub Team Group",
                    ],
                    "access": "org",
                    "capabilities": "updateitemcontrol",
                    "membershipAccess": "collaboration",
                    "snippet": "Members of this group can create, edit, and manage the site, pages, and other content related to hub-groups.",
                }
                collab_group = self._gis.groups.create_from_dict(_collab_group_dict)
                collab_group.protected = True
                self.collab_group_id = collab_group.id
        else:
            # reassign the initiative, site, page items
            self.item.reassign_to(target_owner)
            site = self._hub.sites.get(self.site_id)
            site.item.reassign_to(target_owner)
            site_pages = site.pages.search()
            # If pages exist
            if len(site_pages) > 0:
                for page in site_pages:
                    # Unlink page (deletes if)
                    page.item.reassign_to(target_owner)
        # fetch content group
        content_team = self._gis.groups.get(self.content_group_id)
        # reassign to target_owner
        content_team.reassign_to(target_owner)
        # If it is a Hub Premium initiative, repeat for followers group
        if self._hub._hub_enabled:
            followers_team = self._gis.groups.get(self.followers_group_id)
            followers_team.reassign_to(target_owner)
        return self._gis.content.get(self.itemid)

    def share(
        self, everyone=False, org=False, groups=None, allow_members_to_edit=False
    ):
        """
        Shares an initiative and associated site with the specified list of groups.

        ======================  ========================================================
        **Parameter**            **Description**
        ----------------------  --------------------------------------------------------
        everyone                Optional boolean. Default is False, don't share with
                                everyone.
        ----------------------  --------------------------------------------------------
        org                     Optional boolean. Default is False, don't share with
                                the organization.
        ----------------------  --------------------------------------------------------
        groups                  Optional list of group ids as strings, or a list of
                                arcgis.gis.Group objects, or a comma-separated list of
                                group IDs.
        ----------------------  --------------------------------------------------------
        allow_members_to_edit   Optional boolean. Default is False, to allow item to be
                                shared with groups that allow shared update
        ======================  ========================================================

        :return:
            A dictionary with key "notSharedWith" containing array of groups with which the items could not be shared.
        """
        site = self._gis.sites.get(self.site_id)
        result1 = site.item.share(
            everyone=everyone,
            org=org,
            groups=groups,
            allow_members_to_edit=allow_members_to_edit,
        )
        result2 = self.item.share(
            everyone=everyone,
            org=org,
            groups=groups,
            allow_members_to_edit=allow_members_to_edit,
        )
        print(result1)
        return result2

    def unshare(self, groups: list) -> dict:
        """
        Stops sharing of the initiative and its associated site with the specified list of groups.

        ================  =========================================================================================
        **Parameter**      **Description**
        ----------------  -----------------------------------------------------------------------------------------
        groups            Required list of group names as strings, or a list of arcgis.gis.Group objects,
                          or a comma-separated list of group IDs.
        ================  =========================================================================================

        :return:
            Dictionary with key "notUnsharedFrom" containing array of groups from which the items could not be unshared.
        """
        site = self._gis.sites.get(self.site_id)
        result1 = site.item.unshare(groups=groups)
        result2 = self.item.unshare(groups=groups)
        print(result1)
        return result2

    def update(self, initiative_properties=None):
        """Updates the initiative.


        .. note::
            For initiative_properties, pass in arguments for only the properties you want to be updated.
            All other properties will be untouched.  For example, if you want to update only the
            initiative's description, then only provide the description argument in initiative_properties.


        =====================     ====================================================================
        **Parameter**              **Description**
        ---------------------     --------------------------------------------------------------------
        initiative_properties     Required dictionary. See URL below for the keys and values.
        =====================     ====================================================================


        To find the list of applicable options for argument `initiative_properties`, please see the *Item*
        :meth:`~arcgis.gis.Item.update` documentation.

        :return:
           A boolean indicating success (True) or failure (False).

        .. code-block:: python

            USAGE EXAMPLE: Update an initiative successfully

            initiative1 = myHub.initiatives.get('itemId12345')
            initiative1.update(initiative_properties={'description':'Create your own initiative to organize people around a shared goal.'})

            >> True
        """
        if initiative_properties:
            _initiative_data = self.definition
            for key, value in initiative_properties.items():
                _initiative_data[key] = value
                if key == "title":
                    title = value
                    # Fetch Initiative Collaboration group
                    try:
                        _collab_group = self._gis.groups.get(self.collab_group_id)
                        _collab_group.update(title=title + " Core Team")
                    except:
                        pass
                    # Fetch Followers Group
                    try:
                        _followers_group = self._gis.groups.get(self.followers_group_id)
                        _followers_group.update(title=title + " Followers")
                    except:
                        pass
                    # Fetch Content Group
                    _content_group = self._gis.groups.get(self.content_group_id)
                    # Update title for group
                    _content_group.update(title=title + " Content")
            return self.item.update(_initiative_data)


class InitiativeManager(object):
    """
    Helper class for managing initiatives within a Hub. This class is not created by users directly.
    An instance of this class, called 'initiatives', is available as a property of the Hub object. Users
    call methods on this 'initiatives' object to manipulate (add, get, search, etc) initiatives.
    """

    def __init__(self, hub, initiative=None):
        self._hub = hub
        self._gis = self._hub.gis

    def add(self, title, description=None, site=None):
        """
        Adds a new initiative to the Hub.

        =================       ====================================================================
        **Parameter**            **Description**
        -----------------       --------------------------------------------------------------------
        title                   Required string.
        -----------------       --------------------------------------------------------------------
        description             Optional string.
        -----------------       --------------------------------------------------------------------
        site                    Optional Site object.
        =================       ====================================================================

        :return:
           The :class:`~arcgis.apps.hub.initiatives.Initiative` object if successfully added, None if unsuccessful.

        .. code-block:: python

            USAGE EXAMPLE: Add an initiative successfully

            initiative1 = myHub.initiatives.add(title='Vision Zero Analysis')
            initiative1.item
        """

        # Define initiative
        if description is None:
            description = "Create your own initiative by combining existing applications with a custom site."
        _snippet = "Create your own initiative by combining existing applications with a custom site. Use this initiative to form teams around a problem and invite your community to participate."
        _item_dict = {
            "type": "Hub Initiative",
            "snippet": _snippet,
            "typekeywords": "Hub, hubInitiative, OpenData",
            "title": title,
            "description": description,
            "licenseInfo": "CC-BY-SA",
            "culture": "{{culture}}",
            "properties": {},
        }

        # Defining content, collaboration and followers groups
        _content_group_title = title + " Content"
        _content_group_dict = {
            "title": _content_group_title,
            "tags": [
                "Hub Group",
                "Hub Content Group",
                "Hub Site Group",
                "Hub Initiative Group",
            ],
            "access": "public",
        }
        _collab_group_title = title + " Core Team"
        _collab_group_dict = {
            "title": _collab_group_title,
            "tags": [
                "Hub Group",
                "Hub Initiative Group",
                "Hub Site Group",
                "Hub Core Team Group",
                "Hub Team Group",
            ],
            "access": "org",
            "capabilities": "updateitemcontrol",
            "membershipAccess": "collaboration",
            "snippet": "Members of this group can create, edit, and manage the site, pages, and other content related to hub-groups.",
        }
        _followers_group_title = title + " Followers"
        _followers_group_dict = {
            "title": _followers_group_title,
            "tags": [
                "Hub Group",
                "Hub Initiative Group",
                " Hub Initiative Followers Group",
            ],
            "access": "public",
        }

        # Create groups
        content_group = self._gis.groups.create_from_dict(_content_group_dict)
        # Protect groups from accidental deletion
        content_group.protected = True
        # Adding it to _item_dict
        _item_dict["properties"]["contentGroupId"] = content_group.id
        if self._gis.users.me.role == "org_admin":
            collab_group = self._gis.groups.create_from_dict(_collab_group_dict)
            collab_group.protected = True
            _item_dict["properties"]["collaborationGroupId"] = collab_group.id
        if self._hub._hub_enabled:
            followers_group = self._gis.groups.create_from_dict(_followers_group_dict)
            followers_group.protected = True
            _item_dict["properties"]["followersGroupId"] = followers_group.id

        # Create initiative and share it with collaboration group if exists
        item = self._gis.content.add(_item_dict, owner=self._gis.users.me.username)
        try:
            item.share(groups=[collab_group])
        except:
            pass

        # Create initiative site and set initiative properties
        _initiative = Initiative(self._gis, item)
        # If it is a brand new initiative, create new site
        if site is None:
            site = _initiative.sites.add(title=title)
        # else clone existing site
        else:
            site = _initiative.sites.clone(site, pages=True, title=title)
        item.update(
            item_properties={
                "url": site.url,
                "culture": self._gis.properties.user.culture,
            }
        )
        _initiative.site_url = site.item.url
        item.properties["site_id"] = site.itemid

        # update initiative data
        _item_data = {
            "assets": [
                {
                    "id": "bannerImage",
                    "properties": {
                        "type": "resource",
                        "fileName": "detail-image.jpg",
                        "mimeType": "image/jepg",
                    },
                    "license": {"type": "none"},
                    "display": {"position": {"x": "center", "y": "center"}},
                },
                {
                    "id": "iconDark",
                    "properties": {
                        "type": "resource",
                        "fileName": "icon-dark.png",
                        "mimeType": "image/png",
                    },
                    "license": {"type": "none"},
                },
                {
                    "id": "iconLight",
                    "properties": {
                        "type": "resource",
                        "fileName": "icon-light.png",
                        "mimeType": "image/png",
                    },
                    "license": {"type": "none"},
                },
            ],
            "steps": [
                {
                    "id": "informTools",
                    "title": "Inform the Public",
                    "description": "Share data about your initiative with the public so people can easily find, download and use your data in different formats.",
                    "templateIds": [],
                    "itemIds": [site.itemid],
                },
                {
                    "id": "listenTools",
                    "title": "Listen to the Public",
                    "description": "Create ways to gather citizen feedback to help inform your city officials.",
                    "templateIds": [],
                    "itemIds": [],
                },
                {
                    "id": "monitorTools",
                    "title": "Monitor Progress",
                    "description": "Establish performance measures that incorporate the publics perspective.",
                    "templateIds": [],
                    "itemIds": [],
                },
            ],
            "indicators": [],
            "values": {
                "bannerImage": {
                    "source": "bannerImage",
                    "display": {"position": {"x": "center", "y": "center"}},
                },
            },
        }
        _data = json.dumps(_item_data)
        item.update(item_properties={"text": _data})
        return Initiative(self._gis, item)

    def clone(self, initiative, origin_hub=None, title=None):
        """
        Clone allows for the creation of an initiative that is derived from the current initiative.

        .. note::
            If both your `origin_hub` and `destination_hub` are Hub Basic organizations, please use the
            `clone` method supported under the :class:`~arcgis.apps.hub.sites.SiteManager` class.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        initiative          Required :class:`~arcgis.apps.hub.initiatives.Initiative` object of initiative to be cloned.
        ---------------     --------------------------------------------------------------------
        origin_hub          Optional :class:`~arcgis.apps.hub.hub.Hub` object. Required only for cross-org clones where the
                            initiative being cloned is not an item with public access.
        ---------------     --------------------------------------------------------------------
        title               Optional String.
        ===============     ====================================================================

        :return:
            :class:`~arcgis.apps.hub.Initiative` object

        .. code-block:: python

            USAGE EXAMPLE: Clone an initiative in another organization

            #Connect to Hub
            hub_origin = gis1.hub
            hub_destination = gis2.hub
            #Fetch initiative
            initiative1 = hub_origin.initiatives.get('itemid12345')
            #Clone in another Hub
            initiative_cloned = hub_destination.initiatives.clone(initiative1, origin_hub=hub_origin)
            initiative_cloned.item


            USAGE EXAMPLE: Clone initiative in the same organization

            myhub = gis.hub
            initiative1 = myhub.initiatives.get('itemid12345')
            initiative2 = myhub.initiatives.clone(initiative1, title='New Initiative')

        """
        from datetime import timezone

        now = datetime.now(timezone.utc)
        # Checking if item of correct type has been passed
        if "hubInitiative" not in initiative.item.typeKeywords:
            raise Exception("Incorrect item type. Initiative item needed for cloning.")
        # Checking if initiative or site needs to be cloned
        if self._hub and origin_hub:
            if not self._hub._hub_enabled and not origin_hub._hub_enabled:
                raise Exception(
                    "For Hub Basic organizations, please clone the site instead of initiative."
                )
        # New title
        if title is None:
            title = initiative.title + "-copy-%s" % int(now.timestamp() * 1000)
        # If cloning within same org
        if origin_hub is None:
            origin_hub = self._hub
        # Fetch site (checking if origin_hub is correct or if initiative is public)
        try:
            site = origin_hub.sites.get(initiative.site_id)
        except:
            raise Exception(
                "Please provide origin_hub of the initiative object, if the initiative is not publicly shared."
            )
        # Create new initiative if destination hub is premium
        if self._hub._hub_enabled:
            # new initiative
            new_initiative = self._hub.initiatives.add(title=title, site=site)
            return new_initiative
        else:
            # Create new site if destination hub is basic/enterprise
            new_site = self._hub.sites.clone(site, pages=True, title=title)
            return new_site

    def get(self, initiative_id: str) -> str:
        """
        Returns the initiative object for the specified initiative_id.

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        initiative_id              Required string. The initiative itemid.
        =======================    =============================================================

        :return:
            The :class:`~arcgis.apps.hub.initiatives.Initiative` object if the item is found, None if the item is not found.

        .. code-block:: python

            USAGE EXAMPLE: Fetch an initiative successfully

            initiative1 = myHub.initiatives.get('itemId12345')
            initiative1.item

        """
        initiativeItem = self._gis.content.get(initiative_id)
        if "hubInitiative" in initiativeItem.typeKeywords:
            return Initiative(self._gis, initiativeItem)
        else:
            raise TypeError("Item is not a valid initiative or is inaccessible.")

    def search(self, title=None, owner=None, created=None, modified=None, tags=None):
        """
        Searches for initiatives.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        title               Optional string. Return initiatives with provided string in title.
        ---------------     --------------------------------------------------------------------
        owner               Optional string. Return initiatives owned by a username.
        ---------------     --------------------------------------------------------------------
        created             Optional string. Date the initiative was created.
                            Shown in milliseconds since UNIX epoch.
        ---------------     --------------------------------------------------------------------
        modified            Optional string. Date the initiative was last modified.
                            Shown in milliseconds since UNIX epoch
        ---------------     --------------------------------------------------------------------
        tags                Optional string. User-defined tags that describe the initiative.
        ===============     ====================================================================

        :return:
           A list of matching :class:`~arcgis.apps.hub.Initiative` objects.
        """

        initiativelist = []

        # Build search query
        query = "typekeywords:hubInitiative"
        if title != None:
            query += " AND title:" + title
        if owner != None:
            query += " AND owner:" + owner
        if created != None:
            query += " AND created:" + created
        if modified != None:
            query += " AND modified:" + modified
        if tags != None:
            query += " AND tags:" + tags

        # Search
        items = self._gis.content.search(query=query, max_items=5000)

        # Return searched initiatives
        for item in items:
            initiativelist.append(Initiative(self._gis, item))
        return initiativelist
