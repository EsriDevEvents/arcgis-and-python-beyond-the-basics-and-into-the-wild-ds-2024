########################################################################
from __future__ import annotations
from typing import Any, Union
from arcgis.gis import Item
import json


class CategoryManager(object):
    """
    This class allows for the addition, removal and viewing of category
    schema.

    """

    _url = None
    _gis = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Constructor"""
        self._gis = gis
        self._con = gis._con
        baseurl = gis._portal.resturl
        portal_id = None
        if portal_id is None:
            res = self._con.get("%s/portals/self" % baseurl, params={"f": "json"})
            if "id" in res:
                pid = res["id"]
            else:
                raise Exception("Could not find the portal's ID")
        self._url = "%sportals/%s" % (baseurl, pid)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def schema(self):
        """
        Get/Set the catagory schema for a GIS.

        When schema is used as a getter, then operation returns the GIS'
        defined category schema is any.

        When schema is used as a setter, the parameter:

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        value                      optional list. The schema list.
                                   Syntax Example:
                                   [
                                    {
                                      "title": "Themes",
                                      "categories": [
                                        {
                                          "title": "Basemaps",
                                          "categories": [
                                            {"title": "Partner Basemap"},
                                            {
                                              "title": "Esri Basemaps",
                                              "categories": [
                                                {"title": "Esri Redlands Basemap"},
                                                {"title": "Esri Highland Basemap"}
                                              ]
                                            }
                                          ]
                                        },
                                    {
                                      "title": "Region",
                                      "categories": [
                                        {"title": "US"},
                                        {"title": "World"}
                                      ]
                                    }]}]
        =======================    =============================================================

        """
        url = "%s/categorySchema" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @schema.setter
    def schema(self, value: list[dict[str, Any]]):
        """
        See main ``schema`` property docstring
        """
        params = {"f": "json"}
        if value is not None:
            params["categorySchema"] = json.dumps({"categorySchema": value})
            url = "%s/assignCategorySchema" % self._url

            self._con.post(path=url, postdata=params)
        elif value is None:
            url = "%s/deleteCategorySchema" % self._url
            self._con.post(path=url, postdata=params)

    # ----------------------------------------------------------------------
    def categorize_item(self, item: Union[Item, str], categories: list[str]):
        """
        Assigns or removes a category to a single item.

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        item                       Required Item or Item ID (string). The content within a GIS
                                   that will be updated with a list of categories.
        -----------------------    -------------------------------------------------------------
        categories                 Required list. Assigns a list of string values to the item's
                                   categories
        =======================    =============================================================

        :return: Boolean. True if successful else False

        """
        from arcgis.gis import Item

        res = []
        if isinstance(item, Item):
            if categories is None:
                res.append(item.update(item_properties={"categories": ""}))
            else:
                if isinstance(categories, str):
                    categories = [categories]
                res.append(item.update(item_properties={"categories": categories}))
        return all(res)

    # ----------------------------------------------------------------------
    def add(self, items: list[Item], category: str):
        """
        Adds a category to an existing set of items

        =======================    =============================================================
        **Parameter**               **Description**
        -----------------------    -------------------------------------------------------------
        items                      Required Items. The content within a GIS that will be
                                   updated with a list of categories.
        -----------------------    -------------------------------------------------------------
        category                   Required String. Assigns a category value to the items.
        =======================    =============================================================

        :return: Dictionary indicating 'success' or 'error'

        .. code-block:: python

            >>> item = [gis.content.get("<item id 1>"),
                        gis.content.get("<item id 2>")]
            >>> cs = gis.admin.category_schema
            >>> print(cs.add(items=[item], category="/Categories/TEST3"))
            [{'results': [{'itemId': '<item id 1>', 'success': True}]},
             {'results': [{'itemId': '<item id 2>', 'success': True}]}]


        """

        path = self._gis._portal.resturl + "content/updateItems"
        params = {"f": "json"}
        updates = []
        results = []
        content = self._gis.content
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Item):
                    categories = item.categories
                    categories.append(category)
                    updates.append({item.itemid: {"categories": categories}})
                elif isinstance(item, str):
                    item = content.get(item)
                    categories = item.categories
                    categories.append(category)
                    updates.append({item.itemid: {"categories": categories}})
                del item
        elif isinstance(items, str):
            item = content.get(items)
            categories = item.categories
            categories.append(category)
            updates.append({item.itemid: {"categories": categories}})
        elif isinstance(items, Item):
            categories = items.categories
            categories.append(category)
            updates.append({items.itemid: {"categories": categories}})
        else:
            raise ValueError("Invalid items, must be list of Item, Item, or item id")

        def _chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i : i + n]

        for i in _chunks(l=updates, n=100):
            params["items"] = i

            res = self._gis._con.post(path=path, postdata=params)
            results.append(res)
            del i
        return results

    # ----------------------------------------------------------------------
    def remove(self, items: list[Item], category: str):
        """remove a category to an item or items"""

        path = self._gis._portal.resturl + "content/updateItems"
        params = {"f": "json"}
        updates = []
        results = []
        content = self._gis.content
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Item):
                    categories = item.categories
                    if category in categories:
                        del categories[categories.index(category)]
                        updates.append({item.itemid: {"categories": categories}})
                elif isinstance(item, str):
                    item = content.get(item)
                    categories = item.categories
                    if category in categories:
                        del categories[categories.index(category)]
                        updates.append({item.itemid: {"categories": categories}})
                del item
        elif isinstance(items, str):
            item = content.get(items)
            categories = item.categories
            if category in categories:
                del categories[categories.index(category)]
                updates.append({item.itemid: {"categories": categories}})
        elif isinstance(items, Item):
            categories = items.categories
            if category in categories:
                del categories[categories.index(category)]
                updates.append({items.itemid: {"categories": categories}})
        else:
            raise ValueError("Invalid items, must be list of Item, Item, or item id")

        def _chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i : i + n]

        for i in _chunks(l=updates, n=100):
            params["items"] = i

            res = self._gis._con.post(path=path, postdata=params)
            results.append(res)
            del i
        return results

    # ----------------------------------------------------------------------
    def replace(self, items: list[Item], old_category: str, new_catgory: str):
        """finds and replaces a category value with a new value one"""
        res = self.add(items, new_catgory)
        res = self.remove(items, old_category)
        return res

    # ----------------------------------------------------------------------
    def reset(self, items: list[Item]):
        """deletes all the categories for a given set of items"""
        return self._gis.content.bulk_update(items, {"categories": ""})
