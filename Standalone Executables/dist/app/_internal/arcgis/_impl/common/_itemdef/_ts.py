import os
import uuid
import copy
import shutil
import logging
import tempfile
from typing import List, Any
from arcgis._impl.common._clone import (
    CloneNode,
    _deep_get,
    _ItemDefinition,
    _ItemCreateException,
)
from arcgis._impl.common._clone import (
    _search_org_for_existing_item,
    _share_item_with_groups,
    _TextItemDefinition,
)

try:
    import ujson as json
except ImportError:
    import json

from arcgis.gis.clone import BaseCloneItemDefinition


class _TileItemDefinition(BaseCloneItemDefinition):  # _ItemDefinition):
    """
    Represents the definition of a tile based item within ArcGIS Online or Portal.
    """

    # --------------------------------------------------------------------
    @staticmethod
    def is_tileservice(item) -> bool:
        return (
            "Hosted Service" in item.typeKeywords and "Map Service" in item.typeKeywords
        )

    # --------------------------------------------------------------------
    def clone(self):
        """Clone the item in the target organization."""
        try:
            new_item = None
            original_item = self.info
            if self._search_existing:
                new_item = _search_org_for_existing_item(self.target, self.portal_item)
            if not new_item:
                # Get the item properties from the original item to be applied when the new item is created
                item_properties = self._get_item_properties(self.item_extent)
                data = self.data
                if data:
                    item_properties["text"] = json.dumps(data)

                # Add the new item
                new_item = self._add_new_item(item_properties)
            else:
                logging.info(
                    self.portal_item.title
                    + " not cloned; already existent in target org."
                )
            _share_item_with_groups(
                new_item, self.sharing, self._clone_mapping["Group IDs"]
            )
            self.resolved = True
            self._clone_mapping["Item IDs"][original_item["id"]] = new_item["id"]
            return new_item
        except Exception as ex:
            raise _ItemCreateException(
                "Failed to create {0} {1}: {2}".format(
                    original_item["type"], original_item["title"], str(ex)
                ),
                new_item,
            )

    def _add_new_item(self, item_properties, data=None):
        """Add the new item to the portal"""
        thumbnail = self.thumbnail
        if not thumbnail and self.portal_item:
            temp_dir = os.path.join(self._temp_dir.name, self.info["id"])
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            thumbnail = self.portal_item.download_thumbnail(temp_dir)
        item_id = None
        if self._preserve_item_id and self.target._portal.is_arcgisonline == False:
            item_id = self.portal_item.itemid

        # Get the related source items and clone them (tile packages)
        related_items = self.portal_item.related_items("Service2Data")
        if len(related_items) == 0:
            raise Exception("Could not locate the source tile package.")

        tpk_result = self.target.content.clone_items(related_items, folder=self.folder)
        if len(tpk_result) > 0:
            self.created_items.extend(tpk_result)
        elif tpk_result == []:
            tpk_result = self.target.content.search(
                "source-{}".format(self.portal_item.related_items("Service2Data")[0].id)
            )
        # add the tile package
        if self._preserve_item_id and self.target._portal.is_arcgisonline == False:
            new_item = tpk_result[0].publish(
                build_initial_cache=True, item_id=self.portal_item.id
            )
        else:
            new_item = tpk_result[0].publish(build_initial_cache=True)

        self.created_items.append(new_item)
        self._clone_resources(new_item)
        return new_item
