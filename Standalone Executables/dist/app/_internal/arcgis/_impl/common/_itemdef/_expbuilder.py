import os
import uuid
import copy
import shutil
import tempfile
import logging
from arcgis._impl.common._clone import CloneNode, _deep_get, _ItemDefinition
from arcgis._impl.common._clone import (
    _search_org_for_existing_item,
    _share_item_with_groups,
)
import tempfile

try:
    import ujson as json
except ImportError:
    import json


class _WebExperience(_ItemDefinition):
    """Clones an Web Experience Item"""

    def __init__(
        self,
        target,
        clone_mapping,
        info,
        data=None,
        sharing=None,
        thumbnail=None,
        portal_item=None,
        folder=None,
        item_extent=None,
        search_existing=True,
        owner=None,
        **kwargs,
    ):
        super().__init__(
            target=target,
            clone_mapping=clone_mapping,
            info=info,
            data=data,
            sharing=sharing,
            search_existing=search_existing,
            portal_item=portal_item,
            thumbnail=thumbnail,
            folder=folder,
            item_extent=item_extent,
            owner=owner,
            **kwargs,
        )
        self.info = info
        self._preserve_item_id = kwargs.pop("preserve_item_id", False)
        self._data = data
        self.sharing = sharing
        if not self.sharing:
            self.sharing = {"access": "private", "groups": []}
        self.thumbnail = thumbnail
        self._item_property_names = [
            "title",
            "type",
            "description",
            "snippet",
            "tags",
            "culture",
            "accessInformation",
            "licenseInfo",
            "typeKeywords",
            "extent",
            "url",
            "properties",
        ]
        self.portal_item = portal_item
        self.folder = folder
        self.owner = owner
        self.item_extent = item_extent
        self.created_items = []
        self.resources = kwargs.pop("resources", None)
        self._clone_mapping = clone_mapping
        self.clone_mapping = self._clone_mapping

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
        item_properties["text"] = data
        new_item = self.target.content.add(
            item_properties=item_properties,
            thumbnail=thumbnail,
            folder=self.folder,
            owner=self.owner,
            item_id=item_id,
        )
        self.created_items.append(new_item)
        self._clone_resources(new_item)
        return new_item

    def clone(self):
        def _clone_dict(data_dict, source, target, search_ex):
            new_dict = data_dict
            new_dict["attributes"]["portalUrl"] = target.url
            for k, v in new_dict["dataSources"].items():
                if "itemId" not in v:
                    continue
                v["portalUrl"] = target.url
                orig_id = v["itemId"]
                item = source.content.get(v["itemId"])

                # if predefined in clone mapping
                if orig_id in self._clone_mapping["Item IDs"]:
                    new_id = self._clone_mapping["Item IDs"][orig_id]
                    targ_item = target.content.get(new_id)
                    if targ_item:
                        if targ_item.type == item.type:
                            v["itemId"] = new_id
                            if "url" in v:
                                v["url"] = targ_item.url
                            continue

                # if not, try cloning item
                clone_result = target.content.clone_items(
                    [item],
                    search_existing_items=search_ex,
                    folder=self.folder,
                    owner=self.owner,
                    item_extent=self.item_extent,
                    preserve_item_id=self._preserve_item_id,
                )
                if clone_result:
                    v["itemId"] = clone_result[0].itemid
                    for cloned in clone_result:
                        self.created_items.append(cloned)

                # if it wasn't cloned, search for the existing item
                else:
                    targ_item = _search_org_for_existing_item(self.target, item)
                    v["itemId"] = targ_item.itemid

            return new_dict

        new_item = None
        original_item = self.info
        if self._search_existing:
            new_item = _search_org_for_existing_item(self.target, self.portal_item)
        if not new_item:
            # Get the item properties from the original item to be applied when the new item is created
            item_properties = self._get_item_properties(self.item_extent)
            new_item = self._add_new_item(item_properties)
            if self.resources:
                new_item.resources.add(self.resources, archive=True)
            config_dict = self.portal_item.resources.get("config/config.json")
            new_dict = _clone_dict(
                config_dict, self.portal_item._gis, self.target, self._search_existing
            )
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as tfile:
                json.dump(new_dict, tfile)
                tfile.close()
            new_item.resources.update(
                folder_name="config",
                file_name="config.json",
                file=tfile.name,
            )
            if new_item.url:
                new_item.update(
                    {"url": new_item.url.replace(self.portal_item.id, new_item.id)}
                )
            keywords = new_item.typeKeywords
            for word in keywords:
                if "status" in word:
                    if "Published" in word or "Changed" in word:
                        new_data = _clone_dict(
                            self.portal_item.get_data(),
                            self.portal_item._gis,
                            self.target,
                            True,
                        )
                        new_item.update(item_properties={}, data=new_data)
                    else:
                        new_item.update(
                            item_properties={}, data={"__not_publish": True}
                        )
                    break
            _share_item_with_groups(
                new_item, self.sharing, self._clone_mapping["Group IDs"]
            )
        else:
            logging.info(
                self.portal_item.title + " not cloned; already existent in target org."
            )
        self.resolved = True
        self._clone_mapping["Item IDs"][original_item["id"]] = new_item["id"]
        return new_item
