import os
import json
from collections import namedtuple
from typing import Optional
from arcgis.gis import GIS, Item
from arcgis._impl.common._mixins import PropertyMap


class SnapShot(object):
    """
    A single snapshot instance for a Notebook item.
    """

    _sm = None
    _item = None

    def __init__(self, item: Item, sm: "SnapshotManager", properties: dict):
        self._item = item
        self._sm = sm
        self.properties = properties

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<SnapShot {self.properties['properties']['name']}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<SnapShot {self.properties['properties']['name']}>"

    # ----------------------------------------------------------------------
    def download(self):
        """
        Retrieves a snap shot locally on disk.

        :return: string (path of saved file)


        """
        params = {
            "item": self._item,
            "snapshot": self.properties["resourceKey"],
        }
        return self._sm._download(**params)

    # ----------------------------------------------------------------------
    def save_as_item(self, title: str):
        """
        Converts a Snapshot to a new notebook `Item`.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title                  Required String. The name of the new notebook.
        ==================     ====================================================================

        :return:
            :class:`~arcgis.gis.Item`

        """
        return self._sm._convert(
            item=self._item, snapshot=self.properties["resourceKey"], title=title
        )

    # ----------------------------------------------------------------------
    def restore(
        self,
        title: Optional[str] = None,
        preserve: bool = True,
        description: Optional[str] = None,
    ):
        """
        Rolls back the notebook to a previous snapshot state

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        title                  Optional string. The Item's title.
        ------------------     --------------------------------------------------------------------
        preserve               Optional Bool. If True, the current notebook version is preserved as a snapshot.
        ------------------     --------------------------------------------------------------------
        description            Optional String. Text describing the restoration point.
        ==================     ====================================================================

        :return: Dict
        """
        return self._sm._restore(
            item=self._item,
            title=title,
            snapshot=self.properties["resourceKey"],
            preserve=preserve,
            description=description,
        )

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Deletes a snapshot associated with the notebook item

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to remove snapshots from.
        ------------------     --------------------------------------------------------------------
        snapshot               Required String. The name of the snapshot.
        ==================     ====================================================================

        :return: Boolean

        """
        res = self._sm._delete(item=self._item, snapshot=self.properties["resourceKey"])
        if "status" in res:
            return res["status"] == "success"
        return res


###########################################################################
class SnapshotManager(object):
    """
    Allows for management and creation of snapshots (save points) for ArcGIS Notebooks.
    """

    _gis = None
    _url = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def _convert(self, item, snapshot, title):
        """
        Converts a Snapshot to a new notebook.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to convert.
        ------------------     --------------------------------------------------------------------
        snapshot               Required String. The name of the snapshot.
        ==================     ====================================================================


        :return: Item


        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            url = f"{self._url}/convertToItem"
            params = {
                "f": "json",
                "itemId": item.id,
                "resourceKey": snapshot,
                "notebookTitle": title,
            }
            res = self._gis._con.post(url, params)
            if "itemId" in res:
                return Item(gis=self._gis, itemid=res["itemId"])
            else:
                return res
        else:
            raise ValueError("`item` must be a Notebook")

    # ----------------------------------------------------------------------
    def _download(self, item, snapshot):
        """
        Retrieves a snap shot locally on disk.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to retrieve.
        ------------------     --------------------------------------------------------------------
        snapshot               Required String. The name of the snapshot.
        ==================     ====================================================================

        :return: string (path of saved file)


        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            url = f"{self._url}/download"
            params = {
                "itemId": item.itemid,
                "resourceKey": snapshot,
            }
            return self._gis._con.post(url, params, try_json=False)
        else:
            raise ValueError("`item` must be a Notebook")

    # ----------------------------------------------------------------------
    def create(
        self,
        item: Item,
        name: str,
        description: Optional[str] = None,
        notebook_json: Optional[dict] = None,
        access: bool = False,
    ):
        """
        Creates a Snapshot of a Given Item.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to create a snapshot for.
        ------------------     --------------------------------------------------------------------
        name                   Required String.  The name of the snapshot. This is the identifier
                               used to identify the snapshot.
        ------------------     --------------------------------------------------------------------
        description            Optional String. An piece of text that describes the snapshot.
        ------------------     --------------------------------------------------------------------
        notebook_json          Optional Dict. If you want to store different JSON text other
                               than what is in the current notebook provide it here.
        ------------------     --------------------------------------------------------------------
        access                 Optional Bool. When false, the snapshot will not be publicly available.
        ==================     ====================================================================

        :return: Dict

        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            params = {
                "f": "json",
                "itemId": item.id,
                "name": name,
                "description": description or "",
                "privateAccess": access,
            }

            params["notebookJSON"] = notebook_json or ""
            url = f"{self._url}/create"
            return self._gis._con.post(url, params)
        else:
            raise ValueError("`item` must be a Notebook")

    # ----------------------------------------------------------------------
    def list(self, item: Item):
        """
        Returns a list of SnapShots for a notebook item.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The Notebook :class:`~arcgis.gis.Item` to get all
                               snapshots for.
        ==================     ====================================================================

        :return: List of :class:`~arcgis.gis.nb.SnapShot` objects

        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            params = {
                "f": "json",
                "itemId": item.id,
            }
            url = f"{self._url}/list"
            res = self._gis._con.post(url, params)
            if (
                "status" in res
                and res["status"] == "success"
                and len(res["snapshots"]) > 0
            ):
                return [
                    SnapShot(item=item, sm=self, properties=snap)
                    for snap in res["snapshots"]
                ]
            else:
                return []
        else:
            raise ValueError("`item` must be a Notebook")

    # ----------------------------------------------------------------------
    def _restore(
        self,
        item: Item,
        snapshot: str,
        preserve: bool = True,
        description: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        Rolls back the notebook to a previous snapshot state

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to have rolled back.
        ------------------     --------------------------------------------------------------------
        snapshot               Required String. The name of the snapshot.
        ------------------     --------------------------------------------------------------------
        preserve               Optional Bool. If true, the result is preserved as a snapshot. The
                               default is false.
        ------------------     --------------------------------------------------------------------
        description            Optional String. Text describing the restoration point.
        ------------------     --------------------------------------------------------------------
        title                  Optional string. The title of the item's restored snapshot.
        ==================     ====================================================================

        :return: dict
        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            params = {
                "itemId": item.id,
                "resourceKey": snapshot,
                "preserveCurrentAsSnapshot": preserve,
                "description": description or "",
                "title": title or "",
                "f": "json",
            }
            url = f"{self._url}/restore"
            return self._gis._con.post(url, params)
        else:
            raise ValueError("`item` must be a Notebook")

    # ----------------------------------------------------------------------
    def _delete(self, item: Item, snapshot: str):
        """
        Deletes a snapshot associated with the notebook item

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item                   Required Item. The 'Notebook' typed item to remove snapshots from.
        ------------------     --------------------------------------------------------------------
        snapshot               Required String. The name of the snapshot.
        ==================     ====================================================================

        :return: dict

        """
        if isinstance(item, Item) and item.type.lower() == "notebook":
            params = {"itemId": item.id, "resourceKeys": snapshot, "f": "json"}
            url = f"{self._url}/delete"
            return self._gis._con.post(url, params)
        else:
            raise ValueError("`item` must be a Notebook")
