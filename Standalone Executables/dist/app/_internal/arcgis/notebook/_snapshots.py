from __future__ import annotations
import sys

import concurrent.futures
from arcgis.gis import GIS, Item
from arcgis.gis.nb import NotebookManager
from arcgis.gis.agonb import AGOLNotebookManager, snapshot as _agosnapshot
from arcgis.gis.nb import _snapshot as _entsnapshot


def create_snapshot(
    item: Item,
    name: str,
    *,
    description: str | None = None,
    notebook_json: dict | None = None,
    access: bool = False,
    server_index: int | None = None,
) -> dict | _entsnapshot.SnapShot | _agosnapshot.SnapShot:
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
    ------------------     --------------------------------------------------------------------
    server_index           Optional Int. The ArcGIS Notebook server to use to run the notebook.  This only applies to ArcGIS Enterprise.
    ==================     ====================================================================

    :return: Snapshot | dict (on error)

    """
    gis = item._gis

    mgrs = gis.notebook_server
    if len(mgrs) == 0:
        raise Exception(
            "The user or organization does not have a notebook server configured."
        )
    if gis._portal.is_arcgisonline:
        mgr = mgrs[0]
        sm = mgr.snaphots
        res = sm.create(
            item=item,
            name=name,
            description=description,
            notebook_json=notebook_json,
            access=access,
        )

    else:
        mgr = mgrs[server_index or 0]
        sm = mgr.notebooks.snapshots
        res = sm.create(
            item=item,
            name=name,
            description=description,
            notebook_json=notebook_json,
            access=access,
        )
    if "resourceKey" in res:
        l = [
            s
            for s in item.snapshots
            if s.properties["resourceKey"] == res["resourceKey"]
        ]
        if len(l) > 0:
            return l[0]

    elif "snapshotResourceKey" in res:
        l = [
            s
            for s in item.snapshots
            if s.properties["resourceKey"] == res["snapshotResourceKey"]
        ]
        if len(l) > 0:
            return l[0]

    else:
        return res


def list_snapshots(item: Item) -> list:
    """
    Returns all snapshots associated with a given item.

    :returns: list
    """
    return item.snapshots
