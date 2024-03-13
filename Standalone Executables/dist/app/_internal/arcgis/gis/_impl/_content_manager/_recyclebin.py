from __future__ import annotations
from typing import Any, Iterator
from functools import lru_cache
from arcgis.auth.api import LazyLoader
from arcgis.auth import EsriSession
import logging

json = LazyLoader("json")
requests = LazyLoader("requests")
_arcgis_gis = LazyLoader("arcgis.gis")

_log = logging.getLogger(__name__)


###########################################################################
class RecycleItem:
    """
    This represents a recycled item from the recycling bin.

    .. note::
            This functionality is **only** available to organizations participating
            in the Recycle Bin private beta.

    .. code-block:: python

        # Usage Example:

        >>> gis = GIS(profile="your_online_profile")

        >>> org_user = gis.users.search("gis_user1")[0]
        >>> for r_item in org_user.recyclebin.content:
                print(f"{r_item.properties['title']:15}{r_item.properties['type']:22}{type(r_item)}")

        trees_item1    Service Definition   <class 'arcgis.gis._impl._content_manager._recyclebin.RecycleItem'>
        trees_item1    Feature Service      <class 'arcgis.gis._impl._content_manager._recyclebin.RecycleItem'>
        AR_Counties    Feature Service      <class 'arcgis.gis._impl._content_manager._recyclebin.RecycleItem'>
    """

    _item: _arcgis_gis.Item = None

    # ---------------------------------------------------------------------
    def __init__(self, itemid: str, properties: dict[str, Any], gis: _arcgis_gis.GIS):
        self._itemid: str = itemid
        self._gis: _arcgis_gis.GIS = gis
        self._session: EsriSession = gis.session
        self._properties: dict[str, Any] = properties

    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        return f"< RecycleItem: title:{self.properties['title']} type:{self.properties['type']} >"

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"< RecycleItem: title:{self.properties['title']} type:{self.properties['type']} >"

    # ---------------------------------------------------------------------
    @property
    def properties(self) -> dict[str, Any]:
        """
        The properties of the Item in the recycle bin.
        """
        return self._properties

    # ---------------------------------------------------------------------
    def restore(self) -> _arcgis_gis.Item | None:
        """
        Restores the Item from the recycling bin.

        :return: :class:`~arcgis.gis.Item` | None

        .. code-block:: python

            # Usage Example:

            >>> gis = GIS(profile="your_online_profile")

            >>> gis_user = gis.users.me
            >>> deleted_item = list(gis_user.recyclebin.content)[0]
            >>> restored_item = deleted_item.restore()
        """
        url: str = f"{self._gis._public_rest_url}content/users/{self.properties['owner']}/items/{self.properties['id']}/restore"
        params = {
            "f": "json",
        }
        resp: requests.Response = self._session.post(url, data=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if data.get("success"):
            return _arcgis_gis.Item(gis=self._gis, itemid=data.get("itemId"))
        else:
            _log.info(f"Could not restore {data}")

    # ---------------------------------------------------------------------
    def delete(self) -> bool:
        """
        Permanently removes an Item from the organization

        :return: boolean
        """
        url: str = f"{self._gis._public_rest_url}content/users/{self.properties['owner']}/items/{self.properties['id']}/delete"
        params = {"f": "json", "permanentDelete": json.dumps(True)}
        resp: requests.Response = self._session.post(url, data=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if data.get("success"):
            return True
        else:
            _log.error(f"A problem occurred emptying the recycle bin: {data}")
            return False


###########################################################################
class RecycleBin:
    """
    The `RecycleBin` class allows users to manage items they own that were
    deleted.  Users can :meth:`~arcgis.gis._impl._content_manager.RecycleItem.restore`
    or permanently :meth:`~arcgis.gis._impl._content_manager.RecycleItem.delete`
    items from the recycle bin.

    This class is not meant to be initialized directly, but an instance
    is returned by the :attr:`~arcgis.gis.User.recyclebin` property of the
    :class:`~arcgis.gis.User` class. Users can iterate over the
    :attr:`~arcgis.gis._impl._content_manager.RecycleBin.content`.

    .. note::
            This functionality is **only** available to organizations participating
            in the Recycle Bin private beta.

    .. code-block:: python

        # Usage Example:

        >>> gis = GIS(profile="your_online_profile")

        >>> my_recycle_bin = gis.users.me.recyclebin
    """

    _user: _arcgis_gis.User
    _gis: _arcgis_gis.GIS
    _session: EsriSession

    # ---------------------------------------------------------------------
    def __init__(
        self,
        gis: _arcgis_gis.GIS,
        user: _arcgis_gis.User | str | None = None,
    ):
        self._gis: _arcgis_gis.GIS = gis
        self._session: EsriSession = gis.session
        if user is None:
            self._user = self._gis.users.me
        elif isinstance(user, str):
            self._user = self._gis.users.get(user)
        elif isinstance(user, _arcgis_gis.User):
            self._user = user
        else:
            raise ValueError("The `user` parametre must be a str, User or None.")

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self._user.username}'s {self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self._user.username}'s {self.__class__.__name__}>"

    # ---------------------------------------------------------------------
    @lru_cache(maxsize=5)
    def _supported(self) -> bool:
        """checks if the org supports the recycling bin operations"""
        url: str = f"{self._gis._public_rest_url}portals/self"
        params: dict[str, Any] = {
            "f": "json",
        }
        resp: requests.Response = self._session.get(url, params=params)
        data: dict[str, Any] = resp.json()
        return data.get("recycleBinEnabled", False)

    # ---------------------------------------------------------------------
    @property
    def content(self) -> Iterator[RecycleItem]:
        """
        Lists the content inside the recycling bin.

        :return: Iterator[RecycleItem]

        .. code-block:: python

            # Usage Example:

            >>> gis = GIS(profile="your_online_profile")

            >>> my_user = gis.users.me
            >>> r_bin_content = my_user.recyclebin.content
            >>> type(r_bin_content)

            <class 'generator'>

            >>> for r_item in r_bin_content:
                    print(f"{r_item.properties['title']":15}{r_item.properties['type']}")

            trees_sd        Service Definition
            trees_flc       Feature Service

        """
        if self._supported() == False:
            _log.info("The recyclebin is not supported on this organization.")
            return

        url: str = f"{self._gis._public_rest_url}content/users/{self._user.username}"
        params = {
            "f": "json",
            "foldersContent": json.dumps(True),
            "inRecycleBin": json.dumps(True),
            "start": 1,
            "num": 20,
            "sortField": "modified",
            "sortOrder": "desc",
        }
        resp: requests.Response = self._session.get(url=url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        while True:
            for item in data.get("items", []):
                itemid = item.get("id", None)
                if itemid:
                    yield RecycleItem(itemid=itemid, properties=item, gis=self._gis)
            if data.get("nextStart", -1) == -1 or len(data.get("items", [])) <= 0:
                break
            params["start"] = data.get("nextStart", -1)
            resp: requests.Response = self._session.get(url=url, params=params)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
