from arcgis.gis import User, GIS
from typing import Any, List


class Invitation:
    """Represents a user's invitation"""

    _url: str = None
    _gis: GIS = None
    _properties: dict = None

    def __init__(self, url: str, gis: GIS):
        self._url = url
        self._gis = gis

    @property
    def properties(self) -> dict:
        """returns the details of the invitation"""
        if self._properties is None:
            self._properties = self._gis._con.get(self._url, {"f": "json"})
        return self._properties

    def accept(self) -> dict:
        """
        When a group owner or an administrator invites a user to their
        group, it results in a user invitation. The invited user can
        accept the invitation using the `accept` operation. This
        operation adds the invited user to the group, and the invitation is
        deleted. This operation also creates a notification for the user
        indicating that the user's invitation was accepted.

        :return: Dict[str:str]
        """
        url = f"{self._url}/accept"
        res = self._gis._con.post(url, {"f": "json"})
        return res

    def decline(self) -> dict:
        """
        When a group administrator invites a user to their group, it
        results in a group invitation. The invited user can decline the
        invitation using the `decline` operation. The operation deletes the
        invitation and creates a notification for the user indicating that
        they declined the invitation. The invited user is not added to the
        group.

        :returns: Dict[str:str]
        """
        url = f"{self._url}/decline"
        res = self._gis._con.post(url, {"f": "json"})
        return res


class UserInvitationManager:
    """Provides access to the user's invitations"""

    _user = None
    _gis = None
    _url = None

    def __init__(self, user: User):
        self._user: User = user
        self._gis: GIS = user._gis

        self._url = (
            f"{self._gis._portal.resturl}community/users/{user.username}/invitations"
        )

    @property
    def list(self) -> List[Invitation]:
        """
        returns a list of all invitations

        :return: List of :class:`~arcgis.gis.sharing._users.Invitation` objects

        """
        resp = self._gis._con.get(self._url, {"f": "json"})
        return [
            Invitation(self._url + f"/{invites['id']}", self._gis)
            for invites in resp.get("userInvitations", [])
        ]
