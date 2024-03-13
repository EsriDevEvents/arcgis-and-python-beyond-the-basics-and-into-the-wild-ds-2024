class InvitationManager(object):
    """

    The ``InvitationManager`` provides functionality to see the existing invitations
    set out via email to your organization.

    .. note::
        The manager has the ability to delete
        any invitation sent out by an organization.

    """

    _gis = None
    _url = None
    _invites = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        self._url = url
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< InvitationManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< InvitationManager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __len__(self):
        url = self._url
        params = {"f": "json", "num": 100, "start": 1}
        res = self._gis._con.get(url, params)
        return res["total"]

    # ----------------------------------------------------------------------
    def list(self):
        """
        The ``list`` method retrieves all the organization's invitations.

        :return:
            A List of the organization's invitations

        """
        invites = []
        from arcgis.gis import GIS

        isinstance(self._gis, GIS)
        url = self._url
        params = {"f": "json", "num": 100, "start": 1}
        res = self._gis._con.get(url, params)
        invites = res["invitations"]
        while res["nextStart"] > -1:
            params["start"] += 100
            res = self._gis._con.get(url, params)
            invites.extend(res["invitations"])
        return invites

    def get(self, invite_id: str):
        """
        The ``get`` method retrieves information about a single invitation.

        :return:
            A dictionary

        """
        url = self._url + "/{id}".format(id=invite_id)
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    # ----------------------------------------------------------------------
    def _accept(self, invite_id):
        """Accepts the Invitation"""
        url = f"{self._url}/{invite_id}/accept"
        return self._gis._con.post(url, params={"f": "json"})

    # ----------------------------------------------------------------------
    def _decline(self, invite_id):
        """Reject the Invitation"""
        url = f"{self._url}/{invite_id}/decline"
        return self._gis._con.post(url, params={"f": "json"})

    # ----------------------------------------------------------------------
    def manage_invitations(self, accepts: list = None, declines: list = None) -> dict:
        """
        The ``manage_invitations`` method allows users to Accept/Decline invitations by providing a list of
        invitation IDs.

        :return:
            A List of invitation IDs
        """
        results = []
        if accepts and isinstance(accepts, (list, tuple)):
            accepts = [self._accept(invid) for invid in accepts]
        if declines and isinstance(declines, (list, tuple)):
            declines = [self._decline(invid) for invid in declines]
        return {"accepts": accepts, "declines": declines}

    # ----------------------------------------------------------------------
    def delete(self, invite_id: str):
        """
        The ``delete`` method deletes an invitation by ID

        :return:
            A boolean indicating success (True), or failure (False)
        """
        url = self._url + "/{id}/delete".format(id=invite_id)
        params = {"f": "json"}
        res = self._gis._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res
