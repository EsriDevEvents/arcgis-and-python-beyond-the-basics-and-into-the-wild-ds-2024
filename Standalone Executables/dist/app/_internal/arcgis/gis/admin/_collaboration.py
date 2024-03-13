"""
Classes to manage a GIS Collaboration
"""
from __future__ import annotations
import concurrent.futures
import functools
from .. import GIS, Group
from typing import Optional, Union


def _lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class CollaborationManager(object):
    _gis = None
    _basepath = None
    _pid = None

    def __init__(self, gis: GIS, portal_id: str = None):
        self._gis = gis
        self._portal = gis._portal
        self._pid = portal_id
        if portal_id is None:
            res = self._portal.con.get("portals/self")
            if "id" in res:
                self._pid = res["id"]
            else:
                raise Exception("Could not find the portal's ID")
        self._basepath = "portals/%s" % self._pid

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        description: str,
        workspace_name: str,
        workspace_description: str,
        portal_group_id: str,
        host_contact_first_name: str,
        host_contact_last_name: str,
        host_contact_email_address: str,
        access_mode: str = "sendAndReceive",
    ):
        """
        The create method creates a collaboration. The host
        of the collaboration is the portal where it is created. The initial
        workspace for the collaboration is also created. A portal group in
        the host portal is linked to the workspace. The access mode for the
        host portal is set. The contact information associated with the
        host can be specified; otherwise, the contact information for the
        administrator user performing the operation will be used.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        name                            Required string. Name of the collaboration
        ---------------------------     --------------------------------------------------------------------
        description                     Required string. Description of the collaboration
        ---------------------------     --------------------------------------------------------------------
        workspace_name                  Required string. The name of the initial workspace.
        ---------------------------     --------------------------------------------------------------------
        workspace_description           Required string. The description of the initial workspace.
        ---------------------------     --------------------------------------------------------------------
        portal_group_id                 Required string. ID of group in the portal that will be linked with
                                        the workspace.
        ---------------------------     --------------------------------------------------------------------
        host_contact_first_name         Required string. The first name of the contact person for the
                                        collaboration host portal.
        ---------------------------     --------------------------------------------------------------------
        host_contact_last_name          Required string. The last name of the contact person for the
                                        collaboration host portal.
        ---------------------------     --------------------------------------------------------------------
        host_contact_email_address      Required string. The email address of the contact person for the
                                        collaboration host portal.
        ---------------------------     --------------------------------------------------------------------
        access_mode                     Required string. The organization's access mode to the workspace.
                                        Values: send | receive | sendAndReceive (default)
        ===========================     ====================================================================


        :return: the data item is registered successfully, None otherwise

        """
        if access_mode not in ["send", "receive", "sendAndReceive"]:
            raise Exception(
                "Invalid access_mode. Must be of value: send, "
                + "receive or sendAndReceive."
            )
        params = {
            "f": "json",
            "name": name,
            "description": description,
            "workspaceName": workspace_name,
            "workspaceDescription": workspace_description,
            "portalGroupId": portal_group_id,
            "hostContactFirstname": host_contact_first_name,
            "hostContactLastname": host_contact_last_name,
            "hostContactEmailAddress": host_contact_email_address,
            "accessMode": access_mode,
            "config": {},
        }

        data_path = "%s/createCollaboration" % self._basepath
        res = self._portal.con.post(data_path, params)
        if "collaboration" in res and "id" in res["collaboration"]:
            return Collaboration(
                collab_manager=self,
                collab_id=res["collaboration"]["id"],
                portal_id=self._pid,
            )

    # ----------------------------------------------------------------------
    def accept_invitation(
        self,
        first_name: str,
        last_name: str,
        email: str,
        invitation_file: Optional[str] = None,
        invitation_JSON: Optional[str] = None,
        webauth_username: Optional[str] = None,
        webauth_password: Optional[str] = None,
        webauth_cert_file: Optional[str] = None,
        webauth_cert_password: Optional[str] = None,
    ) -> dict:
        """
        The accept_invitation operation allows a portal to accept a
        collaboration invitation. The invitation file received securely
        from the collaboration host portal must be provided. Once a guest
        accepts an invitation to a collaboration, it must link workspace(s)
        associated with the collaboration to local portal group(s). The
        guest must export a collaboration invitation response file and send
        it to the host. Once the host processes the response, content can
        be shared between the host and guest(s).

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        first_name                      Required string. The first name of the contact person for the guest
                                        portal.
        ---------------------------     --------------------------------------------------------------------
        last_name                       Required string. The last name of the contact person.
        ---------------------------     --------------------------------------------------------------------
        email                           Required string. The email of the contact person.
        ---------------------------     --------------------------------------------------------------------
        invitation_file                 Optional string. The invite file to upload to portal. Use either
                                        this parameter or invitation_JSON.
        ---------------------------     --------------------------------------------------------------------
        invitation_JSON                 Optional string. The same contents as the invitation_file parameter
                                        but passed as a string. Use either this parameter or invitation_file.
        ---------------------------     --------------------------------------------------------------------
        webauth_username                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication user name.
        ---------------------------     --------------------------------------------------------------------
        webauth_password                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication password.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_file               Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate file.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_password           Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate password.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "%s/acceptCollaborationInvitation" % self._basepath
        params = {
            "f": "json",
            "guestContactFirstname": first_name,
            "guestContactLastname": last_name,
            "guestContactEmailAddress": email,
        }
        files = None
        if invitation_file is None and invitation_JSON is None:
            raise ValueError("invitation_file or invitation_JSON must be provided")
        if invitation_file:
            files = {}
            files["invitationFile"] = invitation_file
        if invitation_JSON:
            params["invitationJSON"] = invitation_JSON
        if webauth_cert_file:
            if files is None:
                files = {}
            files["hostWebauthCertificateFile"] = webauth_cert_file
        if webauth_cert_password:
            params["hostWebauthCertPassword"] = webauth_cert_password
        if webauth_password and webauth_username:
            params["hostWebauthUsername"] = webauth_username
            params["hostWebauthPassword"] = webauth_password
        con = self._portal.con
        return con.post(path=data_path, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def list(self) -> list[Collaboration]:
        """gets all collaborations for a portal"""
        data_path = "%s/collaborations" % self._basepath
        params = {"f": "json", "num": 100, "start": 1}
        res = self._portal.con.get(data_path, params)
        collabs = []
        collab_ids = []
        while len(res["collaborations"]) > 0:
            for collab in res["collaborations"]:
                collab_ids.append(collab)
            if res["nextStart"] == -1:
                with concurrent.futures.ThreadPoolExecutor(25) as tp:
                    jobs = {
                        tp.submit(
                            Collaboration,
                            **{
                                "collab_manager": self,
                                "collab_id": collab["id"],
                                "portal_id": self._gis.properties.id,
                            },
                        ): collab
                        for collab in collab_ids
                    }
                    for future in concurrent.futures.as_completed(jobs):
                        collab = jobs[future]
                        try:
                            collabs.append(future.result())
                        except Exception as exc:
                            print("%r generated an exception: %s" % (collab, exc))
                return collabs
            else:
                params["start"] = res["nextStart"]
                res = self._portal.con.get(data_path, params)
        return collabs

    # ----------------------------------------------------------------------
    def validate_invitation(
        self,
        first_name: str,
        last_name: str,
        email: str,
        invitation_file: Optional[str] = None,
        invitation_JSON: Optional[str] = None,
        webauth_username: Optional[str] = None,
        webauth_password: Optional[str] = None,
        webauth_cert_file: Optional[str] = None,
        webauth_cert_password: Optional[str] = None,
    ) -> dict:
        """
        The validate_invitation method allows a portal to
        validate a collaboration invitation. The invitation file received
        securely from the collaboration host portal must be provided.
        Validation checks include checking that the invitation is for the
        intended recipient.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        first_name                      Required string. The first name of the contact person for the guest
                                        portal.
        ---------------------------     --------------------------------------------------------------------
        last_name                       Required string. The last name of the contact person.
        ---------------------------     --------------------------------------------------------------------
        email                           Required string. The email of the contact person.
        ---------------------------     --------------------------------------------------------------------
        invitation_file                 Optional string. The invite file to upload to portal. Use either
                                        this parameter or invitation_JSON.
        ---------------------------     --------------------------------------------------------------------
        invitation_JSON                 Optional string. The same contents as the invitation_file parameter
                                        but passed as a string. Use either this parameter or invitation_file.
        ---------------------------     --------------------------------------------------------------------
        webauth_username                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication user name.
        ---------------------------     --------------------------------------------------------------------
        webauth_password                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication password.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_file               Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate file.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_password           Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate password.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "%s/validateCollaborationInvitation" % self._basepath
        params = {
            "f": "json",
            "guestContactFirstname": first_name,
            "guestContactLastname": last_name,
            "guestContactEmailAddress": email,
        }
        files = None
        if invitation_file is None and invitation_JSON is None:
            raise ValueError("invitation_file or invitation_JSON must be provided")
        if invitation_file:
            files = {}
            files["invitationFile"] = invitation_file
        if invitation_JSON:
            params["invitationJSON"] = invitation_JSON
        if webauth_cert_file:
            if files is None:
                files = {}
            files["hostWebauthCertificateFile"] = webauth_cert_file
        if webauth_cert_password:
            params["hostWebauthCertPassword"] = webauth_cert_password
        if webauth_password and webauth_username:
            params["hostWebauthUsername"] = webauth_username
            params["hostWebauthPassword"] = webauth_password
        con = self._portal.con
        return con.post(path=data_path, postdata=params, files=files)

    # ----------------------------------------------------------------------
    def collaborate_with(
        self, guest_gis: GIS, collaboration_name: str, collaboration_description: str
    ) -> bool:
        """
        A high level method to quickly establish a collaboration between two GIS. This method uses defaults
        wherever applicable and internally calls the `create`, `accept_invitation` and `invite_participant` methods.
        This method will create a new group and a new workspace in both the host and guest GIS for this collaboration.
        Invitation and response files created during the collaborations will be downloaded to the current working
        directory.

        **Use the other methods if you need fine-grained control over how the collaboration is set up.**

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        guest_gis                       Required GIS. GIS object of the guest org or Enterprise.
        ---------------------------     --------------------------------------------------------------------
        collaboration_name              Required string. A generic name for the collaboration. This name is
                                        used with prefixes such as wksp_<your_collab_name>,
                                        grp_<your_collab_name> to create the collaboration workspace and
                                        groups.
        ---------------------------     --------------------------------------------------------------------
        collaboration_description       Optional string. A generic description for the collaboration.
        ===========================     ====================================================================

        :return: boolean



        """

        # create a group in the host
        host_group = self._gis.groups.create(
            title="grp_" + collaboration_name,
            tags="collaboration",
            description="Group for " + collaboration_description,
        )

        # create a collaboration in the host
        host_first_name = ""
        host_last_name = ""
        host_email = ""
        if hasattr(self._gis.users.me, "firstName"):
            host_first_name = self._gis.users.me.firstName
            host_last_name = self._gis.users.me.lastName
        elif hasattr(self._gis.users.me, "fullName"):
            sp = self._gis.users.me.fullName.split()
            host_first_name = sp[0]
            if len(sp) > 1:
                host_last_name = sp[1]
            else:
                host_last_name = host_first_name
        if hasattr(self._gis.users.me, "email"):
            host_email = self._gis.users.me.email

        host_collab = self.create(
            name="collab_" + collaboration_name,
            description=collaboration_description,
            workspace_name="wksp_" + collaboration_name,
            workspace_description="Workspace for " + collaboration_description,
            portal_group_id=host_group.id,
            host_contact_first_name=host_first_name,
            host_contact_last_name=host_last_name,
            host_contact_email_address=host_email,
        )

        # Invite guest GIS as participant
        config = [{host_collab.workspaces[0]["id"]: "sendAndReceive"}]
        invite_file = host_collab.invite_participant(config, guest_gis=guest_gis)

        # Create a group in guest GIS
        guest_group = guest_gis.groups.create(
            title="grp_" + collaboration_name,
            tags="collaboration",
            description="Group for " + collaboration_description,
        )

        # Accept invitation in guest GIS
        guest_first_name = ""
        guest_last_name = ""
        guest_email = ""
        if hasattr(guest_gis.users.me, "firstName"):
            guest_first_name = guest_gis.users.me.firstName
            guest_last_name = guest_gis.users.me.lastName
        elif hasattr(guest_gis.users.me, "fullName"):
            sp = self._gis.users.me.fullName.split()
            guest_first_name = sp[0]
            if len(sp) > 1:
                guest_last_name = sp[1]
            else:
                guest_last_name = guest_first_name
        if hasattr(guest_gis.users.me, "email"):
            guest_email = guest_gis.users.me.email
        response = guest_gis.admin.collaborations.accept_invitation(
            first_name=guest_first_name,
            last_name=guest_last_name,
            email=guest_email,
            invitation_file=invite_file,
        )

        # Export response from guest GIS
        guest_collab = None
        response_file = None
        if response["success"]:
            guest_collab = Collaboration(guest_gis.admin.collaborations, host_collab.id)
            response_file = guest_collab.export_invitation("./")
        else:
            raise Exception("Unable to accept collaboration in the guest GIS")

        # Add guest group to guest collab
        group_add_result = guest_collab.add_group_to_workspace(
            guest_group, guest_collab.workspaces[0]
        )

        # Accept response back in the host GIS
        host_collab.import_invitation_response(response_file)

        return True


###########################################################################
class Collaboration(dict):
    """
    The collaboration resource returns information about the collaboration
    with a specified ID.
    """

    _id = None
    _cm = None  # CollaborationManager
    _baseurl = None
    _portal = None

    def __init__(self, collab_manager, collab_id, portal_id=None):
        dict.__init__(self)
        self._id = collab_id
        self._cm = collab_manager
        self._portal = collab_manager._gis._portal
        if portal_id is None:
            res = self._portal.con.get("portals/self")
            if "id" in res:
                portal_id = res["id"]
            else:
                raise Exception("Could not find the portal's ID")
        self._basepath = "portals/%s/collaborations/%s" % (portal_id, collab_id)
        params = {"f": "json"}
        datadict = self._portal.con.post(self._basepath, params, verify_cert=False)

        if datadict:
            self.__dict__.update(datadict)
            super(Collaboration, self).update(datadict)

    def _refresh(self):
        """refreshes the properties"""
        params = {"f": "json"}
        datadict = self._portal.con.post(self._basepath, params, verify_cert=False)

        if datadict:
            self.__dict__.update(datadict)
            super(Collaboration, self).update(datadict)

    # ----------------------------------------------------------------------
    def __getattr__(
        self, name
    ):  # support group attributes as group.access, group.owner, group.phone etc
        try:
            return dict.__getitem__(self, name)
        except:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    # ----------------------------------------------------------------------
    def __getitem__(
        self, k
    ):  # support group attributes as dictionary keys on this object, eg. group['owner']
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            params = {"f": "json"}
            datadict = self._portal.con.post(self._basepath, params, verify_cert=False)
            super(Collaboration, self).update(datadict)
            self.__dict__.update(datadict)
            return dict.__getitem__(self, k)

    # ----------------------------------------------------------------------
    def add_workspace(
        self, name: str, description: str, config: dict, portal_group_id: str
    ) -> dict:
        """
        The add_workspace resource adds a new workspace to a
        portal-to-portal collaboration. Only collaboration hosts can create
        new workspaces.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        name                            Required string. The name of the workspace.
        ---------------------------     --------------------------------------------------------------------
        description                     Required string. Brief description of the workspace.
        ---------------------------     --------------------------------------------------------------------
        portal_group_id                 Required string. The ID of the portal group linked with the
                                        workspace.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        from arcgis.gis import Group

        if isinstance(portal_group_id, Group):
            portal_group_id = portal_group_id.groupid
        params = {
            "f": "json",
            "collaborationWorkspaceName": name,
            "collaborationWorkspaceDescription": description,
            "config": config,
            "portalGroupId": portal_group_id,
        }
        path = "%s/%s" % (self._basepath, "addWorkspace")
        return self._portal.con.post(path, params, verify_cert=False)

    # ----------------------------------------------------------------------
    def get_invitation(self, invitation_id: str) -> dict:
        """
        The get_invitation operation returns the information about an
        invitation to participate in a portal-to-portal collaboration for a
        particular invitation with the specified ID.
        """
        params = {"f": "json"}
        path = "%s/%s/%s" % (self._basepath, "invitations", invitation_id)
        return self._portal.con.get(path, params)

    # ----------------------------------------------------------------------
    def get_workspace(self, workspace_id: str) -> dict:
        """
        The workspace resource provides information about the collaboration
        workspace with a specified ID.
        """
        params = {"f": "json"}
        path = "%s/%s/%s" % (self._basepath, "workspaces", workspace_id)
        return self._portal.con.get(path, params)

    # ----------------------------------------------------------------------
    @property
    def invitations(self) -> list:
        """The invitations operation returns the invitation information for
        all the invitations generated by a portal-to-portal collaboration
        host.
        """
        params = {"f": "json", "start": 1, "nun": 100}
        invs = []
        path = "%s/%s" % (self._basepath, "invitations")
        res = self._portal.con.get(path, params)
        while len(res["collaborationInvitations"]) > 0:
            invs += res["collaborationInvitations"]
            params["start"] = res["nextStart"]
            if res["nextStart"] == -1:
                return invs
            res = self._portal.con.get(path, params)
        return invs

    # ----------------------------------------------------------------------
    def delete(self) -> bool:
        """
        The delete operation deletes a portal-to-portal collaboration from
        the host portal. This stops any sharing set up from the
        collaboration. The collaboration will be removed on guest portals
        on the next refresh of their content based on the collaboration
        sharing schedule. Guests cannot delete collaborations, but they can
        discontinue participation in a collaboration via the
        removeParticipation endpoint.
        """
        params = {"f": "json"}
        data_path = "%s/delete" % self._basepath
        resp = self._portal.con.post(data_path, params)
        if "success" in resp:
            return resp["success"]
        return resp

    # ----------------------------------------------------------------------
    def remove_workspace(self, workspace_id: str) -> dict:
        """
        The delete operation deletes a collaboration workspace. This
        immediately disables further replication of data to and from the
        portal and the collaboration participants.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Optional string. UID of the workspace to remove from the
                                        collaboration.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'



        """
        params = {"f": "json"}
        data_path = "%s/workpaces/%s/delete" % (self._basepath, workspace_id)
        return self._portal.con.post(data_path, params)

    # ----------------------------------------------------------------------
    @_lazy_property
    def workspaces(self) -> list:
        """
        The workspaces resource lists all the workspaces in a given
        collaboration. A workspace is a virtual space in the collaboration
        to which each participating portal is either sending or receiving
        content. Workspaces can only be created by the collaboration owner.
        """
        data_path = "%s/workspaces" % self._basepath
        params = {"f": "json", "num": 100, "start": 1}
        res = self._portal.con.get(data_path, params)
        workspaces = []
        while len(res["workspaces"]) > 0:
            workspaces += res["workspaces"]
            params["start"] = res["nextStart"]
            if res["nextStart"] == -1:
                return workspaces
            res = self._portal.con.get(data_path, params)
        return workspaces

    # ----------------------------------------------------------------------
    def export_invitation(self, out_folder: str) -> dict:
        """
        The exportInvitationResponse operation exports a collaboration
        invitation response file from a collaboration guest portal. The
        exported response file must be sent via email or through other
        communication channels that are established in your organization to
        the inviting portal's administrator. The inviting portal's
        administrator will then import your response file to complete the
        establishment of trust between your portals.
        It is important that the contents of this response file are not
        intercepted and tampered with by any unknown entity.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        out_folder                      Required string. Save location of the file.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/exportInvitationResponse" % self._basepath
        return self._portal.con.post(
            data_path, params, out_folder=out_folder, verify_cert=False
        )

    # ----------------------------------------------------------------------
    def import_invitation_response(
        self,
        response_file: str,
        webauth_username: Optional[str] = None,
        webauth_password: Optional[str] = None,
        webauth_cert_file: Optional[str] = None,
        webauth_cert_password: Optional[str] = None,
    ) -> dict:
        """
        The importInvitationResponse operation imports an invitation
        response file from a portal collaboration guest. The operation is
        performed on the portal that serves as the collaboration host. Once
        an invitation response is imported, trust between the host and the
        guest is established. Sharing of content between participants can
        proceed from this point.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        response_file                   Required string. File path to the response file.
        ---------------------------     --------------------------------------------------------------------
        webauth_username                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication user name.
        ---------------------------     --------------------------------------------------------------------
        webauth_password                Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication password.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_file               Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate file.
        ---------------------------     --------------------------------------------------------------------
        webauth_cert_password           Optional string. If the collaboration host requires web-tier
                                        authentication, optionally use this parameter to provide the host's
                                        web-tier authentication certificate password.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/importInvitationResponse" % self._basepath
        files = {"invitationResponseFile": response_file}
        if webauth_cert_file:
            files["guestWebauthCertificateFile"] = webauth_cert_file
        if webauth_cert_password:
            params["guestWebauthCertPassword"] = webauth_cert_password
        if webauth_username and webauth_password:
            params["guestWebauthUsername"] = webauth_username
            params["guestWebauthPassword"] = webauth_password
        con = self._portal.con
        return con.post(path=data_path, postdata=params, files=files, verify_cert=False)

    # ----------------------------------------------------------------------
    def invalidate(self, invitation_id: str) -> dict:
        """
        The invalidate operation invalidates a previously generated
        portal-to-portal collaboration invitation. If a guest accepts this
        invitation and sends an invitation response for it, the response
        will not import successfully on the collaboration host.
        """
        params = {"f": "json"}
        data_path = "%s/invitations/%s/invalidate" % (self._basepath, invitation_id)
        con = self._portal.con
        return con.post(data_path, params, verify_cert=False)

    # ----------------------------------------------------------------------
    def invite_participant(
        self,
        config_json: dict,
        expiration: int = 24,
        guest_portal_url: Optional[str] = None,
        guest_gis: Optional[GIS] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        As a collaboration host, once you have set up a new collaboration,
        you are ready to invite other portals as participants in your
        collaboration. The inviteParticipant operation allows you to invite
        other portals to your collaboration by creating an invitation file.
        You need to send this invitation file to the administrator of the
        portal you are inviting to your collaboration. This can be done via
        email or through other communication channels that are established
        in your organization. It is important that the contents of this
        invitation file are not intercepted and tampered with by any
        unknown entity. The invitation file is in the format
        collaboration-<guestHostDomain>.invite.
        The administrator of the participant will accept the invitation by
        importing the invitation file into their portal. Their acceptance
        is returned to you as another file that you must import into your
        portal using the import_invitation_response operation. This will
        establish trust between your portal and that of your participant.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        config_json                     Required dict. A dict containing a map of access modes for the
                                        participant in each of the collaboration workspaces.
                                        Defined as: send | receive | sendAndReceive

                                        :Example:

                                        config_json = [
                                          {"workspace_id" : "send"},
                                          {"workspace_id2" : "receive"},
                                          {"workspace_id3" : "sendAndReceive"}
                                        ]
        ---------------------------     --------------------------------------------------------------------
        expiration                      Optional integer. The time in UTC when the invitation to collaborate
                                        should expire.
        ---------------------------     --------------------------------------------------------------------
        guest_portal_url                Optional string. The URL of the participating org or Enterprise that
                                        you want to invite to the collaboration.
        ---------------------------     --------------------------------------------------------------------
        guest_gis                       Optional GIS. GIS object to the guest collaboration site.
        ---------------------------     --------------------------------------------------------------------
        save_path                       Optional string. Path to download the invitation file to.
        ===========================     ====================================================================

        :return: Contents of a file that contains the invitation information


        """
        if guest_gis is None and guest_portal_url is None:
            raise ValueError("A GIS object or URL is required")
        if guest_portal_url is None and guest_gis:
            guest_portal_url = guest_gis._portal.url
        data_path = "%s/inviteParticipant" % self._basepath
        params = {
            "f": "json",
            "guestPortalUrl": guest_portal_url,
            "collaborationWorkspacesParticipantConfigJSON": config_json,
            "expiration": expiration,
        }
        con = self._portal.con
        return con.post(
            path=data_path, postdata=params, verify_cert=False, out_folder=save_path
        )

    # ----------------------------------------------------------------------
    def get_participant(self, portal_id: str) -> dict:
        """
        The participant operation provides information about the
        collaboration participant with a specified ID.
        """
        data_path = "%s/participants/%s" % (self._basepath, portal_id)
        params = {"f": "json"}
        con = self._portal.con
        return con.get(data_path, params)

    # ----------------------------------------------------------------------
    def participants(self) -> dict:
        """
        The participants resource provides information about all of the
        participants in a portal-to-portal collaboration.
        """
        data_path = "%s/participants" % self._basepath
        params = {"f": "json"}
        con = self._portal.con
        return con.get(data_path, params)

    # ----------------------------------------------------------------------
    def update_item_delete_policy(
        self,
        participant_id: str,
        delete_contributed_items: bool = False,
        delete_received_items: bool = False,
    ) -> dict:
        """
        The participants resource provides information about all of the
        participants in a portal-to-portal collaboration.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        participant_id                  Required String. The participant unique id to update.
        ---------------------------     --------------------------------------------------------------------
        delete_contributed_items        Optional Boolean.  When a participant leaves or deletes a collaboration, this property determines whether contributed items will be deleted or maintained.
        ---------------------------     --------------------------------------------------------------------
        delete_received_items           Optional Boolean.  When a participant leaves or deletes a collaboration, this property determines whether received items will be deleted or maintained.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "%s/participants/%s/updateItemDeletePolicy" % (
            self._basepath,
            participant_id,
        )
        params = {
            "f": "json",
            "deleteContributedItems": delete_contributed_items,
            "deleteReceivedItems": delete_received_items,
        }
        con = self._portal.con
        return con.post(data_path, params)

    # ----------------------------------------------------------------------
    def add_group_to_workspace(self, portal_group: str, workspace: str) -> dict:
        """
        This operation adds a group to a workspace that participates in a portal-to-portal collaboration. Content shared
         to the portal group is shared to other participants in the collaboration.


        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        portal_group                    Required Group of string. Group ID or object to add to the workspace.
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        group_id = None
        if isinstance(portal_group, Group):
            group_id = portal_group.groupid
        elif isinstance(portal_group, str):
            group_id = portal_group

        data_path = "{}/workspaces/{}/updatePortalGroupLink".format(
            self._basepath, workspace["id"]
        )
        params = {
            "f": "json",
            "portalGroupId": group_id,
            "enableRealtimeSync": True,
            "copyFeatureServiceData": False,
        }
        con = self._portal.con
        result = con.post(path=data_path, postdata=params, verify_cert=False)
        return result

    # ----------------------------------------------------------------------
    def _force_sync(self, workspace) -> dict:
        """
        Undocumented. This operation will force sync the collaboration and its workspaces
        :param workspace:
        :return:
        """
        config_sync_data_path = "{}/configSync".format(self._basepath)
        config_sync_status = self._portal.con.get(config_sync_data_path, {"f": "json"})

        if config_sync_status["success"]:
            # proceed to workspace sync
            workspace_sync_data_path = "{}/workspaces/{}/sync".format(
                self._basepath, workspace["id"]
            )
            wksp_sync_status = self._portal.con.post(
                workspace_sync_data_path, postdata={"f": "json"}, verify_cert=False
            )
            return wksp_sync_status
        else:
            raise RuntimeError("Error force syncing")

    # ----------------------------------------------------------------------
    def refresh(self, invitation_id: str) -> dict:
        """
        The refresh operation refreshes a previously generated
        portal-to-portal collaboration invitation. The new invitation file
        is provided via a multipart POST response. The expiration for the
        invitation is extended an additional 72 hours from the current
        time.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        invitation_id                   Required string. ID of the invitation to refresh
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/invitations/%s/refresh" % (self._basepath, invitation_id)
        con = self._portal.con
        return con.post(path=data_path, postdata=params, verify_cert=False)

    # ----------------------------------------------------------------------
    def remove_participation(self) -> dict:
        """
        The removeParticipation operation removes collaboration
        participation by a guest from a collaboration, allowing a guest to
        exit a collaboration. This immediately disables further
        replication of data to and from the portal and the other
        collaboration participants.
        """
        data_path = "%s/removeParticipation" % self._basepath
        params = {"f": "json"}
        con = self._portal.con
        return con.post(path=data_path, postdata=params, verify_cert=False)

    # ----------------------------------------------------------------------
    def remove_participant(self, portal_id: str) -> dict:
        """
        The remove operation allows a collaboration host to remove a
        participant from a portal-to-portal collaboration.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        portal_id                       Required string. ID of the portal to remove.
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/participants/%s/remove" % (self._basepath, portal_id)
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def remove_portal_group_link(self, workspace_id: str) -> dict:
        """
        The remove_portal_group_link operation removes the link between a
        collaboration workspace and a portal group. Replication of content
        discontinues when the link is removed.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/removePortalGroupLink" % (
            self._basepath,
            workspace_id,
        )
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def schedule(self, workspace_id: str) -> dict:
        """
        Collaboration guests can use the schedule resource to return a job
        schedule for synchronized items in a collaboration workspace. The
        response is a single JSON object that represents a job schedule.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ===========================     ====================================================================


        :return: A dictionary of the job schedule

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/schedule" % (self._basepath, workspace_id)
        con = self._portal.con
        return con.get(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def pause_schedule(self, workspace_id: str) -> bool:
        """
        Suspends the scheduling job for synchronized items in a collaboration workspace.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/schedule/pause" % (self._basepath, workspace_id)
        con = self._portal.con
        res = con.post(path=data_path, postdata=params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def delete_schedule(self, workspace_id: str) -> bool:
        """
        Removes the scheduling job for synchronized items in a collaboration workspace.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/schedule/delete" % (self._basepath, workspace_id)
        con = self._portal.con
        res = con.post(path=data_path, postdata=params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def resume_schedule(self, workspace_id: str) -> bool:
        """
        Resumes a paused scheduled synchronization.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ===========================     ====================================================================

        :return: Boolean. True if successful else False.

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/schedule/resume" % (self._basepath, workspace_id)
        con = self._portal.con
        res = con.post(path=data_path, postdata=params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def update_schedule(
        self,
        workspace_id: str,
        start_time: int,
        interval: int = 24,
        repeat_count: int = -1,
    ) -> bool:
        """
        Collaboration guests can use the schedule resource to return a job
        schedule for synchronized items in a collaboration workspace. The
        response is a single JSON object that represents a job schedule.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ---------------------------     --------------------------------------------------------------------
        start_time                      Required Integer. A job's scheduled start time. The startTime is in Unix time in milliseconds. The default is the current time of the request call.
        ---------------------------     --------------------------------------------------------------------
        interval                        Optional Integer. A positive integer that represents time (in hours) between each job trigger. The default interval is 24 hours.
        ---------------------------     --------------------------------------------------------------------
        repeat_count                    Optional Integer. A positive integer or -1 which represents how many times to keep re-triggering this job after which it will be automatically deleted. The default is -1 which means repeat indefinitely.
        ===========================     ====================================================================


        :return: Boolean. True if successful else False.

        """
        params = {
            "f": "json",
            "startTime": start_time,
            "interval": interval,
            "repeatCount": repeat_count,
        }
        data_path = "%s/workspaces/%s/schedule/update" % (self._basepath, workspace_id)
        con = self._portal.con
        res = con.post(path=data_path, postdata=params)
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def sync(self, workspace_id: int, run_async: bool = False) -> dict:
        """
        The sync endpoint is provided to allow execution of a data sync on
        a particular workspace. The operation is allowed on the participant
        that is designated to initiate sync operations as determined during
        trust establishment between the collaboration host and a guest
        participant. Typically, the guest participant is designated to
        initiate sync operations. Note that if a scheduled sync operation
        is already in progress a new sync is not started unless the current
        sync operation is finished.

        When running sync in synchronous mode, the client will be blocked
        until the operation is completed. Invoking sync in synchronous mode
        is good for quickly syncing an item (that is not large) if the
        client does not want to wait for the next scheduled sync.

        Asynchronous mode allows a client to get response immediately so it
        does not have to wait and is not blocked from performing other
        tasks.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to remove from the link.
        ---------------------------     --------------------------------------------------------------------
        run_async                       Optional Boolean.  When true, the job will run asynchronously.
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json", "async": run_async}
        data_path = "%s/workspaces/%s/sync" % (self._basepath, workspace_id)
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def sync_status(self, workspace_id: str) -> list:
        """
        Provides a status summary of each scheduled sync for items in a collaboration workspace.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to examine `sync` jobs.
        ===========================     ====================================================================

        :return: List[Dict]

        """
        params = {"f": "json"}
        data_path = f"{self._basepath}/workspaces/{workspace_id}/syncStatus"
        con = self._portal.con
        resp = con.get(path=data_path, postdata=params)
        return resp.get("status") or resp

    # ----------------------------------------------------------------------
    def sync_details(self, workspace_id: str, sync_id: str) -> dict:
        """
        Provides a detailed description of status for a selected sync ID.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. Workspace ID to examine `sync` jobs.
        ---------------------------     --------------------------------------------------------------------
        sync_id                         Required String. When a sync is performed, an ID is generated to
                                        track the status of the synchronization of the collaboration.
        ===========================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        params = {"f": "json"}
        data_path = "%s/workspaces/%s/syncStatus/%s" % (
            self._basepath,
            workspace_id,
            sync_id,
        )
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def update_collaboration(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """
        The updateInfo operation updates certain properties of a
        collaboration, primarily its name, description, and configuration
        properties. The updates are propagated to guests when the next
        scheduled refresh of content occurs.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        name                            Optional string. Name of the collaboration
        ---------------------------     --------------------------------------------------------------------
        description                     Optional string. The description of the collaboration
        ---------------------------     --------------------------------------------------------------------
        config                          Optional dict. The configuration properties of the collaboration
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "%s/updateInfo" % self._basepath
        params = {"f": "json"}
        if name:
            params["name"] = name
        if description:
            params["description"] = description
        if config:
            params["config"] = config
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[dict] = None,
        max_item_size: Optional[int] = None,
        max_replication_size: Optional[int] = None,
        copy_by_ref_on_fail: bool = False,
    ) -> dict:
        """
        The updateInfo operation updates certain collaboration workspace
        properties.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. UID of the workspace
        ---------------------------     --------------------------------------------------------------------
        name                            Optional string. The name of the workspace
        ---------------------------     --------------------------------------------------------------------
        description                     Optional string. A brief set of texts that explains the workspace
        ---------------------------     --------------------------------------------------------------------
        config                          Optional dict. The configuration details of the new workspace.
                                        Removed at 10.6.
        ---------------------------     --------------------------------------------------------------------
        max_item_size                   Optional Integer.  The maximum item size in MBs.
        ---------------------------     --------------------------------------------------------------------
        max_replication_size            Optional Integer.  The maximum replication item size in MBs.
        ---------------------------     --------------------------------------------------------------------
        copy_by_ref_on_fail             Optional Boolean.  Determines whether a failed attempt to copy
                                        should revert to sharing by reference. For example, in cases where
                                        the imposed size limit has been exceeded.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "%s/workspaces/%s/updateInfo" % (self._basepath, workspace_id)
        params = {"f": "json"}
        if name:
            params["name"] = name
        if description:
            params["description"] = description
        if config:
            params["config"] = config
        if max_item_size:
            params["maxItemSizeInMB"] = max_item_size
        if max_replication_size:
            params["maxReplicationPackageSizeInMB"] = max_replication_size
        params["copyByRefIfCopyFail"] = copy_by_ref_on_fail
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def update_access_modes(
        self, portal_id: str, workspace_access_json: Union[str, dict]
    ) -> dict:
        """
        The update_access_modes operation updates the access mode for a
        specific participant in a portal-to-portal collaboration.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        portal_id                       Required string. UID of the Portal
        ---------------------------     --------------------------------------------------------------------
        workspace_access_json           Required dict/string. JSON describing the participant's access mode.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        data_path = "/participants/%s/updateParticipantAccessModes" % portal_id
        params = {"f": "json"}
        params["collaborationWorkspacesAccessJSON"] = workspace_access_json
        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def update_portal_group_link(
        self,
        workspace_id: str,
        portal_id: str,
        enable_realtime_sync: bool = True,
        copy_feature_service_data: bool = True,
        copy_by_ref_on_fail: bool = True,
        enable_bidirectional_sync: bool = True,
    ) -> dict:
        """
        The `update_portal_group_link` operation updates the group linked with a
        workspace for a participant in a portal-to-portal collaboration.
        Content shared to the portal group is shared to other participants
        in the collaboration.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        workspace_id                    Required string. UID of the workspace
        ---------------------------     --------------------------------------------------------------------
        portal_id                       Required string. UID of the Portal
        ---------------------------     --------------------------------------------------------------------
        enable_realtime_sync            Optional boolean. Determines whether the content shared with the
                                        group is shared to other collaboration participants in real time,
                                        updating whenever changes are made, or whether the content is
                                        shared based on a schedule set by the collaboration host.
        ---------------------------     --------------------------------------------------------------------
        copy_feature_service_data       Optional boolean.  Boolean value used when Feature Service data is
                                        shared in a group that is linked to a distributed collaboration
                                        workspace. When set to "true" Feature Service data will be copied
                                        to collaboration participants.
        ---------------------------     --------------------------------------------------------------------
        copy_by_ref_on_fail             Optional boolean. If the copy feature service data fails, and set to
                                        `True`, the enterprise will reference the data instead of copying it.
                                        This is supported on **10.9+**.
        ---------------------------     --------------------------------------------------------------------
        enable_bidirectional_sync       Optional boolean. When set to true, edits to shared feature services
                                        can be allowed two-way to eligible participants.
                                        This is supported on **10.9+**.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """

        data_path = f"{self._basepath}/workspaces/{workspace_id}/updatePortalGroupLink"
        params = {
            "f": "json",
            "portalGroupId": portal_id,
            "enableRealtimeSync": enable_realtime_sync,
            "copyFeatureServiceData": copy_feature_service_data,
            "copyByRefIfCopyFail": copy_by_ref_on_fail,
            "enableFeatureServiceBidirectionalSync": enable_bidirectional_sync,
        }

        con = self._portal.con
        return con.post(path=data_path, postdata=params)

    # ----------------------------------------------------------------------
    def validate_invitation_response(self, response_file: str) -> dict:
        """
        Prior to importing a collaboration invitation response, the
        invitation response file can be validated by using the
        validate_invitation_response operation to check for the existence
        of the collaboration and validity of the invitation response file.

        ===========================     ====================================================================
        **Parameter**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        response_file                   Required string. Path to the collaboration response file.
        ===========================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        files = {"invitationResponseFile": response_file}
        params = {"f": "json"}
        data_path = "%s/validateInvitationResponse" % self._basepath
        con = self._portal.con
        return con.post(path=data_path, postdata=params, files=files)
