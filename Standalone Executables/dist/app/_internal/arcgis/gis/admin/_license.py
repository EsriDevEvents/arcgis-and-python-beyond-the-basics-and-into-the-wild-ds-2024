"""
Entry point to working with licensing on Portal or ArcGIS Online
"""
from __future__ import annotations
import datetime
from .._impl._con import Connection
from ..._impl.common._mixins import PropertyMap
from ...gis import GIS, User
from ._base import BasePortalAdmin


########################################################################
class LicenseManager(BasePortalAdmin):
    """
    Provides tools to work and manage licenses in ArcGIS Online and
    ArcGIS Enterprise (Portal)

    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    url                 required string, the web address of the site to
                        manage licenses.
                        example:
                        https://<org url>/<wa>/sharing/rest/portals/self/purchases
    ---------------     ----------------------------------------------------
    gis                 required GIS, the gis connection object
    ===============     ====================================================

    :return:
       :class:`~arcgis.gis.admin.LicenseManager` Object
    """

    _con = None
    _url = None
    _json_dict = None
    _json = None
    _properties = None

    def __init__(self, url, gis=None, initialize=True, **kwargs):
        """class initializer"""
        super(LicenseManager, self).__init__(url=url, gis=gis)
        self._url = url
        if isinstance(gis, Connection):
            self._con = gis
        elif isinstance(gis, GIS):
            self._gis = gis
            self._con = gis._con
        else:
            raise ValueError("connection must be of type GIS or Connection")
        if initialize:
            self._init(connection=self._con)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< License Manager @ {url} >".format(url=self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def provisions(
        self,
        user: User,
        all_available: bool = False,
        included_expired: bool = True,
        return_client_ids: bool = False,
    ) -> list:
        """
        Allows administrators to manage a user's list of provsional Add-On Licenses.

        """

        if isinstance(user, User):
            user = user.username
        url = "%s/community/users/%s/provisionedListings" % (
            self._gis._portal.resturl,
            user,
        )

        params = {
            "f": "json",
            "returnAppClientIds": return_client_ids,
            "returnAllProvisions": all_available,
            "includeExpired": included_expired,
            "start": 1,
            "num": 100,
        }
        res = self._con.get(url, params)
        provs = res["provisionedListings"]
        while res["nextStart"] > -1:
            params["start"] = res["nextStart"]
            res = self._con.get(url, params)
            provs.extend(res["provisionedListings"])
            if res["nextStart"] == -1:
                break

        return provs

    # ----------------------------------------------------------------------
    def get(self, name: str):
        """
        Retrieves a license by it's name (title)

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        name                required string, name of the entitlement to locate
                            on the organization.
                            example:
                            name="arcgis pro"
        ===============     ====================================================

        :return:
           List of :class:`~arcgis.gis.admin.License` objects

        """
        licenses = self.all()
        for l in licenses:
            if (
                "listing" in l.properties
                and "title" in l.properties["listing"]
                and l.properties["listing"]["title"].lower() == name.lower()
            ):
                return l
            del l
        del licenses
        return None

    # ----------------------------------------------------------------------
    def all(self) -> list:
        """
        Returns all Licenses registered with an organization

        :return:
           List of :class:`~arcgis.gis.admin.License` objects

        """
        licenses = []
        if self._properties is None:
            self._init()
        if "purchases" in self.properties:
            purchases = self.properties["purchases"]
            for purchase in purchases:
                licenses.append(License(gis=self._gis, info=purchase))
        if "trials" in self.properties:
            purchases = self.properties["trials"]
            for purchase in purchases:
                licenses.append(License(gis=self._gis, info=purchase))
        return licenses

    # ----------------------------------------------------------------------
    @property
    def bundles(self) -> list:
        """
        Returns a list of Application Bundles for an Organization

        :return:
           List of :class:`~arcgis.gis.admin.Bundle` objects

        """
        if self._gis.version < [6, 4]:
            raise NotImplementedError("`bundles` not implemented before version 6.4")

        url = "{base}/portals/self/appBundles".format(base=self._gis._portal.resturl)
        params = {"f": "json", "num": 100, "start": 1}

        res = self._con.get(url, params)
        buns = res["appBundles"]
        while res["nextStart"] != -1:
            params["start"] = res["nextStart"]
            res = self._gis._con.get(url, params)
            buns += res["appBundles"]
            if res["nextStart"] == -1:
                break
        return [
            Bundle(
                url="{base}content/listings/{id}".format(
                    base=self._gis._portal.resturl, id=b["appBundleItemId"]
                ),
                properties=b,
                gis=self._gis,
            )
            for b in buns
        ]

    # ----------------------------------------------------------------------
    @property
    def offline_pro(self) -> bool:
        """
        Administrators can get/set the disconnect settings for the ArcGIS Pro licensing.
        A value of True means that a user can check out a license from the enterprise
        inorder to use it in a disconnected setting.  By setting `offline_pro` to False,
        the enterprise users cannot check out licenses to work in a disconnected setting
        for ArcGIS Pro.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required bool.
                            Value: True | False
        ===============     ====================================================================

        :return: Boolean

        """
        lic = self.get("arcgis pro")
        return lic.properties.provision.canDisconnect

    # ----------------------------------------------------------------------
    @offline_pro.setter
    def offline_pro(self, value: bool):
        """
        See main ``offline_pro`` property docstring
        """
        import json

        lic = self.get("arcgis pro")
        url = "{base}content/listings/{itemid}/setDisconnectSettings".format(
            base=self._gis._portal.resturl,
            itemid=lic.properties.provision.itemId,
        )
        params = {
            "f": "json",
            "canDisconnect": json.dumps(value),
            "maxDisconnectDuration": -1,
        }
        res = self._con.post(url, params)
        if "success" in res and res["success"] == False:
            raise Exception("Could not update the dicconnect settings.")
        elif "success" not in res:
            raise Exception("Could not update the dicconnect settings: %s" % res)


########################################################################
class Bundle(object):
    """
    This represents a single instance of an application bundle
    """

    _users = None
    _con = None
    _url = None
    _properties = None
    _gis = None
    _id = None

    # ----------------------------------------------------------------------
    def __init__(self, url, properties=None, gis=None):
        """Constructor"""
        import os

        if gis is None:
            import arcgis

            gis = arcgis.env.active_gis
        self._gis = gis
        self._url = url
        self._con = gis._con
        self._id = os.path.basename(url)
        self._properties = properties

    # ----------------------------------------------------------------------
    def _find(self, appid):
        """if properties are missing, populate the app bundle's properties"""
        url = "{base}/portals/self/appBundles".format(base=self._gis._portal.resturl)
        params = {"f": "json", "num": 100, "start": 1}

        res = self._con.get(url, params)
        buns = res["appBundles"]
        while res["nextStart"] != -1:
            params["start"] = res["nextStart"]
            res = self._gis._con.get(url, params)
            buns += res["appBundles"]
            if res["nextStart"] == -1:
                break
        for b in buns:
            if b["appBundleItemId"] == self._id:
                return b
        raise ValueError("Invalid Application Bundle.")

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the application bundles properties"""
        if self._properties is None:
            try:
                params = {"f": "json"}
                r = self._con.get(self._url, params)
                self._properties = PropertyMap(r)
            except:
                self._properties = PropertyMap({})
        elif isinstance(self._properties, dict):
            self._properties = PropertyMap(self._properties)
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def users(self):
        """returns a list of users assigned the application bundle"""
        url = f"{self._gis._portal.resturl}/portals/self/users/search"
        params = {
            "f": "json",
            "sortField": "fullname",
            "sortOrder": "asc",
            "q": f"appbundle: {self.properties.appBundleItemId}",
            "start": 1,
            "num": 60,
            "total": 0,
            "nextStart": -1,
        }
        res = self._con.get(url, params)
        final = dict(res)
        while res["nextStart"] > 0:
            params["start"] = res["nextStart"]
            res = self._con.get(url, params)
            final["results"].extend(res["results"])
            if res["nextStart"] == -1:
                break
        from arcgis.gis import User

        users = [
            User(gis=self._gis, username=user["username"], userdict=None)
            for user in final["results"]
        ]
        return users

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.users)

    # ----------------------------------------------------------------------
    def __str__(self):
        """ """
        return "< AppBundle: %s >" % self.properties["name"]

    # ----------------------------------------------------------------------
    def __repr__(self):
        """ """
        return "< AppBundle: %s >" % self.properties["name"]

    # ----------------------------------------------------------------------
    def assign(self, users: list):
        """
        Assigns the current application bundle to a list of users

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        users               Required List. A list of user names or User objects
                            to assign the current application bundle to.
        ===============     ====================================================


        :return: Boolean. True if successful else False

        """
        if isinstance(users, (tuple, set, list)) == False:
            users = [users]
        from arcgis.gis import User

        url = "%s%s" % (self._url, "/provisionUserAppBundle")
        params = {"f": "json", "users": None, "revoke": False}
        us = []
        for user in users:
            if isinstance(user, str):
                us.append(user)
            elif isinstance(user, User):
                us.append(user.username)
        params["users"] = ",".join(us)
        res = self._con.post(url, params)
        self._users = None
        self._properties = None
        if "success" in res:
            return res["success"]
        return res

    # ----------------------------------------------------------------------
    def revoke(self, users: list):
        """
        Revokes the current application bundle to a list of users

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        users               Required List. A list of user names or User objects
                            to remove the current application bundle to.
        ===============     ====================================================


        :return: Boolean. True if successful else False.

        """
        if isinstance(users, (tuple, set, list)) == False:
            users = [users]
        from arcgis.gis import User

        url = "%s%s" % (self._url, "/provisionUserAppBundle")
        params = {"f": "json", "users": None, "revoke": True}
        us = []
        self._users = None
        self._properties = None
        for user in users:
            if isinstance(user, str):
                us.append(user)
            elif isinstance(user, User):
                us.append(user.username)
        params["users"] = ",".join(us)
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"]
        return res


###########################################################################
class License(object):
    """
    Represents a single entitlement for a given organization.


    ===============     ====================================================
    **Parameter**        **Description**
    ---------------     ----------------------------------------------------
    gis                 Required GIS, the gis connection object
    ---------------     ----------------------------------------------------
    info                Required dictionary, the information provided by
                        the organization's site containing the provision
                        and listing information.
    ===============     ====================================================

    :return:
       :class:`~arcgis.gis.admin.License` object

    """

    _properties = None
    _gis = None
    _con = None

    # ----------------------------------------------------------------------
    def __init__(self, gis, info):
        """Constructor"""
        self._gis = gis
        self._con = gis._con
        self._properties = PropertyMap(info)

    # ----------------------------------------------------------------------
    def __str__(self):
        try:
            return "< %s %s @ %s >" % (
                self.properties["listing"]["title"],
                type(self).__name__,
                self._gis._portal.resturl,
            )
        except:
            return "<%s at %s >" % (
                type(self).__name__,
                self._gis._portal.resturl,
            )

    # ----------------------------------------------------------------------
    def __repr__(self):
        try:
            return "<%s %s @ %s >" % (
                self.properties["listing"]["title"],
                type(self).__name__,
                self._gis._portal.resturl,
            )
        except:
            return "<%s at %s >" % (
                type(self).__name__,
                self._gis._portal.resturl,
            )

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def report(self):
        """
        returns a Panda's Dataframe of the licensing count.
        """
        import pandas as pd

        data = []
        columns = ["Entitlement", "Total", "Assigned", "Remaining", "Users"]
        if (
            "provision" in self.properties
            and "orgEntitlements" in self.properties["provision"]
        ):
            for k, v in self.properties["provision"]["orgEntitlements"][
                "entitlements"
            ].items():
                counter = 0
                user_list = []
                for u in self.all():
                    if k in u["entitlements"]:
                        counter += 1
                        if u["lastLogin"] not in [None, -1]:
                            last_used = datetime.datetime.fromtimestamp(
                                u["lastLogin"] / 1000
                            ).strftime("%B %d, %Y")
                        else:
                            last_used = None
                        user_list.append({"user": u["username"], "lastUsed": last_used})
                row = [k, v["num"], counter, v["num"] - counter, user_list]
                data.append(row)
                del k, v
        return pd.DataFrame(data=data, columns=columns)

    # ----------------------------------------------------------------------
    def plot(self):
        """returns a simple bar chart of assigned and remaining entitlements"""
        report = self.report
        try:
            return report.plot(
                x=report["Entitlement"],
                y=["Assigned", "Remaining"],
                kind="bar",
                stacked=True,
            ).legend(loc="best")
        except:
            report.set_index(
                "Entitlement",
                drop=True,
                append=False,
                inplace=True,
                verify_integrity=False,
            )
            return report.plot(
                y=["Assigned", "Remaining"], kind="bar", stacked=True
            ).legend(loc="best")

    # ----------------------------------------------------------------------
    def all(self):
        """
        returns a list of all usernames and their entitlements for this license
        """
        item_id = self.properties["listing"]["itemId"]
        url = "%scontent/listings/%s/userEntitlements" % (
            self._gis._portal.resturl,
            item_id,
        )
        start = 1
        num = 100
        params = {"start": start, "num": num}
        user_entitlements = []
        res = self._con.get(url, params)
        user_entitlements += res["userEntitlements"]
        if "nextStart" in res:
            while res["nextStart"] > 0:
                start += num
                params = {"start": start, "num": num}
                res = self._con.get(url, params)
                user_entitlements += res["userEntitlements"]
        return user_entitlements

    # ----------------------------------------------------------------------
    def check(self, user: str) -> list:
        """
        Checks if the entitlement is assigned or not.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        user                Required string, the name of the user you want to
                            examine the entitlements for.
        ===============     ====================================================

        :return: list
        """
        if hasattr(user, "username"):
            user = user.username
        if "listing" in self.properties:
            item_id = self.properties["listing"]["itemId"]
        else:
            return []
        # elif 'provision' in self.properties and 'itemId' in self.properties['provision']:
        #    item_id = self.properties['provision']['itemId']

        url = "%scontent/listings/%s/userEntitlements/%s" % (
            self._gis._portal.resturl,
            item_id,
            user,
        )
        params = {"f": "json"}
        resp = self._con.get(url, params)
        if (
            "userEntitlements" in resp
            and resp["userEntitlements"]
            and "entitlements" in resp["userEntitlements"]
        ):
            return resp["userEntitlements"]["entitlements"]
        return []

    # ----------------------------------------------------------------------
    def user_entitlement(self, username: str):
        """
        Checks if a user has the entitlement assigned to them

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            Required string, the name of the user you want to
                            examine the entitlements for.
        ===============     ====================================================

        :return:
           dictionary
        """
        item_id = self.properties["listing"]["itemId"]
        url = "%scontent/listings/%s/userEntitlements" % (
            self._gis._portal.resturl,
            item_id,
        )
        start = 1
        num = 100
        params = {"start": start, "num": num}
        user_entitlements = []
        res = self._con.get(url, params)
        for u in res["userEntitlements"]:
            if u["username"].lower() == username.lower():
                return u
        if "nextStart" in res:
            while res["nextStart"] > 0:
                start += num
                params = {"start": start, "num": num}
                res = self._con.get(url, params)
                for u in res["userEntitlements"]:
                    if u["username"].lower() == username.lower():
                        return u
        return {}

    # ----------------------------------------------------------------------
    def assign(
        self,
        username: str,
        entitlements: list[str] | str,
        suppress_email: bool = True,
        overwrite: bool = True,
    ):
        """
        grants a user an entitlement.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            Required string, the name of the user you wish to
                            assign an entitlement to.
        ---------------     ----------------------------------------------------
        entitlements        Required list of strings or strings, of entitlements values.
        ---------------     ----------------------------------------------------
        suppress_email      Optional boolean, if True, the org will not notify
                            a user that their entitlements has changed (default)
                            If False, the org will send an email notifying a
                            user that their entitlements have changed.
        ---------------     ----------------------------------------------------
        overwrite           Optional boolean, if True, existing entitlements
                            for the user are dropped
        ===============     ====================================================

        :return:
           Boolean. True if successful else False.
        """
        item_id = self.properties["listing"]["itemId"]
        if hasattr(username, "username"):
            username = username.username
        if isinstance(entitlements, str):
            entitlements = entitlements.split(",")

        if not overwrite:
            existing = self.user_entitlement(username)
            if existing and "entitlements" in existing:
                entitlement_set = set(existing["entitlements"])
                for e in entitlements:
                    entitlement_set.add(e)
                entitlements = list(entitlement_set)

        params = {
            "f": "json",
            "userEntitlements": {
                "users": [username],
                "entitlements": entitlements,
            },
        }
        if suppress_email is not None:
            params["suppressCustomerEmail"] = suppress_email
        url = "%scontent/listings/%s/provisionUserEntitlements" % (
            self._gis._portal.resturl,
            item_id,
        )
        res = self._con.post(url, params)
        if "success" in res:
            return res["success"] == True
        return res

    # ----------------------------------------------------------------------
    def revoke(
        self,
        username: str,
        entitlements: list[str] | str,
        suppress_email: bool = True,
    ):
        """
        removes a specific license from a given entitlement

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        username            Required string, the name of the user you wish to
                            assign an entitlement to.
        ---------------     ----------------------------------------------------
        entitlments         Required list of strings or string, a list of entitlements values,
                            if * is given, all entitlements will be revoked
        ---------------     ----------------------------------------------------
        suppress_email      Optional boolean, if True, the org will not notify
                            a user that their entitlements has changed (default)
                            If False, the org will send an email notifying a
                            user that their entitlements have changed.
        ===============     ====================================================

        :return:
           boolean
        """
        if entitlements == "*":
            return self.assign(
                username=username,
                entitlements=[],
                suppress_email=suppress_email,
            )
        if isinstance(entitlements, str):
            entitlements = entitlements.split(",")
        if isinstance(entitlements, list):
            es = self.check(user=username)

            if len(es) > 0:
                lookup = {e.lower(): e for e in es}
                es = [e.lower() for e in es]
                if isinstance(entitlements, str):
                    entitlements = [entitlements]
                entitlements = list(set(es) - set([e.lower() for e in entitlements]))
                es2 = []
                for e in entitlements:
                    es2.append(lookup[e])
                return self.assign(
                    username=username,
                    entitlements=es2,
                    suppress_email=suppress_email,
                )
        return False
