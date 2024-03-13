import os
import json
from typing import Optional, Union
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import GIS


class WebhookManager(object):
    """
    Creates and manages ArcGIS Enterprise webhooks.  Webhooks allow you to be
    automatically notified when events associated with items, groups, and
    users occur. Once a webhook has been triggered, an HTTP request is
    made to a user-defined URL to provide information regarding the event.
    """

    _con = None
    _gis = None
    _url = None
    _properties = None

    def __init__(self, url, gis):
        self._url = url
        self._gis = gis
        isinstance(self._gis, GIS)
        self._con = self._gis._con

    # ----------------------------------------------------------------------
    def _init(self):
        """initializer"""
        params = {"f": "json"}
        res = self._con.get(self._url, params)
        self._properties = PropertyMap(res)

    # ----------------------------------------------------------------------
    def __str__(self):
        from urllib.parse import urlparse

        return "<WebhookManager @ {id}>".format(id=urlparse(self._url).netloc)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the Webhook properties"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def settings(self):
        """
        There are several advanced parameters that can be used to configure
        the connection behavior of your webhook. These parameters will be
        applied to all of the configured webhooks in your Portal. Use the
        Update operation to modify any of the parameters.



        ** Dictionary Key/Values **

        =================================  ===============================================================================
        **Parameter**                       **Description**
        ---------------------------------  -------------------------------------------------------------------------------
        notificationAttempts               Required Integer. This will determine how many attempts will be made to deliver
                                           a payload.
        ---------------------------------  -------------------------------------------------------------------------------
        otificationTimeOutInSeconds        Required Integer. The length of time (in seconds) that Portal will wait to
                                           receive a response. The max response is 60.
        ---------------------------------  -------------------------------------------------------------------------------
        notificationElapsedTimeInSeconds   Required Integer. The amount of time between each payload delivery attempt. By
                                           default, this is set to 30 seconds and can be set to a maximum of 100 seconds
                                           and a minimum of one second.
        =================================  ===============================================================================

        returns: dict

        """
        url = "%s/settings" % self._url
        params = {"f": "json"}
        return self._con.get(url, params)

    # ----------------------------------------------------------------------
    @settings.setter
    def settings(self, value):
        """
        See main ``settings`` property docstring
        """
        url = "%s/settings/update" % self._url
        params = {"f": "json"}

        params = {
            "notificationAttempts": 4,
            "notificationTimeOutInSeconds": 11,
            "notificationElapsedTimeInSeconds": 6,
            "f": "json",
        }
        for k, v in value.items():
            params[k] = v
        self._con.post(url, params)

    # ----------------------------------------------------------------------
    def get(self, name: str):
        """finds a single instance of a webhook by name"""
        for wh in self.list():
            if wh.properties.name.lower() == name.lower():
                return wh
            del wh
        return

    # ----------------------------------------------------------------------
    def create(
        self,
        name: str,
        url: str,
        events: Union[list, str] = "ALL",
        number_of_failures: int = 5,
        days_in_past: int = 5,
        secret: Optional[str] = None,
        properties: Optional[dict] = None,
    ):
        """
        Creates a WebHook to monitor REST endpoints and report activities

        =================================  ===============================================================================
        **Parameter**                       **Description**
        ---------------------------------  -------------------------------------------------------------------------------
        name                               Required String. The name of the webhook.
        ---------------------------------  -------------------------------------------------------------------------------
        url                                Required String. This is the URL to which the webhook will deliver payloads to.
        ---------------------------------  -------------------------------------------------------------------------------
        events                             Otional List or String.  The events accepts a list or all events can be
                                           monitored. This is done by passing "ALL" in as the events.  If a list is
                                           provided, a specific endpoint can be monitored.

                                            **Item Trigger Events**

                                            +------------------------------------------------+-------------------------+
                                            | **Trigger event**                              | **URI example**         |
                                            +------------------------------------------------+-------------------------+
                                            | All trigger events for all items               | /items                  |
                                            +------------------------------------------------+-------------------------+
                                            | Add item to the portal                         | /items/add              |
                                            +------------------------------------------------+-------------------------+
                                            | All trigger events for a specific item         | /items/<itemID>         |
                                            +------------------------------------------------+-------------------------+
                                            | Delete a specific item                         | /items/<itemID>/delete  |
                                            +------------------------------------------------+-------------------------+
                                            | Update a specific item's properties            | /items/<itemID>/update  |
                                            +------------------------------------------------+-------------------------+
                                            | Move an item or changing ownership of the item | /items/<itemID>/move    |
                                            +------------------------------------------------+-------------------------+
                                            | Publish a specific item                        | /items/<itemID>/publish |
                                            +------------------------------------------------+-------------------------+
                                            | Share a specific item                          | /items/<itemID>/share   |
                                            +------------------------------------------------+-------------------------+
                                            | Unshare a specific item                        | /items/<itemID>/unshare |
                                            +------------------------------------------------+-------------------------+

                                            **Group Trigger Events**

                                            +------------------------------------------------+-------------------------------+
                                            | **Trigger event**                              | **URI example**               |
                                            +------------------------------------------------+-------------------------------+
                                            | All trigger events for all groups              | /groups                       |
                                            +------------------------------------------------+-------------------------------+
                                            | Add group                                      | /groups/add                   |
                                            +------------------------------------------------+-------------------------------+
                                            | All trigger events for a specific group        | /groups/<groupID>             |
                                            +------------------------------------------------+-------------------------------+
                                            | Update a specific group                        | /groups/<groupID>/update      |
                                            +------------------------------------------------+-------------------------------+
                                            | Delete a specific group                        | /groups/<groupID>/delete      |
                                            +------------------------------------------------+-------------------------------+
                                            | Enable Delete Protection for a specific group  | /groups/<groupID>/protect     |
                                            +------------------------------------------------+-------------------------------+
                                            | Disable Delete Protection for a specific group | /groups/<groupID>/unprotect   |
                                            +------------------------------------------------+-------------------------------+
                                            | Invite a user to a specific group              | /groups/<groupID>/invite      |
                                            +------------------------------------------------+-------------------------------+
                                            | Add a user to a specific group                 | /groups/<groupID>/addUsers    |
                                            +------------------------------------------------+-------------------------------+
                                            | Remove a user from a specific group            | /groups/<groupID>/removeUsers |
                                            +------------------------------------------------+-------------------------------+
                                            | Update a user's role in a specific group       | /groups/<groupID>/updateUsers |
                                            +------------------------------------------------+-------------------------------+


                                            **User Trigger Events**

                                            +----------------------------------------------------+---------------------------+
                                            | **Trigger event**                                  | **URI example**           |
                                            +----------------------------------------------------+---------------------------+
                                            | All trigger events for all users in the portal     | /users                    |
                                            +----------------------------------------------------+---------------------------+
                                            | All trigger events associated with a specific user | /users/<username>         |
                                            +----------------------------------------------------+---------------------------+
                                            | Delete a specific user                             | /users/<username>/delete  |
                                            +----------------------------------------------------+---------------------------+
                                            | Update a specific user's profile                   | /users/<username>/update  |
                                            +----------------------------------------------------+---------------------------+
                                            | Disable a specific user's account                  | /users/<username>/disable |
                                            +----------------------------------------------------+---------------------------+
                                            | Enable a specific user's account                   | /users/<username>/enable  |
                                            +----------------------------------------------------+---------------------------+

                                           Example Syntax: ['/users', '/groups/abcd1234....']

        ---------------------------------  -------------------------------------------------------------------------------
        number_of_failures                 Optional Integer. The number of failures to allow before the service
        ---------------------------------  -------------------------------------------------------------------------------
        days_in_past                       Option Integer. The number of days to report back on.
        ---------------------------------  -------------------------------------------------------------------------------
        secret                             Optional String. Add a Secret to your payload that can be used to authenticate
                                           the message on your receiver.
        ---------------------------------  -------------------------------------------------------------------------------
        properties                         Optional Dict. At 10.9.1+ users can provide additional configuration properties.
        =================================  ===============================================================================

        :returns a :class:`WebHook<arcgis.gis.admin.Webhook>` instance

        .. code-block:: python

            # Example using Zapier as the payload URL

            from arcgis.gis import GIS

            gis = GIS(profile="your_profile", verify_cert=False)

            wh_mgr = gis.admin.webhooks
            wh = wh_mgr.create(name="Webhook_from_API",
                               url="https://hooks.zapier.com/hooks/catch/6694048/odqj9o3/",
                               events=["/items/981e98b949d9432ebf26433f40948cec/move",
                                       "/items/981e98b949d9432ebf26433f40948cec/update"]

        See `Webhook Blog Post <https://www.esri.com/arcgis-blog/products/arcgis-enterprise/administration/webhooks-dev-summit-2019/>`_ for a detailed explanation.

        """
        if secret is None:
            secret = ""
        purl = "%s/createWebhook" % self._url
        config = {
            "deactivationPolicy": {
                "numberOfFailures": number_of_failures,
                "daysInPast": days_in_past,
            }
        }
        if properties:
            config["properties"] = properties
        params = {
            "f": "json",
            "name": name,
            "url": url,
            "secret": secret,
            "config": config,
        }
        if str(events).lower() == "all":
            params["changes"] = "allChanges"
            res = self._con.post(purl, params)

        elif isinstance(events, list):
            params["changes"] = "manualChanges"
            params["events"] = ",".join(events)
            res = self._con.post(purl, params)
        if "success" in res and res["success"]:
            return self.get(name=name)
        return None

    # ----------------------------------------------------------------------
    def list(self) -> list:
        """Returns a list of WebHook objects"""
        hooks: list[Webhook] = []
        self._properties = None
        if self._gis.version < [10, 3]:
            for wh in self.properties.webhooks:
                try:
                    url = "%s/%s" % (self._url, wh["id"])
                    hooks.append(Webhook(url=url, gis=self._gis))
                except:
                    pass
            return hooks
        else:
            params: dict = {
                "f": "json",
                "start": 1,
                "num": 25,
                "sortField": None,
                "sortOrder": None,
            }
            res: dict = self._con.get(self._url, params)
            hooks.extend(
                [
                    Webhook(url, self._gis)
                    for url in [
                        "%s/%s" % (self._url, wh["id"]) for wh in res["webhooks"]
                    ]
                ]
            )
            while res["nextStart"] != -1:
                params["start"] = res["nextStart"]
                res: dict = self._con.get(self._url, params)
                hooks.extend(
                    [
                        Webhook(url, self._gis)
                        for url in [
                            "%s/%s" % (self._url, wh["id"]) for wh in res["webhooks"]
                        ]
                    ]
                )
            return hooks


########################################################################
class Webhook(object):
    """a single webhook"""

    _con = None
    _gis = None
    _url = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis):
        """Constructor"""
        self._url = url
        self._gis = gis
        self._con = gis._con

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<WebHook @ {name}>".format(name=self.properties.name)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def _init(self):
        """Constructor"""
        if self._properties is None:
            self._properties = PropertyMap(self._con.get(self._url, {"f": "json"}))

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """ """
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Removes the current webhook from the system.

        :return: Boolean

        """
        url = self._url + "/delete"
        params = {"f": "json"}
        return self._con.post(url, params)["success"]

    # ----------------------------------------------------------------------
    @property
    def notifications(self):
        """
        The `notifications`` will display information pertaining to
        trigger events associated with the specific webhook. You can use
        this table to monitor your webhook and the details of any delivered
        payloads such as the time the webhook was triggered, the response
        received from the payload URL, and the delivered payload data.

        :return: List

        """
        url = "%s/notificationStatus" % self._url
        messages = []
        params = {"num": 100, "start": 1, "f": "json"}
        res = self._con.post(url, params)
        if len(res["WebhookStatus"]) > 0:
            messages += res["WebhookStatus"]
        while res["nextStart"] != -1:
            params["start"] = res["nextStart"]
            res = self._con.post(url, params)
            messages += res["WebhookStatus"]
            if res["nextStart"] == -1:
                return messages
        return messages

    # ----------------------------------------------------------------------
    def deactivate(self):
        """
        Temporarily pause the webhook. This will stop the webhook from
        delivering payloads when it is invoked. The webhook will be
        automatically deactivated when the deactivation policy is met.

        :return: boolean
        """
        url = self._url + "/deactivate"
        params = {"f": "json"}
        res = self._con.post(url, params)
        self._properties = None
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def activate(self):
        """
        Restarts a deactivated webhook. When activated, payloads
        will be delivered to the payload URL when the webhook is invoked.
        """
        url = self._url + "/activate"
        params = {"f": "json"}
        res = self._con.post(url, params)
        self._properties = None
        if "success" in res:
            return res["success"]
        return False

    # ----------------------------------------------------------------------
    def update(
        self,
        name: Optional[str] = None,
        url: Optional[str] = None,
        events: Optional[Union[list, str]] = None,
        number_of_failures: Optional[int] = None,
        days_in_past: Optional[int] = None,
        secret: Optional[str] = None,
        properties: Optional[dict] = None,
    ):
        """
        The Update Webhook operation allows administrators to update any of
        the parameters of their webhook.

        =================================  ===============================================================================
        **Parameter**                       **Description**
        ---------------------------------  -------------------------------------------------------------------------------
        name                               Required String. The name of the webhook.
        ---------------------------------  -------------------------------------------------------------------------------
        url                                Required String. This is the URL to which the webhook will deliver payloads to.
        ---------------------------------  -------------------------------------------------------------------------------
        events                             Otional List or String.  The events accepts a list of all events that can be
                                           monitored. This is done by passing "ALL" in as the events.  If a list is
                                           provided, a specific endpoint can be monitored.

                                           **Item Trigger Events**

                                            +------------------------------------------------+-------------------------+
                                            | **Trigger event**                              | **URI example**         |
                                            +------------------------------------------------+-------------------------+
                                            | All trigger events for all items               | /items                  |
                                            +------------------------------------------------+-------------------------+
                                            | Add item to the portal                         | /items/add              |
                                            +------------------------------------------------+-------------------------+
                                            | All trigger events for a specific item         | /items/<itemID>         |
                                            +------------------------------------------------+-------------------------+
                                            | Delete a specific item                         | /items/<itemID>/delete  |
                                            +------------------------------------------------+-------------------------+
                                            | Update a specific item's properties            | /items/<itemID>/update  |
                                            +------------------------------------------------+-------------------------+
                                            | Move an item or changing ownership of the item | /items/<itemID>/move    |
                                            +------------------------------------------------+-------------------------+
                                            | Publish a specific item                        | /items/<itemID>/publish |
                                            +------------------------------------------------+-------------------------+
                                            | Share a specific item                          | /items/<itemID>/share   |
                                            +------------------------------------------------+-------------------------+
                                            | Unshare a specific item                        | /items/<itemID>/unshare |
                                            +------------------------------------------------+-------------------------+

                                            **Group Trigger Events**

                                            +------------------------------------------------+-------------------------------+
                                            | **Trigger event**                              | **URI example**               |
                                            +------------------------------------------------+-------------------------------+
                                            | All trigger events for all groups              | /groups                       |
                                            +------------------------------------------------+-------------------------------+
                                            | Add group                                      | /groups/add                   |
                                            +------------------------------------------------+-------------------------------+
                                            | All trigger events for a specific group        | /groups/<groupID>             |
                                            +------------------------------------------------+-------------------------------+
                                            | Update a specific group                        | /groups/<groupID>/update      |
                                            +------------------------------------------------+-------------------------------+
                                            | Delete a specific group                        | /groups/<groupID>/delete      |
                                            +------------------------------------------------+-------------------------------+
                                            | Enable Delete Protection for a specific group  | /groups/<groupID>/protect     |
                                            +------------------------------------------------+-------------------------------+
                                            | Disable Delete Protection for a specific group | /groups/<groupID>/unprotect   |
                                            +------------------------------------------------+-------------------------------+
                                            | Invite a user to a specific group              | /groups/<groupID>/invite      |
                                            +------------------------------------------------+-------------------------------+
                                            | Add a user to a specific group                 | /groups/<groupID>/addUsers    |
                                            +------------------------------------------------+-------------------------------+
                                            | Remove a user from a specific group            | /groups/<groupID>/removeUsers |
                                            +------------------------------------------------+-------------------------------+
                                            | Update a user's role in a specific group       | /groups/<groupID>/updateUsers |
                                            +------------------------------------------------+-------------------------------+


                                            **User Trigger Events**

                                            +----------------------------------------------------+---------------------------+
                                            | **Trigger event**                                  | **URI example**           |
                                            +----------------------------------------------------+---------------------------+
                                            | All trigger events for all users in the portal     | /users                    |
                                            +----------------------------------------------------+---------------------------+
                                            | All trigger events associated with a specific user | /users/<username>         |
                                            +----------------------------------------------------+---------------------------+
                                            | Delete a specific user                             | /users/<username>/delete  |
                                            +----------------------------------------------------+---------------------------+
                                            | Update a specific user's profile                   | /users/<username>/update  |
                                            +----------------------------------------------------+---------------------------+
                                            | Disable a specific user's account                  | /users/<username>/disable |
                                            +----------------------------------------------------+---------------------------+
                                            | Enable a specific user's account                   | /users/<username>/enable  |
                                            +----------------------------------------------------+---------------------------+

                                           .. code-block:: python

                                               #Example Usage:

                                               >>> events = ['/users', '/groups/abcd1234....']

        ---------------------------------  -------------------------------------------------------------------------------
        number_of_failures                 Optional Integer. The number of failures to allow before the webhook is
                                           deactivated.
        ---------------------------------  -------------------------------------------------------------------------------
        days_in_past                       Option Integer. The number of days to report back on.
        ---------------------------------  -------------------------------------------------------------------------------
        secret                             Optional String. Add a secret to your payload that can be used to authenticate
                                           the message on your receiver.
        =================================  ===============================================================================

        :returns Boolean

        """

        if name is None:
            name = self.properties["name"]
        if "secret" in self.properties:
            if secret is None:
                secret = self.properties.secret
            elif secret == "":
                secret = ""
        else:
            if secret is None or secret == "":
                secret = None
        if url is None:
            url = self.properties.payloadUrl
        if number_of_failures is None:
            number_of_failures = (
                self.properties.config.deactivationPolicy.numberOfFailures
            )
        if days_in_past is None:
            days_in_past = self.properties.config.deactivationPolicy.daysInPast
        if events is None:
            events = ",".join(list(self.properties.events))
        params = {
            "f": "json",
            "name": name,
            "url": url,
            "secret": secret,
            "config": dict(self.properties["config"]),
        }
        if number_of_failures:
            params["config"]["deactivationPolicy"][
                "numberOfFailures"
            ] = number_of_failures
        if days_in_past:
            params["config"]["deactivationPolicy"]["daysInPast"] = days_in_past
        if properties:
            params["config"]["deactivationPolicy"]["properties"].update(properties)
        params["events"] = events
        purl = self._url + "/update"

        res = self._con.post(purl, params)
        self._properties = None
        if "success" in res:
            return res["success"]
        return False
