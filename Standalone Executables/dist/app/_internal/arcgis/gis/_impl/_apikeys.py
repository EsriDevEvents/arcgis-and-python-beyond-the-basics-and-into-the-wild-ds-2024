from __future__ import annotations
from typing import Optional, Union
from arcgis._impl.common._isd import InsensitiveDict


###########################################################################
class APIKey(object):
    """
    The ``APIKey`` class is a single instance of a registered access key for
    performing certain operations based on permissions.

    Users can create an APIKey instance as shown below:

    .. code-block:: python

            # Getting from a list of keys
            >>> key1 = gis.api_keys.keys[0]

            # Getting a key using
            >>> key2 = gis.api_keys.get('key_value')
    """

    _gis = None
    _item = None
    _properties = None

    def __init__(self, item, gis):
        self._item = item
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<API Key {self._item.title}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        The ``properties`` property retrieves the properties of the current APIKey object.

        :return:
            A dictionary containin the properties (if any) of the current APIKey object.
        """
        if self._properties is None:
            self._properties = InsensitiveDict(self._item.app_info)
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def apikey(self):
        """
        The ``apikey`` property retrieves the API Key value for the current key.

        :return:
            String
        """
        return self.properties.apiKey

    # ----------------------------------------------------------------------
    def delete(self):
        """
        The ``delete`` method deletes the current APIKey object permanently.

        :return:
            A boolean indicating success (True), or failure (False)
        """
        return self._item.delete()

    # ----------------------------------------------------------------------
    def reset(self):
        """
        Resets the API Key for the Item. The call will return the information
        with the new API Key information.

        :return:
            A dictionary with the APIKey object information

        """
        url = f"{self._gis._portal.resturl}oauth2/apps/{self.properties.client_id}/resetApiKey"
        params = {"f": "json"}
        self._properties = None
        return self._gis._con.post(url, params)

    # ----------------------------------------------------------------------
    def update(self, http_referers: list[str] = None, privileges: list[str] = None):
        """
        The ``update`` method updates the current APIKey object's properties

        ================  ===============================================================================
        **Parameter**     **Description**
        ----------------  -------------------------------------------------------------------------------
        http_referers     Optional List. A list of the http referrers for which usage of the
                          API Key will be restricted to.

                          **Example**

                          ```
                          [
                          "https://foo.com",
                          "https://bar.com"
                          ]
                          ```

                          Note: Http Referrers can be configured for non apiKey type apps as
                          well. The list configured here will be used to validate the app
                          tokens sent in while accessing the sharing API. The referrer checks
                          will not be applied to user tokens.
        ----------------  -------------------------------------------------------------------------------
        privileges        Optional List. A list of the privileges that will be available for
                          this API key.

                          **Example**

                          ```

                          [
                          "portal:apikey:basemaps",
                          "portal:app:access:item:itemId",
                          "premium:user:geocode",
                          "premium:user:networkanalysis"
                          ]

                          ```

                          Note: Privileges can be configured for non  `API Key` type apps as
                          well. The list configured here will be used to grant access to items
                          when item endpoint is accessed with app tokens. The checks will not
                          be applied to user tokens and they can continue accessing items
                          based on the current item sharing model. With app tokens, all items
                          of app owner can be accessed if the privileges list is not
                          configured.
        ================  ===============================================================================


        :return:
            A dictionary

        .. code-block:: python

            # Usage Example

            >>> key1 = gis.api_keys.keys[0]
            >>> key1.update(http_referers = ["https://foo.com", "https://bar.com"],
            >>>                 privileges = ["portal:apikey:basemaps",
            >>>                               "portal:app:access:item:itemId",
            >>>                               "premium:user:geocode",
            >>>                               "premium:user:networkanalysis"])

        """
        url = f"{ self._gis._portal.resturl}oauth2/apps/{self.properties.client_id}/update"
        if http_referers is None and privileges is None:
            return self.properties
        params = {"f": "json"}
        if http_referers:
            params["httpReferrers"] = http_referers
        if privileges:
            params["priveleges"] = privileges
        self._properties = None
        return self._gis._con.post(url, params)


###########################################################################
class APIKeyManager(object):
    """
    The ``APIKeyManager`` creates, manages and updates :class:`~arcgis.gis._impl.APIKey` objects for ArcGIS Online.
    """

    _gis = None
    _url = None

    def __init__(self, gis):
        from .. import GIS

        assert isinstance(gis, GIS)
        self._gis = gis
        self._base_url = f"{ self._gis._portal.resturl}"

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"< API Key Manager @ {self._gis._portal.resturl} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def get(self, api_key: APIKey = None, title: Optional[str] = None):
        """
        The ``get`` method retrieves an :class:`~arcgis.gis._impl.APIKey`
        object based on the Key Value or its title.

        .. code-block:: python

            # Usage Example - Getting API Key using key string

            >>> gis.api_keys.get(api_key='key_string')

            # Getting api key using key Item's title

            >>> gis.api_keys.get(title='project1_key1')

        :return:
            An :class:`~arcgis.gis._impl.APIKey` object
        """
        if api_key:
            for key in self.keys:
                if key.properties.apikey.lower() == api_key.lower():
                    return key
        elif title:
            from arcgis.gis import Item

            for key in self.keys:
                i = Item(itemid=key.properties.itemid, gis=self._gis)
                if title.lower() == i.title.lower():
                    return key
        return None

    # ----------------------------------------------------------------------
    def create(
        self,
        title: str,
        tags: Union[str, list[str]],
        description: Optional[str] = None,
        http_referers: Optional[list[str]] = None,
        redirect_uris: Optional[list[str]] = None,
        privileges: Optional[list[str]] = None,
    ):
        """
        The ``create`` method generates a new :class:`~arcgis.gis._impl.APIKey` objects for the Organization.

        ================  ===============================================================================
        **Parameter**     **Description**
        ----------------  -------------------------------------------------------------------------------
        title             Required String. The name of the API Key Item.
        ----------------  -------------------------------------------------------------------------------
        tags              Required String. A comma seperated list of descriptive words describing the
                          API Key item.
        ----------------  -------------------------------------------------------------------------------
        description       Optional String. A description of what the API Key is going to be used for.
        ----------------  -------------------------------------------------------------------------------
        http_referers     Optional List. A list of the http referrers for which usage of the
                          API Key will be restricted to.

                          **Example**

                          ```
                          [
                          "https://foo.com",
                          "https://bar.com"
                          ]
                          ```

                          Note: Http Referrers can be configured for non apiKey type apps as
                          well. The list configured here will be used to validate the app
                          tokens sent in while accessing the sharing API. The referrer checks
                          will not be applied to user tokens.
        ----------------  -------------------------------------------------------------------------------
        redirect_uris     Optional list.  The URIs where the access_token or authorization
                          code will be delivered upon successful authorization. The
                          redirect_uri specified during authorization must match one of the
                          registered URIs, otherwise authorization will be rejected.

                          A special value of urn:ietf:wg:oauth:2.0:oob can also be specified
                          for authorization grants. This will result in the authorization
                          code being delivered to a portal URL (/oauth2/approval). This
                          value is typically used by apps that don't have a web server or a
                          custom URI scheme where the code can be delivered.

                          The value is a JSON string array.

        ----------------  -------------------------------------------------------------------------------
        privileges        Optional List. A list of the privileges that will be available for
                          this API key.


                          .. note::
                            Privileges can be configured for non  `API Key` type apps as
                            well. The list configured here will be used to grant access to items
                            when item endpoint is accessed with app tokens. The checks will not
                            be applied to user tokens and they can continue accessing items
                            based on the current item sharing model. With app tokens, all items
                            of app owner can be accessed if the privileges list is not
                            configured.
        ================  ===============================================================================

        :return:
            An :class:`~arcgis.gis._impl.APIKey` object

        .. code-block:: python

            # Usage Example

            >>> gis.api_keys.create(title ="title_name", tags = "tags, apiKey, Manager",
            >>>                     http_referers = ["https://foo.com", "https://bar.com"],
            >>>                     privleges = ["portal:apikey:basemaps", "portal:app:access:item:itemId",
            >>>                                        "premium:user:geocode", "premium:user:networkanalysis"])
        """
        api_item = self._gis.content.add(
            {
                "title": title,
                "type": "API Key",
                "tags": tags,
                "description": description or "",
            }
        )
        if privileges is None:
            privileges = [
                "premium:user:geocode:temporary",
                "portal:apikey:basemaps",
            ]
        result = api_item.register(
            app_type="apikey",
            redirect_uris=redirect_uris,
            http_referers=http_referers,
            privileges=privileges,
        )
        return APIKey(item=api_item, gis=self._gis)

    # ----------------------------------------------------------------------
    def validate(self, api_key: APIKey, privileges: Optional[list[str]] = None):
        """

        The ``validate`` method checks if an :class:`~arcgis.gis._impl.APIKey` object has a specific privilege.

        ================  ===============================================================================
        **Parameter**     **Description**
        ----------------  -------------------------------------------------------------------------------
        api_key           Required :class:`~arcgis.gis._impl.APIKey`.  The key to validate against.
        ----------------  -------------------------------------------------------------------------------
        privileges        Optional List. The list of the privileges to check for.  The list consists of
                          a list of string values.
        ================  ===============================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.APIKeyManager.validate(ApiKey1)

        """
        if isinstance(privileges, (list, tuple)):
            privileges = ",".join(privileges)

        url = f"{self._base_url}oauth2/validateApiKey"
        params = {
            "f": "json",
            "key": api_key.properties.apiKey,
            "privilege": privileges,
        }
        res = self._gis._con.post(url, params)
        if "valid" in res:
            return res["valid"]
        return res

    # ----------------------------------------------------------------------
    @property
    def keys(self):
        """

        The ``keys`` property retrieves a tuple of :class:`~arcgis.gis._impl.APIKey` objects registered with the
        Organization.

        :return:
            A `tuple <https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences>`_ of
            :class:`~arcgis.gis._impl.APIKey` objects

        """
        url = f"{self._base_url}portals/self/apiKeys"
        start = 0
        params = {"f": "json", "num": 100, "start": start}
        from .. import Item

        res = self._gis._con.post(url, params)
        k = [
            APIKey(Item(gis=self._gis, itemid=k["itemId"]), gis=self._gis)
            for k in res["apiKeys"]
        ]
        while res["nextStart"] > 0:
            start += 100
            params["start"] = start
            res = self._gis._con.post(url, params)
            k.extend(
                k=[
                    APIKey(
                        Item(gis=self._gis, itemid=k["itemId"]),
                        gis=self._gis,
                    )
                    for k in res["apiKeys"]
                ]
            )
            if res["nextStart"] <= 0:
                break
        return tuple(k)
