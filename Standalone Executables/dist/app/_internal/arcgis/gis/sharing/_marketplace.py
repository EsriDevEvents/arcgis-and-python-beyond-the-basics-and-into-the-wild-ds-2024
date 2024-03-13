from __future__ import annotations
from arcgis.gis import GIS


class MarketPlaceManager:
    """Provides the ability for the manager to list and unlist marketplace items"""

    _gis: GIS = None
    _url: str = None

    def __init__(self, gis: GIS):
        self._gis = gis
        self._url = f"{gis._portal.resturl}content"

    # ----------------------------------------------------------------------
    def list(self, itemid: str) -> dict:
        """
        This operation lists the item in the marketplace.

        This operation is only available to organizations that have permissions
        to list items in the marketplace. The permissions are returned with the
        Portal Self response.

        The listing properties must be specified for the item before
        calling this operation. This operation will fail if listing
        properties have not already been specified.

        Listing an item will set its listed property to true.

        This operation is available to the user and to the administrator
        of the organization to which the user belongs.

        :return: A dictionary indicating success or failure and the item id of the listed item.
        """
        params = {"f": "json"}
        url = f"{self._url}/users/{self._gis.users.me.username}/items/{itemid}/list"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def unlist(self, itemid: str) -> dict:
        """
        This operation unlists a previously listed item from the marketplace.

        Unlisting an item will reset its listed property to false.

        This operation is available to the user and the administrator of
        the organization to which the user belongs.

        :return: A dictionary indicating success or failure and the item id of the listed item.
        """
        params = {"f": "json"}
        url = f"{self._url}/users/{self._gis.users.me.username}/items/{itemid}/unlist"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def listings(
        self,
        query: str = "*",
        my_listings: bool = False,
        start: int = 1,
        num: int = 10,
        sort_field: str | None = None,
        sort_order: str = "asc",
    ) -> dict:
        """
        This operation searches for marketplace listings. The searches are performed
        against a high performance index that indexes the most popular fields of a listing.

        By default, this search spans all public listings in the marketplace. However,
        if you're logged in as a vendor org admin and you specify the my_listings=true parameter,
        it then searches all public and private listings in your organization.


        ======================      ====================================================================
        **Parameter**                **Description**
        ----------------------      --------------------------------------------------------------------
        query                       Optional string. The query string to use to search.
        ----------------------      --------------------------------------------------------------------
        my_listings                 Optional boolean.  If True and you're logged in as
                                    a vendor org admin, it searches all public and private
                                    listings in your organization. Note that if my_listings=True,
                                    the query parameter is optional. Default is False.
        ----------------------      --------------------------------------------------------------------
        start                       Optional integer. The number of the first entry in the
                                    result set response. The index number is 1-based.

                                    The default value of start is 1. (i.e., the first search result).

                                    The start parameter, along with the num parameter,
                                    can be used to paginate the search results.
        ----------------------      --------------------------------------------------------------------
        num                         Optional integer. The maximum number of results to be
                                    included in the result set response.

                                    The default value is 10, and the maximum allowed value is 100.

                                    The start parameter, along with the num parameter,
                                    can be used to paginate the search results.

                                    Note that the actual number of returned results may be
                                    less than num. This happens when the number of results
                                    remaining after start is less than num.
        ----------------------      --------------------------------------------------------------------
        sort_field                  Optional string. The field to sort by. You can also sort
                                    by multiple fields (comma separated) for listings, sort
                                    field names are case-insensitive.

                                    Supported sort field names are:
                                    "title", "created", "listingpublisheddate", "type", "owner",
                                    "avgrating", "numratings", "numcomments", and "numviews"
        ----------------------      --------------------------------------------------------------------
        sort_order                  Optional string. Describes whether the order returns
                                    in ascending(asc) or descending(desc) order. Default is asc.
        ======================      ====================================================================


        :return:
            A dictionary with response syntax of:

                | {
                | "query": "<query string>",
                | "total": <total number of results>,
                | "start": <results in first set>,
                | "num": <number of results per page>,
                | "nextStart": <result number of next page>,
                | "listings": [{<listing1>}, {<listing2>}]
                | }

        """
        url = f"{self._url}/listings"
        params = {
            "f": "json",
            "q": query,
            "mylistings": my_listings,
            "start": start,
            "num": num,
            "sortField": sort_field,
            "sortOrder": sort_order,
        }
        return self._gis._portal.con.get(url, params)

    # ----------------------------------------------------------------------
    def listing(self, itemid: str) -> dict:
        """
        A listing in the marketplace. The listing and its corresponding item share the same ID.

        =====================       ========================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------
        itemid                      Required String. The item id.
        =====================       ========================================

        :return:
            A dictionary of the listed item with properties.
        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}"
        return self._gis._portal.con.get(url, params)

    # ----------------------------------------------------------------------
    def delete_provision(self, itemid: str) -> dict:
        """
        This operation deletes all provisions to this item for
        the specified purchaser.

        This operation cannot be invoked if the item has not been provisioned
        to the specified purchaser.

        .. note::
            Only vendor org admins can invoke this operation.

        =====================       ========================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------
        itemid                      Required String. The item id.
        =====================       ========================================

        :return:
            A dictionary with syntax:

                | {
                | "success": <true | false>,
                | "itemId": "<itemId>",
                | "purchaserOrgId": "<purchaserOrgId>"
                | }

        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/deleteProvision"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def express_interest(self, itemid: str) -> dict:
        """
        A purchaser can express interest in a marketplace listing by
        invoking this operation.

        This operation cannot be invoked if the item has already been
        purchased or if the purchaser has previously expressed interest.

        Only administrators and members with request purchase information
        privilege of purchasing orgs can invoke this operation.

        Note that interests cannot be expressed for free listings, because
        they can be directly purchased by purchasing org admins or members
        with request purchase information privilege.

        =====================       ========================================
        **Parameter**                **Description**
        ---------------------       ----------------------------------------
        itemid                      Required String. The item id.
        =====================       ========================================

        :return:
            A dictionary of the listed item with properties.
        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/interest"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def provision_org_entitlements(
        self,
        itemid: str,
        purchaser_org_id: str,
        purchaser_subscription_id: str | None = None,
        org_entitlements: dict | None = None,
    ) -> dict:
        """
        For a license-by-user listing, selling organization administrator
        or members can use this operation to provision entitlements to a
        purchasing organization. It can only be made if the item has already
        been purchased, or is being tried by the purchasing org.

        This operation is HTTPS only for Esri apps that require a signature,
        otherwise it can be either for provider apps.

        It can only be invoked by org admins or members with request
        purchase information privilege.

        ==========================      ==================================================================================================================================================================
        **Parameter**                    **Description**
        --------------------------      ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        itemid                          Required String. The item id.
        --------------------------      ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        purchaser_org_id                Required String. The org ID of the purchasing organization
        --------------------------      ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        purchaser_subscription_id       Optional String. The subscription(SMS) ID of the purchasing organization.
        --------------------------      ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        org_entitlements                Required Dictionary. A JSON object representing the set of entitlements available to the purchasing org.

                                        Example:

                                            | {
                                            | "maxUsers": 10,
                                            | "entitlements": {
                                            | "standard": {"num": 8}, ('standard' is an entitlement string that uniquely identifies entitlement, listingID is used typically for provider apps)
                                            | "advanced": {"num": 2},
                                            | "spatialAnalyst": {"num": 2}
                                            | }
                                            | }

        ==========================      ==================================================================================================================================================================

        :return: A dictionary of the provision item.
        """
        params = {
            "f": "json",
            "purchaserOrgId": purchaser_org_id,
            "purchaserSubscriptionId": purchaser_subscription_id,
            "orgEntitlements": org_entitlements,
        }
        url = f"{self._url}/listings/{itemid}"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def provision_user_entitlements(self, itemid: str, user_entitlements: dict) -> bool:
        """
        For a license-by-user listing, purchasing organization administrator
        can use this operation to provision entitlements to org
        members. It can only be made if the item has already been purchased,
        or is being tried by the purchasing org. A maximum of 25 users can
        be provisioned in one request.

        =====================       ====================================================================================================================================================================================
        **Parameter**                **Description**
        ---------------------       ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        itemid                      Required String. The item id.
        ---------------------       ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        user_entitlements           Required Dictionary. A JSON object representing the set of entitlements
                                    assigned to the specified set of users.

                                    Example:

                                        | {
                                        | "users": ["username1", "username2"],
                                        | "entitlements": ["standard", "networkAnalyst"] ("standard" is an entitlement string that uniquely identifies entitlement, listing itemId is used typically for provider apps)
                                        | }

                                    Only members of the purchasing org can be specified in the request.

                                    Specified entitlements are assigned to all specified users. If different sets of
                                    entitlements are to be assigned to different users, multiple requests
                                    with this operation are required.

                                    When there is no entitlements specified, it will revoke access to
                                    the item completely for the specified users.

                                    The total number of currently provisioned users plus users specified in requests
                                    should be no larger than the maximum number of users allowed for the purchasing org.
        =====================       ====================================================================================================================================================================================

        :return:
            A boolean indicating success (True) or failure (False).
        """
        params = {"f": "json", "userEntitlements": user_entitlements}
        url = f"{self._url}/listings/{itemid}"
        res = self._gis._portal.con.post(url, params)
        return res["success"]

    # ----------------------------------------------------------------------
    def purchases(self, status: str = "active") -> dict:
        """
        The purchases resource returns a list of purchases, trials, and
        interests expressed by this organization for items in the marketplace.

        =====================       ====================================================================================
        **Parameter**                **Description**
        ---------------------       ------------------------------------------------------------------------------------
        status                      Optional String. Status of the listings to be returned. The default value is active.

                                    Accepted values are:
                                        * active: Only listings that are currently active will be returned
                                        * expired: Only listings that have already expired will be returned
                                        * all: Both active and expired listings will be returned
        =====================       ====================================================================================

        :return: A dictionary depicting the purchases, trials, and interests.
        """
        params = {"f": "json", "status": status}
        url = f"{self._gis._portal.resturl}portals/self/purchases"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def purchase(
        self,
        itemid: str,
        purchase_org_id: str | None = None,
        provisioned_itemid: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """
        =====================       ====================================================================================
        **Parameter**                **Description**
        ---------------------       ------------------------------------------------------------------------------------
        itemid                      Required String. The item id.
        ---------------------       ------------------------------------------------------------------------------------
        purchase_org_id             Required String. The org ID of the purchaser organization. This parameter is
                                    required only when the call is made by the vendor. It is ignored otherwise.
        ---------------------       ------------------------------------------------------------------------------------
        provisioned_itemid          Required String. The ID of the item to be provisioned if different from the one
                                    listed.

                                    Note that the listed item and the provisioned item must be related by the
                                    `Listed2Provisioned` relationship otherwise it will result in an error.

                                    This parameter is allowed only when the call is made by the vendor.
                                    It is ignored otherwise.
        ---------------------       ------------------------------------------------------------------------------------
        end_date                    Required String. The end/expiry date of this purchase if any. If this parameter is
                                    not specified, it implies an unexpiring purchase. The end date specified
                                    should be in milliseconds from epoch.
        =====================       ====================================================================================

        :return: A dictionary of the provision item.
        """

        params = {
            "f": "json",
            "purchaseOrgId": purchase_org_id,
            "provisionedItemId": provisioned_itemid,
            "endDate": end_date,
        }
        url = f"{self._url}/listings/{itemid}/purchase"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def trial(self, itemid) -> dict:
        """
        A purchaser can start a trial for a marketplace listing by invoking this operation.

        This operation is only supported for listings that support trials
        (whose trialSupported property is true). Once started, the trial will be valid
        for the duration of the trial specified on the listing (the trialDuration property).

        This operation cannot be invoked if the item has already been purchased or
        if the purchaser has started a trial previously.

        Only admins or members with request purchase information privilege of
        purchasing orgs can invoke this operation.

        =====================       =================================
        **Parameter**                **Description**
        ---------------------       ---------------------------------
        itemid                      Required String. The item id.
        =====================       =================================

        :return: A dictionary of the provision item.
        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/trial"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def comments(self, itemid: str):
        """Lists all comments for the item"""
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/comments"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def user_entitlements(self, itemid: str) -> dict:
        """
        This operation allows purchasing organization administrators or a
        user with Manage Licenses privilege to retrieve all user entitlements
        assigned to users in their organization.

        =====================       ==============================
        **Parameter**                **Description**
        ---------------------       ------------------------------
        itemid                      Required String. The item id.
        =====================       ==============================

        :return: A JSON document representing the set of entitlements assigned to the specified set of users.
        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/userEntitlements"
        try:
            return self._gis._portal.con.post(url, params)
        except:
            return None

    # ----------------------------------------------------------------------
    def user_entitlement(self, itemid: str, username: str) -> dict:
        """
        This resource allows user, purchasing organization administrator, and
        members with the manage licenses privilege to retrieve entitlements assigned to the user.

        =====================       ==============================
        **Parameter**                **Description**
        ---------------------       ------------------------------
        itemid                      Required String.
        ---------------------       ------------------------------
        username                    Required String.
        =====================       ==============================
        """
        params = {"f": "json"}
        url = f"{self._url}/listings/{itemid}/userEntitlements/{username}"
        return self._gis._portal.con.post(url, params)

    # ----------------------------------------------------------------------
    def customer_list(
        self,
        itemid: str | None = None,
        orgname: str | None = None,
        status: str = "all",
        type: str = "PURCHASE",
        modified: str | None = None,
        sort_fields: str | None = None,
        sort_order: str = "asc",
        include_listing: bool = True,
        num: int = 10,
        start: int = 1,
    ) -> dict:
        """
        The customers_list resource returns a list of purchases, trials, and
        interests expressed by customers for items listed by this organization
        in the marketplace. This operation allows filtering and sorting of provisions.

        =====================       ====================================================================================
        **Parameter**                **Description**
        ---------------------       ------------------------------------------------------------------------------------
        itemid                      Optional String. The item id of the provision to be returned.
        ---------------------       ------------------------------------------------------------------------------------
        orgname                     Optional String. Purchaser organization name of the provisions to be returned.
        ---------------------       ------------------------------------------------------------------------------------
        status                      Optional String. Status of the listings to be returned. The default value is active.

                                    Accepted values are:
                                        * active: Only listings that are currently active will be returned
                                        * expired: Only listings that have already expired will be returned
                                        * all: Both active and expired listings will be returned
        ---------------------       ------------------------------------------------------------------------------------
        type                        Optional String. Access type of the provisions to be returned:
                                        * REQUEST: Only provisions that have been requested will be returned.
                                        * TRIAL: Only trial provisions will be returned.
                                        * PURCHASE: Only subscription provisions will be returned.
                                        * REQUESTANDTRIAL: Both provisions that have been requested and trial provisions will be returned.
                                        * REQUESTANDPURCHASE: Both provisions that have been requested and subscriptions will be returned.
                                        * TRIALANDPURCHASE: Both trial provisions and subscriptions will be returned.

                                    Values: "REQUEST" | "TRIAL" | "PURCHASE" | "REQUESTANDTRIAL" | "REQUESTANDPURCHASE" | "TRIALANDPURCHASE"
        ---------------------       ------------------------------------------------------------------------------------
        modified                    Optional String. The last modified date of the provisions to be returned. The date
                                    specified should be in milliseconds from epoch.
        ---------------------       ------------------------------------------------------------------------------------
        sort_field                  Optional String. The fields to sort provisions by. The allowed sort field names are
                                    orgname, created, endDate, and modified.
        ---------------------       ------------------------------------------------------------------------------------
        sort_order                  Optional String. Describe whether the order returns in ascending (asc) or
                                    descending (desc) order. The default is asc.
        ---------------------       ------------------------------------------------------------------------------------
        include_listing             Optional Boolean. If True, listing objects are included in the provision response.
                                    The default is True.
        ---------------------       ------------------------------------------------------------------------------------
        num                         Optional Integer. The maximum number of provisions to be included in the result
                                    set response. The default value is 10, and the maximum allowed value is 100.
                                    The start parameter, along with the num parameter, can be used to paginate the
                                    query results. Note that the actual number of returned results may be less than num.
                                    This happens when the number of results remaining after start is less than num.
        ---------------------       ------------------------------------------------------------------------------------
        start                       Optional Integer. The number of the first entry in the result set response. The
                                    index number is 1-based. The default value of start is 1. (i.e. the first search result).
                                    The start parameter, along with the num parameter, can be used to paginate the
                                    query results.
        =====================       ====================================================================================

        :return: A dictionary.
        """
        params = {
            "f": "json",
            "status": status,
            "type": type,
            "itemId": itemid,
            "orgname": orgname,
            "modified": modified,
            "sortFields": sort_fields,
            "sortOrder": sort_order,
            "includeListing": include_listing,
            "num": num,
            "start": start,
        }
        url = f"{self._gis._portal.resturl}portals/self/customers"
        return self._gis._portal.con.get(url, params)

    # ----------------------------------------------------------------------
