import sys
import copy
import json


# --------------------------------------------------------------------------
def _search(
    gis,
    query,
    stype="content",
    max_items=100,
    bbox=None,
    categories=None,
    category_filter=None,
    start=1,
    sort_field="avgrating",
    sort_order="asc",
    count_fields=None,
    count_size=None,
    group_id=None,
    as_dict=False,
    enrich=None,
):
    """
    Generalized advanced search method.  This method allows for the query and
    searching of users, groups, group content, and general content.  This method
    allows for the control of the finer grained operations hidden by the 'search'
    method where a user can control and define how all information is returned.

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    gis                 Required GIS. The connection object.
    ----------------    ---------------------------------------------------------------
    query               Required String.  The search query.
    ----------------    ---------------------------------------------------------------
    stype               Required String. The search type to find. This tells the
                        internal method what type of return object should be used.
                        Allowed values: `content`, `users`, `group_content`, and `groups`.
    ----------------    ---------------------------------------------------------------
    bbox                Optional String. This is the xmin,ymin,xmax,ymax bounding box to
                        limit the search in.  Items like documents do not have bounding
                        boxes and will not be included in the search.
    ----------------    ---------------------------------------------------------------
    categories          Optional String. A comma separated list of up to 8 org content
                        categories to search items. Exact full path of each category is
                        required, OR relationship between the categories specified.

                        Each request allows a maximum of 8 categories parameters with
                        AND relationship between the different categories parameters
                        called.
    ----------------    ---------------------------------------------------------------
    category_filters    Optional String. A comma separated list of up to 3 category
                        terms to search items that have matching categories. Up to 2
                        `category_filters` parameter are allowed per request. It can
                        not be used together with categories to search in a request.
    ----------------    ---------------------------------------------------------------
    start               Optional Int. The starting position to search from.  This is
                        only required if paging is needed.
    ----------------    ---------------------------------------------------------------
    sort_field          Optional String. Responses from the `search` operation can be
                        sorted on various fields. `avgrating` is the default.
    ----------------    ---------------------------------------------------------------
    sort_order          Optional String. The sequence into which a collection of
                        records are arranged after they have been sorted. The allowed
                        values are: asc for ascending and desc for descending.
    ----------------    ---------------------------------------------------------------
    count_fields        Optional String. A comma separated list of fields to count.
                        Maximum count fields allowed per request is 3. Supported count
                        fields: `tags`, `type`, `access`, `contentstatus`, and
                        `categories`.
    ----------------    ---------------------------------------------------------------
    count_size          Optional Int. The maximum number of field values to count for
                        each `count_fields`. The default value is None, and maximum size
                        allowed is 200.
    ----------------    ---------------------------------------------------------------
    group_id            Optional String. The unique `id` of the group to search for
                        content in. This is only used if `group_content` is used for
                        searching.
    ----------------    ---------------------------------------------------------------
    as_dict             Required Boolean. If True, the response comes back as a dictionary.
    ================    ===============================================================

    """
    from arcgis.gis import GIS, Item, User, Group

    if gis is None:
        import arcgis

        gis = arcgis.env.active_gis

    if max_items == -1:
        page_size = 100
    else:
        page_size = min(max_items, 100)
    max_items = page_size
    items = []
    params = {
        "f": "json",
        "q": query,
        "start": start,
        "num": page_size,
        "sortField": sort_field,
        "sortOrder": sort_order,
    }
    stype = str(stype).lower()
    if categories:
        params["categories"] = json.dumps(categories)
    if category_filter:
        params["categoryFilters"] = category_filter
    if count_fields:
        params["countFields"] = count_fields
    if count_size:
        params["countSize"] = count_size
    if bbox:
        if isinstance(bbox, (tuple, list)):
            bbox = ",".join([str(b) for b in bbox])
        params["bbox"] = bbox
    if stype in {"content", "item", "items"}:
        url = "{base}search".format(base=gis._portal.resturl)
        if enrich:
            params["enrich"] = enrich
    elif stype == "group_content" and group_id:
        url = "{base}content/groups/{gid}/search".format(
            base=gis._portal.resturl, gid=group_id
        )
    elif stype == "group_content" and group_id is None:
        raise
    elif stype == "portal_users":
        allowed_keys = {
            "q",
            "start",
            "num",
            "sortField",
            "sortOrder",
            "f",
            "token",
        }
        for k in list(params.keys()):
            if not k in allowed_keys:
                del params[k]
            del k
        url = "{base}portals/self/users".format(base=gis._portal.resturl)
    elif stype in {"user", "groups", "users", "group"}:
        allowed_keys = {
            "q",
            "start",
            "num",
            "sortField",
            "sortOrder",
            "f",
            "token",
        }
        for k in list(params.keys()):
            if not k in allowed_keys:
                del params[k]
            del k
        if stype in ["user", "users"]:
            url = "{base}community/users".format(base=gis._portal.resturl)
        elif stype in {"group", "groups"}:
            url = "{base}community/groups".format(base=gis._portal.resturl)
    count = 0
    res = gis._con.post(url, params)
    results = copy.deepcopy(res)
    count += int(res["num"])
    nextstart = int(res["nextStart"])
    while (count < max_items and max_items > 0) or (nextstart > 0 and max_items == -1):
        params["start"] = res["nextStart"]
        if len(results["results"]) >= max_items and max_items != -1:
            break
        res = gis._con.post(url, params)
        results["results"].extend(res["results"])
        count += int(res["num"])
        nextstart = int(res["nextStart"])
        if len(res["results"]) == 0:
            break
    if len(results["results"]) > max_items and max_items != -1:
        results["results"] = results["results"][:max_items]
    ## Clean up Response
    ##
    results["results"] = _handle_response(
        as_dict=as_dict, gis=gis, res=results, stype=stype
    )
    return results


# --------------------------------------------------------------------------
def _handle_response(res, stype, gis, as_dict):
    """
    Handles returning the data in the proper format

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    res                 Required Dict.  The response dictionary from the query call.
    ----------------    ---------------------------------------------------------------
    stype               Required String. The search type to find. This tells the
                        internal method what type of return object should be used.
    ----------------    ---------------------------------------------------------------
    gis                 Required GIS. The connection object.
    ----------------    ---------------------------------------------------------------
    as_dict             Required Boolean. If True, the response comes back as a dictionary.
    ================    ===============================================================

    :return: List

    """
    from arcgis.gis import GIS, Item, User, Group

    if as_dict:
        return res["results"]
    elif str(stype).lower() in {"content", "item", "items", "group_content"}:
        return [Item(itemid=r["id"], itemdict=r, gis=gis) for r in res["results"]]
    elif str(stype).lower() in ["user", "users", "accounts", "account"]:
        return [
            User(gis=gis, username=r["username"], userdict=res) for r in res["results"]
        ]
    elif str(stype).lower() in ["groups", "group"]:
        return [Group(groupdict=r, groupid=r["id"], gis=gis) for r in res["results"]]
    return res["results"]
