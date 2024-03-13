from arcgis.gis import GIS
import datetime
import pandas as pd


def date_range_search_string(
    from_datetime: datetime.datetime = None,
    to_datetime: datetime.datetime = None,
) -> str:
    """
    Returns a search string for filtering data within a specified date range.

    Args:
        from_datetime (Optional[datetime.datetime]): The starting date and time of the range.
            If not provided, the range will start from the epoch (0).
        to_datetime (Optional[datetime.datetime]): The ending date and time of the range.
            If not provided, the range will end at the current date and time.

    Returns:
        str: The search string in the format "[from_datetime TO to_datetime]".

    """
    # If we have a from_datetime, convert it to milliseconds since the epoch
    if from_datetime:
        from_datetime = int(from_datetime.timestamp() * 1000)
    else:
        from_datetime = 0
    if to_datetime:
        to_datetime = int(to_datetime.timestamp() * 1000)
    else:
        to_datetime = datetime.datetime.now()
    return f"[{from_datetime} TO {to_datetime}]"


def items_search(
    gis: GIS,
    append_search_string: str = None,
    owner: str = None,
    group: str = None,
    tag: str = None,
    content_status: str = None,
    created_from: datetime.datetime = None,
    created_to: datetime.datetime = None,
    modified_from: datetime.datetime = None,
    modified_to: datetime.datetime = None,
    output_path: str = None,
):
    """
    Searches for items in ArcGIS Online organization or Portal for ArcGIS based on the specified criteria.

    Args:
        gis (GIS): The GIS object representing the ArcGIS Online organization or Portal for ArcGIS.
        append_search_string (str, optional): Additional search string to be appended to the main search string. Defaults to None.
        owner (str, optional): Username of the item owner. Defaults to None.
        group (str, optional): Name of the group to filter the search results by. Defaults to None.
        tag (str, optional): Tag to filter the search results by. Defaults to None.
        content_status (str, optional): Content status to filter the search results by. Must be one of ["deprecated", "org_authoritative", "public_authoritative"]. Defaults to None.
        created_from (datetime.datetime, optional): Start date of the item creation range. Defaults to None.
        created_to (datetime.datetime, optional): End date of the item creation range. Defaults to None.
        modified_from (datetime.datetime, optional): Start date of the item modification range. Defaults to None.
        modified_to (datetime.datetime, optional): End date of the item modification range. Defaults to None.

    Returns:
        list: List of search results matching the specified criteria.
    """
    # First, building the search string

    search_string = f'(orgid:"{gis.properties["id"]}")'

    # If we have a content status, we need to add it to the search string
    if content_status:
        # Valid content status values
        allowed_content_status = [
            "deprecated",
            "org_authoritative",
            "public_authoritative",
            None,
        ]
        # If the content status is not valid, raise an error
        if content_status not in allowed_content_status:
            raise ValueError(
                f"Invalid content status. Must be one of {allowed_content_status}"
            )
        search_string += f" AND contentStatus:{content_status}"

    # If we have a username, we need to add it to the search string
    if owner:
        search_string += f" AND owner:{owner}"

    # If we have a group, we need to get the group ID and add it to the search string
    if group:
        group_id = gis.groups.get(group)
        search_string += f" AND group:{group_id}"

    # If we have a tag, we need to add it to the search string
    if tag:
        search_string += f" AND tags:{tag}"

    # If we have a date range, we need to add it to the search string
    if created_from or created_to:
        search_string += (
            f" AND created: {date_range_search_string(created_from, created_to)}"
        )

    if modified_from or modified_to:
        search_string += (
            f" AND modified: {date_range_search_string(modified_from, modified_to)}"
        )

    # If we have an additional search string, we need to add it to the search string
    if append_search_string:
        search_string += f" AND ({append_search_string})"

    # Perform the search
    search_results = gis.content.advanced_search(
        query=search_string,
        as_dict=True,
        max_items=10000,
    )

    if output_path:
        # Use pandas to write the search results to an Excel file
        pd.DataFrame(search_results["results"]).to_excel(output_path, index=False)

    return search_results
