from __future__ import annotations

import requests
from enum import Enum
from typing import Any, Iterator
from functools import lru_cache
from arcgis.gis import GIS
from arcgis.auth import EsriSession

__all__ = ["get_places_api"]


###########################################################################
class PlaceIdEnums(Enum):
    """
    When obtaining a single place by ID, the enumerations can filter and
    return specific information about a specific place ID.  To obtain a
    place ID, you must perform a query first.
    """

    ALL = "all"
    ADDITIONALLOCATIONS = "additionalLocations"
    ADDITIONALLOCATIONS_DROPOFF = "additionalLocations:dropOff"
    ADDITIONALLOCATIONS_FRONTDOOR = "additionalLocations:frontDoor"
    ADDITIONALLOCATIONS_ROAD = "additionalLocations:road"
    ADDITIONALLOCATIONS_ROOF = "additionalLocations:roof"
    ADDRESS = "address"
    ADDRESS_ADMINREGION = "address:adminRegion"
    ADDRESS_CENSUSBLOCKID = "address:censusBlockId"
    ADDRESS_COUNTRY = "address:country"
    ADDRESS_DESIGNATEDMARKETAREA = "address:designatedMarketArea"
    ADDRESS_EXTENDED = "address:extended"
    ADDRESS_LOCALITY = "address:locality"
    ADDRESS_NEIGHBORHOOD = "address:neighborhood"
    ADDRESS_POBOX = "address:poBox"
    ADDRESS_POSTCODE = "address:postcode"
    ADDRESS_POSTTOWN = "address:postTown"
    ADDRESS_REGION = "address:region"
    ADDRESS_STREETADDRESS = "address:streetAddress"
    CATEGORIES = "categories"
    CONTACTINFO = "contactInfo"
    CONTACTINFO_EMAIL = "contactInfo:email"
    CONTACTINFO_FAX = "contactInfo:fax"
    CONTACTINFO_TELEPHONE = "contactInfo:telephone"
    CONTACTINFO_WEBSITE = "contactInfo:website"
    CHAINS = "chains"
    DESCRIPTION = "description"
    HOURS = "hours"
    HOURS_OPENING = "hours:opening"
    HOURS_OPENINGTEXT = "hours:openingText"
    HOURS_POPULAR = "hours:popular"
    LOCATION = "location"
    NAME = "name"
    RATING = "rating"
    RATING_PRICE = "rating:price"
    RATING_USER = "rating:user"
    SOCIALMEDIA = "socialMedia"
    SOCIALMEDIA_FACEBOOKID = "socialMedia:facebookId"
    SOCIALMEDIA_INSTAGRAM = "socialMedia:instagram"
    SOCIALMEDIA_TWITTER = "socialMedia:twitter"


###########################################################################
class PlacesAPI:
    """
    The places service is a ready-to-use location service that can search
    for businesses and geographic locations around the world. It allows
    you to find, locate, and discover detailed information about each place.
    """

    _gis: GIS = None
    _urls: dict = None

    # ---------------------------------------------------------------------
    def __init__(self, gis: GIS) -> None:
        assert gis._portal.is_arcgisonline, "The GIS must be ArcGIS Online."
        assert not gis.users.me is None or gis.properties.get(
            "appInfo", None
        ), "You must be signed into the GIS to use the Places API"
        assert self._check_privileges(
            gis=gis
        ), "The current GIS does not have PlaceAPI permissions"
        self._gis = gis
        self._urls = {
            "base_url": "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1",
            "near-point": "/places/near-point",
            "within-extent": "/places/within-extent",
            "categories": "/categories",
            "places": "/places",
        }
        self.session: EsriSession = gis._con._session

    @lru_cache(maxsize=255)
    def _check_privileges(self, gis: GIS) -> bool:
        """
        Checks to see if the current login has the proper permisions to use
        the Places API

        :return: bool

        """
        user: "User" = gis.users.me
        properties: dict[str, Any] = dict(gis.properties)
        if user:
            if "premium:user:places" in user.privileges:
                return True
        else:
            if "premium:user:places" in properties.get("appInfo", {}).get(
                "privileges", []
            ):
                return True
        return False

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"< {self.__class__.__name__} @ {self._urls['base_url']} >"

    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        return f"< {self.__class__.__name__} @ {self._urls['base_url']} >"

    # ---------------------------------------------------------------------
    def examine_category(self, category: str) -> dict[str, Any]:
        """
        Get the category details for a category ID.

        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        category                   Required String. The category ID to examine.
        ======================     ===============================================================

        :return: Dictionary
        """
        url: str = f"{self._urls['base_url']}{self._urls['categories']}/{category}"
        params = {
            "f": "json",
        }
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    # ---------------------------------------------------------------------
    def find_category(self, query: str) -> dict[str, Any]:
        """
        Return the name and category ID of all categories, or categories
        which satisfy a filter.

        A category describes a type of place, such as "movie theater" or
        "zoo". The places service has over 1,000 categories (or types) of
        place. The categories fall into ten general groups: Arts and
        Entertainment, Business and Professional Services, Community and
        Government, Dining and Drinking, Events, Health and Medicine,
        Landmarks and Outdoors, Retail, Sports and Recreation, and Travel
        and Transportation.


        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        query                      Required String. The filter string used to find the matching
                                   categories.
        ======================     ===============================================================

        :return: Dictionary

        """
        url: str = f"{self._urls['base_url']}{self._urls['categories']}"
        params = {
            "f": "json",
            "filter": query,
            "token": self.session.auth.token,
        }
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    # ---------------------------------------------------------------------
    def search_by_radius(
        self,
        point: list[float] | list[int],
        radius: float | int = 1000,
        categories: list[str] | None = None,
        search_text: str | None = None,
        page_size: int = 10,
    ) -> Iterator[dict[str, Any]]:
        """
        Search for places near a point or location by radius.

        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        point                      list/tuple[float]. The X/Y coordinates centroid to search by
                                   radius by.  The coordinates must be in WGS-1984 (Lat/Long).

                                   Example: [-73.991997,40.743648]
        ----------------------     ---------------------------------------------------------------
        categories                 Optional list[str]. The category IDs to examine.
        ----------------------     ---------------------------------------------------------------
        search_text                Optional str. The free search text for places against names, categories etc.
        ----------------------     ---------------------------------------------------------------
        page_size                  Optional Integer. The amount of records to return per query. The default is 10.
        ======================     ===============================================================

        :yield: dict[str,Any]

        """
        x, y = point
        params: dict[str, Any] = {
            "f": "json",
            "x": x,
            "y": y,
            "radius": radius,
            "searchText": search_text or "",
            "pageSize": page_size,
        }
        if search_text is None:
            search_text = ""
        if not categories is None:
            params["categoriesIds"] = ",".join(categories)
        url: str = f"{self._urls['base_url']}{self._urls['near-point']}"
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        for result in data.get("results", []):
            yield result
        while (
            not data.get("pagination", {}).get("nextUrl") is None
            and len(data.get("results", [])) != 0
        ):
            url = data.get("pagination", {}).get("nextUrl")
            resp = self.session.get(url=url)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            if len(data.get("results", [])) == 0:
                return
            for result in data.get("results", []):
                yield result

    # ---------------------------------------------------------------------
    def search_by_extent(
        self,
        bbox: list[float] | list[int],
        categories: list[str] | None = None,
        search_text: str | None = None,
        page_size: int = 10,
    ) -> Iterator[dict[str, Any]]:
        """
        Search for places within an extent (bounding box).

        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        bbox                       list/tuple[float]. The min X/Y and max X/Y coordinates to
                                   search within. Coordinates must be in WGS-1984 (Lat/Long).

                                   Example: [-54,-75,54,75]
        ----------------------     ---------------------------------------------------------------
        categories                 Optional list[str]. The category IDs to examine.
        ----------------------     ---------------------------------------------------------------
        search_text                Optional str. The free search text for places against names, categories etc.
        ----------------------     ---------------------------------------------------------------
        page_size                  Optional Integer. The amount of records to return per query. The default is 10.
        ======================     ===============================================================

        :return: Iterator[dict[str,Any]]

        """
        data: dict[str, Any]
        resp: requests.Response
        xmin: float | int
        ymin: float | int
        xmax: float | int
        ymax: float | int

        if search_text is None:
            search_text = ""
        xmin, ymin, xmax, ymax = bbox
        params: dict[str, Any] = {
            "f": "json",
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "searchText": search_text,
            "pageSize": page_size,
        }
        if not categories is None:
            params["categoriesIds"] = ",".join(categories)
        url: str = f"{self._urls['base_url']}{self._urls['within-extent']}"
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        for result in data.get("results", []):
            yield result
        while (
            not data.get("pagination", {}).get("nextUrl") is None
            and len(data.get("results", [])) != 0
        ):
            url = data.get("pagination", {}).get("nextUrl")
            resp = self.session.get(url=url)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            if len(data.get("results", [])) == 0:
                return
            for result in data.get("results", []):
                yield result

    # ---------------------------------------------------------------------
    def get_place_by_id(
        self,
        placeid: str,
        filters: list[PlaceIdEnums] | None = None,
    ) -> dict[str, Any]:
        """
        Get place details including name, address, description, and other attributes.

        ======================     ===============================================================
        **Parameter**               **Description**
        ----------------------     ---------------------------------------------------------------
        placeid                    Required String. The Id of the place for which you want to fetch additional details.
        ----------------------     ---------------------------------------------------------------
        fields                     Optional list[string]. The array of fields that define the attributes to return for a place.
        ======================     ===============================================================

        :returns: dict[str,Any]

        """
        if filters is None:
            filters = [PlaceIdEnums.ALL]
        url: str = f"{self._urls['base_url']}{self._urls['places']}/{placeid}"
        filters: list[str] = [f.value for f in filters if isinstance(f, PlaceIdEnums)]
        params = {
            "f": "json",
            "requestedFields": ",".join(filters),
        }
        resp: requests.Response = self.session.get(url=url, params=params)
        resp.raise_for_status()
        return resp.json()


###########################################################################


# -------------------------------------------------------------------------
@lru_cache(maxsize=50)
def get_places_api(gis: GIS) -> PlacesAPI:
    """
    Returns the PlacesAPI class for a given GIS object

    :return:
        An instance of the :class:`~arcgis.geocoding.PlacesAPI` for the
        :class:`~arcgis.gis.GIS`
    """
    return PlacesAPI(gis=gis)
