"""
Types and functions for geocoding.
"""
from __future__ import annotations
import copy
from typing import Any, Optional, Union
from arcgis.features.layer import FeatureLayer
from ..gis import GIS, _GISResource, Item
import arcgis.env
import logging
from ..features import FeatureSet
from ..geometry import Geometry, Point, SpatialReference
from arcgis._impl.common._utils import _validate_url, chunks

_LOGGER = logging.getLogger(__name__)


class Geocoder(_GISResource):
    """
    The ``Geocoder`` class represents Geocoder objects.
    ``Geocoder`` objects can find point locations of addresses, business names, and so on.
    The output points can be visualized on a map, inserted as stops for a route,
    or loaded as input for spatial analysis. It is also used to generate
    batch results for a set of addresses, as well as for reverse geocoding,
    i.e. determining the address at a particular x/y location.

    .. note::
        A :class:`~arcgis.gis.GIS` includes one or more geocoders, which can be queried using `get_geocoders(gis)`.

    .. note::
        ``Geocoders`` shared as :class:`~arcgis.gis.Item` objects in the GIS can be obtained using
        `Geocoder.fromitem(item)`.

    .. note::
        ``Geocoder`` objects may also be created using the constructor by passing in their location, such as
        a url to a ``Geocoding Service``.
    """

    def __init__(self, location, gis=None):
        """
        Creates a Geocoder from a location, such as a url to a Geocoding Service.
        :param location: geocoder location, such as a url to a Geocoding Service.
        :param gis: the gis to which the geocoder belongs
        """
        super(Geocoder, self).__init__(location, gis)
        try:
            from arcgis.gis.server._service._adminfactory import (
                AdminServiceGen,
            )

            self.service = AdminServiceGen(service=self, gis=gis)
        except:
            pass
        try:
            self._address_field = self.properties.singleLineAddressField.name
        except:
            pass  # print("Geocoder does not support single line address input")

    @classmethod
    def fromitem(cls, item: Item):
        """
        The ``fromitem`` method creates a ``Geocoder`` from an :class:`~arcgis.gis.Item` in the
        class:`~arcgis.gis.GIS` instance.

        =================== ====================================================
        **Parameter**        **Description**
        ------------------- ----------------------------------------------------
        item                A required :class:`~arcgis.gis.Item` object. The
                            ``Item`` to convert to a ``Geocoder`` object.

                            .. note::
                                The :class:`~arcgis.gis.Item` must be of type
                                ``Geocoding Service``.
        =================== ====================================================

        :return:
            A :class:`~arcgis.geocoding.Geocoder` object.

        """
        if not item.type == "Geocoding Service":
            raise TypeError(
                "item must be a type of Geocoding Service, not " + item.type
            )
        url = _validate_url(item.url, item._gis)
        return cls(url, item._gis)

    def _geocode(
        self,
        address,
        search_extent=None,
        location=None,
        distance=None,
        out_sr=None,
        category=None,
        out_fields="*",
        max_locations=20,
        magic_key=None,
        for_storage=False,
        as_featureset=False,
        match_out_of_range=True,
        location_type="street",
        lang_code=None,
        source_country=None,
    ):
        """
        The geocode method geocodes one location per request.

        ====================     ====================================================
        **Parameter**             **Description**
        --------------------     ----------------------------------------------------
        address                  Required list of strings or dictionaries.
                                 Specifies the location to be geocoded. This can be
                                 a string containing the street address, place name,
                                 postal code, or POI.

                                 Alternatively, this can be a dictionary containing
                                 the various address fields accepted by the
                                 corresponding geocoder. These fields are listed in
                                 the addressFields property of the associated
                                 geocoder. For example, if the address_fields of a
                                 geocoder includes fields with the following names:
                                 Street, City, State and Zone, then the address
                                 argument is of the form:
                                 {
                                    Street: "1234 W Main St",
                                    City: "Small Town",
                                    State: "WA",
                                    Zone: "99027"
                                 }
        --------------------     ----------------------------------------------------
        search_extent            Optional string. A set of bounding box coordinates
                                 that limit the search area to a specific region.
                                 This is especially useful for applications in which
                                 a user will search for places and addresses only
                                 within the current map extent.
        --------------------     ----------------------------------------------------
        location                 Optional [x,y]. Defines an origin point location that
                                 is used with the distance parameter to sort
                                 geocoding candidates based upon their proximity to
                                 the location.
        --------------------     ----------------------------------------------------
        distance                 Optional float. Specifies the radius of an area
                                 around a point location which is used to boost the
                                 rank of geocoding candidates so that candidates
                                 closest to the location are returned first. The
                                 distance value is in meters.
        --------------------     ----------------------------------------------------
        out_sr                   Optional dictionary. The spatial reference of the
                                 x/y coordinates returned by a geocode request. This
                                 is useful for applications using a map with a spatial
                                 reference different than that of the geocode service.
        --------------------     ----------------------------------------------------
        category                 Optional string. A place or address type which can
                                 be used to filter find results. The parameter
                                 supports input of single category values or multiple
                                 comma-separated values. The category parameter can be
                                 passed in a request with or without the text
                                 parameter.
        --------------------     ----------------------------------------------------
        out_fields               Optional string. Name of all the fields to inlcude.
                                 The default is "*" which means all fields.
        --------------------     ----------------------------------------------------
        max_locations            Optional integer. The number of locations to be
                                 returned from the service. The default is 20.
        --------------------     ----------------------------------------------------
        magic_key                The find operation retrieves results quicker when
                                 you pass a valid text and magickey value.
        --------------------     ----------------------------------------------------
        for_storage              Specifies whether the results of the operation will
                                 be persisted. The default value is false, which
                                 indicates the results of the operation can't be
                                 stored, but they can be temporarily displayed on a
                                 map for instance. If you store the results, in a
                                 database for example, you need to set this parameter
                                 to true.
        --------------------     ----------------------------------------------------
        geocoder                 Optional, the geocoder to be used. If not specified,
                                 the active GIS's first geocoder is used.
        --------------------     ----------------------------------------------------
        as_featureset            Optional boolean. If True, the result set will be a
                                 FeatureSet instead of a dictionary. False is default
        --------------------     ----------------------------------------------------
        match_out_of_range       Optional Boolean. Provides better spatial accuracy
                                 for inexact street addresses by specifying whether
                                 matches will be returned when the input number is
                                 outside of the house range defined for the input
                                 street. Out of range matches will be defined as
                                 Addr_type=StreetAddressExt. Input house numbers
                                 that exceed the range on a street segment by more
                                 than 100 will not result in `StreetAddressExt`
                                 matches. The default value of this parameter is
                                 True.
        --------------------     ----------------------------------------------------
        location_type            Optional Str. Specifies whether the rooftop point or
                                 street entrance is used as the output geometry of
                                 PointAddress matches. By default, street is used,
                                 which is useful in routing scenarios, as the rooftop
                                 location of some addresses may be offset from a
                                 street by a large distance. However, for map display
                                 purposes, you may want to use rooftop instead,
                                 especially when large buildings or landmarks are
                                 geocoded. The `location_type` parameter only affects
                                 the location object in the JSON response and does
                                 not change the x,y or DisplayX/DisplayY attribute
                                 values.

                                 Values: `street` or `rooftop`
        --------------------     ----------------------------------------------------
        lang_code                Optional str. Sets the language in which geocode
                                 results are returned.
        --------------------     ----------------------------------------------------
        source_country           Optional str. Limits the returned candidates to the
                                 specified country or countries for either single-field
                                 or multifield requests. Acceptable values include
                                 the 3-character country code.
        ====================     ====================================================

        :return:
           dictionary or FeatureSet

        """
        url = self.url + "/findAddressCandidates"

        params = {
            "f": "json",
        }

        if address is not None:
            if isinstance(address, str):
                params[self._address_field] = address
            elif isinstance(address, dict):
                params.update(address)
            else:
                print(
                    "address should be a string (single line address) or dictionary "
                    "(with address fields as keys)"
                )

        if not magic_key is None:
            params["magicKey"] = magic_key
        if not search_extent is None:
            params["searchExtent"] = search_extent
        if not location is None and isinstance(location, list):
            params["location"] = "%s,%s" % (location[0], location[1])
        elif location is not None:
            params["location"] = location
        if not distance is None:
            params["distance"] = distance
        if not out_sr is None:
            params["outSR"] = out_sr
        if not category is None:
            params["category"] = category
        if out_fields is None:
            params["outFields"] = "*"
        else:
            params["outFields"] = out_fields
        if not max_locations is None:
            params["maxLocations"] = max_locations
        if not for_storage is None:
            params["forStorage"] = for_storage
        if not match_out_of_range is None:
            params["matchOutOfRange"] = match_out_of_range
        if not location_type is None:
            params["locationType"] = location_type
        if lang_code:
            params["langCode"] = lang_code
        if source_country:
            params["sourceCountry"] = source_country
        resp = self._con.post(url, params, token=self._token)

        if resp is not None and as_featureset:
            features = []
            sr = resp["spatialReference"]
            for c in resp["candidates"]:
                geom = c["location"]
                geom["spatialReference"] = sr
                features.append(
                    {
                        "geometry": Geometry(geom),
                        "attributes": c["attributes"],
                    }
                )

            return FeatureSet(
                features=features, spatial_reference=sr
            )  # resp['candidates']
        elif resp is not None and as_featureset == False:
            return resp["candidates"]
        else:
            return []

    def _reverse_geocode(
        self,
        location,
        distance=None,
        out_sr=None,
        lang_code=None,
        return_intersection=False,
        for_storage=False,
        as_featureset=False,
        feature_types=None,
        location_type="street",
    ):
        """
        The reverseGeocode operation determines the address at a particular
        x/y location. You pass the coordinates of a point location to the
        geocoding service, and the service returns the address that is
        closest to the location.


        Input:
           location - a list defined as [X,Y] or a JSON Point
        """
        params = {"f": "json"}
        url = self.url + "/reverseGeocode"
        if isinstance(location, list):
            params["location"] = "%s,%s" % (location[0], location[1])
        elif isinstance(location, dict):
            params["location"] = location
        else:
            raise Exception("Invalid location")

        if distance is not None:
            params["distance"] = distance
        if out_sr is not None:
            params["outSR"] = out_sr
        if lang_code is not None:
            params["langCode"] = lang_code
        if return_intersection:
            params["returnIntersection"] = return_intersection
        if for_storage:
            params["forStorage"] = for_storage
        if feature_types:
            if isinstance(feature_types, list):
                feature_types = ",".join(feature_types)
            params["featureTypes"] = feature_types
        if location_type:
            params["locationType"] = location_type

        resp = self._con.post(url, params, token=self._token)
        if resp is not None and as_featureset:
            geom = copy.copy(resp["location"])
            del resp["location"]
            fs = FeatureSet(
                features=[
                    {
                        "geometry": Geometry(geom),
                        "attributes": resp["address"],
                    }
                ]
            )
            return fs
        return resp

    def _batch_geocode(
        self,
        addresses: list,
        source_country: Optional[str] = None,
        category: Optional[str] = None,
        out_sr: Optional[str] = None,
        as_featureset: Optional[bool] = False,
        match_out_of_range: Optional[bool] = True,
        location_type: Optional[str] = "street",
        search_extent: Optional[str] = None,
        lang_code: Optional[str] = "EN",
        preferred_label_values: Optional[str] = None,
        out_fields: Optional[str] = None,
    ):
        """
        The batch_geocode() method geocodes an entire list of addresses.
        Geocoding many addresses at once is also known as bulk geocoding.


        Inputs:
           addresses - A list of addresses to be geocoded.
           For passing in the location name as a single line of text -
           single field batch geocoding - use a string.
           For passing in the location name as multiple lines of text
           multifield batch geocoding - use the address fields described
           in the Geocoder documentation.
            The maximum number of addresses that can be geocoded in a
            single request is limited to the SuggestedBatchSize property of
            the locator.
            Syntax:
             addresses = ["380 New York St, Redlands, CA",
             "1 World Way, Los Angeles, CA",
             "1200 Getty Center Drive, Los Angeles, CA",
             "5905 Wilshire Boulevard, Los Angeles, CA",
             "100 Universal City Plaza, Universal City, CA 91608",
             "4800 Oak Grove Dr, Pasadena, CA 91109"]

             OR

             addresses= [{
                "Address": "380 New York St.",
                "City": "Redlands",
                "Region": "CA",
                "Postal": "92373"
            },{
                "Address": "1 World Way",
                "City": "Los Angeles",
                "Region": "CA",
                "Postal": "90045"
            }]

           sourceCountry - The sourceCountry parameter is only supported by
            geocode services published using StreetMap Premium locators.
            Added at 10.3 and only supported by geocode services published
            with ArcGIS 10.3 for Server and later versions.
           category - The category parameter is only supported by geocode
            services published using StreetMap Premium locators.
           outSR - The well-known ID of the spatial reference, or a spatial
            reference json object for the returned addresses. For a list of
            valid WKID values, see Projected coordinate systems and
            Geographic coordinate systems.
        """
        if (
            "locatorProperties" in self.properties
            and "MaxBatchSize" in self.properties.locatorProperties
        ):
            max_batch_size = self.properties.locatorProperties.MaxBatchSize
        else:
            max_batch_size = 1000
        params = {"f": "json"}
        url = self.url + "/geocodeAddresses"
        if out_sr is not None:
            params["outSR"] = out_sr
        if source_country is not None:
            params["sourceCountry"] = source_country
        if category is not None:
            params["category"] = category

        addr_recordset = []
        if len(addresses) <= max_batch_size:
            for index in range(len(addresses)):
                address = addresses[index]

                attributes = {"OBJECTID": index}
                if isinstance(address, str):
                    attributes[self._address_field] = address
                elif isinstance(address, dict):
                    attributes.update(address)
                else:
                    print("Unsupported address: " + str(address))
                    print(
                        "address should be a string (single line address) or dictionary "
                        "(with address fields as keys)"
                    )

                addr_rec = {"attributes": attributes}
                addr_recordset.append(addr_rec)

            params["addresses"] = {"records": addr_recordset}
            params["matchOutOfRange"] = match_out_of_range
            params["locationType"] = location_type
            if search_extent is not None:
                params["searchExtent"] = search_extent
            params["langCode"] = lang_code
            if preferred_label_values is not None:
                params["preferredLabelValues"] = preferred_label_values
            if out_fields is not None:
                params["outFields"] = out_fields
            resp = self._con.post(url, params)
            if resp is not None and as_featureset:
                sr = resp["spatialReference"]

                matches = [None] * len(addresses)
                locations = resp["locations"]
                for idx, location in enumerate(locations):
                    geom = copy.copy(location.get("location", None))
                    if geom and "spatialReference" not in geom:
                        geom["spatialReference"] = sr
                    att = location["attributes"]
                    if geom:
                        matches[idx] = {
                            "geometry": Geometry(geom),
                            "attributes": att,
                        }
                    else:
                        matches[idx] = {"geometry": None, "attributes": att}
                return FeatureSet(features=matches, spatial_reference=sr)
            elif resp is not None and as_featureset == False:
                matches = [None] * len(addresses)
                locations = resp["locations"]
                for idx, location in enumerate(locations):
                    matches[idx] = location
                return matches
            else:
                return []
        else:
            result = [
                self._batch_geocode(
                    addresses=chunk,
                    source_country=source_country,
                    category=category,
                    out_sr=out_sr,
                    as_featureset=as_featureset,
                    match_out_of_range=match_out_of_range,
                    location_type=location_type,
                    search_extent=search_extent,
                    lang_code=lang_code,
                    preferred_label_values=preferred_label_values,
                )
                for chunk in chunks(addresses, max_batch_size)
            ]
            if len(result) == 0:
                result = [[]]
            if as_featureset:
                if len(result) > 1:
                    parent = result[0]
                    for fs in result[1:]:
                        parent.features.extend(fs.features)
                    return parent
                elif len(result) == 1:
                    return result[0]
                else:
                    return None
            else:
                try:
                    return [y for x in result for y in x]
                except:
                    return []

    def _find_best_match(
        self,
        address,
        search_extent=None,
        location=None,
        distance=None,
        out_sr=None,
        category=None,
        out_fields="*",
        magic_key=None,
        for_storage=False,
    ):
        """Returns the (latitude, longitude) or (y, x) coordinates of the best match for specified address"""
        candidates = self._geocode(
            address,
            search_extent,
            location,
            distance,
            out_sr,
            category,
            out_fields,
            1,
            magic_key,
            for_storage,
        )
        if candidates:
            location = candidates[0]["location"]
            return location["y"], location["x"]

    def _suggest(
        self,
        text,
        location=None,
        distance=None,
        category=None,
        search_extent=None,
        max_suggestions=5,
        country_code=None,
    ):
        """
        The suggest operation is performed on a geocoder.
        The result of this operation is a resource representing a list of
        suggested matches for the input text. This resource provides the
        matching text as well as a unique ID value, which links a
        suggestion to a specific place or address.
        A geocode service must meet the following requirements to support
        the suggest operation:
          The address locator from which the geocode service was published
          must support suggestions. Only address locators created using
          ArcGIS 10.3 for Desktop and later can support suggestions. See
          the Create Address Locator geoprocessing tool help topic for more
          information.
          The geocode service must have the Suggest capability enabled.
          Only geocode services published using ArcGIS 10.3 for Server or
          later support the Suggest capability.
        The suggest operation allows character-by-character auto-complete
        suggestions to be generated for user input in a client application.
        This capability facilitates the interactive search user experience
        by reducing the number of characters that need to be typed before
        a suggested match is obtained. A client application can provide a
        list of suggestions that is updated with each character typed by a
        user until the address they are looking for appears in the list.
        Inputs:
           text - The input text provided by a user that is used by the
            suggest operation to generate a list of possible matches. This
            is a required parameter.
           location -  Defines an origin point location that is used with
            the distance parameter to sort suggested candidates based on
            their proximity to the location. The distance parameter
            specifies the radial distance from the location in meters. The
            priority of candidates within this radius is boosted relative
            to those outside the radius.
            This is useful in mobile applications where a user wants to
            search for places in the vicinity of their current GPS
            location. It is also useful for web mapping applications where
            a user wants to find places within or near the map extent.
            The location parameter can be specified without specifying a
            distance. If distance is not specified, it defaults to 2000
            meters.
            The object can be an common.geometry.Point or X/Y list object
           distance - Specifies the radius around the point defined in the
            location parameter to create an area, which is used to boost
            the rank of suggested candidates so that candidates closest to
            the location are returned first. The distance value is in
            meters.
            If the distance parameter is specified, the location parameter
            must be specified as well.
            It is important to note that the location and distance
            parameters allow searches to extend beyond the specified search
            radius. They are not used to filter results, but rather to rank
            resulting candidates based on their distance from a location.
           category - The category parameter is only supported by geocode
            services published using StreetMap Premium locators.
        """
        params = {"f": "json", "text": text}
        url = self.url + "/suggest"

        if isinstance(location, list):
            params["location"] = "%s,%s" % (location[0], location[1])
        elif isinstance(location, dict):
            params["location"] = dict(location)

        if not category is None:
            params["category"] = category
        if not distance is None and isinstance(distance, (int, float)):
            params["distance"] = distance
        if search_extent:
            params["searchExtent"] = search_extent
        if max_suggestions is not None and isinstance(max_suggestions, int):
            params["maxSuggestions"] = max_suggestions
        if country_code and isinstance(country_code, str):
            params["countryCode"] = country_code
        resp = self._con.post(url, params, token=self._token)
        return resp


def get_geocoders(gis: GIS):
    """
    The ``get_geocoders`` method is used to query the list of geocoders registered with the :class:`~arcgis.gis.GIS`.

    .. note::
        A ``GIS`` includes one or more :class:`~arcgis.geocoding.Geocoder` objects.

    :param gis: the GIS whose registered geocoders are to be queried

    =================== ====================================================
    **Parameter**        **Description**
    ------------------- ----------------------------------------------------
    gis                 A required :class:`~arcgis.gis.Gis` object. The
                        ``GIS`` whose registered ``geocoders`` are to be
                        queried.

    =================== ====================================================

    :return:
        A list of :class:`~arcgis.geocoding.Geocoder` objects registered with the ``GIS``.
    """
    geocoders = []
    try:
        geocode_services = gis.properties["helperServices"]["geocode"]
        for geocode_service in geocode_services:
            try:
                url = _validate_url(geocode_service["url"], gis=gis)
                geocoders.append(Geocoder(url, gis))
            except RuntimeError as runtime_error:
                _LOGGER.warning("Unable to use Geocoder at " + geocode_service["url"])
                _LOGGER.warning(str(runtime_error))
    except KeyError:
        pass
    return geocoders


# ----------------------------------------------------------------------
def analyze_geocode_input(
    input_table_or_item: Union[Item, str, dict[str, str]],
    geocode_service_url: Optional[Union[str, Geocoder]] = None,
    column_names: Optional[str] = None,
    input_file_parameters: Optional[dict[str, str]] = None,
    locale: str = "en",
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
):
    """
    The ``analyze_geocode_input`` function takes in a geocode input (either a table or file of
    addresses) and returns an output dictionary that includes a suggested field mapping. It supports CSV,
    XLS, or table input. The table can be from a big data file share or from a feature service. The
    task generates a suggested field mapping based on the input fields and the geocoding service
    candidate fields and returns it in a ``geocode_parameters`` dictionary. This ``geocode_parameters``
    dictionary output is the an input to the ``Batch Geocode`` tool. The output ``geocode_parameters``
    dictionary also includes field info (name, length, and type) as well as additional information
    that helps the geocode tool parse the input file or table.

    =====================     ================================================================
    **Parameter**              **Description**
    ---------------------     ----------------------------------------------------------------
    input_table_or_item       required :class:`~arcgis.gis.Item`, string or dictionary.
                              The input to analyze for geocoding.

                              For tables:

                               The input table specification must include the following:

                               - A URL to an input table
                               - A service token to access the table
                               Note that if the table is a hosted table on the same portal,
                               serviceToken is not required.

                               Example: {"url":"<table url>","serviceToken":"<token>"}

                              For File Items:

                              The input file should be a portal item. Input the itemid of
                              the item in the portal. The format of the item in the portal
                              can be in one of the following formats:

                              - CSV
                              - Microsoft Excel spreadsheet (XLSX)
                              Example: {"itemid": "<itemid of file>" }
    ---------------------     ----------------------------------------------------------------
    geocode_service_url       Optional string or ``Geocoder`` object.  The geocode service
                              that you want to geocode your addresses against.
    ---------------------     ----------------------------------------------------------------
    column_names              Optional string.  Only used when input table or ``Item`` has no
                              header row.
                              Example: address,city,state,zip
    ---------------------     ----------------------------------------------------------------
    input_file_parameters     Optional dictionary. Enter information about how to parse the
                              file. If you are using an input table instead of an Item as
                              input, this parameter can be left blank.
                              Any of the key values in the dictionary can be left blank using
                              the "".

                              Values:

                              ``fileType`` - Enter CSV or XLS for the file format of file
                              Item.
                              ``headerRowExists`` - Enter true if your file has a header row,
                                                false if it does not.
                              ``columnDelimiter`` - Enter SPACE, TAB, COMMA, PIPE, or SEMICOLON.
                              ``textQualifier`` - Enter either SINGLE_QUOTE or DOUBLE_QUOTE.

                              Example: {"fileType":"xlsx","headerRowExists":"true",
                                        "columnDelimiter":"","textQualifier":""}
    ---------------------     ----------------------------------------------------------------
    locale                    Optional string. Enter the 2-letter ("en") or 4-letter ("ar-il")
                              specific locale if geocodeInput is in a language other than
                              English.
    ---------------------     ----------------------------------------------------------------
    context                   Optional dictionary.
                              Context contains additional settings that affect task execution.
                              ``analyze_geocode_input`` has the following two settings:
                              1. Extent (extent) - A bounding box that defines the analysis
                                 area. Only those points in inputLayer that intersect the
                                 bounding box are analyzed.
                              2. Output Spatial Reference (outSR) - The output features are
                                 projected into the output spatial reference.
    ---------------------     ----------------------------------------------------------------
    gis                       Optional ``GIS``. Connection to the site. If None is given, the
                              active ``GIS`` is used.
    =====================     ================================================================

    :return:
        A dictionary


    .. code block:: python

        :Usage Example:

        >>>res = analyze_geocode_input(geocode_service_url=my_geocoder_url,
                                       input_table_or_item={"itemid" : "abc123545asv"},
                                       input_file_parameters={"fileType":"csv","headerRowExists":"true",
                                                             "columnDelimiter":"","textQualifier":""})
        >>> print(res)
        {'header_row_exists': True, 'field_info': '[["Address", "TEXT", 255], ["City", "TEXT", 255],
        ["State", "TEXT", 255], ["ZipCode", "TEXT", 255]]', 'file_type': 'csv', 'field_mapping': '[["Address", ""],
        ["City", "City"], ["State", "State"], ["ZipCode", ""]]', 'column_names': '',
        'column_delimiter': '', 'text_qualifier': '', 'singleline_field': 'Single Line Input'}

    .. code block:: python
        :Usage Example 2:

        >>> table_lyr = Table(url="http://testsite.com/server/rest/services/Hosted/addresses/FeatureServer/0", gis=gis)
        >>> res = analyze_geocode_input()

    """
    import json
    from arcgis.gis import Item
    from arcgis.features.layer import Layer
    from arcgis.geoprocessing._tool import Toolbox

    gis = arcgis.env.active_gis if gis is None else gis
    analyze_geocode_url = gis.properties.helperServices.asyncGeocode.url
    tbx = Toolbox(url=analyze_geocode_url, gis=gis)

    if geocode_service_url is None:
        gcs = gis.properties.helperServices.geocode
        for gc in gcs:
            if "batch" in gc and gc["batch"]:
                geocode_service_url = gc["url"]
                break
            del gc
        del gcs
    elif isinstance(geocode_service_url, Geocoder):
        geocode_service_url = geocode_service_url.url
    elif isinstance(geocode_service_url, str) == False:
        raise ValueError("Invalid geocoder service given.")
    if geocode_service_url is None:
        raise ValueError(
            "The registered geocoders are not valid to use with this tool."
        )

    kwargs = {
        "geocode_service_url": geocode_service_url,
        "input_table": "",
        "input_file_item": None,
        "column_names": column_names,
        "input_file_parameters": input_file_parameters,
        "locale": locale,
        "context": context,
    }

    if isinstance(input_table_or_item, Item):
        kwargs["input_file_item"] = {"itemid": input_table_or_item.itemid}
        if input_file_parameters is None:
            kwargs["input_file_parameters"] = json.dumps(
                {
                    "fileType": input_table_or_item.type.lower(),
                    "headerRowExists": "true",
                    "columnDelimiter": "",
                    "textQualifier": "",
                }
            )
    elif isinstance(input_table_or_item, str):
        kwargs["input_file_item"] = {"itemid": input_table_or_item}
        if input_file_parameters is None:
            item = gis.content.search("id: %s" % input_table_or_item)[0]
            kwargs["input_file_parameters"] = json.dumps(
                {
                    "fileType": item.type.lower(),
                    "headerRowExists": "true",
                    "columnDelimiter": "",
                    "textQualifier": "",
                }
            )
    elif isinstance(input_table_or_item, dict):
        if "url" in input_table_or_item:
            kwargs["input_table"] = input_table_or_item
        elif "itemid" in input_table_or_item:
            kwargs["input_file_item"] = input_table_or_item
            if input_file_parameters is None:
                item = gis.content.search("id: %s" % input_table_or_item["itemid"])[0]
                kwargs["input_file_parameters"] = json.dumps(
                    {
                        "fileType": item.type.lower(),
                        "headerRowExists": "true",
                        "columnDelimiter": "",
                        "textQualifier": "",
                    }
                )
    elif isinstance(input_table_or_item, Layer):
        lyr_dict = input_table_or_item._lyr_dict
        if "type" in lyr_dict:
            lyr_dict.pop("type")
        kwargs["input_table"] = lyr_dict
    for k, v in list(kwargs.items()):
        if v is None:
            kwargs.pop(k)
    return tbx.analyze_geocode_input(**kwargs)


# ----------------------------------------------------------------------
def geocode_from_items(
    input_data: Union[Item, str, FeatureLayer],
    output_type: str = "Feature Layer",
    geocode_service_url: Optional[Union[str, Geocoder]] = None,
    geocode_parameters: Optional[dict[str, Any]] = None,
    country: Optional[str] = None,
    output_fields: Optional[str] = None,
    header_rows_to_skip: int = 1,
    output_name: Optional[str] = None,
    category: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    gis: Optional[GIS] = None,
):
    """
    The ``geocode_from_items`` method creates :class:`~arcgis.geocoding.Geocoder` objects from an
    :class:`~arcgis.gis.Item` or ``Layer`` objects.

    .. note::
        ``geocode_from_items`` geocodes the entire file regardless of size.

    =====================     ================================================================
    **Parameter**              **Description**
    ---------------------     ----------------------------------------------------------------
    input_data                required Item, string, Layer. Data to geocode.
    ---------------------     ----------------------------------------------------------------
    output_type               optional string.  Export item types.  Allowed values are "CSV",
                              "XLS", or "FeatureLayer".

                              .. note::
                                The default for ``output_type`` is "FeatureLayer".
    ---------------------     ----------------------------------------------------------------
    geocode_service_url       optional string of Geocoder. Optional
                              :class:`~arcgis.geocoding.Geocoder` to use to
                              spatially enable the dataset.
    ---------------------     ----------------------------------------------------------------
    geocode_parameters        optional dictionary.  This includes parameters that help parse
                              the input data, as well the field lengths and a field mapping.
                              This value is the output from the ``analyze_geocode_input``
                              available on your server designated to geocode. It is important
                              to inspect the field mapping closely and adjust them accordingly
                              before submitting your job, otherwise your geocoding results may
                              not be accurate. It is recommended to use the output from
                              ``analyze_geocode_input`` and modify the field mapping instead of
                              constructing this dictionary by hand.

                              **Values**

                              ``field_info`` - A list of triples with the field names of your input
                              data, the field type (usually TEXT), and the allowed length
                              (usually 255).

                              Example: [['ObjectID', 'TEXT', 255], ['Address', 'TEXT', 255],
                                       ['Region', 'TEXT', 255], ['Postal', 'TEXT', 255]]

                              ``header_row_exists`` - Enter true or false.

                              ``column_names`` - Submit the column names of your data if your data
                              does not have a header row.

                              ``field_mapping`` - Field mapping between each input field and
                              candidate fields on the geocoding service.
                              Example: [['ObjectID', 'OBJECTID'], ['Address', 'Address'],
                                          ['Region', 'Region'], ['Postal', 'Postal']]
    ---------------------     ----------------------------------------------------------------
    country                   optional string.  If all your data is in one country, this helps
                              improve performance for locators that accept that variable.
    ---------------------     ----------------------------------------------------------------
    output_fields             optional string. Enter the output fields from the geocoding
                              service that you want returned in the results, separated by
                              commas. To output all available outputFields, leave this
                              parameter blank.

                              Example: score,match_addr,x,y
    ---------------------     ----------------------------------------------------------------
    header_rows_to_skip       optional integer. Describes on which row your data begins in
                              your file or table. The default is 1 (since the first row
                              contains the headers). The default is 1.
    ---------------------     ----------------------------------------------------------------
    output_name               optional string, The task will create a feature service of the
                              results. You define the name of the service.
    ---------------------     ----------------------------------------------------------------
    category                  optional string. Enter a category for more precise geocoding
                              results, if applicable. Some geocoding services do not support
                              category, and the available options depend on your geocode service.
    ---------------------     ----------------------------------------------------------------
    context                   optional dictionary. Context contains additional settings that
                              affect task execution. Batch Geocode has the following two
                              settings:

                              1. Extent (extent) - A bounding box that defines the analysis
                                 area. Only those points in inputLayer that intersect the
                                 bounding box are analyzed.
                              2. Output Spatial Reference (outSR) - The output features are
                                 projected into the output spatial reference.

                              Syntax:
                              {
                              "extent" : {extent}
                              "outSR" : {spatial reference}
                              }
    ---------------------     ----------------------------------------------------------------
    gis                       optional ``GIS``, the :class:`~arcgis.gis.GIS` on which this
                              tool runs.

                              .. note::
                                If not specified, the active ``GIS`` is used.
    =====================     ================================================================

    .. code-block:: python

        # Usage Example
        >>> fl_item = geocode_from_items(csv_item, output_type='Feature Layer',
                             geocode_parameters={"field_info": ['Addresses', 'TEXT', 255],
                                                 "column_names": ["Addresses"],
                                                 "field_mapping": ['Addresses', 'Address']
                                                 },
                             output_name="address_file_matching",
                             gis=gis)
        >>> type(fl_item)
        <:class:`~arcgis.gis.Item`>

    :return:
        A :class:`~arcgis.gis.Item` object.
    """

    import json
    import uuid
    from arcgis.gis import Item
    from arcgis.geocoding import Geocoder
    from arcgis.features.layer import Layer
    from arcgis.geoprocessing._tool import Toolbox
    from arcgis.geoanalytics._util import _create_output_service

    uid = uuid.uuid4().hex[:5]
    locator_parameters = None

    kwargs = {
        "geocode_parameters": geocode_parameters,
        "geocode_service_url": geocode_service_url,
        "output_type": output_type,
        "input_table": "",
        "input_file_item": None,
        "source_country": country,
        "category": category,
        "output_fields": output_fields,
        "header_rows_to_skip": header_rows_to_skip,
        "output_name": output_name,
        "context": context,
        "locator_parameters": locator_parameters,
    }

    _item_type = "SERVICE"
    item = None
    if gis is None:
        gis = arcgis.env.active_gis
    url = gis.properties.helperServices.asyncGeocode.url
    tbx = Toolbox(url=url, gis=gis)

    if output_type is None:
        output_type = "CSV"
    if output_type.lower() == "xlsx":
        output_type = "XLS"
    if output_type.lower() == "feature layer":
        output_type = "Feature Service"
    if output_type not in ["CSV", "XLS", "Feature Service"]:
        raise ValueError("Invalid output_type: %s" % output_type)

    if geocode_service_url is None:
        gcs = gis.properties.helperServices.geocode
        for gc in gcs:
            if "batch" in gc and gc["batch"]:
                geocode_service_url = gc["url"]
                kwargs["geocode_service_url"] = gc["url"]
                break
            del gc
        del gcs
    elif isinstance(geocode_service_url, Geocoder):
        geocode_service_url = geocode_service_url.url
        kwargs["geocode_service_url"] = geocode_service_url.url
    elif isinstance(geocode_service_url, str) == False:
        raise ValueError("Invalid geocoder service given.")
    if geocode_service_url is None:
        raise ValueError(
            "The registered geocoders are not valid to use with this tool."
        )

    if (
        isinstance(input_data, Item)
        and input_data.type == "Feature Service"
        and input_data.tables
    ):
        lyr = input_data.tables[0]
        kwargs["input_table"] = {"url": lyr.url}
        if gis._con.token:
            kwargs["input_table"]["serviceToken"] = gis._con.token
        if geocode_parameters is None:
            kwargs["geocode_parameters"] = analyze_geocode_input(
                input_table_or_item=lyr,
                geocode_service_url=geocode_service_url,
                gis=gis,
            )

    elif isinstance(input_data, Item):
        _item_type = input_data.type
        kwargs["input_file_item"] = {"itemid": input_data.itemid}
    elif isinstance(input_data, dict):
        if "url" in input_data:
            _item_type = "SERVICE"
            kwargs["input_table"] = input_data
        elif "itemid" in input_data:
            item = gis.content.search("id: %s" % input_data["itemid"])[0]
            _item_type = item.type
            kwargs["input_file_item"] = input_data
        pass
    elif isinstance(input_data, str):
        item = gis.content.search("id: %s" % input_data)[0]
        _item_type = item.type
        kwargs["input_file_item"] = {"itemid": input_data}
    elif isinstance(input_data, Layer):
        lyr = input_data._lyr_dict
        if "type" in lyr:
            del lyr["type"]
        kwargs["input_table"] = lyr
    else:
        raise ValueError("Invalid input_data")

    # Figure out input_data
    if geocode_parameters is None and _item_type in [
        "CSV",
        "csv",
        "XLS",
        "xls",
        "XLSX",
        "xlsx",
    ]:
        if header_rows_to_skip is None:
            hre = "false"
        else:
            hre = "true"
        kwargs["geocode_parameters"] = analyze_geocode_input(
            input_table_or_item=kwargs["input_file_item"],
            geocode_service_url=geocode_service_url,
            gis=gis,
        )
    if output_type in ["Feature Layer", "Feature Service", "FeatureLayer"]:
        output_type = "Feature Service"
        kwargs["output_type"] = "Feature Service"
        if output_name is None:
            kwargs["output_name"] = {
                "serviceProperties": {"name": "Geocoded_Feature_Service_%s" % uid}
            }
        else:
            kwargs["output_name"] = {
                "serviceProperties": {
                    "name": "Geocoded_Feature_Service_%s"
                    % output_name.replace(" ", "_")
                }
            }
    elif output_type in ["XLS", "xls", "XLSX", "xlsx"]:
        if output_name:
            kwargs["output_name"] = {
                "itemProperties": {
                    "title": "Geocoded Results %s" % output_name,
                    "description": "Geocoded results for %s generated from running the Geocode Locations from Table solution."
                    % output_name,
                    "tags": "Analysis Result, Geocode Locations From Table",
                    "snippet": "Excel File generated from Geocode Locations From Table",
                    "folderId": "",
                }
            }
        else:
            output_name = "Geocoded_Result_ %s" % uid
            kwargs["output_name"] = {
                "itemProperties": {
                    "title": "Geocoded Results %s" % output_name,
                    "description": "Geocoded results for %s generated from running the Geocode Locations from Table solution."
                    % output_name,
                    "tags": "Analysis Result, Geocode Locations From Table",
                    "snippet": "Excel File generated from Geocode Locations From Table",
                    "folderId": "",
                }
            }
    elif output_type in ["CSV", "csv"]:
        if output_name:
            kwargs["output_name"] = {
                "itemProperties": {
                    "title": "Geocoded Results %s" % output_name,
                    "description": "Geocoded results for %s generated from running the Geocode Locations from Table solution."
                    % output_name,
                    "tags": "Analysis Result, Geocode Locations From Table",
                    "snippet": "CSV File generated from Geocode Locations From Table",
                    "folderId": "",
                }
            }
        else:
            output_name = "Geocoded_Result_%s" % uid
            kwargs["output_name"] = {
                "itemProperties": {
                    "title": "Geocoded Results %s" % output_name,
                    "description": "Geocoded results for %s generated from running the Geocode Locations from Table solution."
                    % output_name,
                    "tags": "Analysis Result, Geocode Locations From Table",
                    "snippet": "CSV File generated from Geocode Locations From Table",
                    "folderId": "",
                }
            }

    for k, v in list(kwargs.items()):
        if v is None:
            kwargs.pop(k)

    res = tbx.batch_geocode(**kwargs)
    if "itemId" in res:
        return gis.content.get(res["itemId"])
    elif hasattr(res, "geocode_result"):
        item_id = res.geocode_result.get("itemId", None)
        if item_id:
            return gis.content.get(item_id)
        else:
            res
    else:
        return res


def geocode(
    address: Union[list[str], dict[str, str]],
    search_extent: Optional[str] = None,
    location: Optional[Union[list, tuple]] = None,
    distance: Optional[int] = None,
    out_sr: Optional[dict[str, Any]] = None,
    category: Optional[str] = None,
    out_fields: str = "*",
    max_locations: int = 20,
    magic_key: Optional[str] = None,
    for_storage: bool = False,
    geocoder: Optional[Geocoder] = None,
    as_featureset: bool = False,
    match_out_of_range: bool = True,
    location_type: str = "street",
    lang_code: Optional[str] = None,
    source_country: Optional[str] = None,
):
    """
    The ``geocode`` function geocodes one location per request.

    ====================     ====================================================
    **Parameter**             **Description**
    --------------------     ----------------------------------------------------
    address                  Required list of strings or dictionaries.
                             Specifies the location to be geocoded. This can be
                             a string containing the street address, place name,
                             postal code, or POI.

                             Alternatively, this can be a dictionary containing
                             the various address fields accepted by the
                             corresponding geocoder. These fields are listed in
                             the addressFields property of the associated
                             geocoder. For example, if the address_fields of a
                             geocoder includes fields with the following names:
                             Street, City, State and Zone, then the address
                             argument is of the form:

                             {
                               Street: "1234 W Main St",
                               City: "Small Town",
                               State: "WA",
                               Zone: "99027"
                             }
    --------------------     ----------------------------------------------------
    search_extent            Optional string, A set of bounding box coordinates
                             that limit the search area to a specific region.
                             This is especially useful for applications in which
                             a user will search for places and addresses only
                             within the current map extent.
    --------------------     ----------------------------------------------------
    location                 Optional [x,y], Defines an origin point location that
                             is used with the distance parameter to sort
                             geocoding candidates based upon their proximity to
                             the location.
    --------------------     ----------------------------------------------------
    distance                 Optional float, Specifies the radius of an area
                             around a point location which is used to boost the
                             rank of geocoding candidates so that candidates
                             closest to the location are returned first. The
                             distance value is in meters.
    --------------------     ----------------------------------------------------
    out_sr                   Optional dictionary, The spatial reference of the
                             x/y coordinates returned by a geocode request. This
                             is useful for applications using a map with a spatial
                             reference different than that of the geocode service.
    --------------------     ----------------------------------------------------
    category                 Optional string, A place or address type which can
                             be used to filter find results. The parameter
                             supports input of single category values or multiple
                             comma-separated values. The category parameter can be
                             passed in a request with or without the text
                             parameter.
    --------------------     ----------------------------------------------------
    out_fields               Optional string, name of all the fields to include.
                             The default is "*" which means all fields.
    --------------------     ----------------------------------------------------
    max_location             Optional integer, The number of locations to be
                             returned from the service. The default is 20.
    --------------------     ----------------------------------------------------
    magic_key                Optional string. The find operation retrieves
                             results quicker when you pass a valid text and
                             `magic_key` value.
    --------------------     ----------------------------------------------------
    for_storage              Optional Boolean. Specifies whether the results of
                             the operation will
                             be persisted. The default value is false, which
                             indicates the results of the operation can't be
                             stored, but they can be temporarily displayed on a
                             map for instance.

                             .. note::
                                If you store the results, in a
                                database for example, you need to set this parameter
                                to ``True``.
    --------------------     ----------------------------------------------------
    geocoder                 Optional, the :class:`~arcgis.geocoding.Geocoder` to
                             be used.

                             .. note::
                                If not specified, the active
                                :class:`~arcgis.gis.GIS` object's
                                first geocoder is used.
    --------------------     ----------------------------------------------------
    as_featureset            Optional boolean, If ``True``, the result set is
                             returned as a :class:`~arcgis.features.FeatureSet`
                             object, else it is a dictionary.
    --------------------     ----------------------------------------------------
    match_out_of_range       Optional Boolean. Provides better spatial accuracy
                             for inexact street addresses by specifying whether
                             matches will be returned when the input number is
                             outside of the house range defined for the input
                             street. Out of range matches will be defined as
                             Addr_type=StreetAddressExt. Input house numbers
                             that exceed the range on a street segment by more
                             than 100 will not result in `StreetAddressExt`
                             matches. The default value of this parameter is
                             True.
    --------------------     ----------------------------------------------------
    location_type            Optional Str. Specifies whether the rooftop point or
                             street entrance is used as the output geometry of
                             PointAddress matches. By default, street is used,
                             which is useful in routing scenarios, as the rooftop
                             location of some addresses may be offset from a
                             street by a large distance. However, for map display
                             purposes, you may want to use rooftop instead,
                             especially when large buildings or landmarks are
                             geocoded. The `location_type` parameter only affects
                             the location object in the JSON response and does
                             not change the x,y or DisplayX/DisplayY attribute
                             values.

                             Values: `street` or `rooftop`
    --------------------     ----------------------------------------------------
    lang_code                Optional str. Sets the language in which geocode
                             results are returned.
    --------------------     ----------------------------------------------------
    source_country           Optional str. Limits the returned candidates to the
                             specified country or countries for either single-field
                             or multifield requests. Acceptable values include
                             the 3-character country code.
    ====================     ====================================================

    .. code-block:: python

        # Usage Example
        >>> geocoded = geocode(addresses = {
                                                    Street: "1234 W Main St",
                                                    City: "Small Town",
                                                    State: "WA",
                                                    Zone: "99027"
                                                    },
                                            distance = 1000,
                                            max_locations = 50,
                                            as_featureset = True,
                                            match_out_of_range = True,
                                            location_type = "Street"
                                            )
        >>> type(geocoded)
        <:class:`~arcgis.features.FeatureSet>

    :return:
       A dictionary or :class:`~arcgis.features.FeatureSet` object.

    """
    # as_featureset = False
    if geocoder is None:
        geocoder = arcgis.env.active_gis._tools.geocoders[0]
    return geocoder._geocode(
        address,
        search_extent,
        location,
        distance,
        out_sr,
        category,
        out_fields,
        max_locations,
        magic_key,
        for_storage,
        as_featureset,
        match_out_of_range=match_out_of_range,
        location_type=location_type,
        lang_code=lang_code,
        source_country=source_country,
    )


def reverse_geocode(
    location: Union[list, dict, Point],
    distance: Optional[float] = None,
    out_sr: Optional[Union[int, SpatialReference]] = None,
    lang_code: Optional[str] = None,
    return_intersection: bool = False,
    for_storage: bool = False,
    geocoder: Optional[Geocoder] = None,
    feature_types: Optional[str] = None,
    roof_top: str = "street",
):
    """
    The ``reverse_geocode`` operation determines the address at a particular
    x/y location. You pass the coordinates of a point location to the
    geocoding service, and the service returns the address that is
    closest to the location.

    =================== ====================================================
    **Parameter**        **Description**
    ------------------- ----------------------------------------------------
    location            Required location input as list, dict (with or without SpatialReference),
                        or :class:`~arcgis.geometry.Point` object.
    ------------------- ----------------------------------------------------
    distance            optional float, radial distance in meters to
                        search for an address.

                        .. note::
                            The default for ``distance`` is 100 meters.
    ------------------- ----------------------------------------------------
    out_sr              optional integer or
                        :class:`~arcgis.geometry.SpatialReference` of the
                        x/y coordinate returned.
    ------------------- ----------------------------------------------------
    lang_code           optional string. Sets the language in which geocode
                        results are returned. This is useful for ensuring
                        that results are returned in the expected language.
                        If the lang_code parameter isn't included in a
                        request, or if it is included but there are no
                        matching features with the input language code, the
                        resultant match is returned in the language code of
                        the primary matched components from the input search
                        string.
    ------------------- ----------------------------------------------------
    return_intersection optional Boolean, which specifies whether the
                        service should return the nearest street
                        intersection or the nearest address to the input
                        location
    ------------------- ----------------------------------------------------
    for_storage         optional boolean, specifies whether the results of
                        the operation will be persisted
    ------------------- ----------------------------------------------------
    geocoder            optional :class:`~arcgis.geocoding.Geocoder`,
                        the geocoder to be used.

                        .. note::
                            If not specified, the active ``GIS`` instances
                            first ``Geocoder`` is used.
    ------------------- ----------------------------------------------------
    feature_types       Optional String. Limits the possible match types
                        performed by the `reverse_geocode` method. If a
                        single value is included, the search tolerance for
                        the input feature type is 500 meters. If multiple
                        values (separated by a comma, with no spaces) are
                        included, the default search distances specified in
                        the feature type hierarchy table are applied.

                        Values: `StreetInt, DistanceMarker, StreetAddress,
                                StreetName, POI, PointAddress, Postal, and
                                Locality`
    ------------------- ----------------------------------------------------
    location_type       Optional string. Specifies whether the rooftop point
                        or street entrance is used as the output geometry of
                        point address matches. By default,
                        ``street`` is used,
                        which is useful in routing scenarios, as the rooftop
                        location of some addresses may be offset from a
                        street by a large distance. However, for map display
                        purposes, you may want to use ``rooftop`` instead,
                        especially when large buildings or landmarks are
                        geocoded. The ``location_type`` parameter only
                        affects the location object in the JSON response
                        and does not change the x,y or
                        ``DisplayX/DisplayY`` attribute values.

                        Values: ``street``, ``rooftop``
    =================== ====================================================

    .. code-block:: python

        # Usage Example
        >>> reversed = Geocoder.reverse_geocode(location = point1,
                                                distance = 50,
                                                for_storage = True,
                                                feature_types = "StreetName",
                                                location_type = "street")
        >>> type(reversed)
        <Dictionary>

    :return:
       A dictionary
    """

    if geocoder is None:
        geocoder = arcgis.env.active_gis._tools.geocoders[0]
        assert isinstance(geocoder, Geocoder)
    return geocoder._reverse_geocode(
        location=location,
        distance=distance,
        out_sr=out_sr,
        lang_code=lang_code,
        return_intersection=return_intersection,
        for_storage=for_storage,
        feature_types=feature_types,
        location_type=roof_top,
    )


def batch_geocode(
    addresses: Union[list[str], dict[str, str]],
    source_country: Optional[str] = None,
    category: Optional[str] = None,
    out_sr: Optional[dict] = None,
    geocoder: Optional[Geocoder] = None,
    as_featureset: bool = False,
    match_out_of_range: bool = True,
    location_type: str = "street",
    search_extent: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None,
    lang_code: str = "EN",
    preferred_label_values: Optional[str] = None,
    out_fields: Optional[str] = None,
):
    """
    The ``batch_geocode`` function geocodes an entire list of addresses.

    .. note::
        Geocoding many addresses at once is also known as bulk geocoding.

    =========================     ================================================================
    **Parameter**                  **Description**
    -------------------------     ----------------------------------------------------------------
    addresses                     Required list of strings or dictionaries.
                                  A list of addresses to be geocoded.
                                  For passing in the location name as a single line of text -
                                  single field batch geocoding - use a string.
                                  For passing in the location name as multiple lines of text
                                  multifield batch geocoding - use the address fields described
                                  in the Geocoder documentation.

                                  .. note::
                                    The maximum number of addresses that can be geocoded in a
                                    single request is limited to the SuggestedBatchSize property of
                                    the locator.

                                  Syntax:
                                  addresses = ["380 New York St, Redlands, CA",
                                    "1 World Way, Los Angeles, CA",
                                    "1200 Getty Center Drive, Los Angeles, CA",
                                    "5905 Wilshire Boulevard, Los Angeles, CA",
                                    "100 Universal City Plaza, Universal City, CA 91608",
                                    "4800 Oak Grove Dr, Pasadena, CA 91109"]

                                  OR

                                  addresses= [{
                                       "Address": "380 New York St.",
                                       "City": "Redlands",
                                       "Region": "CA",
                                       "Postal": "92373"
                                   },{
                                       "OBJECTID": 2,
                                       "Address": "1 World Way",
                                       "City": "Los Angeles",
                                       "Region": "CA",
                                       "Postal": "90045"
                                   }]
    -------------------------     ----------------------------------------------------------------
    source_country                Optional string, The ``source_country`` parameter is
                                  only supported by geocoders published using StreetMap
                                  Premium locators.

                                  .. note::
                                    Added at 10.3 and only supported by geocoders published
                                    with ArcGIS 10.3 for Server and later versions.

    -------------------------     ----------------------------------------------------------------
    category                      Optional String. The ``category`` parameter is only supported by geocode
                                  services published using StreetMap Premium locators.
    -------------------------     ----------------------------------------------------------------
    out_sr                        Optional dictionary, The spatial reference of the
                                  x/y coordinates returned by a geocode request. This
                                  is useful for applications using a map with a spatial
                                  reference different than that of the geocode service.
    -------------------------     ----------------------------------------------------------------
    as_featureset                 Optional boolean, if True, the result set is
                                  returned as a FeatureSet object, else it is a
                                  dictionary.
    -------------------------     ----------------------------------------------------------------
    geocoder                      Optional :class:`~arcgis.geocoding.Geocoder`,
                                  the geocoder to be used.

                                  .. note::
                                    If not specified, the active ``GIS`` instances
                                    first ``Geocoder`` is used.
    -------------------------     ----------------------------------------------------------------
    match_out_of_range            Optional, A Boolean which specifies if StreetAddress matches should
                                  be returned even when the input house number is outside of the house
                                  number range defined for the input street.
    -------------------------     ----------------------------------------------------------------
    location_type                 Optional, Specifies if the output geometry of PointAddress matches
                                  should be the rooftop point or street entrance location. Valid values
                                  are rooftop and street.
    -------------------------     ----------------------------------------------------------------
    search_extent                 Optional, a set of bounding box coordinates that limit the search
                                  area to a specific region. The input can either be a comma-separated
                                  list of coordinates defining the bounding box or a JSON envelope
                                  object.
    -------------------------     ----------------------------------------------------------------
    lang_code                     Optional, sets the language in which geocode results are returned.
                                  See the table of supported countries for valid language code values
                                  in each country.
    -------------------------     ----------------------------------------------------------------
    preferred_label_values        Optional, allows simple configuration of output fields returned
                                  in a response from the World Geocoding Service by specifying which
                                  address component values should be included in output fields. Supports
                                  a single value or a comma-delimited collection of values as input.
                                  e.g. ='matchedCity,primaryStreet'
    -------------------------     ----------------------------------------------------------------
    out_fields                    Optional String. A string of comma seperated fields names used to
                                  limit the return attributes of a geocoded location.
    =========================     ================================================================

    .. code-block:: python

        # Usage Example
        >>> batched = batch_geocode(addresses = ["380 New York St, Redlands, CA",
                                                            "1 World Way, Los Angeles, CA",
                                                            "1200 Getty Center Drive, Los Angeles, CA",
                                                            "5905 Wilshire Boulevard, Los Angeles, CA",
                                                            "100 Universal City Plaza, Universal City, CA 91608",
                                                            "4800 Oak Grove Dr, Pasadena, CA 91109"]
                                            as_featureset = True,
                                            match_out_of_range = True,
                                            )
        >>> type(batched)
        <:class:`~arcgis.features.FeatureSet>

    :return:
      A dictionary or :class:`~arcgis.features.FeatureSet`
    """
    if geocoder is None:
        geocoder = arcgis.env.active_gis._tools.geocoders[0]
    return geocoder._batch_geocode(
        addresses,
        source_country,
        category,
        out_sr,
        as_featureset,
        match_out_of_range,
        location_type,
        search_extent,
        lang_code,
        preferred_label_values,
        out_fields,
    )


def suggest(
    text: str,
    location: Optional[dict[str, Any]] = None,
    distance: Optional[float] = None,
    category: Optional[str] = None,
    geocoder: Optional[Geocoder] = None,
    search_extent: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None,
    max_suggestions: int = 5,
    country_code: Optional[str] = None,
):
    """
    The ``suggest`` method retrieves a resource representing a list of
    suggested matches for the input text. This resource provides the
    matching text as well as a unique ID value, which links a
    suggestion to a specific place or address.
    A geocoder must meet the following requirements to support
    the suggest operation:

    1. The address locator from which the geocoder was published
      must support suggestions.

    .. note::
        Only address locators created using
        ArcGIS 10.3 for Desktop and later can support suggestions. See
        the Create Address Locator geoprocessing tool help topic for more
        information.

    2. The geocoder must have the Suggest capability enabled.

    .. note::
            Only ``geocoders`` published using ArcGIS 10.3 for Server or
        later support the Suggest capability.

    The ``suggest`` operation allows character-by-character auto-complete
    suggestions to be generated for user input in a client application.
    This capability facilitates the interactive search user experience
    by reducing the number of characters that need to be typed before
    a suggested match is obtained. A client application can provide a
    list of suggestions that is updated with each character typed by a
    user until the address they are looking for appears in the list.

    ===============     =================================================================
    **Parameter**        **Description**
    ---------------     -----------------------------------------------------------------
    text                The input text provided by a user that is used by the
                        suggest operation to generate a list of possible
                        matches. This is a required parameter.
    ---------------     -----------------------------------------------------------------
    location            Optional x/y dictionary. Defines an origin point location that is used
                        with the distance parameter to sort suggested candidates
                        based on their proximity to the location. The
                        distance parameter specifies the radial distance from
                        the location in meters. The priority of candidates
                        within this radius is boosted relative to those
                        outside the radius.
                        This is useful in mobile applications where a user
                        wants to search for places in the vicinity of their
                        current GPS location. It is also useful for web
                        mapping applications where a user wants to find
                        places within or near the map extent.

                        .. note::
                            The ``location`` parameter can be specified without
                            specifying a ``distance``. If distance is not specified,
                            it defaults to 2000 meters.

    ---------------     -----------------------------------------------------------------
    distance            Optional float. Specifies the radius around the point defined in the
                        location parameter to create an area, which is used to boost
                        the rank of suggested candidates so that candidates closest to
                        the location are returned first. The distance value is in
                        meters.

                        .. note::
                            If the ``distance`` parameter is specified, the ``location``
                            parameter must be specified as well.

                        It is important to note that the ``location`` and ``distance``
                        parameters allow searches to extend beyond the specified search
                        radius. They are not used to filter results, but rather to rank
                        resulting candidates based on their distance from a location.
    ---------------     -----------------------------------------------------------------
    category            The category parameter is only supported by geocode
                        services published using StreetMap Premium locators.
    ---------------     -----------------------------------------------------------------
    geocoder            Optional :class:`~arcgis.geocoding.Geocoder` - the geocoder to
                        be used. If not specified, the active
                        :class:`~arcgis.gis.GIS` object's first geocoder is used.
    ---------------     -----------------------------------------------------------------
    search_extent       Optional String/Dict. A set of bounding box coordinates that
                        limit the search area to a specific region. You can specify the
                        spatial reference of the `search_extent` coordinates, which is
                        necessary if the map spatial reference is different than that of
                        the geocoding service; otherwise, the spatial reference of the
                        map coordinates is assumed to be the same as that of the
                        geocoding service. The input can either be a comma-separated list
                        of coordinates defining the bounding box or a JSON envelope
                        object.

                        .. note::
                            The ``search_extent`` coordinates should always use a
                            period as the decimal separator, even in countries where
                            traditionally a comma is used.
    ---------------     -----------------------------------------------------------------
    max_suggestions     Optional Int.  The maximum number of suggestions returned by the
                        suggest operation, up to the maximum number allowed by the
                        service.

                        .. note::
                            If ``maxSuggestions`` is not included in the suggest
                            request, the default value is 5. The maximum suggestions value
                            can be modified in the source address locator.
    ---------------     -----------------------------------------------------------------
    country_code        Optional Str. Limits the returned suggestions to values in a
                        particular country. Valid two- and three-character country code
                        values for each country are available in geocode coverage.

                        .. note::
                            When the ``country_code`` parameter is specified in a
                            suggest request, the corresponding ``geocode`` call must
                            also include the ``country_code`` parameter with the
                            same value.
    ===============     =================================================================

    .. code-block:: python

        # Usage Example
        >>> suggested = suggest(text = "geocoding_text"
                                        location = point1,
                                        distance = 5000,
                                        max_suggestions = 10
                                        )
        >>> type(suggested)
        <Dictionary>

    :return:
        A dictionary
    """
    if geocoder is None:
        geocoder = arcgis.env.active_gis._tools.geocoders[0]
    return geocoder._suggest(
        text,
        location,
        distance,
        category,
        search_extent=search_extent,
        max_suggestions=max_suggestions,
        country_code=country_code,
    )
