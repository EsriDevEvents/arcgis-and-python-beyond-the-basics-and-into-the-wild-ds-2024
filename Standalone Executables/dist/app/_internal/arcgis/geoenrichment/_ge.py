import json
import os
from typing import Optional, Union
import pandas as pd
import arcgis
from arcgis.gis import GIS
from arcgis.features import FeatureSet
from arcgis.geometry import Envelope

try:
    from arcgis.features.geo import _is_geoenabled
except:

    def _is_geoenabled(o):
        return False


###########################################################################
class _GeoEnrichment(object):
    """
    The GeoEnrichment class provides the ability to get facts about a location or area. Using
    GeoEnrichment, you can get information about the people, places, and businesses in a specific
    area or within a certain distance or drive time from a location. More specifically, by
    submitting a point or polygon to the GeoEnrichment class, you can retrieve the demographics and
    other relevant characteristics associated with the surrounding area. You can also use the
    GeoEnrichment class to obtain additional geographic context (for example, the ZIP Code of a
    location) and geographic boundaries (for example, the geometry for a drive-time service area).

    This service enables you to answer questions about locations that you can't answer with maps
    alone. For example: What kind of people live here? What do people like to do in this area? What
    are their habits and lifestyles? What kind of businesses are in this area?

    Site analysis is a popular application of this type of data enrichment. For example, the
    GeoEnrichment class can be leveraged to study the population that would be affected by the
    development of a new community center within their neighborhood. With the service, the proposed
    site can be submitted, and the demographics and other relevant characteristics associated with
    the area around the site will be returned.

    Study areas
    ===========

    The GeoEnrichment class uses the concept of a study area to define the location of the point or
    area that you want to enrich with additional information. If one or more points is input as a
    study area, the service will create a one-mile ring buffer around the points or points to
    collect and append enrichment data. You can optionally change the ring buffer size or create
    drive-time service areas around a point.

    The service is capable of enriching study areas in the following ways:

    - Input XY locations-One or more input points (latitude and longitude) can be provided to the
      service to set the study areas that you want to enrich with additional information. You can
      create a buffer ring or drive-time service area around the points to aggregate data for the
      study areas.
    - Input polygons-You can enrich a single area defined by a polygon feature.
    - Named statistical areas-You can enrich areas defined by a named administrative boundary area
      and include the associated geometry in the response. Rather than specifying the polygon
      feature, identifiers (IDs) are used to specify named statistical areas such as states,
      provinces, counties, postal codes, and the like. Named administrative boundary areas can be
      looked up with the StandardGeographyQuery administrative boundary lookup service.
    - Network service areas-You can create drive time service areas around points as well as other
      advanced service areas such as walking and trucking.
    - Street address locations-You can also enrich input addresses. Enter an address, point of
      interest, place name, or other supported location as a single string ("380 New York St,
      Redlands, CA"), and the service returns a coordinate pair representing the address
      ("-117.1956, 34.0576"). Once a matched address is returned, you can create a buffer ring or
      drive-time service area around the point to aggregate data for the study area.

    Data collections
    ================
    The GeoEnrichment class uses the concept of a data collection to define the data attributes
    returned by the enrichment service. More specifically, a data collection is a preassembled list
    of attributes that will be used to enrich the input features. Collection attributes can describe
    various types of information, such as demographic characteristics and geographic context, of the
    locations or areas submitted as input features.

    Service properties
    ==================
    Discover what capabilities are available in service properties as well as the service limits.
    For example, these service properties would let you know what services endpoints and child
    resources are available. It will also allow the Business Analyst Web app or Business Analyst
    Desktop to discover if custom reports, stored as portal items, are available in the service.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    Required GIS object.  The ArcGIS Online GIS object.
    ------------------     --------------------------------------------------------------------
    product                Optional string.  The value of
    ------------------     --------------------------------------------------------------------
    language_code          Optional string.  The desired returned language.
    ==================     ====================================================================
    """

    _appID = None
    _langCode = "en-us"
    _countries = None
    _countries_dict = None
    _gis = None
    _portal = None
    _base_url = None
    _limits = None
    _url_standard_geography_query = "/StandardGeographyLevels"
    _url_standard_geography_query_execute = "/StandardGeographyQuery/execute"
    _url_getVariables = "/GetVariables/execute"
    _url_create_report = "/GeoEnrichment/createReport"
    _url_list_reports = "/Geoenrichment/Reports"
    _url_enrich_data = "/Geoenrichment/Enrich"
    _url_data_collection = "/Geoenrichment/dataCollections"

    # ----------------------------------------------------------------------
    def __init__(
        self,
        gis: GIS,
        url: Optional[str] = None,
        product: str = "bao",
        language_code: Optional[str] = None,
    ):
        """initializer"""
        # if gis._portal.is_logged_in == False:
        #     raise Exception('User must be logged in to use the '+ \
        #                     'GeoEnrichment API')
        gis = arcgis.env.active_gis if gis is None else gis
        self._gis = gis
        self._portal = gis._portal
        if url is None:
            hs = dict(self._gis.properties["helperServices"])
            if "geoenrichment" in hs:
                self._base_url = hs["geoenrichment"]["url"]
            else:
                self._base_url = "http://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver"
            if self._gis._is_hosted_nb_home:
                self._base_url = self._validate_url(self._base_url)
        else:
            self._base_url = url
        self._appID = "esripythonapi"
        if language_code is None:
            self._langCode = language_code

    # ----------------------------------------------------------------------
    def _validate_url(self, url):
        res = self._gis._private_service_url(url)
        if "privateServiceUrl" in res:
            return res["privateServiceUrl"]
        else:
            return res["serviceUrl"]
        return url

    # ----------------------------------------------------------------------
    def _explode(self, df, lst_cols, fill_value=""):
        """internal method to help flatten out data sources"""
        import numpy as np

        # make sure `lst_cols` is a list
        if lst_cols and not isinstance(lst_cols, list):
            lst_cols = [lst_cols]
        # all columns except `lst_cols`
        idx_cols = df.columns.difference(lst_cols)

        # calculate lengths of lists
        lens = df[lst_cols[0]].str.len()

        if (lens > 0).all():
            # ALL lists in cells aren't empty
            return (
                pd.DataFrame(
                    {
                        col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                        for col in idx_cols
                    }
                )
                .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
                .loc[:, df.columns]
            )
        else:
            # at least one list in cells is empty
            return (
                pd.DataFrame(
                    {
                        col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                        for col in idx_cols
                    }
                )
                .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
                .append(df.loc[lens == 0, idx_cols])
                .fillna(fill_value)
                .loc[:, df.columns]
            )

    # ----------------------------------------------------------------------
    def countries(self, as_df: bool = False):
        """
        returns a list or Pandas' DataFrame of available countries that have GeoEnrichment data.
        """
        import pandas as pd

        if self._countries_dict is None:
            params = {"f": "json"}
            url = self._base_url + "/Geoenrichment/Countries"
            if self._gis._con.token:
                params["token"] = self._gis._con.token
            res = self._gis._con.post(url, params)
            self._countries_dict = res["countries"]
        if as_df:
            return pd.DataFrame(self._countries_dict)
        else:
            return self._countries_dict

    # ----------------------------------------------------------------------
    def country_info(self, country: str, as_dict=False):
        """
        Returns report information for a desired country using the country code.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                Required string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country. This can be the two letter country code or the coutries
                               full name.
        ------------------     --------------------------------------------------------------------
        as_dict                Optional bool. If True then will return as a dictionary. Default is
                               False and a pandas DataFrame will be returned.
        ==================     ====================================================================

        :return: Pandas' DataFrame or Dictionary (as_dict == True)
        """
        params = {"f": "json"}
        if self._gis._con.token:
            params["token"] = self._gis._con.token
        countries = self.countries()
        if len(country) > 2:
            q = self.countries()["Full_Name"].str.upper() == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Name: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        else:
            q = self.countries()["Country_Code"] == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Code: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        url = "%s/Geoenrichment/Countries/%s" % (self._base_url, country)
        res = self._gis._con.post(url, params)
        if as_dict == True:
            return res
        else:
            try:
                return pd.DataFrame.from_dict(res["countries"])
            except:
                return None
        return

    # ----------------------------------------------------------------------
    def report_metadata(self, country: str):
        """
        This method returns information about a given country's available reports and provides
        detailed metadata about each report.

        :Usage:
        >>> ge = gis.enrichment
        >>> df = ge.report_metadata("al")
        # returns basic report metadata for Albania

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                Required string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country. This can be the two letter country code or the coutries
                               full name.
        ==================     ====================================================================

        :return: Pandas' DataFrame
        """
        import pandas as pd

        params = {"f": "json"}
        if self._gis._con.token:
            params["token"] = self._gis._con.token
        url = self._base_url + "/Geoenrichment/Reports/%s" % country
        res = self._gis._con.post(url, params)
        meta = []
        cols = []
        for r in res["reports"]:
            if len(cols) == 0:
                cols = ["reportID"]
                for k in r["metadata"].keys():
                    cols.append(k)
            row = {"reportID": r["reportID"]}
            for k, v in r["metadata"].items():
                row[k] = v
            meta.append(row)
        return pd.DataFrame(meta)

    # ----------------------------------------------------------------------
    def report_info(self, country: str, report_id: str, as_dict: bool = False):
        """
        Returns a detailed description of a given report for a given country
        """
        countries = self.countries()
        if len(country) > 2:
            q = self.countries()["Full_Name"].str.upper() == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Name: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        else:
            q = self.countries()["Country_Code"] == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Code: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        params = {"f": "json"}
        url = self._base_url + "/Geoenrichment/Reports/%s/%s" % (
            country,
            report_id,
        )
        if self._gis._con.token:
            params["token"] = self._gis._con.token
        res = self._gis._con.post(url, params)
        if as_dict == True:
            return res
        elif "reports" in res:
            return pd.DataFrame.from_dict(res["reports"])
        return res

    # ----------------------------------------------------------------------
    def data_collections(
        self,
        country: Optional[str] = None,
        collection_name: Optional[str] = None,
        variables: Optional[Union[str, list]] = None,
        out_fields: str = "*",
        hide_nulls: bool = True,
        as_dict: bool = False,
    ):
        """
        The GeoEnrichment class uses the concept of a data collection to define the data
        attributes returned by the enrichment service. Each data collection has a unique name
        that acts as an ID that is passed in the data_collections parameter of the GeoEnrichment
        service.

        Some data collections (such as default) can be used in all supported countries. Other data
        collections may only be available in one or a collection of countries. Data collections may
        only be available in a subset of countries because of differences in the demographic data
        that is available for each country. A list of data collections for all available countries
        can be generated with the data collection discover method seen below.
        Return a list of data collections that can be run for any country.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                optional string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country.
        ------------------     --------------------------------------------------------------------
        collection_name        Optional string. Name of the data collection to examine.
        ------------------     --------------------------------------------------------------------
        variables              Optional string/list. This parameter to specifies a list of field
                               names that include variables for the derivative statistics.
        ------------------     --------------------------------------------------------------------
        out_fields             Optional string. This parameter is a string of comma seperate field
                               names.
        ------------------     --------------------------------------------------------------------
        hide_nulls             Optional boolean. parameter to return only values that are not NULL
                               in the output response. Adding the optional suppress_nulls parameter
                               to any data collections discovery method will reduce the size of the
                               output that is returned.
        ------------------     --------------------------------------------------------------------
        as_dict                Optional bool. If True then will return as a dictionary. Default is
                               False and a pandas DataFrame will be returned.
        ==================     ====================================================================

        :return: A Pandas DataFrame unless as_dict=True then a dictionary, describing the requested return data.
        """
        import pandas as pd

        params = {"f": "json", "langCode": self._langCode}
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        if variables is not None:
            params["addDerivativeVariables"] = variables
        if out_fields is not None:
            params["outFields"] = out_fields
        if hide_nulls is not None:
            params["suppressNullValues"] = hide_nulls
        if country is not None:
            url = "%s%s/%s" % (
                self._base_url,
                self._url_data_collection,
                country,
            )
            if collection_name is not None:
                url = "%s%s/%s/%s" % (
                    self._base_url,
                    self._url_data_collection,
                    country,
                    collection_name,
                )
        else:
            url = "%s%s" % (self._base_url, self._url_data_collection)
        res = self._gis._con.get(path=url, params=params)
        if as_dict == True:
            return res
        dfs = []
        for dc in res["DataCollections"]:
            dfs.append(pd.DataFrame(dc["data"]))
            del dc
        if len(dfs) > 1:
            df = pd.concat(dfs)
            df.reset_index(inplace=True, drop=True)
            return df
        else:
            return dfs[0]
        return res

    # ----------------------------------------------------------------------
    def find_report(self, country: str, as_df: bool = False):
        """
        Returns a list of reports by a country code

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                Optional string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country. This should be a two country code name.
                               Example: United States as US
        ------------------     --------------------------------------------------------------------
        as_dict                Optional bool. If True then will return as a dictionary. Default is
                               False and a pandas DataFrame will be returned.
        ==================     ====================================================================

        :return: Panda's DataFrame
        """
        import pandas as pd

        url = self._base_url + self._url_list_reports + "/%s" % country
        params = {
            "f": "json",
        }
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        res = self._gis._con.post(path=url, postdata=params)
        if "reports" in res:
            if as_df:
                return pd.DataFrame(res["reports"])
            else:
                return res["reports"]
        return res

    # ----------------------------------------------------------------------
    def enrich(
        self,
        study_areas: Union[list, dict],
        data_collections: Optional[list] = None,
        analysis_variables: Optional[list] = None,
        add_derivative_variables: Optional[list] = None,
        options: Optional[dict] = None,
        use_data: Optional[dict] = None,
        intersecting_geographies: Optional[dict] = None,
        return_geometry: bool = True,
        in_sr: Optional[int] = None,
        out_sr: Optional[int] = None,
        suppress_nulls: bool = False,
        for_storage: bool = True,
        as_featureset: bool = False,
    ):
        """
        The GeoEnrichment class uses the concept of a study area to
        define the location of the point or area that you want to enrich
        with additional information. If one or many points are input as
        a study area, the service will create a 1-mile ring buffer around
        the point to collect and append enrichment data. You can optionally
        change the ring buffer size or create drive-time service areas
        around the point. The most common method to determine the center
        point for a study areas is a set of one or many point locations
        defined as XY locations. More specifically, one or many input
        points (latitude and longitude) can be provided to the service to
        set the study areas that you want to enrich with additional
        information. You can create a buffer ring or drive-time service
        area around the points to aggregate data for the study areas. You
        can also return enrichment data for buffers around input line
        features.

        =========================     ====================================================================
        **Parameter**                  **Description**
        -------------------------     --------------------------------------------------------------------
        study_areas                   Required list/dictionary. This parameter is used to specify a list
                                      of input features to be enriched. Study areas can be input XY point
                                      locations.
        -------------------------     --------------------------------------------------------------------
        data_collections              Optional list. A Data Collection is a preassembled list of
                                      attributes that will be used to enrich the input features.
                                      Enrichment attributes can describe various types of information such
                                      as demographic characteristics and geographic context of the
                                      locations or areas submitted as input features in study_areas.
        -------------------------     --------------------------------------------------------------------
        analysis_variables            Optional list. A Data Collection is a preassembled list of
                                      attributes that will be used to enrich the input features. With the
                                      analysis_variables parameter you can return a subset of variables
                                      enrichment attributes can describe various types of information such
                                      as demographic characteristics and geographic context of the
                                      locations or areas submitted as input features in study_areas.
        -------------------------     --------------------------------------------------------------------
        add_derivative_variables      Optional list. This parameter is used to specify an array of string
                                      values that describe what derivative variables to include in the
                                      output.
        -------------------------     --------------------------------------------------------------------
        options                       Optional dictionary. This parameter is used to specify enrichment
                                      behavior. For points described as map coordinates, a 1-mile ring
                                      area centered on each site will be used by default. You can use this
                                      parameter to change these default settings.
                                      With this parameter, the caller can override the default behavior
                                      describing how the enrichment attributes are appended to the input
                                      features described in study_areas. For example, you can change the
                                      output ring buffer to 5 miles, change the number of output buffers
                                      created around each point, and also change the output buffer type to
                                      a drive-time service area rather than a simple ring buffer.
        -------------------------     --------------------------------------------------------------------
        use_data                      Optional dictionary. The parameter is used to explicitly specify the
                                      country or dataset to query.
        -------------------------     --------------------------------------------------------------------
        intersecting_geographies      Optional parameter to explicitly define the geographic layers used
                                      to provide geographic context during the enrichment process. For
                                      example, you can use this optional parameter to return the U.S.
                                      county and ZIP Code that each input study area intersects.
                                      You can intersect input features defined in the study_areas
                                      parameter with standard geography layers that are provided by the
                                      GeoEnrichment class for each country. You can also intersect
                                      features from a publicly available feature service.
        -------------------------     --------------------------------------------------------------------
        return_geometry               Optional boolean. A parameter to request the output geometries in
                                      the response.
        -------------------------     --------------------------------------------------------------------
        in_sr                         Optional integer. A parameter used to define the input geometries in
                                      the study_areas parameter in a specified spatial reference system.
        -------------------------     --------------------------------------------------------------------
        out_sr                        Optional integer. A parameter to request the output geometries in a
                                      specified spatial reference system.
        -------------------------     --------------------------------------------------------------------
        suppress_nulls                Optional boolean. A parameter to return only values that are not
                                      NULL in the output response. Adding the optional suppress_nulls
                                      parameter to any data collections discovery method will reduce the
                                      size of the output that is returned.
        -------------------------     --------------------------------------------------------------------
        for_storage                   Optional boolean. A parameter to define if GeoEnrichment output is
                                      being stored. The price for using the Enrich method varies according
                                      to whether the data returned is being persisted, i.e. being stored,
                                      or whether it is merely being used in an interactive context and is
                                      discarded after being viewed. If the data is being stored, the terms
                                      of use for the GeoEnrichment class require that you specify the
                                      for_storage parameter to true.
        -------------------------     --------------------------------------------------------------------
        as_featureset                 Optional boolean.  The default is False. If True, the result will be
                                      a liar of :class:`~arcgis.features.FeatureSet` object instead of a
                                      SpatailDataFrame or Pandas' DataFrame.
        =========================     ====================================================================

        :return: Spatially Enabled DataFrame, Panda's DataFrame, or a dictionary (on error)
        """
        if _is_geoenabled(study_areas):
            study_areas = [{"FeatureSet": study_areas.spatial.__feature_set__}]
        elif isinstance(study_areas, FeatureSet):
            study_areas = [{"FeatureSet": study_areas.sdf.spatial.__feature_set__}]
        params = {
            "langCode": self._langCode,
            "f": "json",
            "suppressNullValues": suppress_nulls,
            "studyareas": study_areas,
            "forStorage": for_storage,
            "appID": self._appID,
        }
        params["returnGeometry"] = return_geometry
        if options is not None:
            params["studyAreasOptions"] = options
        if data_collections is not None:
            params["dataCollections"] = data_collections
        if add_derivative_variables is not None:
            params["addDerivativeVariables"] = add_derivative_variables
        if out_sr is not None:
            params["outSR"] = out_sr
        if in_sr is not None:
            params["inSR"] = in_sr

        if use_data is not None:
            params["useData"] = use_data
        if intersecting_geographies is not None:
            params["intersectingGeographies"] = intersecting_geographies
        if analysis_variables is not None:
            params["analysisVariables"] = analysis_variables
        url = "%s%s" % (self._base_url, self._url_enrich_data)
        if self._gis._portal.is_arcgisonline and self._gis._con._session.auth.token:
            params["token"] = self._gis._con._session.auth.token
        res = self._gis._con.post(path=url, postdata=params)
        if as_featureset == False:
            import pandas as pd

            dfs = []
            if "results" in res:
                for result in res["results"]:
                    if "value" in result and "FeatureSet" in result["value"]:
                        for f in result["value"]["FeatureSet"]:
                            dfs.append(FeatureSet.from_dict(f).sdf)
                            del f
                    del result
            if len(dfs) > 1:
                df = pd.concat(dfs)
                df.reset_index(inplace=True, drop=True)
                return df
            elif len(dfs) == 1:
                return dfs[0]
            else:
                return res
        else:
            dfs = []
            if "results" in res:
                for result in res["results"]:
                    if "value" in result and "FeatureSet" in result["value"]:
                        for f in result["value"]["FeatureSet"]:
                            dfs.append(FeatureSet.from_dict(f))
                            del f
                    del result
                return dfs
            else:
                return res

    # ----------------------------------------------------------------------
    def get_variables(
        self,
        country: str,
        dataset: Optional[Union[str, list]] = None,
        text: Optional[str] = None,
        as_dict: bool = False,
    ):
        """
        The GeoEnrichment get_variables method allows you to search the data
        collections for variables that contain specific keywords.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        country                    Optional string. Specifies the source country for the search. Use
                                   this parameter to limit the search and query of standard geographic
                                   features to one country. This parameter supports both the
                                   two-digit and three-digit country codes illustrated in the
                                   coverage table.

                                   Example 1 - Set source country to the United States:
                                   country=US

                                   Example 2 - Set source country to the Canada:
                                   country=CA

                                   Additional notes
                                   Currently, the service is available for Canada, the United States
                                   and a number of European countries. Other countries will be added
                                   in the near future.
        ----------------------     --------------------------------------------------------------------
        dataset                    Optional string/list. Optional parameter to specify a specific
                                   dataset within a defined country. This parameter will not be used
                                   in the Beta release. In the future, some countries may have two or
                                   more datasets that may have different vintages and standard
                                   geography areas. For example, in the United States, there may be
                                   an optional dataset with historic census data from previous years.
                                   Examples
                                   dataset=USA_ESRI_2013
        ----------------------     --------------------------------------------------------------------
        text                       Optional string. Use this parameter to specify the text to query and
                                   search the data collections for the country and datasets specified.
                                   You can use this parameter to query and find specific keywords that
                                   are contained in a data collection.
        ----------------------     --------------------------------------------------------------------
        as_dict                    Optional boolean. If true, the result is returned as a python
                                   dictionary, else it's returned as a Panda's DataFrame
        ======================     ====================================================================

        returns: Pandas' DataFrame
        """
        import pandas as pd

        url = "%s%s" % (self._base_url, self._url_getVariables)
        params = {
            "f": "json",
            "langCode": self._langCode,
            "sourceCountry": country,
        }
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        if not text is None:
            params["searchText"] = text
        if not dataset is None:
            params["optionalCountryDataset"] = dataset
        if self._gis._portal.is_arcgisonline and self._gis._con._session.auth.token:
            params["token"] = self._gis._con._session.auth.token
        res = self._gis._con.post(path=url, postdata=params)
        if as_dict == True:
            return res
        dfs = []
        if "results" in res:
            for result in res["results"]:
                if "value" in result:
                    dfs.append(pd.DataFrame.from_dict(data=result["value"]))
                del result
        df = pd.concat(dfs)
        df.reset_index(inplace=True, drop=True)
        return df

    # ----------------------------------------------------------------------
    def select_businesses(
        self,
        type_filters: Optional[list] = None,
        feature_limit: int = 1000,
        feature_offset: int = 0,
        exact_match: bool = False,
        search_string: Optional[str] = None,
        spatial_filter: Optional[dict] = None,
        simple_search: bool = False,
        dataset_id: Optional[str] = None,
        full_error_message: bool = False,
        out_sr: int = 4326,
        return_geometry: bool = False,
        as_featureset: bool = False,
    ):
        """
        The select_businesses method returns business points matching a given search criteria.
        Business points can be selected using any combination of three search criteria: search
        string, spatial filter and business type. A business point will be selected if it matches
        all search criteria specified.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        type_filters               Optional list. List of business type filters restricting the search.
                                   For USA, either the NAICS or SIC filter is useful as a business type
                                   filter. If both filters are specified in the type_filters parameter
                                   value, selected business points will match both of them.
        ----------------------     --------------------------------------------------------------------
        feature_limit              Optional integer. The limit of returned business points.
        ----------------------     --------------------------------------------------------------------
        feature_offset             Optional integer. Start the results on the number of the record
                                   specified.
        ----------------------     --------------------------------------------------------------------
        exact_match                Optional boolean. True value of the parameter means the exact match
                                   of the string to search.
        ----------------------     --------------------------------------------------------------------
        search_string              Optional string. A string of characters which is used in the search
                                   query.
        ----------------------     --------------------------------------------------------------------
        spatial_filter             Optional SpatialFilter. A spatial filter restricting the search.
        ----------------------     --------------------------------------------------------------------
        simple_search              Optional boolean. A spatial filter restricting the search. True
                                   value of the parameter means a simple search (e.g., in company
                                   names only).
        ----------------------     --------------------------------------------------------------------
        dataset_id                 Optional string. ID of the active dataset.
        ----------------------     --------------------------------------------------------------------
        full_error_message         Optional boolean. Parameter for composing error message.
        ----------------------     --------------------------------------------------------------------
        out_sr                     Optional integer. Parameter specifying the spatial reference to
                                   return the output dataframe.
        ----------------------     --------------------------------------------------------------------
        return_geometry            Optional boolean. When true, geometries are returned with the
                                   response.
        ----------------------     --------------------------------------------------------------------
        as_featureset              Optional boolean. If False (default) the return type is a Spatail
                                   DataFrame, else it is a :class:`~arcgis.features.FeatureSet`
        ======================     ====================================================================

        returns: DataFrame (Spatial or Pandas) or dictionary on error.
        """
        url = self._base_url + "/SelectBusinesses/execute"
        params = {"f": "json", "langCode": self._langCode, "appID": self._appID}
        if return_geometry is not None:
            params["returnGeometry"] = return_geometry
        if out_sr is not None:
            params["outputSpatialReference"] = out_sr
        if full_error_message is not None:
            params["isFullErrorMessage"] = full_error_message
        if dataset_id is not None:
            params["activeDatasetID"] = dataset_id
        if simple_search is not None:
            params["useSimpleSearch"] = simple_search
        if spatial_filter is not None:
            params["spatialFilter"] = spatial_filter
        if search_string is not None:
            params["searchString"] = search_string
        if exact_match is not None:
            params["matchExactly"] = exact_match
        if feature_offset is not None:
            params["featureOffset"] = feature_offset
        if feature_limit is not None:
            params["featureLimit"] = feature_limit
        if type_filters is not None:
            params["businessTypeFilters"] = type_filters
        if self._gis._portal.is_arcgisonline and self._gis._con._session.auth.token:
            params["token"] = self._gis._con._session.auth.token
        res = self._gis._con.post(path=url, postdata=params)
        dfs = []
        if as_featureset == False:
            import pandas as pd

            if "results" in res:
                for result in res["results"]:
                    if "value" in result:
                        dfs.append(FeatureSet.from_dict(result["value"]).sdf)
                    del result
            if len(dfs) > 1:
                df = pd.concat(dfs)
                df.reset_index(inplace=True, drop=True)
                return df
            elif len(dfs) == 1:
                return dfs[0]
            else:
                return res
        else:
            if "results" in res:
                for result in res["results"]:
                    if "value" in result:
                        dfs.append(FeatureSet.from_dict(result["value"]))
                    del result
            if len(dfs) == 1:
                return dfs[0]
            if len(dfs) > 1:
                return dfs
            else:
                return res

    # ----------------------------------------------------------------------
    def create_report(
        self,
        study_areas: list,
        report: Optional[str] = None,
        export_format: str = "pdf",
        report_fields: Optional[str] = None,
        options: Optional[dict] = None,
        return_type: Optional[dict] = None,
        use_data: Optional[dict] = None,
        in_sr: int = 4326,
        out_folder: Optional[str] = None,
        out_name: Optional[str] = None,
    ):
        """
        The Create Report method allows you to create many types of high quality reports for a
        variety of use cases describing the input area. If a point is used as a study area, the
        service will create a 1-mile ring buffer around the point to collect and append enrichment
        data. Optionally, you can create a buffer ring or drive-time service area around points of
        interest to generate PDF or Excel reports containing relevant information for the area on
        demographics, consumer spending, tapestry market, business or market potential.

        Report options are available and can be used to describe and gain a better understanding
        about the market, customers / clients and competition associated with an area of interest.


        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        study_areas            Required list. Required parameter: Study areas may be defined by
                               input points, polygons, administrative boundaries or addresses.
        ------------------     --------------------------------------------------------------------
        report                 Optional string. identify the id of the report. This may be one of
                               the many default reports available along with our demographic data
                               collections or a customized report. Custom report templates are
                               stored in an ArcGIS Online organization as a Report Template item.
                               The organization URL and a valid ArcGIS Online authentication token
                               is required for security purposes to access these templates. If no
                               report is specified, the default report is census profile for United
                               States and a general demographic summary report for most countries.
        ------------------     --------------------------------------------------------------------
        export_format          Optional parameter to specify the format of the generated report.
                               Supported formats include PDF and XLSX.
        ------------------     --------------------------------------------------------------------
        report_fields          Optional parameter specifies additional choices to customize
                               reports. Below is an example of the position on the report header
                               for each field.
        ------------------     --------------------------------------------------------------------
        options                Optional parameter to specify the properties for the study area
                               buffer. For a full list of valid buffer properties values and
                               further examples review the Input XY Locations' options parameter.

                               By default a 1 mile radius buffer will be applied to point(s) and
                               address locations to define a study area.
        ------------------     --------------------------------------------------------------------
        return_type            Optional parameter used for storing an output report item to Portal
                               for ArcGIS instead of returning a report to a customer via binary
                               stream. The attributes are used by Portal to determine where and how
                               an item is stored. Parameter attributes include: user, folder,
                               title, item_properties, URL, token, and referrer.
                               Example

                               Creating a new output in a Portal for ArcGIS Instance:

                               return_type = {'user' : 'testUser',
                                              'folder' : 'FolderName',
                                              'title' : 'Report Title',
                                              'item_properties' : '<properties>',
                                              'url' : 'https://hostname.domain.com/webadaptor',
                                              'token' : 'token', 'referrer' : 'referrer'}
        ------------------     --------------------------------------------------------------------
        use_data               Optional dictionary. This parameter explicitly specify the country
                               or dataset to query. When all input features specified in the
                               study_areas parameter describe locations or areas that lie in the
                               same country or dataset, this parameter can be specified to provide
                               an additional 'performance hint' to the service.

                               By default, the service will automatically determine the country or
                               dataset that is associated with each location or area submitted in
                               the study_areas parameter. Specifying a specific dataset or country
                               through this parameter will potentially improve response time.

                               By default, the data apportionment method is determined by the size
                               of the study area. Small study areas use block apportionment for
                               higher accuracy whereas large study areas (100 miles or more) will
                               use a cascading centroid apportionment method to maintain
                               performance. This default behavior can be overridden by using the
                               detailed_aggregation parameter.
        ------------------     --------------------------------------------------------------------
        in_sr                  Optional parameter to define the input geometries in the study_areas
                               parameter in a specified spatial reference system.
                               When input points are defined in the study_areas parameter, this
                               optional parameter can be specified to explicitly indicate the
                               spatial reference system of the point features. The parameter value
                               can be specified as the well-known ID describing the projected
                               coordinate system or geographic coordinate system.
                               The default is 4326
        ------------------     --------------------------------------------------------------------
        out_name               Optional string.  Name of the output file
        ------------------     --------------------------------------------------------------------
        out_folder             Optional string. Name of the save folder
        ==================     ====================================================================
        """
        url = "%s%s" % (self._base_url, self._url_create_report)
        params = {
            "f": "bin",
            "studyAreas": study_areas,
            "appID": self._appID,
            "format": export_format,
        }
        if self._gis._portal.is_arcgisonline and self._gis._con._session.auth.token:
            params["token"] = self._gis._con._session.auth.token
        if report is not None:
            params["report"] = report
        if report_fields is not None:
            params["reportFields"] = report_fields
        if options is not None:
            params["studyAreasOptions"] = options
        if self._gis._portal.is_arcgisonline and return_type is not None:
            params["returnType"] = return_type
        if in_sr:
            params["inSR"] = in_sr
        if use_data is not None:
            params["useData"] = use_data
        # result is always a file path because error response will be parsed inside the method
        # due to try_json=True and file_name=None parameters
        report_file_path = self._gis._con.post(
            path=url,
            out_folder=out_folder,
            postdata=params,
            try_json=True,
        )
        if out_name:
            # rename created file
            updated_report_file_path = os.path.join(out_folder, out_name)
            if os.path.isfile(updated_report_file_path):
                os.remove(updated_report_file_path)
            os.rename(report_file_path, updated_report_file_path)
            return updated_report_file_path

        return report_file_path

    # ----------------------------------------------------------------------
    def standard_geography_levels(self, country: str):
        """
        For a given country, the standard geography level returns information
        relating to the area in question.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                Required string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country. This can be the two letter country code or the coutries
                               full name.
        ==================     ====================================================================

        :return: dictionary

        """
        as_dict = True
        countries = self.countries()
        if len(country) > 2:
            q = self.countries()["Full_Name"].str.upper() == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Name: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        else:
            q = self.countries()["Country_Code"] == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Code: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        params = {"f": "json"}
        url = self._base_url + "/Geoenrichment/standardgeographylevels/%s" % (country)
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        res = self._gis._con.post(url, params)
        if as_dict == True:
            return res
        return res

    # ----------------------------------------------------------------------
    def standard_geography_level_info(self, country: str, hierarchy: str):
        """

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        country                Required string. lets the user supply an optional name of a country
                               in order to get information about the data collections in that given
                               country. This can be the two letter country code or the coutries
                               full name.
        ------------------     --------------------------------------------------------------------
        hierarchy              Optional parameter to specify a specific hierarchy within a defined country.
        ==================     ====================================================================

        :return: A dictionary
        """
        as_dict = True
        countries = self.countries()
        if len(country) > 2:
            q = self.countries()["Full_Name"].str.upper() == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Name: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        else:
            q = self.countries()["Country_Code"] == str(country).upper()
            if len(countries[q]) == 0:
                raise ValueError("Invalid Country Code: %s" % country)
            country = countries[q]["Country_Code"].tolist()[0]
        params = {"f": "json"}
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        url = self._base_url + "/Geoenrichment/standardgeographylevels/%s/%s" % (
            country,
            hierarchy,
        )
        if self._gis._con.token:
            params["token"] = self._gis._con.token
        res = self._gis._con.post(url, params)
        if as_dict == True:
            return res
        else:
            return pd.DataFrame.from_dict(res)
        return res

    # ----------------------------------------------------------------------
    @property
    def limits(self):
        """
        Provides the limits of the current GeoEnrichment Service.  This will allow
        users to determine how to break up the calls accordingly to ensure all data
        is returned.

        :return: Pandas' DataFrame
        """
        if self._limits is None:
            limits_resp = self._gis._con.get(
                f"{self._gis.properties.helperServices.geoenrichment.url}/Geoenrichment/ServiceLimits"
            )
            if "serviceLimits" in limits_resp:
                self._limits = pd.DataFrame(limits_resp["serviceLimits"]["value"])
            else:
                return limits_resp
        return self._limits

    # ----------------------------------------------------------------------
    def standard_geography_query(
        self,
        source_country: Optional[str] = None,
        country_dataset: Optional[str] = None,
        layers: Optional[Union[list, str]] = None,
        ids: Optional[Union[list, str]] = None,
        geoquery: Optional[Union[list, str]] = None,
        return_sub_geography: bool = False,
        sub_geography_layer: Optional[Union[list, str]] = None,
        sub_geography_query: Optional[str] = None,
        out_sr: int = 4326,
        return_geometry: bool = False,
        return_centroids: bool = False,
        generalization_level: int = 0,
        use_fuzzy_search: bool = False,
        feature_limit: int = 5000,
        as_featureset: bool = False,
    ):
        """
        The GeoEnrichment class provides a helper method that returns standard geography IDs and
        features for the supported geographic levels in the United States and Canada.
        The GeoEnrichment class uses the concept of a study area to define the location of the point
        or area that you want to enrich with additional information. Locations can also be passed as
        one or many named statistical areas. This form of a study area lets you define an area by
        the ID of a standard geographic statistical feature, such as a census or postal area. For
        example, to obtain enrichment information for a U.S. state, county or ZIP Code or a Canadian
        province or postal code, the Standard Geography Query helper method allows you to search and
        query standard geography areas so that they can be used in the GeoEnrichment method to
        obtain facts about the location.
        The most common workflow for this service is to find a FIPS (standard geography ID) for a
        geographic name. For example, you can use this service to find the FIPS for the county of
        San Diego which is 06073. You can then use this FIPS ID within the GeoEnrichment class study
        area definition to get geometry and optional demographic data for the county. This study
        area definition is passed as a parameter to the GeoEnrichment class to return data defined
        in the enrichment pack and optionally return geometry for the feature.

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        source_country             Optional string. to specify the source country for the search. Use
                                   this parameter to limit the search and query of standard geographic
                                   features to one country. This parameter supports both the two-digit
                                   and three-digit country codes illustrated in the coverage table.
        ----------------------     --------------------------------------------------------------------
        country_dataset            Optional string. parameter to specify a specific dataset within a
                                   defined country.
        ----------------------     --------------------------------------------------------------------
        layers                     Optional list/string. Parameter specifies which standard geography
                                   layers are being queried or searched. If this parameter is not
                                   provided, all layers within the defined country will be queried.
        ----------------------     --------------------------------------------------------------------
        ids                        Optional parameter to specify which IDs for the standard geography
                                   layers are being queried or searched. You can use this parameter to
                                   return attributes and/or geometry for standard geographic areas for
                                   administrative areas where you already know the ID, for example, if
                                   you know the Federal Information Processing Standard (FIPS) Codes for
                                   a U.S. state or county; or, in Canada, to return the geometry and
                                   attributes for a Forward Sortation Area (FSA).
                                   Example:
                                   Return the state of California where the layers parameter is set to
                                   layers=['US.States']
                                   then set ids=["06"]
        ----------------------     --------------------------------------------------------------------
        geoquery                   Optional string/list. This parameter specifies the text to query
                                   and search the standard geography layers specified. You can use this
                                   parameter to query and find standard geography features that meet an
                                   input term, for example, for a list of all the U.S. counties that
                                   contain the word "orange". The geoquery parameter can be a string
                                   that contains one or more words.
        ----------------------     --------------------------------------------------------------------
        return_sub_geography       Optional boolean. Use this optional parameter to return all the
                                   subgeographic areas that are within a parent geography.
                                   For example, you could return all the U.S. counties for a given
                                   U.S. state or you could return all the Canadian postal areas
                                   (FSAs) within a Census Metropolitan Area (city).
                                   When this parameter is set to true, the output features will be
                                   defined in the sub_geography_layer. The output geometries will be
                                   in the spatial reference system defined by out_sr.
        ----------------------     --------------------------------------------------------------------
        sub_geography_layer        Optional string/list. Use this optional parameter to return all the
                                   subgeographic areas that are within a parent geography. For example,
                                   you could return all the U.S. counties within a given U.S. state or
                                   you could return all the Canadian postal areas (FSAs) within a
                                   Census Metropolitan Areas (city).
                                   When this parameter is set to true, the output features will be
                                   defined in the sub_geography_layer. The output geometries will be
                                   in the spatial reference system defined by out_sr.
        ----------------------     --------------------------------------------------------------------
        sub_geography_query        Optional string.User this parameter to filter the results of the
                                   subgeography features that are returned by a search term.
                                   You can use this parameter to query and find subgeography
                                   features that meet an input term. This parameter is used to
                                   filter the list of subgeography features that are within a
                                   parent geography. For example, you may want a list of all the
                                   ZIP Codes that are within "San Diego County" and filter the
                                   results so that only ZIP Codes that start with "921" are
                                   included in the output response. The subgeography query is a
                                   string that contains one or more words.
        ----------------------     --------------------------------------------------------------------
        out_sr                     Optional integer Use this parameter to request the output geometries
                                   in a specified spatial reference system.
        ----------------------     --------------------------------------------------------------------
        return_geometry            Optional boolean. Use this parameter to request the output
                                   geometries in the response.  The return type will become a Spatial
                                   DataFrame instead of a Panda's DataFrame.
        ----------------------     --------------------------------------------------------------------
        return_centroids           Optional Boolean.  Use this parameter to request the output geometry
                                   to return the center point for each feature.
        ----------------------     --------------------------------------------------------------------
        generalization_level       Optional integer that specifies the level of generalization or
                                   detail in the area representations of the administrative boundary or
                                   standard geographic data layers.
                                   Values must be whole integers from 0 through 6, where 0 is most
                                   detailed and 6 is most generalized.
        ----------------------     --------------------------------------------------------------------
        use_fuzzy_search           Optional Boolean parameter to define if text provided in the
                                   geoquery parameter should utilize fuzzy search logic. Fuzzy searches
                                   are based on the Levenshtein Distance or Edit Distance algorithm.
        ----------------------     --------------------------------------------------------------------
        feature_limit              Optional integer value where you can limit the number of features
                                   that are returned from the geoquery.
        ----------------------     --------------------------------------------------------------------
        as_featureset              Optional boolean. If False (default) the return type is a Spatail
                                   DataFrame, else it is a :class:`~arcgis.features.FeatureSet`
        ======================     ====================================================================

        :return: Spatial or Pandas Dataframe on success, dictionary on failure.

        """
        url = self._base_url + self._url_standard_geography_query_execute
        params = {"f": "json", "langCode": self._langCode, "appID": self._appID}
        if self._gis._portal.is_arcgisonline and self._gis._con.token:
            params["token"] = self._gis._con.token
        if not source_country is None:
            params["sourceCountry"] = source_country
        if not country_dataset is None:
            params["optionalCountryDataset"] = country_dataset
        if not layers is None:
            params["geographylayers"] = layers
        if not ids is None:
            params["geographyids"] = json.dumps(ids)
        if not geoquery is None:
            params["geographyQuery"] = geoquery
        if not return_sub_geography is None and isinstance(return_sub_geography, bool):
            params["returnSubGeographyLayer"] = return_sub_geography
        if not sub_geography_layer is None:
            params["subGeographyLayer"] = sub_geography_layer
        if not sub_geography_query is None:
            params["subGeographyQuery"] = sub_geography_query
        if not out_sr is None and isinstance(out_sr, int):
            params["outSR"] = out_sr
        if not return_geometry is None and isinstance(return_geometry, bool):
            params["returnGeometry"] = return_geometry
        if not return_centroids is None and isinstance(return_centroids, bool):
            params["returnCentroids"] = return_centroids
        if not generalization_level is None and isinstance(generalization_level, int):
            params["generalizationLevel"] = generalization_level
        if not use_fuzzy_search is None and isinstance(use_fuzzy_search, bool):
            params["useFuzzySearch"] = json.dumps(use_fuzzy_search)
        if feature_limit is None:
            feature_limit = 5000
        elif isinstance(feature_limit, int):
            params["featureLimit"] = feature_limit
        else:
            params["featureLimit"] = 5000
        if self._gis._con.token:
            params["token"] = self._gis._con.token

        res = self._gis._con.post(path=url, postdata=params)
        dfs = []
        if as_featureset == False:
            import pandas as pd

            if "results" in res:
                for result in res["results"]:
                    if "value" in result and "dataType" in result:
                        dfs.append(FeatureSet.from_dict(result["value"]).sdf)
            if len(dfs) > 0:
                df = pd.concat(dfs)
                df.reset_index(inplace=True, drop=True)
                return df
            elif len(dfs) == 1:
                return dfs[0]
            return res
        else:
            if "results" in res:
                for result in res["results"]:
                    if "value" in result and "dataType" in result:
                        dfs.append(FeatureSet.from_dict(result["value"]))
            if len(dfs) > 0:
                return dfs
            elif len(dfs) == 1:
                return dfs[0]
            return res
