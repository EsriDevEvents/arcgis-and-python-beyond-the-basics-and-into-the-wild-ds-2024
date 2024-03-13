import asyncio
import json
import re
import uuid
from collections import namedtuple
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Union, Awaitable, Iterable, Optional
from warnings import warn

import pandas as pd

from arcgis._impl.common._utils import _lazy_property as lazy_property
from arcgis.features import GeoAccessor, FeatureSet
from arcgis.geometry import Geometry, SpatialReference
from arcgis.gis import GIS
from arcgis.network.analysis import get_travel_modes
from ._spatial import change_spatial_reference
from ._utils import (
    add_proximity_to_enrich_feature_list,
    extract_from_kwargs,
    get_helper_service_url,
    get_sanitized_names,
    get_spatially_enabled_dataframe,
    is_dict_geometry,
    is_dict_featureset,
    local_vs_gis,
    pep8ify,
    pro_at_least_version,
    run_async,
    set_source,
    validate_network_travel_mode,
)

__all__ = ["BusinessAnalyst", "Country"]


class AOI(object):
    """
    An AOI (area of interest) delineates the area being used for enrichment.
    Currently this is implemented as a Country, but is classed as a parent
    so later areas of interest can be delineated spanning country borders.
    """

    def __init__(self, source, **kwargs):
        # set the enrichment property
        self._ba = self._ba = (
            kwargs["enrichment"] if "enrichment" in kwargs else BusinessAnalyst(source)
        )

        # set the source
        self.source = self._ba.source

        # placeholder for properties
        self.properties: namedtuple = None

    def __repr__(self):
        repr_str = f"<{type(self).__name__} ({self.source})>"
        return repr_str

    @property
    def source(self) -> Union[str, GIS]:
        """
        Source being used.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        in_source               Optional either the 'local' keyword or an instantiated ``GIS`` object
                                instance.
        ==================      ====================================================================
        """
        return self._source

    @source.setter
    def source(self, in_source: Optional[Union[str, GIS]] = None) -> None:
        self._source = set_source(in_source)

        # if working with a GIS object instance, we need to set a few extra properties
        if isinstance(self._source, GIS):
            # run a few checks and get the helper service for geoenrichment
            self._base_url = get_helper_service_url(self.source, "geoenrichment")

            # run a check for nuances of hosted notebooks
            if self._source._is_hosted_nb_home:
                # check if there is a private service url (only true for hosted notebooks)
                res = self._source._private_service_url(self._base_url)

                # if there is a private service url returned in the response, change to this
                self._base_url = (
                    res["privateServiceUrl"]
                    if "privateServiceUrl" in res
                    else res["serviceUrl"]
                )

    @lazy_property
    @local_vs_gis
    def enrich_variables(self):
        """Pandas DataFrame of available enrichment enrich_variables."""
        pass

    def _enrich_variables_gis(self):
        """GIS implementation of enrich_variables property."""
        # if the AOI has an iso3 property, then is a Country, and should be added as part of the request
        iso3 = self.__dict__["iso3"] if hasattr(self, "iso3") else None

        # get the enrich variables
        ev = self._ba._get_enrich_variables_gis(iso3)

        return ev

    def get_enrich_variables_from_iterable(
        self, enrich_variables: Union[Iterable, pd.Series], **kwargs
    ) -> pd.DataFrame:
        """Get a dataframe of enrich enrich_variables associated with the list of enrich_variables
        passed in. This is especially useful when needing aliases (*human readable
        names*), or are interested in enriching more data using previously enriched
        data as a template.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        enrich_variables        Required Iterable (normally a list) of enrich_variables correlating to
                                enrichment enrich_variables. These variable names can be simply the name, the
                                name prefixed by the collection separated by a dot, or the output from
                                enrichment in ArcGIS Pro with the field name modified to fit field naming
                                and length constraints.
        ==================      ====================================================================

        :returns:
            Pandas DataFrame of enrich enrich_variables with the different available aliases.

        .. code-block:: python

            from pathlib import Path

            import arcpy
            from business_analyst import BusinessAnalyst, Country

            # paths
            gdb = Path(r'C:/path/to/geodatabase.gdb')
            enriched_fc_pth = gdb/'enriched_data'
            features_to_be_enriched = gdb/'block_groups_pdx'
            new_fc_pth = gdb/'block_groups_pdx_enriched'

            # get a list of column names from previously enriched data
            attr_lst = [c.name for c in arcpy.ListFields(str(enriched_fc_pth))

            # get a country to work in
            cntry = BusinessAnalyst('local').get_country('USA')

            # get dataframe of enrich_variables used for previously enriched data
            enrich_vars = cntry.get_enrich_variables_from_name_list(attr_lst)

            # enrich block groups in new area of interest using the same enrich_variables
            bg_df = pd.DataFrame.spatial.from_featureclass(features_to_be_enriched)
            enrich_df = cntry.enrich(new_fc_pth)

            # save the enriched data
            enrich_df.spatial.to_featureclass(new_fc_pth)
        """
        # call the method from the parent business analyst instance
        ev = self._ba.get_enrich_variables_from_iterable(
            enrich_variables, country=self, **kwargs
        )

        return ev

    @lazy_property
    @local_vs_gis
    def geography_levels(self) -> pd.DataFrame:
        """Dataframe of available geography levels."""
        pass

    def get_geography_level(
        self,
        selector: Optional[Union[str, pd.DataFrame]] = None,
        selection_field: str = "NAME",
        query_string: Optional[str] = None,
        output_spatial_reference: Optional[Union[SpatialReference, dict, int]] = 4326,
        return_geometry: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Get a DataFrame at an available geography_level level.

        Args:
            selector: Spatially Enabled DataFrame or string value used to select features.
                If a specific value can be identified using a string, even if
                just part of the field value, you can insert it here.
            selection_field: This is the field to be searched for the string values
                input into selector. It defaults to ``NAME``.
            query_string: If a more custom query is desired to filter the output, please
                use SQL here to specify the query. The normal query is ``UPPER(NAME) LIKE
                UPPER('%<selector>%')``. However, if a more specific query is needed, this
                can be used as the starting point to get more specific.
            output_spatial_reference: Desired spatial reference for returned data. This can be
                a spatial reference WKID as an integer, a dictionary representing the spatial
                reference, or a Spatial Reference object. The default is WGS84 (WKID 4326).
            return_geometry: Boolean indicating if geometry should be returned. While
                typically the case, there are instances where it is useful to not
                retrieve the geometries. This includes testing to create a query  only
                retrieving one area of interest.

        Returns:
            Pandas DataFrame of values fulfilling selection.
        """
        pass

    @lazy_property
    @local_vs_gis
    def travel_modes(self) -> pd.DataFrame:
        """Dataframe of available travel modes."""
        pass

    def enrich(
        self,
        geographies: Union[pd.DataFrame, Iterable, Path],
        enrich_variables: Union[pd.DataFrame, Iterable],
        return_geometry: bool = True,
        standard_geography_level: Optional[Union[int, str]] = None,
        standard_geography_id_column: Optional[str] = None,
        proximity_type: Optional[str] = None,
        proximity_value: Optional[Union[float, int]] = None,
        proximity_metric: Optional[str] = None,
        output_spatial_reference: Union[int, dict, SpatialReference] = None,
        estimate_credits: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Path, float]:
        """
        Get demographic data apportioned to the input geographies based on population density
        weighting. Input geographies can be polygons or points. If providing point geographies,
        an area surrounding every point will be used to determine the area for enrichment.

        .. note::

            By default, point area is typically a buffered straight-line distance. However,
            if a transportation network is available, more accurate proximity metrics may be
            available, such as drive distance or drive time. This is the case if using ``local``
            (ArcGIS Pro with Business Analyst and local data) or a ``GIS`` object connected to ArcGIS
            Online, and very well may also the be the case if using an instance of ArcGIS Enterprise.

        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        geographies                         Required geographic areas or points to be enriched.
                                            enrich_variables: Enrichment enrich_variables to be used,
                                            typically discovered using the "enrich_variables" property.
        -----------------------------       --------------------------------------------------------------------
        return_geometry                     Optional boolean indicating if geometry is desired in the output.
                                            Default is True.
        -----------------------------       --------------------------------------------------------------------
        standard_geography_level            If the input geographies are a standard geography level,
                                            it can be specified using either the standard geography index or the standard
                                            geography identifier retrieved using the Country.geography_levels property.
        -----------------------------       --------------------------------------------------------------------
        standard_geography_id_column        Column with values uniquely identifying the input
                                            geographies using a standard level of geography. For example, in the United
                                            States, typically block groups are used for analysis if possible, and these
                                            block groups all have a unique identifier, typically referred to as the FIPS.
                                            If you have this value in a column of your data, it will *dramatically* speed
                                            up the enrichment process if you specify it in this parameter.
        -----------------------------       --------------------------------------------------------------------
        proximity_type                      Type of area to create around each point.
        -----------------------------       --------------------------------------------------------------------
        proximity_value                     Scalar value representing the proximity around each point to
                                            be used for creating an area for enrichment. For instance, if using 1.2 km,
                                            the input for this parameter is 1.2. The default is 1.
        -----------------------------       --------------------------------------------------------------------
        proximity_metric                    Scalar metric defining the proximity_value. Again, if
                                            1.2 km, the input for this metric will be kilometers. The default is
                                            ``kilometers``.
        -----------------------------       --------------------------------------------------------------------
        output_spatial_reference            Desired spatial reference for returned data. This can be
                                            a spatial reference WKID as an integer, a dictionary representing the spatial
                                            reference, or a Spatial Reference object. The default is WGS84 (WKID 4326).
        -----------------------------       --------------------------------------------------------------------
        estimate_credits                    While only useful for ArcGIS Online, enables populating the
                                            parameters just as you would for enriching using ArcGIS Online, and getting an
                                            estimate of the number of credits, which will be consumed if the enrich
                                            operation is performed. If this parameter is populated, the function will *not*
                                            perform the enrich operation or consume credits. Rather, it will *only* provide
                                            a credit consumption estimate.
        =============================       ====================================================================

        Returns:
            Pandas DataFrame, path to the output Feature Class or table, or float of predicted
            credit consumption.
        """
        enrich_df = self._ba.enrich(
            geographies=geographies,
            enrich_variables=enrich_variables,
            country=self,
            standard_geography_level=standard_geography_level,
            standard_geography_id_column=standard_geography_id_column,
            proximity_type=proximity_type,
            proximity_value=proximity_value,
            proximity_metric=proximity_metric,
            return_geometry=return_geometry,
            output_spatial_reference=output_spatial_reference,
            estimate_credits=estimate_credits,
            **kwargs,
        )
        return enrich_df


class Country(AOI):
    """
    Country enables access to Business Analyst functionality. Business Analyst
    data is available by iso3 using both ``local`` (ArcGIS Pro with the Business
    Analyst extension and local data) and ``GIS`` sources.

    =============================       ====================================================================
    **Parameter**                        **Description**
    -----------------------------       --------------------------------------------------------------------
    iso3                                The country's ISO3 identifier.
    -----------------------------       --------------------------------------------------------------------
    source                              Optional ``GIS`` object or ``local`` keyword specifying the Business
                                        Analyst data and analysis source. If ``local``, the Python
                                        environment *must* have ``arcpy`` installed with bindings to ArcGIS
                                        Pro with the Business Analyst extension. If connecting to a ``GIS``
                                        instance, both ArcGIS Enterprise with Business Analyst and ArcGIS
                                        Online are supported. However, please be aware, any geoenrichment or
                                        analysis *will* consume ArcGIS Online credits.
    -----------------------------       --------------------------------------------------------------------
    year                                Optional integer explicitly specifying the year to reference.
                                        This is only honored if using local resources and the specified
                                        year is available.
    =============================       ====================================================================

    """

    def __init__(
        self,
        iso3: str,
        source: Optional[Union[str, GIS]] = None,
        year: Optional[int] = None,
        **kwargs,
    ) -> None:
        # invoke the AOI init method - mostly just sets the _ba property
        super().__init__(source, **kwargs)

        # set the iso3 property based on the iso3
        self.iso3 = self._ba._standardize_country_str(iso3)

        # use the iso3 to filter the available countries to a dataframe of just the country requested.
        sel_df = self._ba.countries[self._ba.countries["iso3"] == self.iso3]

        # if the data source is local, but no year was provided, get the year
        if self.source == "local" and year is None:
            year = sel_df.vintage.max()

        # if the year is provided, validate
        elif self.source == "local" and year is not None:
            # just in case it was accidentally input as a string
            year = int(year) if isinstance(year, str) else year

            # get list of valid values for this country
            lcl_yr_vals = list(
                self._ba.countries[self._ba.countries["iso3"] == self.iso3]["vintage"]
            )

            # validate year is available for given country
            assert year in lcl_yr_vals, (
                f'The year you provided, "{year}" is not among the available '
                f'years ({", ".join([str(v) for v in lcl_yr_vals])}) for "{self.iso3.upper()}".'
            )

        # if source is GIS and year is provided, warn will be ignored
        elif isinstance(self.source, GIS) and year is not None:
            warn(
                "Explicitly specifying a year (vintage) is not supported when using a GIS instance as the source."
            )

        # get the first row from the selected dataframe, filtering based on year if local, for the properties
        properties = (
            sel_df[sel_df.vintage == year].iloc[0]
            if self.source == "local"
            else sel_df.iloc[0]
        )

        # if local, set a few more properties unique to local
        if self.source == "local":
            # to avoid confusion, add year as alias to vintage
            properties["year"] = properties["vintage"]

            # set the path to the network dataset
            properties["network_path"] = self._get_network_dataset_path_local(
                properties
            )

            # ensure analysis uses correct data source
            self._set_local_ba_analysis_source(properties)

        # convert the properties to a namedtuple and set as property of object
        self.properties = namedtuple("properties", properties.index)(*properties)

    def __repr__(self):
        repr_str = f"<{type(self).__name__} - {self.iso3}"

        if self.source == "local":
            repr_str = f"{repr_str} {self.properties.vintage} (local)>"
        else:
            repr_str = f"{repr_str} ({self.source})>"

        return repr_str

    @staticmethod
    def _get_network_dataset_path_local(properties) -> str:
        """Get the path to the network dataset for this country."""
        import arcpy

        # get a dictionary of dataset properties
        ds_dict = {ds["id"]: ds for ds in arcpy._ba.getLocalDatasets()}

        # get the path to the network dataset using the country id (captures both country and year)
        src_pth = ds_dict[properties.country_id]["networkDatasetPath"]

        return src_pth

    @staticmethod
    def _set_local_ba_analysis_source(properties) -> None:
        """Ensure analysis uses the correct data source."""
        import arcpy

        arcpy.env.baDataSource = f"LOCAL;;{properties.country_id}"
        arcpy.env.baNetworkSource = properties.network_path

    def _enrich_variables_local(self) -> pd.DataFrame:
        """Local implementation of enrich_variables property and only available by iso3 currently."""
        # ensure using correct data sources
        self._set_local_ba_analysis_source(self.properties)

        # lazy load
        from arcpy._ba import ListVariables

        # pull out the iso3 dataframe to a variable for clarity
        cntry_df = self._ba.countries

        # create a filter to select the right iso3 and year dataset
        dataset_fltr = (cntry_df["iso3"] == self.iso3) & (
            cntry_df["vintage"] == int(self.properties.vintage)
        )

        # get the iso3 identifier needed for listing enrich_variables
        ba_id = cntry_df[dataset_fltr].iloc[0]["country_id"]

        # retrieve variable objects
        var_gen = ListVariables(ba_id)

        # use a list comprehension to unpack the properties of the enrich_variables into a dataframe
        var_df = pd.DataFrame(
            (
                (v.Name, v.Alias, v.DataCollectionID, v.FullName, v.OutputFieldName)
                for v in var_gen
            ),
            columns=[
                "name",
                "alias",
                "data_collection",
                "enrich_name",
                "enrich_field_name",
            ],
        )

        return var_df

    def _geography_levels_local(self):
        """Local implementation of geography_levels."""
        import arcpy

        # ensure using correct data sources
        self._set_local_ba_analysis_source(self.properties)

        # get a dataframe of level properties for the country
        geo_lvl_df = pd.DataFrame.from_records(
            [
                lvl.__dict__
                for lvl in arcpy._ba.ListGeographyLevels(self.properties.country_id)
            ]
        )

        # reverse sorting order so smallest is at the top
        geo_lvl_df = geo_lvl_df.iloc[::-1].reset_index(drop=True)

        # calculate a field for use in the accessor
        geo_lvl_df.insert(
            0,
            "level_name",
            geo_lvl_df["LevelID"].apply(lambda val: pep8ify(val.split(".")[1])),
        )

        # purge columns and include AdminLevel if present (added in Pro 2.9)
        out_col_lst = [
            "level_name",
            "SingularName",
            "PluralName",
            "LevelName",
            "LayerID",
            "IDField",
            "NameField",
        ]
        if "AdminLevel" in list(geo_lvl_df.columns):
            out_col_lst = out_col_lst + ["AdminLevel"]
        geo_lvl_df.drop(
            columns=[c for c in geo_lvl_df.columns if c not in out_col_lst],
            inplace=True,
        )

        # rename fields for consistency
        out_rename_dict = {
            "SingularName": "singular_name",
            "PluralName": "plural_name",
            "LevelName": "alias",
            "LayerID": "level_id",
            "IDField": "id_field",
            "NameField": "name_field",
            "AdminLevel": "admin_level",
        }
        geo_lvl_df.rename(columns=out_rename_dict, inplace=True, errors="ignore")

        return geo_lvl_df

    def _geography_levels_gis(self):
        """GIS implementation of geography levels."""
        # unpack the geoenrichment url from the properties
        enrich_url = self.source.properties.helperServices.geoenrichment.url

        # construct the url to the standard geography levels
        url = f"{enrich_url}/Geoenrichment/standardgeographylevels"

        # get the geography levels from the enrichment server
        res_json = self.source._con.post(url, {"f": "json"})

        # unpack the geography levels from the json
        geog_lvls = res_json["geographyLevels"]

        # get matching geography levels out of the list of countries
        for lvl in geog_lvls:
            if lvl["countryID"] == self.properties.iso2:
                geog = lvl
                break  # once found, bust out of the loop

        # get the hierarchical levels out as a dataframe
        geo_lvl_df = pd.DataFrame(geog["hierarchies"][0]["levels"])

        # reverse the sorting so the smallest is at the top
        geo_lvl_df = geo_lvl_df.iloc[::-1].reset_index(drop=True)

        # create consistent naming convention
        geo_lvl_df["level_name"] = geo_lvl_df["id"].apply(
            lambda val: pep8ify(val.split(".")[1])
        )

        # clean up the field names so they follow more pythonic conventions and are consistent
        geo_lvl_df = geo_lvl_df[
            ["level_name", "singularName", "pluralName", "name", "id", "adminLevel"]
        ].copy()
        geo_lvl_df.columns = [
            "level_name",
            "singular_name",
            "plural_name",
            "alias",
            "level_id",
            "admin_level",
        ]

        return geo_lvl_df

    def _travel_modes_local(self):
        """Local implementation of travel modes."""
        # ensure using correct data sources
        self._set_local_ba_analysis_source(self.properties)

        # get a list of useful information about each travel mode
        mode_lst = [
            [
                mode.name,
                mode.description,
                mode.type,
                mode.impedance,
                mode.timeAttributeName,
                mode.distanceAttributeName,
            ]
            for mode in self._travel_modes_dict_local.values()
        ]

        # create a dataframe of the travel modes
        mode_df = pd.DataFrame(
            mode_lst,
            columns=[
                "alias",
                "description",
                "type",
                "impedance",
                "time_attribute_name",
                "distance_attribute_name",
            ],
        )

        # add a pep8tified name for each travel mode
        mode_name = mode_df["alias"].apply(lambda val: pep8ify(val))
        mode_df.insert(0, "name", mode_name)

        # calculate impedance category
        imped_cat = mode_df["impedance"].apply(
            lambda val: "temporal" if val.lower().endswith("time") else "distance"
        )
        insert_idx = mode_df.columns.get_loc("impedance") + 1
        mode_df.insert(insert_idx, "impedance_category", imped_cat)

        return mode_df

    @lazy_property
    def _travel_modes_dict_local(self) -> dict:
        """Since looking up the travel modes is time consuming, use a lazy property to cache it."""
        import arcpy

        return arcpy.nax.GetTravelModes(self.properties.network_path)

    def _travel_modes_gis(self):
        """Web GIS implementation of travel modes."""
        # get the travel modes FeatureSet
        trvl_df = self._ba.travel_modes

        return trvl_df

    def _enrich_local(
        self,
        geographies: Union[pd.DataFrame, Iterable, Path],
        enrich_variables: Union[pd.DataFrame, Iterable],
        return_geometry: bool = True,
        standard_geography_level: Optional[Union[int, str]] = None,
        standard_geography_id_column: Optional[str] = None,
        proximity_type: Optional[str] = None,
        proximity_value: Optional[Union[float, int]] = None,
        proximity_metric: Optional[str] = None,
        output_spatial_reference: Union[int, dict, SpatialReference] = 4326,
        estimate_credits: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Local enrich method implementation."""
        # ensure using correct data sources
        self._set_local_ba_analysis_source(self.properties)

        enrich_res = self._ba.enrich(
            geographies=geographies,
            enrich_variables=enrich_variables,
            country=self,
            standard_geography_level=standard_geography_level,
            standard_geography_id_column=standard_geography_id_column,
            proximity_type=proximity_type,
            proximity_value=proximity_value,
            proximity_metric=proximity_metric,
            return_geometry=return_geometry,
            output_spatial_reference=output_spatial_reference,
            estimate_credits=estimate_credits,
        )
        return enrich_res


class BusinessAnalyst(object):
    """
    The BusinessAnalyst object enables access to Business Analyst functionality thorough a specified
    source. A source can either be in an environment with ArcGIS Pro with Business Analyst and local
    data (``local``) or a Web GIS (``GIS``) object instance. The Web GIS can reference either an
    instance of ArcGIS Enterprise with Business Analyst or ArcGIS Online.

    .. note::

        If the source is not explicitly set, the ``BusinessAnalyst`` object will first try to use
        ``local``, ArcGIS Pro with the Business Analyst extension. If this is not available,
        ``BusinessAnalyst`` will try to use a ``GIS`` object instance already in the session. If
        neither is available, and a ``source`` is not set, this will invoke an error.

    .. warning::

        GeoEnrichment (adding demographic enrich_variables) using ArcGIS Online *does* cost credits.
        Country (``BusinessAnalyst.countries``) and variable (:func:`~arcgis.geoenrichment.Country.enrich_variables`)
        introspection does *not* cost any credits.

    =============================       ====================================================================
    **Parameter**                        **Description**
    -----------------------------       --------------------------------------------------------------------
    source                              Optional ``GIS`` object or ``local`` keyword specifying the Business
                                        Analyst data and analysis source. If ``local``, the Python
                                        environment *must* have ``arcpy`` installed with bindings to ArcGIS
                                        Pro with the Business Analyst extension. If connecting to a ``GIS``
                                        instance, both ArcGIS Enterprise with Business Analyst and ArcGIS
                                        Online are supported. However, please be aware, any geoenrichment or
                                        analysis *will* consume ArcGIS Online credits.
    =============================       ====================================================================

    """

    def __init__(self, source: Optional[Union[str, GIS]] = None) -> None:
        # set the source, defaulting, based on what is available, to local or active_gis, or simply error if neither
        self.source = source

    def __repr__(self):
        repr_str = f"<{type(self).__name__} ({self.source})>"
        return repr_str

    @property
    def source(self) -> Union[str, GIS]:
        """
        Source being used.

        Args:
            in_source: Either the 'local' keyword or an instantiated ``GIS`` object
                instance.
        """
        return self._source

    @source.setter
    def source(self, in_source: Optional[Union[str, GIS]] = None) -> None:
        self._source = set_source(in_source)

        # if working with a GIS object instance, we need to set a few extra properties
        if isinstance(self._source, GIS):
            # run a few checks and get the helper service for geoenrichment
            self._base_url = get_helper_service_url(self.source, "geoenrichment")

            # run a check for nuances of hosted notebooks
            if self._source._is_hosted_nb_home:
                # check if there is a private service url (only true for hosted notebooks)
                res = self._source._private_service_url(self._base_url)

                # if there is a private service url returned in the response, change the url being used to this
                self._base_url = (
                    res["privateServiceUrl"]
                    if "privateServiceUrl" in res
                    else res["serviceUrl"]
                )

    @lazy_property
    @local_vs_gis
    def countries(self) -> pd.DataFrame:
        """DataFrame of available countries with relevant metadata columns based on the source."""
        pass

    def _countries_local(self) -> pd.DataFrame:
        """Local countries implementation."""
        # lazy load to avoid import issues
        import arcpy._ba

        # get a generator of dataset objects
        ds_lst = list(arcpy._ba.ListDatasets())

        # throw error if no local datasets are available
        assert len(ds_lst), (
            "No datasets are available locally. If you want to locate available countries on a "
            "Web GIS, please provide a GIS object instance for the source parameter when creating "
            "the BusinessAnalyst object."
        )

        # organize all the iso3 dataset properties
        cntry_lst = [
            (
                ds.CountryInfo.Name,
                ds.Version,
                ds.CountryInfo.ISO2,
                ds.CountryInfo.ISO3,
                ds.DataSourceID,
                ds.ID,
                None,
            )
            for ds in ds_lst
        ]

        # create a dataframe of the iso3 properties
        cntry_df = pd.DataFrame(
            cntry_lst,
            columns=[
                "name",
                "vintage",
                "iso2",
                "iso3",
                "data_source_id",
                "country_id",
                "hierarchies",
            ],
        )

        # convert the vintage years to integer
        cntry_df["vintage"] = cntry_df["vintage"].astype("int64")

        # ensure the values are in order by iso3 and year
        cntry_df.sort_values(["iso3", "vintage"], inplace=True)

        # organize the columns
        cntry_df = cntry_df[
            [
                "iso2",
                "iso3",
                "name",
                "vintage",
                "country_id",
                "data_source_id",
                "hierarchies",
            ]
        ]

        return cntry_df

    def _countries_gis(self) -> pd.DataFrame:
        """GIS countries implementation."""
        # make sure countries are available
        ge_err_msg = (
            "The provided GIS instance does not appear to have geoenrichment enabled and configured, "
            "so no countries are available."
        )
        assert "geoenrichment" in self.source.properties.helperServices, ge_err_msg
        assert isinstance(
            self.source.properties.helperServices.geoenrichment["url"], str
        ), ge_err_msg

        # extract out the geoenrichment url
        ge_url = self.source.properties.helperServices.geoenrichment["url"]
        if self.source._is_hosted_nb_home:
            res = self.source._private_service_url(ge_url)
            ge_url = (
                res["privateServiceUrl"]
                if "privateServiceUrl" in res
                else res["serviceUrl"]
            )

        # get a list of countries available on the Web GIS for enrichment
        url = f"{ge_url}/Geoenrichment/Countries"
        cntry_res = self.source._con.post(url, {"f": "json"})
        cntry_dict = cntry_res["countries"]

        # convert the dictionary to a dataframe
        cntry_df = pd.DataFrame(cntry_dict)

        # clean up some column names for consistency
        cntry_df.rename(
            {
                "id": "iso2",
                "abbr3": "iso3",
                "altName": "alt_name",
                "defaultDatasetID": "default_dataset",
                "hierarchies": "hierarchy",
            },
            inplace=True,
            axis=1,
        )
        keep_cols = [
            "iso2",
            "iso3",
            "name",
            "alt_name",
            "datasets",
            "default_dataset",
            "continent",
            "hierarchy",
        ]
        cntry_df = cntry_df[keep_cols]

        # clean up column for hierarchies to only keep alias if simple
        alias_names = []
        for i, v in cntry_df["hierarchy"].items():
            cntry_hier = []
            for hier in v:
                cntry_hier.append(hier["ID"])
            alias_names.append(cntry_hier)
        cntry_df["hierarchy"] = alias_names

        return cntry_df

    def _get_hierarchies_df(self, country_string: str):
        """Internal helper method to get the dataframe of hierarchies for each country"""
        # make sure countries are available
        ge_err_msg = (
            "The provided GIS instance does not appear to have geoenrichment enabled and configured, "
            "so no countries are available."
        )
        assert "geoenrichment" in self.source.properties.helperServices, ge_err_msg
        assert isinstance(
            self.source.properties.helperServices.geoenrichment["url"], str
        ), ge_err_msg

        # extract out the geoenrichment url
        ge_url = self.source.properties.helperServices.geoenrichment["url"]
        if self.source._is_hosted_nb_home:
            res = self.source._private_service_url(ge_url)
            ge_url = (
                res["privateServiceUrl"]
                if "privateServiceUrl" in res
                else res["serviceUrl"]
            )

        # get a list of countries available on the Web GIS for enrichment
        url = f"{ge_url}/Geoenrichment/Countries"
        cntry_res = self.source._con.post(url, {"f": "json"})
        cntry_dict = cntry_res["countries"]

        # convert the dictionary to a dataframe
        cntry_df = pd.DataFrame(cntry_dict)

        # clean up some column names for consistency
        cntry_df.rename(
            {
                "abbr3": "iso3",
            },
            inplace=True,
            axis=1,
        )
        keep_cols = ["iso3", "hierarchies"]
        cntry_df = cntry_df[keep_cols]

        # Get dataframe for specific country we are working with
        cntry_interest_df = cntry_df[cntry_df["iso3"] == country_string]
        # Get only the hierarchy column value and create own dataframe from it
        hierarchy_df = pd.DataFrame(cntry_interest_df.iloc[0]["hierarchies"])

        keep_cols = [
            "ID",
            "alias",
            "shortDescription",
            "datasets",
            "levelsInfo",
            "variablesInfo",
            "hasInterestingFactsStatistics",
        ]
        hierarchy_df = hierarchy_df[keep_cols]
        return hierarchy_df

    def _standardize_country_str(self, country_string: str) -> str:
        """Internal helper method to standardize the input for iso3 identifier strings to ISO3."""
        # cast to lowercase to sidestep any issues with case
        cntry_str = country_string.lower()

        # filter functions for getting the iso3 iso3 value
        iso3_fltr = self.countries["iso3"].str.lower() == cntry_str
        iso2_fltr = self.countries["iso2"].str.lower() == cntry_str
        name_fltr = self.countries["name"].str.lower() == cntry_str

        # construct the filter, using alias if working online
        cntry_fltr = iso3_fltr | iso2_fltr | name_fltr
        if isinstance(self.source, GIS):
            alias_fltr = self.countries["alt_name"].str.lower() == cntry_str
            cntry_fltr = cntry_fltr | alias_fltr

        # query available countries to see if the requested iso3 is available
        fltr_df = self.countries[cntry_fltr]

        if len(fltr_df.index) > 0:
            iso3_str = fltr_df.iloc[0]["iso3"]
        else:
            raise ValueError(
                f'The provided iso3 code, "{country_string}" does not appear to be available. Please '
                f"choose from the available country iso3 codes discovered using the "
                f"BusinessAnalyst.countries property."
            )

        return iso3_str

    def get_country(self, iso3: str, year: Optional[int] = None) -> Country:
        """
        Get a Country object instance.
        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        iso3                                Required String. The country's ISO3 identifier.
        -----------------------------       --------------------------------------------------------------------
        year                                Optional integer explicitly specifying the year to reference.
                                            This is only honored if using local resources and the specified
                                            year is available.
        =============================       ====================================================================

        Returns:
            Country object instance.

        A Country object instance can be created to work with Business Analyst data installed
        either locally with ArcGIS Pro or remotely through a Web GIS using a GIS object instance.

        .. code-block:: python

            from business_analyst import BusinessAnalyst
            ba = BusinessAnalyst('local')  # using ArcGIS Pro with Business Analyst
            usa = ba.get_country('USA')  # USA with most current data installed on machine

        .. code-block:: python

            from business_analyst import BusinessAnalyst
            ba = BusinessAnalyst('local')
            usa = ba.get_country('USA', year=2019)  # explicitly specifying the year with local data

        .. code-block:: python

            from arcgis.gis import GIS
            from business_analyst import BusinessAnalyst
            gis = GIS(username='my_username', password='$up3r$3cr3tP@$$w0rd')
            ba = BusinessAnalyst(gis)  # using ArcGIS Online
            usa = ba.get_country('USA')

        Once instantiated, available enrichment enrich_variables can be discovered and used for
        geoenrichment.

        .. code-block:: python

            # get the enrichment enrich_variables as a Pandas DataFrame
            evars = usa.enrich_variables

            # filter to just current year key enrich_variables
            kvars = [
                (evars.name.str.endswith('CY'))
                & (evars.data_collection.str.startswith('Key')
            ]

        Then, based on the environment being used for GeoEnrichment, these variable identifiers
        can be formatted for input into the respective enrich functions, `EnrichLayer`_ in Pro
        or `enrich`_ in the Python API. In the code below, the local and the Web GIS blocks will
        produce very similar outputs with the only differences being the column names.

        .. code-block:: python

            from arcgis.features import GeoAccessor
            import pandas as pd

            input_fc = 'C:/path/to/data.gdb/feature_class'
            output_pth = 'C:/path/to/data.gdb/enrich_features'

            # local geoenrichment
            import arcpy
            kvars_lcl = ';'.join(kvars.enrich_name)  # combine into semicolon separated string
            enrich_pth = arcpy.ba.EnrichLayer(input_fc, out_feature_class=output_pth, enrich_variables=kvars_lcl)[0]
            enrich_df = pd.DataFrame.spatial.from_featureclass(enrich_pth)  # dataframe for more analysis

            # WebGIS geoenrichment
            from arcgis.geoenrichment import enrich
            kvars_gis = ';'.join(kvars.name)
            in_df = pd.DataFrame.spatial.from_featureclass(input_fc)
            enrich_df = enrich(in_df, analysis_variables=kvars_gis, return_geometry=False)
            enrich_pth = enrich_df.spatial.to_table(output_pth)  # saving so do not have to re-run

        .. _EnrichLayer: https://pro.arcgis.com/en/pro-app/latest/tool-reference/business-analyst/enrich-layer-advanced.htm
        .. _enrich: https://developers.arcgis.com/python/api-reference/arcgis.geoenrichment.html#enrich

        """
        # standardize iso3 string to iso3 identifier
        iso3 = self._standardize_country_str(iso3)

        # create a iso3 object instance
        cntry = Country(iso3, year=year, enrichment=self)

        return cntry

    @lazy_property
    @local_vs_gis
    def enrich_variables(self) -> pd.DataFrame:
        """Available enrichment variables."""
        pass

    def _enrich_variables_gis(self) -> pd.DataFrame:
        """Web GIS implementation of enrich_variables property"""
        # retrieve the enrich variables using the method
        ev = self._get_enrich_variables_gis()

        return ev

    @lru_cache(maxsize=255)
    def _get_enrich_variables_gis(self, iso3: Optional[str] = None) -> pd.DataFrame:
        """Provide method to return enrich variables at both the BusinessAnalyst and AOI (Country) levels."""
        # construct the url with the option to simply not explicitly specify a iso3
        url = f"{self._base_url}/Geoenrichment/DataCollections/"
        if iso3 is not None:
            url = f"{url}{iso3}"

        # get the data collections from the GIS enrichment REST endpoint
        res = self.source._con.get(url, params={"f": "json"})
        msg_dc = (
            "Could not retrieve enrichment enrich_variables (DataCollections) from "
            "the GIS instance."
        )
        assert "DataCollections" in res.keys(), msg_dc

        # list to store all the dataframes as they are created for each data collection
        mstr_lst = []

        # iterate the data collections
        for col in res["DataCollections"]:
            # create a dataframe of the enrich_variables, keep only needed columns, and add the data collection name
            coll_df = pd.json_normalize(col["data"])[
                ["id", "alias", "description", "vintage", "units"]
            ]
            coll_df["data_collection"] = col["dataCollectionID"]

            # schema cleanup
            coll_df.rename(columns={"id": "name"}, inplace=True)
            coll_df = coll_df[
                ["name", "alias", "data_collection", "description", "vintage", "units"]
            ]

            # append the list
            mstr_lst.append(coll_df)

        # combine all the dataframes into a single master dataframe
        var_df = pd.concat(mstr_lst)

        # create the column for enrichment
        var_df.insert(3, "enrich_name", var_df.data_collection + "." + var_df.name)

        # create column for matching to previously enriched column names
        regex = re.compile(r"(^[0-9]+)")
        fld_vals = var_df.enrich_name.apply(
            lambda val: regex.sub(r"F\1", val.replace(".", "_"))
        )
        var_df.insert(4, "enrich_field_name", fld_vals)

        # reset the index so it looks like a single dataframe
        var_df.reset_index(inplace=True, drop=True)

        return var_df

    def get_enrich_variables_from_iterable(
        self, enrich_variables: Union[Iterable, pd.Series], **kwargs
    ) -> pd.DataFrame:
        """
        Get a dataframe of enrich enrich_variables associated with the list of enrich_variables
        passed in. This is especially useful when needing aliases (*human readable
        names*), or are interested in enriching more data using previously enriched
        data as a template.

        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        enrich_variables                    Iterable (normally a list) of enrich_variables correlating to
                                            enrichment enrich_variables. These variable names can be simply the name, the
                                            name prefixed by the collection separated by a dot, or the output from
                                            enrichment in ArcGIS Pro with the field name modified to fit field naming
                                            and length constraints.
        =============================       ====================================================================

        Returns:
            Pandas DataFrame of enrich enrich_variables with the different available aliases.

        .. code-block:: python

            from pathlib import Path

            import arcpy
            from arcgis.apps.ba import BusinessAnalyst

            # paths
            gdb = Path(r'C:/path/to/geodatabase.gdb')
            enriched_fc_pth = gdb/'enriched_data'
            features_to_be_enriched = gdb/'block_groups_pdx'
            new_fc_pth = gdb/'block_groups_pdx_enriched'

            # ArcGIS Online credentials
            usr = 'username'
            pswd = 'P@$$w0rd!'

            # create a business analyst instance
            gis = GIS(username=usr, password=pswd)
            ba = BusinessAnalyst(gis)

            # get a list of column names from previously enriched data
            attr_lst = [c.name for c in arcpy.ListFields(str(enriched_fc_pth))

            # get dataframe of enrich_variables used for previously enriched data
            enrich_vars = cntry.get_enrich_variables_from_name_list(attr_lst)

            # enrich block groups in new area of interest using the same enrich_variables
            bg_df = pd.DataFrame.spatial.from_featureclass(features_to_be_enriched)
            enrich_df = cntry.enrich(bg_df)

            # save the enriched data
            enrich_df.spatial.to_featureclass(new_fc_pth)
        """
        # validate the series input
        ev_msg = "Only an list or list-like object can be used as input as the enrich_variables input."
        ev_valid = not isinstance(enrich_variables, pd.DataFrame) and isinstance(
            enrich_variables, (Iterable, pd.Series)
        )
        assert ev_valid, ev_msg

        # if a country is included in the kwargs, pluck it out and get the right ba source
        if "country" in kwargs.keys():
            cntry: Country = kwargs["country"]
            iso3 = cntry.iso3
            ba = cntry._ba
        else:
            cntry, iso3 = None, None
            ba = self

        # based on the source, get the available enrichment variables
        if isinstance(ba.source, GIS):
            ev = ba._get_enrich_variables_gis(iso3)
        else:
            ev = cntry.enrich_variables

        # if the input is not a series, make it so
        if isinstance(enrich_variables, Iterable):
            enrich_variables = pd.Series(enrich_variables)

        # make the enrich variables lowercase to address potential missed matches due simply to case
        enrich_variables = enrich_variables.str.lower()

        # columns from enrich variables dataframe to search for matches in
        match_cols = ["enrich_name", "enrich_field_name", "name", "alias"]

        # iteratively check for matches and if matches found, bingo out
        for col in match_cols:
            sel_vars = (
                ev[ev[col].str.lower().isin(enrich_variables)]
                .drop_duplicates(col)
                .reset_index(drop=True)
            )
            if len(sel_vars):
                break

        # if no matches found, try sanitizing the names and see if this works
        if len(sel_vars) == 0:
            enrich_variables = get_sanitized_names(enrich_variables)
            for col in match_cols:
                sel_vars = (
                    ev[ev[col].str.lower().isin(enrich_variables)]
                    .drop_duplicates(col)
                    .reset_index(drop=True)
                )
                if len(sel_vars):
                    break

        # make sure something was found, but don't break the runtime
        if "suppress_warn" not in kwargs.keys() and len(sel_vars) == 0:
            warn(f"It appears none of the input enrich enrich_variables were found.")

        return sel_vars

    def _enrich_variable_preprocessing(
        self, enrich_variables: Union[Iterable, pd.DataFrame], **kwargs
    ) -> str:
        """Provide flexibility for enrich variable preprocessing by enabling enrich enrich_variables
        to be specified in a variety of iterables, but always provide a standardized variable
        DataFrame as output.

        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        enrich_variables                    Iterable (normally a list) or pd.DataFrame
                                            of enrich_variables correlating to
                                            enrichment enrich_variables. These variable names can be simply the name, the
                                            name prefixed by the collection separated by a dot, or the output from
                                            enrichment in ArcGIS Pro with the field name modified to fit field naming
                                            and length constraints.
        =============================       ====================================================================

        Returns:
            Pandas DataFrame of enrich enrich_variables.
        """
        # enrich variable dataframe column name
        enrich_str_col = "enrich_name"
        enrich_nm_col = "name"

        # if just a single variable is provided pipe it into a pandas series
        if isinstance(enrich_variables, str):
            enrich_variables = pd.Series([enrich_variables])

        # toss the enrich_variables into a pandas Series if an iterable was passed in
        elif isinstance(enrich_variables, Iterable) and not isinstance(
            enrich_variables, pd.DataFrame
        ):
            enrich_variables = pd.Series(enrich_variables)

        # if the enrich dataframe is passed in, check to make sure it has what we need, the right columns
        if isinstance(enrich_variables, pd.DataFrame):
            assert enrich_str_col in enrich_variables.columns, (
                f"It appears the dataframe used for enrichment does"
                f" not have the column with enrich string names "
                f"({enrich_str_col})."
            )
            assert enrich_nm_col in enrich_variables.columns, (
                f"It appears the dataframe used for enrichment does "
                f"not have the column with the enrich enrich_variables names "
                f"({enrich_nm_col})."
            )
            enrich_vars_df = enrich_variables

        # otherwise, create an enrich enrich_variables dataframe from the enrich series for a few more checks
        else:
            # get the enrich enrich_variables dataframe
            enrich_vars_df = self.get_enrich_variables_from_iterable(
                enrich_variables, **kwargs
            )

        # now, drop any duplicates so we're not getting the same variable twice from different data collections
        enrich_vars_df = enrich_vars_df.drop_duplicates("name").reset_index(drop=True)

        # note any enrich_variables submitted, but not found
        if len(enrich_variables) > len(enrich_vars_df.index):
            missing_count = len(enrich_variables) - len(enrich_vars_df.index)
            warn(
                "Some of the enrich_variables provided are not available for enrichment "
                f"(missing count: {missing_count:,}).",
                UserWarning,
            )

        # check to make sure there are enrich_variables for enrichment
        if len(enrich_vars_df.index) == 0:
            raise Exception(
                "There appear to be no enrich_variables selected for enrichment."
            )

        # get the string names needed in a list
        ev_lst = list(enrich_vars_df[enrich_str_col])

        # format output based on enrichment source
        enrich_vars = (
            json.dumps(ev_lst) if isinstance(self.source, GIS) else ";".join(ev_lst)
        )

        return enrich_vars

    @lazy_property
    @local_vs_gis
    def travel_modes(self) -> pd.DataFrame:
        """Dataframe of available travel modes."""
        pass

    def _travel_modes_gis(self):
        """Web GIS implementation of travel modes."""
        # get the travel modes FeatureSet
        trvl_fs = get_travel_modes(self.source).supported_travel_modes

        # pluck out the travel modes metadata from the TravelMode attribute in the FeatureSet
        trvl_df = pd.DataFrame.from_records(
            [json.loads(itm.attributes["TravelMode"]) for itm in trvl_fs.features]
        )

        # put the travel mode dict as a string (less the description) into a column - useful when needing for REST calls
        trvl_df["travel_mode_dict"] = [
            json.dumps(val)
            for val in trvl_df.drop(columns="description").to_dict("records")
        ]

        # make the column names pep8 compliant since in happy Python land
        trvl_df.columns = [pep8ify(c) for c in trvl_df.columns]

        # move the name into the alias and pep8ify the name
        trvl_df["alias"] = trvl_df["name"]
        trvl_df["name"] = trvl_df["name"].apply(lambda val: pep8ify(val))

        # rename a few columns for the sake of consistency with results returned from local network introspection
        trvl_df.rename(
            columns={"impedance_attribute_name": "impedance", "id": "travel_mode_id"},
            inplace=True,
        )

        # calculate impedance categories for ease of filtering in some workflows
        trvl_df["impedance_category"] = trvl_df["impedance"].apply(
            lambda val: ("temporal" if val.endswith("Time") else "distance")
            if pd.notna(val)
            else val
        )

        # reorganize the column order
        trvl_df = trvl_df[
            [
                "name",
                "alias",
                "description",
                "type",
                "impedance",
                "impedance_category",
                "time_attribute_name",
                "distance_attribute_name",
                "travel_mode_id",
                "travel_mode_dict",
            ]
        ]

        return trvl_df

    def _can_use_arrow(self, geo) -> bool:
        if not isinstance(geo, pd.DataFrame):
            return False  # not a DataFrame

        if not geo.spatial.validate():
            return False  # not a valid SeDF

        if len(geo.spatial.geometry_type) != 1:
            return False  # more than one geometry type

        if geo.spatial.geometry_type[0] != "polygon":
            return False  # only polygonal DF are supported at this point

        return True

    def _enrich_using_arrow(self, in_sedf, variables) -> pd.DataFrame:
        # create a data frame with two columns: object id and shape in WKB
        input_shape_series = in_sedf.loc[:, in_sedf.spatial.name]
        df_input = pd.DataFrame({in_sedf.spatial.name: input_shape_series})
        df_input.spatial.set_geometry(in_sedf.spatial.name)

        # come up with index field that doesn't exist yet
        orig_index_field = str(uuid.uuid4())
        # enrichArrowTable returns records in arbitrary order, ORIG_INDEX is in the original order
        # ORIG_INDEX starts with 0
        df_input[orig_index_field] = range(0, len(in_sedf))

        geo_accessor = GeoAccessor(df_input)
        arrow_table = geo_accessor.to_arrow()

        import arcpy._ba

        output_table = arcpy._ba.enrichArrowTable(arrow_table, variables, False)

        enrich_result_df = output_table.to_pandas()

        input_copy_df = in_sedf.copy()
        input_copy_df[orig_index_field] = range(0, len(in_sedf))

        # we need to rename that field to avoid clashes
        enrich_result_df.rename(columns={"ORIG_INDEX": orig_index_field}, inplace=True)

        if "ORIG_OID" in enrich_result_df:
            enrich_result_df.drop(["ORIG_OID"], axis=1, inplace=True)

        # join based on objectid
        merged_df = input_copy_df.merge(enrich_result_df, on=orig_index_field)
        merged_df.drop([orig_index_field], axis=1, inplace=True)

        # rearrange columns to move SHAPE to the last one
        orig_cols = merged_df.columns.tolist()
        new_cols = [c for c in orig_cols if c != in_sedf.spatial.name] + [
            in_sedf.spatial.name
        ]
        final_df = merged_df[new_cols]

        # return new SeDF based on final_df; shape column name is the same as before
        final_df.spatial.set_geometry(in_sedf.spatial.name)
        return final_df

    def enrich(
        self,
        geographies: Union[pd.DataFrame, Geometry, Iterable, Path],
        enrich_variables: Union[pd.DataFrame, Iterable],
        proximity_type: Optional[str] = None,
        proximity_value: Optional[Union[float, int]] = None,
        proximity_metric: Optional[str] = None,
        return_geometry: bool = True,
        output_spatial_reference: Union[int, dict, SpatialReference] = None,
        estimate_credits: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Enrich enables retrieving apportioned demographic factors for input geographies.

        =============================       ====================================================================
        **Parameter**                        **Description**
        -----------------------------       --------------------------------------------------------------------
        geographies                         Input geographies desired to get demographic variables for. Normally
                                            these will be geometries included as part of a spatially enabled Pandas
                                            Data Frame, but if using standard geographies, this can also be just an
                                            iterable of standard geography identifiers.
        -----------------------------       --------------------------------------------------------------------
        enrich_variables                    Variables desired to be retrieved.
        -----------------------------       --------------------------------------------------------------------
        proximity_type                      If the input geographies are points, retrieving enriched
                                            variables requires delineating a zone around each point to use for apportioning
                                            demographic factors to each input geography. Default is ``straight_line``.
        -----------------------------       --------------------------------------------------------------------
        proximity_value                     If the input geographies are points, this is the value used
                                            to create a zone around the points for apportioning demographic factors. For
                                            instance, if specifying five miles, this parameter value will be ``5``. Default
                                            is ``1``.
        -----------------------------       --------------------------------------------------------------------
        proximity_metric                    If the input geographies are points, this is the metric
                                            defining the proximity value. For instance, if specifying one kilometer, this
                                            value will be ``kilometers``. Default is ``kilometers``.
        -----------------------------       --------------------------------------------------------------------
        return_geometry                     Whether or not it is desired to have geometries returned as part
                                            of the returned dataframe. Default is ``True``.
        -----------------------------       --------------------------------------------------------------------
        output_spatial_reference            If the geometry is being returned, and a geometry other
                                            than WGS84 is desired, please provide it here. The default is
                                            ``{'wkid': 4326}`` (WGS84).
        -----------------------------       --------------------------------------------------------------------
        estimate_credits                    If the source for the Business Analyst instance is ArcGIS Online,
                                            this enables estimation of credit consumption before actually performing the
                                            enrich task.
        =============================       ====================================================================

        Returns:
            Pandas Data Frame
        """
        from arcgis.geoenrichment.enrichment import NamedArea

        # pull out country specific parameters from the kwargs
        country, kwargs = extract_from_kwargs("country", kwargs)
        standard_geography_level, kwargs = extract_from_kwargs(
            "standard_geography_level", kwargs
        )
        standard_geography_id_column, kwargs = extract_from_kwargs(
            "standard_geography_id_column", kwargs
        )

        # if the geographies input is just a single geometry, convert to list to work with
        if isinstance(geographies, Geometry):
            geographies = [geographies]

        # if a dict is passed directly in, check if it is Geometry or a FeatureSet
        if isinstance(geographies, dict):
            if is_dict_geometry(geographies):
                geographies = Geometry(geographies)
            elif is_dict_featureset(geographies):
                geographies = FeatureSet(geographies)
            # dict of named areas
            elif isinstance(list(geographies.values())[0], NamedArea):
                named_areas = list(geographies.values())
                geographies = [named_area.geometry for named_area in named_areas]

        if isinstance(geographies, Iterable) and not isinstance(
            geographies, pd.DataFrame
        ):
            # if a list of geometries is passed in dict form, convert to Geometry objects
            if isinstance(geographies[0], dict):
                if is_dict_geometry(geographies[0]):
                    geographies = [Geometry(g_dict) for g_dict in geographies]

        # if a FeatureSet, convert to spatially enabled data frame
        if isinstance(geographies, FeatureSet):
            geographies = geographies.sdf

        # flag if a dataframe
        geo_is_df = isinstance(geographies, pd.DataFrame)

        # flag if is list of dictionaries, which allowed through untouched
        geo_is_dict = False
        if isinstance(geographies, Iterable):
            first_geo = (
                geographies.iloc[0]
                if isinstance(geographies, pd.DataFrame)
                else geographies[0]
            )
            if isinstance(first_geo, dict):
                geo_is_dict = True

        # check if a spatially enabled dataframe, if standard geography identifiers are not provided
        if geo_is_df and standard_geography_id_column is None and not geo_is_dict:
            assert geographies.spatial.validate(), (
                "If providing a Pandas DataFrame for enrichment, you must either "
                "ensure it is a valid spatially enabled dataframe or provide a "
                "column containing standard geography identifiers."
            )

        # if a dataframe and the geography id column is provided, pluck out the standard geography identifiers
        elif geo_is_df and standard_geography_id_column and not geo_is_dict:
            geographies = geographies[standard_geography_id_column]

        if geo_is_df and output_spatial_reference is None:
            output_spatial_reference = geographies.spatial.sr
        elif geo_is_dict and output_spatial_reference is None:
            if "spatialReference" in first_geo:
                output_spatial_reference = first_geo["spatialReference"]
        # ensure if specifying a standard geography id column, the standard geography level is also provided
        if standard_geography_id_column is not None:
            assert standard_geography_level is not None, (
                "If providing a standard_geography_id_column, you also need "
                "to provide a standard_geography_level."
            )
            assert (
                country is not None
            ), "If using a standard geography, you must also provide a country."

        # make sure the standard geography level is either an integer index or recognized string
        if isinstance(standard_geography_level, int):
            msg_cntry_idx = (
                "If providing an integer index for the standard_geography_level, it must be within "
                "the available range."
            )
            assert (
                standard_geography_level in country.geography_levels.index
            ), msg_cntry_idx

        elif isinstance(standard_geography_level, str):
            # crawl through all the available columns looking for a match and retrieve the index
            lvl_mtch = False
            for c in country.geography_levels.columns:
                std_geo_lwr = standard_geography_level.lower()
                mtch_srs = (
                    country.geography_levels[c].str.lower().str.contains(std_geo_lwr)
                )
                if mtch_srs.any():
                    lvl_mtch = True
                    standard_geography_level = country.geography_levels[mtch_srs].index[
                        0
                    ]
                    break

            # make sure the geography level was found
            assert lvl_mtch, (
                "If providing a standard_geography_level, it must match one of the available "
                "geography levels."
            )

        # if the geographies is not a path and not a dataframe, convert the iterable to a list for consistency later
        if not isinstance(geographies, (pd.DataFrame, Path)) and not isinstance(
            geographies, str
        ):
            geographies = list(geographies)

        # get enrichment variables to validate against depending on the enrichment variable source
        ev_df = (
            country.enrich_variables if country is not None else self.enrich_variables
        )

        # if no variables submitted, provide defaults flexibly based on the parent
        if enrich_variables is None:
            # pluck out enrich variables into a shorter variable
            ev = self.enrich_variables

            # get the current year key variables
            enrich_variables = (
                ev[
                    (ev.name.str.lower().str.contains("cy"))
                    & (ev.data_collection.str.lower().str.contains("key"))
                ]
                .drop_duplicates("name")
                .reset_index(drop=True)
            )

            # ensure something is found, dropping current year if nothing found
            if len(enrich_variables.index) == 0:
                enrich_variables = (
                    ev[(ev.data_collection.str.lower().str.contains("key"))]
                    .drop_duplicates("name")
                    .reset_index(drop=True)
                )

        # if a list of enrichment variables was provided, ensure they are valid
        if not isinstance(enrich_variables, pd.DataFrame):
            # iteratively go through all the columns and try to find matching variables
            for c in ev_df:
                enrich_vars = [v.lower() for v in enrich_variables]
                e_df = ev_df[ev_df[c].str.lower().isin(enrich_vars)].reset_index(
                    drop=True
                )
                if len(e_df.index):
                    break

            # pitch a fit if no variables were found
            if len(e_df.index) == 0:
                raise ValueError(
                    "None of the provided enrich_variables appear to be available."
                )

            # provide a message if not all enrich_variables were found
            if len(enrich_variables) > len(e_df.index):
                warn(
                    f"Not all input enrich_variables were matched ({len(e_df.index):,}/{len(enrich_variables):,})"
                )

        # set the default proximity metric if none provided
        if proximity_metric is None:
            proximity_metric = "kilometers"

        e_df = self._enrich(
            geographies,
            enrich_variables,
            country,
            standard_geography_level,
            proximity_type,
            proximity_value,
            proximity_metric,
            return_geometry,
            output_spatial_reference,
            estimate_credits,
            **kwargs,
        )

        return e_df

    @local_vs_gis
    def _enrich(
        self,
        geographies: Union[pd.DataFrame, Geometry, Iterable, Path],
        enrich_variables: pd.DataFrame,
        country: Optional[Country] = None,
        standard_geography_level: Optional[Union[int, str]] = None,
        proximity_type: str = "straight_line",
        proximity_value: Union[float, int] = 1,
        proximity_metric: str = "Kilometers",
        return_geometry: bool = True,
        output_spatial_reference: Union[int, dict, SpatialReference] = 4326,
        estimate_credits: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Redirect enabling all the front end validation to be handled in the non-underscore function."""
        pass

    def _enrich_local(
        self,
        geographies: Union[pd.DataFrame, Geometry, Iterable, Path],
        enrich_variables: pd.DataFrame,
        country: Optional[Country] = None,
        standard_geography_level: Optional[Union[int, str]] = None,
        proximity_type: Optional[str] = None,
        proximity_value: Optional[Union[float, int]] = None,
        proximity_metric: Optional[str] = None,
        return_geometry: bool = True,
        output_spatial_reference: Union[int, dict, SpatialReference] = 4326,
        estimate_credits: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Local enrich implementation."""
        assert estimate_credits is False, (
            "Cannot estimate credits for local analysis since using a local source does "
            "not use credits."
        )

        # if unsupported local analysis parameters provided, error
        unsupported_params = [
            "comparison_levels",
            "add_derivative_variables",
            "intersecting_geographies",
        ]
        invalid_params = [k for k in kwargs.keys() if k in unsupported_params]
        if len(invalid_params):
            raise ValueError(
                f"{', '.join(invalid_params)} input parameters are not supported with local analysis."
            )

        # also, raw json input is also not supported
        if isinstance(geographies, (Iterable, pd.Series)) and not isinstance(
            geographies, pd.DataFrame
        ):
            if not isinstance(geographies[0], Geometry) and isinstance(
                geographies[0], dict
            ):
                raise ValueError(
                    "Raw JSON is not supported as input for geoenrichment when using a 'local' source."
                )

        # late import to avoid collisions
        import arcpy

        # ensure no conflicts with intermediate datasets
        arcpy.env.overwriteOutput = True

        # get the variables property formatted for the Enrich Layer tool
        if country is not None:
            evars = self._enrich_variable_preprocessing(
                enrich_variables, country=country
            )
        else:
            evars = self._enrich_variable_preprocessing(enrich_variables)

        # if the standard geographies were provided
        if standard_geography_level is not None:
            # if input is a dataframe, ensure correctly set up
            if isinstance(geographies, pd.DataFrame):
                std_geo_in = get_spatially_enabled_dataframe(
                    geographies
                ).spatial.to_featureclass("memory/tmp_in_geo")

            # if input is any other iterable or Series, convert to comma separated string
            elif isinstance(geographies, (Iterable, pd.Series)):
                std_geo_in = ", ".join(list(geographies))

            # convert the standard geography to the correct string for use in later geoprocessing
            standard_geography_level = country.geography_levels.iloc[
                standard_geography_level
            ]["level_id"]

            # retrieve the standard geographies based on the identifiers
            geographies = arcpy.ba.StandardGeographyTA(
                geography_level=standard_geography_level,
                out_feature_class="memory/tmp_std_geo",
                input_type="LIST",
                ids_list=std_geo_in,
            )[0]

            # convert to spatially enabled dataframe
            geographies = GeoAccessor.from_featureclass(geographies)

        # otherwise, make sure input in consistent format
        else:
            geographies = get_spatially_enabled_dataframe(geographies)

            # if z-enabled, de-enable so enrich can work...conversion does not work with z-enabled features
            if geographies[geographies.spatial.name].iloc[0].has_z:
                geographies[geographies.spatial.name] = geographies[
                    geographies.spatial.name
                ].apply(lambda geom: Geometry(geom.__geo_interface__))

        # if a proximity type is provided, validate
        if proximity_type is not None:
            proximity_type = validate_network_travel_mode(
                country, proximity_type, proximity_metric
            )

            # provide defaults if nothing provided for any of the proximity metrics
            proximity_type = (
                "Straight Line" if proximity_type is None else proximity_type
            )
            proximity_metric = (
                "kilometers" if proximity_metric is None else proximity_metric
            )
            proximity_value = 1 if proximity_value == 1 else proximity_value

            # ensure if the geometry is lines, the proximity_type is not a network travel mode
            if (
                geographies.spatial.geometry_type[0] == "polyline"
                and proximity_type is not None
            ):
                assert (
                    proximity_type == "Straight Line"
                ), 'Only "Straight Line" proximity_type can be used with line geometry.'

        # handle pre 2.9
        in_geo = (
            geographies
            if pro_at_least_version("2.9")
            else geographies.spatial.to_featureclass(
                f"memory/geo_in_{uuid.uuid4().hex}"
            )
        )

        # now, actually perform enrichment
        use_arrow = pro_at_least_version("3.1") and self._can_use_arrow(in_geo)

        if use_arrow:
            enrich_res = self._enrich_using_arrow(in_sedf=in_geo, variables=evars)
        else:
            enrich_res = arcpy.ba.EnrichLayer(
                in_features=in_geo,
                out_feature_class=f"memory/tmp_enrich_{uuid.uuid4().hex}",
                variables=evars,
                buffer_type=proximity_type,
                distance=proximity_value,
                unit=proximity_metric,
            )

        # handle differences in pre 2.9
        if not pro_at_least_version("2.9"):
            arcpy.management.Delete(in_geo)
            enrich_res = enrich_res[0]

        # cache output depending on what the input was
        enrich_res = (
            enrich_res if isinstance(geographies, pd.DataFrame) else enrich_res[0]
        )

        # convert the output to a spatially enabled dataframe if necessary (in 3.1 it's done in
        enrich_df = (
            enrich_res if use_arrow else GeoAccessor.from_featureclass(enrich_res)
        )

        if not use_arrow:
            # in some cases from_featureclass returns a data frame that doesn't pass SEDF validation (enrich_df.spatial.validate)
            enrich_df.spatial.set_geometry("SHAPE")

        # standardize columns to ensure results are as expected
        enrich_df.columns = [
            self._standardize_enrich_column_name(c, country) for c in enrich_df.columns
        ]

        column_candidates_to_remove = ["Shape_Area", "Shape_Length"]
        # default value set to True to keep backward compatibility
        sanitize_columns = kwargs.pop("sanitize_columns", True)
        if sanitize_columns:
            enrich_df.columns = [
                pep8ify(c) for c in enrich_df.columns if c != "SHAPE"
            ] + ["SHAPE"]
            column_candidates_to_remove = [
                pep8ify(c) for c in column_candidates_to_remove
            ]

        # start creating a list of columns to remove - beginning with the OBJECTID field
        drop_cols = [c for c in enrich_df.columns if c in column_candidates_to_remove]

        if not use_arrow:
            if sanitize_columns:
                drop_cols.append(pep8ify(arcpy.Describe(enrich_res).OIDFieldName))
            else:
                drop_cols.append(arcpy.Describe(enrich_res).OIDFieldName)

        if not use_arrow:
            # get rid of the temporary output to save memory
            arcpy.management.Delete(enrich_res)

        # if returning geometry
        if return_geometry:
            # ensure the spatial reference is correct

            if output_spatial_reference:
                enrich_df = change_spatial_reference(
                    enrich_df, output_spatial_reference
                )

            # removing unneeded columns - using inplace to preserve all spatial namespace properties
            enrich_df.drop(columns=drop_cols, inplace=True)

        # if not returning geometry, drop the geometry column
        else:
            drop_cols.append(enrich_df.spatial.name)

            # remove unneeded columns - not doing inplace to ensure no 'spatial' namespace remnants
            enrich_df = enrich_df.drop(columns=drop_cols)

        return enrich_df

    def _enrich_gis(
        self,
        geographies: Union[pd.DataFrame, Geometry, Iterable, Path],
        enrich_variables: pd.DataFrame,
        country: Optional[Country] = None,
        standard_geography_level: Optional[Union[int, str]] = None,
        proximity_type: Optional[str] = None,
        proximity_value: Optional[Union[float, int]] = None,
        proximity_metric: Optional[str] = None,
        return_geometry: bool = True,
        output_spatial_reference: Union[int, dict, SpatialReference] = None,
        estimate_credits: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Web GIS implementation for _enrich"""
        # before going any further, make sure can enrich using current user (if any)
        if self.source.users.me is not None:
            has_ge = (
                len(
                    [p for p in self.source.users.me.privileges if "geoenrichment" in p]
                )
                > 0
            )
        elif (
            "appInfo" in self.source.properties
            and self.source.properties.appInfo.privileges
            and "premium:user:geoenrichment"
            in self.source.properties.appInfo.privileges
        ):
            has_ge = True
        else:
            has_ge = False
        assert has_ge, (
            "Cannot enrich due to insufficient permissions. Please ensure the GIS object instance is "
            "created with credentials, a user, with permissions to perform geoenrichment."
        )

        # TODO: implement estimate credits
        if estimate_credits:
            raise NotImplementedError("estimate_credits not yet implemented")

        # pitch a fit with a path
        if isinstance(geographies, Path):
            raise ValueError(
                "Providing a direct path to a table or feature class for enrichment is only supported "
                "when using a local ArcGIS Pro source."
            )

        # best practices batch size
        batch_size = 50

        # construct geoenrichment enrich method url
        ge_url = f"{self._base_url}/Geoenrichment/Enrich"

        # get the enrichment variables as a string ready to submit as a payload parameter
        evars = self._enrich_variable_preprocessing(enrich_variables, country=country)

        if output_spatial_reference is None:
            output_spatial_reference = 4326
        # properly format the output spatial reference
        if isinstance(output_spatial_reference, (int, str)):
            output_spatial_reference = SpatialReference(output_spatial_reference)

        # start building out the package for enrich REST call
        params = {
            "f": "json",
            "analysisVariables": evars,
            "appID": "esripythonapi",
        }

        # if any legacy parameters provided, add them to the payload
        legacy_params = [
            "comparison_levels",
            "add_derivative_variables",
            "intersecting_geographies",
        ]
        for param in [k for k in kwargs.keys() if k in legacy_params]:
            # convert the parameter key from snake case to camel case
            prts = param.split("_")
            param_cml = prts[0] + "".join(prt.title() for prt in prts[1:])

            # tack onto the payload
            params[param_cml] = kwargs[param]

        # if working with a specific country, add this to the payload
        if country is not None:
            hierarchy = kwargs.pop("hierarchy", country.properties.hierarchy[0])
            params["useData"] = json.dumps(
                {"sourceCountry": country.properties.iso3, "hierarchy": hierarchy}
            )

        # get the maximum batch size to ensure is not less than best practices set above
        svc_lmt_url = f'{self.source.properties.helperServices("geoenrichment").url}/Geoenrichment/ServiceLimits'
        svc_lmt_res = self.source._con.get(svc_lmt_url)

        max_batch_size = [
            v["value"]
            for v in svc_lmt_res["serviceLimits"]["value"]
            if v["paramName"] == "maximumStudyAreasNumber"
        ][0]
        batch_size = batch_size if max_batch_size > batch_size else max_batch_size

        # if a string, or list of strings, and no standard geography level is provided, format as address in JSON
        if isinstance(geographies, str) and standard_geography_level is None:
            geographies = [geographies]
        if isinstance(geographies, Iterable) and not isinstance(
            geographies, pd.DataFrame
        ):
            if isinstance(geographies[0], str) and standard_geography_level is None:
                geographies = [
                    {"address": {"text": addr_str}} for addr_str in geographies
                ]

        # convert boolean to string for payload in correct circumstances
        if return_geometry:
            retrieve_geometry = True
            params["returnGeometry"] = "true"
        else:
            retrieve_geometry = False
            params["returnGeometry"] = "false"

        # list to store request parameter payloads
        req_param_lst = []

        # detect and flag if a list of Geometry. String and dictionary objects passed directly in as iterable
        is_dict = False
        is_geom = False
        is_str = False
        if isinstance(geographies, list):
            if isinstance(geographies[0], Geometry):
                is_geom = True
            elif isinstance(geographies[0], dict):
                is_dict = True
            if isinstance(geographies[0], str):
                is_str = True

        # if working with standard geography
        if standard_geography_level is not None:
            assert (
                country is not None
            ), "Standard geography levels can only be used with a Country."

            # pull the geography level out of the geography levels dataframe
            geo_lvl = country.geography_levels.iloc[standard_geography_level][
                "level_id"
            ]

            # use the count of features and the max bach size to create a list of param payloads
            for idx in range(0, len(geographies), batch_size):
                # peel off just the id's for this batch
                batch_id_lst = geographies[idx : idx + batch_size]

                # create the param payload
                params["studyAreas"] = json.dumps(
                    [
                        {
                            "sourceCountry": country.iso3,
                            "layer": geo_lvl,
                            "ids": batch_id_lst,
                        }
                    ]
                )

                # add the payload to the list
                req_param_lst.append(deepcopy(params))

        # if a list of dictionaries, which are not geometries, or a list of strings is being passed in, just send
        elif (is_dict and not is_geom) or is_str:
            if proximity_value is not None:
                prx_src = self if country is None else country
                geographies = add_proximity_to_enrich_feature_list(
                    prx_src,
                    geographies,
                    proximity_type,
                    proximity_metric,
                    proximity_value,
                )
            for idx in range(0, len(geographies), batch_size):
                geo_btch = geographies[idx : idx + batch_size]
                params["studyAreas"] = json.dumps(geo_btch) if is_dict else geo_btch
                req_param_lst.append(deepcopy(params))

        # otherwise, working with geometries, so do this thing
        else:
            # no matter what the input, get a spatially enabled dataframe
            geographies = get_spatially_enabled_dataframe(geographies)

            # tack on the spatial reference to make sure it comes along for the ride
            params["insr"] = json.dumps(geographies.spatial.sr)

            # tack on the output spatial reference as well
            params["outsr"] = json.dumps(output_spatial_reference)

            # check to make sure a valid geometry is present
            geom_typ_lst = [
                gt for gt in geographies.spatial.geometry_type if isinstance(gt, str)
            ]
            assert len(
                geom_typ_lst
            ), "The Dataframe does not appear to have a valid geometry type."

            # make sure there is a unique identifier for combining data later if more than just geometries provided
            if isinstance(geographies, pd.DataFrame):
                geographies["enrich_idx"] = geographies.index

            # use the count of features and the max bach size to create a list of param payloads
            total_cnt = (
                len(geographies.index)
                if isinstance(geographies, pd.DataFrame)
                else len(geographies)
            )
            for idx in range(0, total_cnt, batch_size):
                # get a slice of the input data to enrich, converting to feature set if dataframe
                if isinstance(geographies, pd.DataFrame):
                    in_batch_df = geographies.iloc[idx : idx + batch_size]

                    # format the features for sending - keep it light, just the geometry
                    batch_df = in_batch_df[[in_batch_df.spatial.name, "enrich_idx"]]
                    batch_df.spatial.set_geometry(geographies.spatial.name)
                    batch_features = batch_df.spatial.to_featureset().features
                    feature_lst = [f.as_dict for f in batch_features]

                    # if proximity metrics provided, add the proximity metrics to the list of features
                    if proximity_value is not None:
                        prx_src = self if country is None else country
                        feature_lst = add_proximity_to_enrich_feature_list(
                            prx_src,
                            feature_lst,
                            proximity_type,
                            proximity_metric,
                            proximity_value,
                        )

                # ...and if already list of dict objects, split into parts
                else:
                    feature_lst = geographies[idx : idx + batch_size]

                # convert everything to string for payload
                params["studyAreas"] = json.dumps(feature_lst)

                # add the payload onto the list
                req_param_lst.append(deepcopy(params))

        # bach request asynchronously
        enrich_res_df = run_async(
            _get_enrich_rest, ge_url, req_param_lst, retrieve_geometry, self.source
        )

        # clean up the response dataframe schema
        drop_cols = [c for c in enrich_res_df.columns if "objectid" in c.lower()] + [
            "ID"
        ]
        enrich_res_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # if more than just a list of standard geography id's was the input, combine the input with the results
        if isinstance(geographies, pd.DataFrame):
            src_drop_cols = [geographies.spatial.name] + [
                c for c in geographies.columns if c.lower().startswith("shape_")
            ]
            enrich_df = enrich_res_df.join(
                geographies.drop(columns=src_drop_cols).set_index("enrich_idx"),
                on="enrich_idx",
            ).drop(columns="enrich_idx")
            cols_in_order = [
                c for c in geographies.columns if c not in src_drop_cols
            ] + list(enrich_res_df.columns)
            enrich_df = enrich_df[[c for c in cols_in_order if c != "enrich_idx"]]
        else:
            enrich_df = enrich_res_df

        # free up memory
        del enrich_res_df

        # if geometry, which may not be, but if there is, clean up and make sure everything is as expected
        if "SHAPE" in enrich_df.columns:
            # ensure it is valid to begin with - doubtful after the join
            enrich_df.spatial.set_geometry("SHAPE")

            # shuffle columns so geometry is at the end
            enrich_df = enrich_df[
                [c for c in enrich_df.columns if c != "SHAPE"] + ["SHAPE"]
            ]

            # set the geometry
            enrich_df.spatial.set_geometry("SHAPE")

        # proactively change the column names so no surprises if exporting to a feature class later
        enrich_df.columns = [
            self._standardize_enrich_column_name(c, country) for c in enrich_df.columns
        ]

        # default value set to True to keep backward compatibility
        sanitize_columns = kwargs.pop("sanitize_columns", True)
        if sanitize_columns:
            enrich_df.columns = [
                pep8ify(c) if c != "SHAPE" else c for c in enrich_df.columns
            ]

        # stash useful pieces for potential later access in metadata
        enrich_df.attrs["arcgis_ba"] = self
        enrich_df.attrs["arcgis_aoi"] = country

        return enrich_df

    @lru_cache(maxsize=255)
    def _standardize_enrich_column_name(
        self, column_name: str, country: Optional[Country] = None
    ):
        """Helper function to standardize the output column names so is the same no matter the source."""
        std_src = self if country is None else country
        col_nm = std_src.get_enrich_variables_from_iterable(
            column_name, suppress_warn=True
        )
        col_nm = col_nm.iloc[0]["name"] if len(col_nm.index) > 0 else column_name
        return col_nm


async def _get_enrich_rest(
    ge_url: str, payload_lst: Iterable[dict], retrieve_geometry: bool, source: GIS
) -> Awaitable[pd.DataFrame]:
    """Function enabling batching of enrich rest call asynchronously."""
    # variable for storing results
    enrich_res_itr = []

    # iterate the batched payloads
    for payload in payload_lst:
        # create an event loop for wrapping requests
        loop = asyncio.get_event_loop()

        # get a listener, a future object, and send request to the server
        future = loop.run_in_executor(None, source._con.post, ge_url, payload)

        # hold short for response (but since using async, other requests get queued up)
        res = await future

        # ensure a valid result is received
        if "error" in res["messages"]:
            err = res["messages"]["error"]
            raise Exception(
                "Error in enriching data using Business Analyst Enrich REST endpoint - Error "
                f'Code {err["code"]}: {err["message"]}'
            )

        # pull out the response feature set
        fs = res["results"][0]["value"]["FeatureSet"]
        assert (
            len(fs) > 0
        ), "No results were returned. Please ensure you are using the correct country."

        # if getting geometry back, unpack into spatially enabled dataframe
        if retrieve_geometry:
            r_df = FeatureSet.from_dict(fs[0]).sdf

        # unpack the enriched results - reaching into the FeatureSet for just the attributes - much faster
        else:
            r_df = pd.DataFrame([f["attributes"] for f in fs[0]["features"]])

        # add the dataframe to the list
        enrich_res_itr.append(r_df)

    # combine all the received enriched data and take out the trash
    res_df = pd.concat(enrich_res_itr).reset_index(drop=True)
    del enrich_res_itr

    return res_df
