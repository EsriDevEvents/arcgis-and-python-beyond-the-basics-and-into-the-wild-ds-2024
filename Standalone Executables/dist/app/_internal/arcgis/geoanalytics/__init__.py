"""
The arcgis.geoanalytics module provides types and functions for distributed analysis of large datasets.
These GeoAnalytics tools work with big data registered in the GISs datastores as well as with feature layers.

Use arcgis.geoanalytics.is_analysis_supported(gis) to check if geoanalytics is supported in your GIS.

Note: GeoAnalytics operations use the following context parameters defined in the `arcgis.env` module:

        =========================     ====================================================================
        **Context parameter**         **Description**
        -------------------------     --------------------------------------------------------------------
        out_spatial_reference         Used for setting the output spatial reference
        -------------------------     --------------------------------------------------------------------
        process_spatial_reference     Used for setting the processing spatial reference.
        -------------------------     --------------------------------------------------------------------
        analysis_extent               Used for setting the analysis extent.
        -------------------------     --------------------------------------------------------------------
        output_datastore              Used for setting the output datastore to be used.
        =========================     ====================================================================
"""

from typing import Optional, Union

from arcgis.gis import GIS, Datastore
from . import (
    summarize_data,
    analyze_patterns,
    use_proximity,
    manage_data,
    find_locations,
    data_enrichment,
)


def get_datastores(gis: Optional[GIS] = None):
    """
    Returns a helper object to manage geoanalytics datastores in the GIS.
    If a :class:`gis <arcgis.gis.GIS>` isn't specified, returns a datastore
    manager for the active gis (`arcgis.env.active_gis`).  If the active gis
    has not been configured with a GeoAnalytics Server, the function returns
    `None`.
    """
    import arcgis

    gis = arcgis.env.active_gis if gis is None else gis

    for ds in gis._datastores:
        if "GeoAnalytics" in ds._server["serverFunction"]:
            return ds

    return None


def define_output_datastore(
    datastore: Optional[Union[str, Datastore]] = None, template: Optional[str] = None
):
    """
    Sets the `arcgis.env.output_datastore` by providing the datastore and template name
    to this method. If datastore is None, the `arcgis.env.output_datastore` will reset
    to default.

    ==========================   ===============================================================
    **Parameter**                 **Description**
    --------------------------   ---------------------------------------------------------------
    datastore                    Optional Datastore/String. This specifies the big data file
                                 share to save GeoAnalyticss results to. If specified as None the
                                 `arcgis.env.output_datastore` will reset to default.  Allowed
                                 string values are: `spatiotemporal` or `relational`.
    --------------------------   ---------------------------------------------------------------
    template                     Optional string. When specified, the `template` determines how
                                 GeoAnalytics result schema will be formatted. The output will
                                 be written to a file in the big data file share.
    ==========================   ===============================================================

    :return: Boolean with True indicating success

    """
    import arcgis

    if isinstance(datastore, str) and str(datastore).lower() in [
        "spatiotemporal",
        "relational",
    ]:
        arcgis.env.output_datastore = str(datastore).lower()
        return True
    elif isinstance(datastore, str) and not str(datastore).lower() in [
        "spatiotemporal",
        "relational",
    ]:
        raise ValueError("datastore can only ")
    elif datastore and template:
        if isinstance(datastore, arcgis.gis.Datastore):
            arcgis.env.output_datastore = "{path}:{template}".format(
                path=datastore.path, template=template
            )
        elif isinstance(datastore, str):
            arcgis.env.output_datastore = "{path}:{template}".format(
                path=datastore, template=template
            )
        elif isinstance(datastore, arcgis.gis.server.Datastore):
            path = datastore.properties.path
            arcgis.env.output_datastore = "{path}:{template}".format(
                path=datastore, template=template
            )
        else:
            raise ValueError("Invalid Datastore")
        return True
    elif template and datastore is None:
        raise ValueError("datastore must be specified to set an output template.")
    elif datastore and template is None:
        raise ValueError("template must be specified to set the output_datastore.")
    else:
        arcgis.env.output_datastore = None
        return True


def is_supported(gis: Optional[GIS] = None):
    """
    Returns True if the GIS supports geoanalytics. If a gis isn't specified,
    checks if arcgis.env.active_gis supports geoanalytics
    """
    import arcgis

    gis = arcgis.env.active_gis if gis is None else gis
    if (
        "geoanalytics" in gis.properties.helperServices
        and gis._portal.is_arcgisonline == False
    ):
        return True
    else:
        return False
