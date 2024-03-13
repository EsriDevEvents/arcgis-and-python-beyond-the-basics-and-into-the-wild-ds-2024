import os, copy
from typing import Dict, List, Any, Optional, Union, Iterator
import pandas as pd
from arcgis.gis import GIS
from arcgis.geometry import Geometry
from arcgis import env as _env
from arcgis._impl.common._isd import InsensitiveDict
from functools import lru_cache


###########################################################################
class OGCCollection:
    """
    Represents a single OGC dataset

    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The web address endpoint.
    ----------------  -------------------------------------------------------------------------------
    gis               Optional :class:`~arcgis.gis.GIS`. The connection object.
    ================  ===============================================================================

    """

    _gis = None
    _url = None
    _properties = None

    # ---------------------------------------------------------------------
    def __init__(self, url: str, gis: GIS = None) -> "OGCCollection":
        """Constructor"""
        assert (
            str(url).lower().find("ogcfeatureserver") > -1
            and os.path.basename(url).isdigit()
        )
        if gis is None:
            gis = _env.active_gis or GIS()
        self._gis = gis
        self._url = url

    # ---------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """returns the service properties"""
        params = {"f": "json"}
        try:
            res = self._gis._con.get(self._url, params)
            return InsensitiveDict(res)
        except:
            res = self._gis._con.post(self._url, params)
            return InsensitiveDict(res)

    # ---------------------------------------------------------------------
    def _process_row(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the GeoJSON Geometry to Esri JSON and format the row accordingly"""
        row = {"SHAPE": Geometry(feature["geometry"]) if feature["geometry"] else None}
        row.update(feature["properties"])
        return row

    # ---------------------------------------------------------------------
    def query(
        self,
        query: Optional[str] = None,
        limit: int = 10000,
        bbox: Optional[List[float]] = None,
        bbox_sr: Optional[int] = None,
        time_filter: Optional[str] = None,
        return_all: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Queries the :class:`~arcgis.mapping.ogc.OGCFeatureService` Layer and returns back the information as a Spatially Enabled DataFrame.

        ================  ===============================================================================
        **Parameter**      **Description**
        ----------------  -------------------------------------------------------------------------------
        query             Optional String. A SQL based query applied to the service.
        ----------------  -------------------------------------------------------------------------------
        limit             Optional Integer. The number of records to limit to.  The default is 10,000.
        ----------------  -------------------------------------------------------------------------------
        bbox              Optional List[float]. The bounding box to limit search in.
        ----------------  -------------------------------------------------------------------------------
        bbox_sr           Optional Integer. The coordinate reference system as a WKID.
        ----------------  -------------------------------------------------------------------------------
        time_filter       Optional String. The dates to filter time by.
        ================  ===============================================================================

        :return: Union[Dict[str, Any], pd.DataFrame]
        """
        url = f"{self._url}/items"
        params = {"f": "json"}
        if query:
            params["filter"] = query
        if isinstance(limit, int) and limit >= 0:
            params["limit"] = limit
        if bbox and bbox_sr:
            params["bbox"] = bbox
            params["bbox-crs"] = bbox_sr
        if kwargs.get("crs", None):
            params["crs"] = kwargs.pop("crs")
        params["offset"] = kwargs.pop("offset", 0)
        if time_filter:
            params["datetime"] = time_filter
        for k, v in kwargs.items():
            params[k] = v
        as_dict = kwargs.pop("as_dict", False)
        if as_dict == False:  # returns all records as sedf
            res = self._gis._con.get(url, params)
            if len(res["features"]) == 0:
                return pd.DataFrame([])
            results = [
                self._process_row(feature) for feature in res["features"] if feature
            ]
            params["offset"] += limit
            while True:
                res = self._gis._con.get(url, params)
                if res == {} or res["numberReturned"] == 0:
                    break
                elif return_all == False and len(results) >= limit:
                    results = results[:limit]
                    break
                elif res["numberReturned"] < limit:
                    r1 = [
                        self._process_row(feature)
                        for feature in res["features"]
                        if feature
                    ]
                    results.extend(r1)
                    break
                else:
                    r1 = [
                        self._process_row(feature)
                        for feature in res["features"]
                        if feature
                    ]
                    results.extend(r1)
                params["offset"] += limit
            df = pd.DataFrame(results)
            df.spatial.name
            return df
        else:
            results = self._gis._con.get(url, params)
            if len(results["features"]) == 0:
                return results
            res = copy.deepcopy(results)
            params["offset"] += limit
            while res["numberReturned"] > 0:
                res = self._gis._con.get(url, params)

                if res == {} or res["numberReturned"] == 0:
                    break

                results["features"].extend(res["features"])
                if return_all == False and len(results["features"]) >= limit:
                    results["features"] = results["features"][:limit]
                    break
                elif res["numberReturned"] < limit:
                    break
                params["offset"] += limit
            return results
        return {}

    # ---------------------------------------------------------------------
    def get(self, feature_id: int) -> Dict[str, Any]:
        """
        Gets an individual feature on the service. Needs to correspond
        to an id of the feature.

        :return: Dict[str, Any]
        """
        assert isinstance(feature_id, int)
        url = f"{self._url}/items/{feature_id}"
        params = {"f": "json"}
        return self._gis._con.get(url, params)


###########################################################################
class OGCFeatureService:
    """
    Represents the Hosted OGC Feature Server

    ================  ===============================================================================
    **Parameter**      **Description**
    ----------------  -------------------------------------------------------------------------------
    url               Required String. The web address endpoint.
    ----------------  -------------------------------------------------------------------------------
    gis               Optional :class:`~arcgis.gis.GIS`. The connection object.
    ================  ===============================================================================

    """

    _gis = None
    _url = None
    _properties = None

    # ---------------------------------------------------------------------
    def __init__(self, url, gis=None) -> "OGCFeatureService":
        """Constructor"""
        assert str(url).lower().endswith("ogcfeatureserver")
        if gis is None:
            gis = _env.active_gis or GIS()
        self._gis = gis
        self._url = url

    # ---------------------------------------------------------------------
    @property
    def properties(self) -> InsensitiveDict:
        """returns the service properties"""
        params = {"f": "json"}
        if self._properties is None:
            try:
                res = self._gis._con.get(self._url, params)
                self._properties = InsensitiveDict(res)
            except:
                res = self._gis._con.post(self._url, params)
                self._properties = InsensitiveDict(res)
        return self._properties

    # ---------------------------------------------------------------------
    @property
    @lru_cache(maxsize=100)
    def conformance(self) -> Dict[str, Any]:
        """
        Provides the API conformance with the OGC standard.

        :return: Dict[str, Any]
        """
        url = f"{self._url}/conformance"
        params = {"f": "json"}
        return self._gis._con.get(url, params)

    # ---------------------------------------------------------------------
    @property
    def collections(self) -> Iterator[OGCCollection]:
        """
        Yields all the OGC Feature Service Layers within the service.

        :return: Iterator[:class:`~arcgis.mapping.ogc.OGCCollection`]
        """
        url = f"{self._url}/collections"
        params = {"f": "json"}
        for idx, lyr in enumerate(self._gis._con.get(url, params)["collections"]):
            service_url = f"{url}/{lyr['id']}"
            yield OGCCollection(url=service_url, gis=self._gis)
