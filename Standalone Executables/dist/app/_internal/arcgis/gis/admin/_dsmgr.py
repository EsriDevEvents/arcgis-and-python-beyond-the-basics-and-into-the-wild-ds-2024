from __future__ import annotations
import logging
from enum import Enum
from typing import Any
from functools import lru_cache
from arcgis.auth.tools import LazyLoader

_arcgis = LazyLoader("arcgis")
requests = LazyLoader("requests")
_dt = LazyLoader("datetime")

logger = logging.getLogger(__name__)

__all__ = [
    "DataStoreMetric",
    "DataStoreTimeUnit",
    "DataStoreAggregation",
    "DataStoreMetricsManager",
]


class DataStoreMetric(Enum):
    """The allowed metric values"""

    # Average CPU Time
    AVG_CPU = "db_avg_cpu"
    # Storage Size in MegaBytes
    STORAGE_MB = "db_used_storage_mb"
    # Allocated Storage Size in MegaBytes
    ALLOCATED_STORAGE = "db_allocated_storage_mb"
    # Percent of Total Storage
    PERCENT_STORAGE = "db_percent_storage"
    # Size of Feature Storage in MegaBytes
    FEATURESTORAGE = "db_featurestorage_mb"


class DataStoreTimeUnit(Enum):
    """Time Units for DataStore Metrics class"""

    # a period of twenty-four hours as a unit of time
    DAY = "d"
    # a period of time equal to a twenty-fourth part of a day and night and divided into 60 minutes.
    HOUR = "h"
    # a period of time equal to sixty seconds or a sixtieth of an hour.
    MINUTE = "m"
    #  a unit of time in the International System of Units (SI), historically defined as 1/86400 of a day
    SECOND = "s"
    # a period of time equal to one thousandth of a second.
    MILLISECOND = "ms"


class DataStoreAggregation(Enum):
    # a number expressing the central or typical value in a set of data
    AVG = "avg"
    # determine the total number of (a collection of items).
    COUNT = "count"
    # a maximum amount
    MAX = "max"
    # a minimum amount
    MIN = "min"
    # a quantity calculated to indicate the extent of deviation for a group as a whole.
    STDEV = "stdev"
    # the total amount resulting from the addition of two or more numbers, amounts, or items.
    SUM = "sum"
    # a statistical measurement of the spread between numbers in a data set
    VARIANCE = "variance"


class DataStoreMetricsManager:
    """
    This class allows for ArcGIS Online administrators to query statistics about the
    managed datastore. It is not meant to be initialized directly, but instead an
    instance is returned from the :attr:`~arcgis.gis.admin.AGOLAdminManager.datastore_metrics`
    property.

    .. code-block:: python

        # Usage Example;
        >>> gis = GIS(profile="your_online_admin_profile")

        >>> ago_mgr = gis.admin
        >>> ds_mgr = ago_mgr.datastore_metrics
    """

    _gis: _arcgis.gis.GIS | None = None
    _orgid: str | None = None
    _session: _arcgis.auth.EsriSession | None = None
    _base_url: str = None

    # ---------------------------------------------------------------------
    def __init__(self, gis: _arcgis.gis.GIS):
        self._gis = gis
        self._session = gis._con._session
        self._orgid = gis.properties.id
        self._base_url = gis.properties["helperServices"]["datastoreManagement"]["url"]

    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        return "< Datastore Metrics >"

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ---------------------------------------------------------------------
    def query_resource_usage(self, query_period: str) -> dict[str, list]:
        """
        Allows administrators to view the average and max CPU usage for the
        organization's datastore.

        ====================  =========================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------
        query_period          Required String. The time period to query. The allowed
                              values are: day, week, or hour.
        ====================  =========================================================

        :return:
           dictionary containing the average and max usage.
        """
        query_period = query_period.lower()
        assert query_period in [
            "day",
            "week",
            "hour",
        ], "`query_period` must be a value of `day`, `week` or `hour`."

        unit_lu: dict[str, str] = {
            "day": [
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.AVG.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.DAY.value,
                    "ago": 1,
                    "binSize": 5,
                },
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.MAX.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.DAY.value,
                    "ago": 1,
                    "binSize": 5,
                },
            ],
            "week": [
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.AVG.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.DAY.value,
                    "ago": 7,
                    "binSize": 30,
                },
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.MAX.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.DAY.value,
                    "ago": 7,
                    "binSize": 30,
                },
            ],
            "hour": [
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.AVG.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.HOUR.value,
                    "ago": 1,
                    "binSize": 1,
                },
                {
                    "metric": DataStoreMetric.AVG_CPU.value,
                    "binAggregation": DataStoreAggregation.MAX.value,
                    "binUnit": DataStoreTimeUnit.MINUTE.value,
                    "agoUnit": DataStoreTimeUnit.HOUR.value,
                    "ago": 1,
                    "binSize": 1,
                },
            ],
        }
        params: dict[str, Any] = {
            "f": "json",
        }
        url: str = f"{self._base_url}/{self._orgid}/datastores/featureDataStore"

        return_data: dict = {}
        for param in unit_lu[query_period]:
            params.update(param)
            resp: requests.Response = self._session.get(url=url, params=params)
            resp.raise_for_status()
            data: list = resp.json()
            if params["binAggregation"] == DataStoreAggregation.MAX.value:
                return_data["max"] = [
                    {
                        "ts": _dt.datetime.fromtimestamp(entry["ts"] / 1000.0),
                        "value": entry["value"],
                    }
                    for entry in data
                ]
            else:
                return_data["avg"] = [
                    {
                        "ts": _dt.datetime.fromtimestamp(entry["ts"] / 1000.0),
                        "value": entry["value"],
                    }
                    for entry in data
                ]
        return return_data

    # ---------------------------------------------------------------------
    @property
    def feature_storage(self) -> list[dict[str, Any]]:
        """
        Returns storage percentage of total storage used

        :returns: list[dict[str,Any]]

        """
        params: dict[str, Any] = {
            "f": "json",
            "metric": DataStoreMetric.FEATURESTORAGE.value,
            "ago": 1,
            "agoUnit": DataStoreTimeUnit.HOUR.value,
        }
        url: str = f"{self._base_url}/{self._orgid}/datastores/featureDataStore"
        resp: requests.Response = self._session.get(url=url, params=params)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json()
        return [
            {
                "ts": _dt.datetime.fromtimestamp(entry["ts"] / 1000.0),
                "value": int(round(entry["value"] / 500000, 2) * 100),
            }
            for entry in data
        ]

    # ---------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def query(
        self,
        metric: DataStoreMetric,
        bin_size: float | int,
        bin_unit: DataStoreTimeUnit,
        aggregation: DataStoreAggregation = DataStoreAggregation.SUM,
        start_time: _dt.datetime | None = None,
        end_time: _dt.datetime | None = None,
        ago: int | None = None,
        ago_unit: DataStoreTimeUnit | None = None,
    ) -> list[dict[str, Any]]:
        """
        A query operation used to gather metrics about the ArcGIS Online Datastore

        ====================  =========================================================
        **Parameter**         **Description**
        --------------------  ---------------------------------------------------------
        metric                Required DataStoreMetric. The statistical method to gather.
        --------------------  ---------------------------------------------------------
        bin_size              Required Float or Int. The size of the bin to aggregate on.
        --------------------  ---------------------------------------------------------
        bin_unit              Required DataStoreTimeUnit. The size of the bin.
        --------------------  ---------------------------------------------------------
        aggregation           Required DataStoreAggregation. The type of aggregation to perform.
        --------------------  ---------------------------------------------------------
        start_time            Optional datetime.datetime. The starting date point.
        --------------------  ---------------------------------------------------------
        end_time              Optional datetime.datetime. The ending date point.
        --------------------  ---------------------------------------------------------
        ago                   Optional Int. The time to look back from today.
        --------------------  ---------------------------------------------------------
        ago_unit              Optional DataStoreTimeUnit. The time unit to look back.
        ====================  =========================================================

        :returns: list[dict[str,Any]]
        """
        if ago is None and start_time is None and end_time is None:
            logger.warning("No ago was supplied, defaulting to ago=1")
            ago = 1
            ago_unit = DataStoreTimeUnit.DAY
        params: dict[str, Any] = {
            "f": "json",
        }
        assert isinstance(metric, DataStoreMetric)
        params["metric"] = metric.value
        if start_time:
            assert isinstance(
                start_time, _dt.datetime
            ), "`start_time` must be a datetime object"
            params["startMillis"] = int(start_time.timestamp() * 1000)
        if end_time:
            assert isinstance(
                end_time, _dt.datetime
            ), "`end_time` must be a datetime object"
            params["endMillis"] = int(end_time.timestamp() * 1000)
        if ago:
            assert isinstance(ago, int), "`ags` must be an integer."
            params["ago"] = ago
        if ago_unit:
            assert isinstance(
                ago_unit, DataStoreTimeUnit
            ), "`ago_unit` must be a DataStoreTimeUnit"
            params["agoUnit"] = ago_unit.value
        if bin_unit:
            bin_unit
            assert isinstance(
                bin_unit, DataStoreTimeUnit
            ), "`ago_unit` must be a DataStoreTimeUnit"
            params["binUnit"] = bin_unit.value
        if bin_size:
            assert isinstance(
                bin_size, (int, float)
            ), "`ago_unit` must be a float or int."
            params["binSize"] = bin_size
        if aggregation:
            assert isinstance(
                aggregation, DataStoreAggregation
            ), "`aggregation` must be a float or int."
            params["binAggregation"] = aggregation.value
        if start_time and end_time and ago:
            logger.warning(
                "Ignoring the `ago` parameter due to the presents of start and end times."
            )
            del params["ago"]
        url: str = f"{self._base_url}/{self._orgid}/datastores/featureDataStore"
        resp: requests.Response = self._session.get(url=url, params=params)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json()
        return [
            {
                "ts": _dt.datetime.fromtimestamp(entry["ts"] / 1000.0),
                "value": entry["value"],
            }
            for entry in data
        ]
