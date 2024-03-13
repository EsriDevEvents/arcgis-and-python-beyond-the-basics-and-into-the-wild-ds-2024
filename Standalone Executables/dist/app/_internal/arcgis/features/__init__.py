"""
The ``arcgis.features`` module contains types and functions for working with features and feature layers in the
:class:`~arcgis.gis.GIS` .

Entities located in space with a geometrical representation (such as points, lines or polygons) and a set of properties
can be represented as features. The ``arcgis.features`` module is used for working with feature data, feature layers and
collections of feature layers in the GIS. It also contains the spatial analysis functions which operate against
feature data.

In the ``GIS``, entities located in space with a set of properties can be represented as features.  Features are stored
as feature classes, which represent a set of features located using a single spatial type (point, line, polygon) and a
common set of properties.  This is the geographic extension of the classic tabular or relational representation for
entities - a set of entities is modelled as rows in a table.  Tables represent entity classes with uniform
properties.  In addition to working with entities with location as features, the system can also work with
non-spatial entities as rows in tables.  The system can also model relationships between entities using properties
which act as primary and foreign keys.  A collection of feature classes and tables, with the associated
relationships among the entities, is a feature layer collection. :class:`~arcgis.features.FeatureLayerCollection`
are one of the dataset types contained in a :class:`~arcgis.gis.Datastore`.

.. note::
    Features are not simply entities
    in a dataset.  Features have a visual representation and user experience - on a map, in a 3D scene,
    as entities with a property sheet or popups.
"""

from .feature import Feature, FeatureSet, FeatureCollection
from .layer import FeatureLayer, Table, FeatureLayerCollection
from ._parcel import ParcelFabricManager
from ._utility import UtilityNetworkManager
from ._validation import ValidationManager
from ._trace_configuration import TraceConfiguration
from . import analyze_patterns
from . import enrich_data
from . import find_locations
from . import manage_data
from . import summarize_data
from . import use_proximity
from . import analysis
from . import elevation
from . import hydrology


__all__ = [
    "Feature",
    "FeatureSet",
    "FeatureCollection",
    "FeatureLayer",
    "Table",
    "FeatureLayerCollection",
    "UtilityNetworkManager",
    "ValidationManager",
    "ParcelFabricManager",
    "TraceConfiguration",
]
try:
    from .geo import GeoAccessor, GeoSeriesAccessor

    __all__.extend(["GeoAccessor", "GeoSeriesAccessor"])
except ImportError:
    pass

try:
    from .geo._dask import GeoDaskSeriesAccessor, GeoDaskSpatialAccessor

    __all__.extend(["GeoDaskSeriesAccessor", "GeoDaskSpatialAccessor"])
except:
    pass
