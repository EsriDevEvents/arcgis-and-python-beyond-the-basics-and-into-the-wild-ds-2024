try:
    import pandas as pd
    from arcgis.features.geo._accessor import GeoAccessor
    from arcgis.features.geo._accessor import GeoSeriesAccessor
    from arcgis.features.geo._accessor import _is_geoenabled

    __all__ = ["GeoAccessor", "GeoSeriesAccessor"]
except ImportError:
    pass
