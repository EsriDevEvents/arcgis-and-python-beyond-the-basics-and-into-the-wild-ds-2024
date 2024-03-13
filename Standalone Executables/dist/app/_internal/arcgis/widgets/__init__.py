try:
    from arcgis.widgets._mapview import MapView
except ImportError as e:
    import logging

    log = logging.getLogger()
    import_error = e

    class MapView:
        def __init__(self, *args, **kwargs):
            log.warning(
                "MapView class replaced with a non-functional "
                "placeholder due to the following import error:"
            )
            raise import_error
