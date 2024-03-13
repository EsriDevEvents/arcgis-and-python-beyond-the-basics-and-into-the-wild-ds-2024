from . import hub
from . import workforce
from . import storymap
from . import survey123
from . import tracker
from . import dashboard
from . import expbuilder

from ._url_schemes import build_collector_url
from ._url_schemes import build_field_maps_url
from ._url_schemes import build_explorer_url
from ._url_schemes import build_navigator_url
from ._url_schemes import build_survey123_url
from ._url_schemes import build_tracker_url
from ._url_schemes import build_workforce_url

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
