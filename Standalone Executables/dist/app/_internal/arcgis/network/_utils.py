import json
import logging
from functools import lru_cache
from arcgis.gis import GIS
from arcgis import network
from arcgis._impl.common._utils import _validate_url

_log = logging.getLogger()


# -------------------------------------------------------------------------
def _gp_travel_mode(gis: GIS, travel_mode: str = None) -> str:
    """Calculates the travel mode via the GP Service"""
    output = network.analysis.get_travel_modes(gis=gis)
    if travel_mode is None:
        _log.warning("Travel mode not set, using default travel mode")
        travel_mode = output.default_travel_mode
    matches = [
        feature.attributes["TravelMode"]
        for feature in output.supported_travel_modes.features
        if (
            feature.attributes["TravelModeId"].lower() == travel_mode.lower()
            or feature.attributes["AltName"].lower() == travel_mode.lower()
            or feature.attributes["Name"].lower() == travel_mode.lower()
        )
    ]

    if len(matches) > 0:
        try:
            return json.loads(matches[0])
        except:
            return matches[0]
    else:
        _log.warning(
            f"Cannot find {travel_mode}, using default: {output.default_travel_mode}."
        )
        matches = [
            feature.attributes["TravelMode"]
            for feature in output.supported_travel_modes.features
            if (
                feature.attributes["TravelModeId"].lower()
                == output.default_travel_mode.lower()
                or feature.attributes["AltName"].lower()
                == output.default_travel_mode.lower()
                or feature.attributes["Name"].lower()
                == output.default_travel_mode.lower()
            )
        ]
        return matches[0]


# -------------------------------------------------------------------------
def _route_service_travel_modes(gis, travel_mode: str = None) -> str:
    """gets the default values from the routing service"""
    url = _validate_url(gis.properties.helperServices.route.url, gis)
    route_service = network.RouteLayer(url, gis=gis)
    modes = route_service.retrieve_travel_modes()
    if travel_mode is None:
        travel_mode = modes["defaultTravelMode"]
    fn = lambda tm: travel_mode.lower() in [tm["id"].lower(), tm["name"].lower()]
    res = list(filter(fn, modes["supportedTravelModes"]))
    if len(res) > 0:
        return json.dumps(res[0])
    else:
        travel_mode = modes["defaultTravelMode"]
        res = list(filter(fn, modes["supportedTravelModes"]))
        return json.dumps(res[0])


# -------------------------------------------------------------------------
@lru_cache(maxsize=10)
def find_travel_mode(gis: GIS, travel_mode: str = None) -> str:
    """Gets and Validate the Travel Mode for the Network Analyst Tools"""
    try:
        return _gp_travel_mode(gis, travel_mode)
    except:
        return _route_service_travel_modes(gis, travel_mode)


# -------------------------------------------------------------------------
@lru_cache(maxsize=10)
def default_travel_mode(gis: GIS) -> str:
    """Gets the default travel mode for the GIS"""
    try:
        output = network.analysis.get_travel_modes(gis=gis)
        return output.default_travel_mode
    except:
        url = _validate_url(gis.properties.helperServices.route.url, gis)
        route_service = network.RouteLayer(url, gis=gis)
        modes = route_service.retrieve_travel_modes()
        return modes["defaultTravelMode"]
