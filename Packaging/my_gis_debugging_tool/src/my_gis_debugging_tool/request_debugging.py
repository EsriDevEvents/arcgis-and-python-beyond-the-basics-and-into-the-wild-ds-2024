from . import hooks
from arcgis.gis import GIS


def log_all_requests(gis: GIS):
    """
    Logs all requests made by the GIS object.

    Args:
        gis (GIS): The GIS object to log requests for.

    Returns:
        None
    """

    gis._con._session._session.hooks["response"].append(hooks.log_all_requests)

    return gis


def log_all_requests_detailed(gis: GIS):
    """
    Logs all requests made by the GIS object in a detailed format.

    Args:
        gis (GIS): The GIS object to log requests for.

    Returns:
        None
    """
    gis._con._session._session.hooks["response"].append(hooks.log_all_requests_detailed)

    return gis


def throttle_rate(
    gis: GIS,
    threshold: int = 5,
    peroid: int = 10,
    pause: int = 10,
    log_all_requests: bool = True,
    log_rate: bool = True,
):
    """
    Throttles the rate of requests made through the GIS object.

    Args:
        gis (GIS): The GIS object to throttle the requests for.
        threshold (int, optional): The maximum number of requests allowed within the specified period. Defaults to 5.
        peroid (int, optional): The time period (in seconds) within which the maximum number of requests is allowed. Defaults to 10.
        pause (int, optional): The time (in seconds) to pause between requests when the threshold is reached. Defaults to 10.
        log_all_requests (bool, optional): Whether to log all requests made through the GIS object. Defaults to True.
        log_rate (bool, optional): Whether to log the request rate information. Defaults to True.

    Returns:
        None
    """
    gis._con._session._session.hooks["response"].append(
        hooks.throttle_rate(gis, threshold, peroid, pause, log_all_requests, log_rate)
    )

    return gis


def response_error_handling(gis: GIS):
    """
    Handles response errors for the GIS object.

    Args:
        gis (GIS): The GIS object to handle response errors for.

    Returns:
        None
    """
    gis._con._session._session.hooks["response"].append(hooks.response_error_handling)

    return gis


def clear_hooks(gis: GIS):
    """
    Clears the hooks for the response object in the GIS session.

    Parameters:
        gis (GIS): The GIS object representing the GIS session.

    Returns:
        None
    """
    gis._con._session._session.hooks["response"].clear()

    return gis
