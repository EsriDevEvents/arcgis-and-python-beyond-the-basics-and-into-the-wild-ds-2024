from arcgis.gis import GIS
import requests
import time
from . import _global_settings

class gis_debugger:
    def __init__(selfl , gis: GIS):
        self.gis = gis
        

    def print_message(self, message):
        print(message)



def log_all_requests(response: requests.Response, *args, **kwargs):
    """
    Logs all HTTP responses.

    Args:
        response (requests.Response): The HTTP response to handle.


    Returns:
        requests.Response: The original HTTP response.

    Raises:
        None
    """

    print(f"Networking: {response.status_code} response for {response.url}.")
    return response

def log_all_requests_detailed(response: requests.Response, *args, **kwargs):
    """
    Handles errors in HTTP responses.

    Args:
        response (requests.Response): The HTTP response to handle.

    Returns:
        requests.Response: The original HTTP response.

    Raises:
        None
    """
    print(
        f"Networking: {response.status_code} response for {response.url}\n"
        f"Request details:\n"
        f"Method: {response.request.method}\n"
        f"URL: {response.request.url}\n"
        f"Body: {response.request.body}\n"
        f"Headers: {response.request.headers}\n"
        f"Response: {response.status_code} {response.reason}\n"
        f"Response Text: {response.text}\n"
    )
    return response


def throttle_rate(
    threshold: int = 1500,
    peroid: int = 300,
    pause: int = 300,
    log_all_requests=False,
    log_rate=False,
):
    """
    Check if the number of requests made in the given period exceeds the threshold.
    If it does, pause the program for the specified amount of time.

    Args:
        threshold (int, optional): The maximum number of requests allowed in the given period. Defaults to 1500.
        peroid (int, optional): The time period in seconds. Defaults to 300.
        pause (int, optional): The amount of time to pause the program if the threshold is exceeded. Defaults to 300.
        log_all_requests (bool, optional): Whether to log information about all requests. Defaults to False.
        log_rate (bool, optional): Whether to log the request rate. Defaults to False.

    Returns:
        function: Callable that's used as a response hook.
    """

    def throttle(response: requests.Response, *args, **kwargs):

        if _global_settings.request_peroid_start is None:
            _global_settings.request_peroid_start = time.time()

        _global_settings.request_count += 1

        if _global_settings.request_count_start is None:
            _global_settings.request_count_start = time.time()

        time_elasped = time.time() - _global_settings.request_count_start

        if time_elasped > peroid:
            _global_settings.request_count_start = time.time()
            _global_settings.request_count = 0

        if _global_settings.request_count > threshold:
            print(
                f"Networking: Request count exceeded threshold of {threshold} requests per {peroid} seconds in {time_elasped} seconds, pausing for {pause} seconds."
            )
            time.sleep(pause)
            _global_settings.request_count_start = time.time()
            _global_settings.request_count = 0

        if _global_settings.request_count > 0 and time_elasped > 0:
            rate = _global_settings.request_count / time_elasped

        else:
            rate = 0

        if log_all_requests:
            print(
                f"Netwoking: Request stats: {_global_settings.request_count} request in past {time_elasped} seconds, a rate of {rate} requests/second. Threshold is {threshold} requests/second and period is {peroid} seconds."
            )
        if log_rate:
            print(
                f"Networking: Request rate: {_global_settings.request_count} requests in {peroid} second peroid, rate of {rate} requests/second."
            )

        return response

    return throttle


def response_error_handling(response: requests.Response, *args, **kwargs):
    """
    Handles errors in HTTP responses.

    Args:
        response (requests.Response): The HTTP response to handle.

    Returns:
        requests.Response: The original HTTP response.

    Raises:
        None
    """
    # List of exemptions for certain errors that we don't want to log, each must have a status code and url substring
    exemptions = [
        {
            # Until we find a better way to check for metadata existance, we'll just ignore this error
            "status_code": 403,
            "url_substring": "info/metadata/metadata.xml",
        },
        {
            # Until we find a better way to check for metadata existance, we'll just ignore this error
            "status_code": 400,
            "url_substring": "info/metadata/metadata.xml",
        },
    ]
    if response.status_code != 200 and response.status_code != 302:
        if any(
            [
                exemption["status_code"] == response.status_code
                and exemption["url_substring"] in response.url
                for exemption in exemptions
            ]
        ):
            # Ignore these scenarios
            return response
        print(
            f"Networking: {response.status_code} response for {response.url}\n"
            f"Request details:\n"
            f"Method: {response.request.method}\n"
            f"URL: {response.request.url}\n"
            f"Body: {response.request.body}\n"
            f"Headers: {response.request.headers}\n"
            f"Response: {response.status_code} {response.reason}\n"
            f"Response Text: {response.text}\n"
        )
        if response.status_code == 403 and response.reason == "FORBIDDEN":
            print(
                "Networking: A 403 FORBIDDEN response indicates the requests may be getting blocked. Check any firewalls that may be blocking this."
            )
    return response