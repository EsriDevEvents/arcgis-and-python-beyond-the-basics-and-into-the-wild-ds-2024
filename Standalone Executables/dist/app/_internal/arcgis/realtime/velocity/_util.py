import re

from arcgis import GIS
from typing import Optional, Dict, Union, List


class _Util:
    """
    Private class that provides wrapper functions for connection objects
    (gis._con) xhr functions and some reusable function endpoint calls
    for _start, _stop, and _delete operations on a task.
    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    gis                    An authenticated :class:`arcgis.gis.GIS` object.
    ------------------     --------------------------------------------------------------------
    base_url               Base URL of ArcGIS Velocity.
    ==================     ====================================================================
    """

    _gis = None
    _base_url = None
    _params = None

    def __init__(self, gis: GIS, base_url: str):
        self._gis = gis
        self._base_url = base_url
        self._params = {"authorization": f"token={gis._con.token}"}

    def _get_request(self, path) -> Dict:
        """
        Private wrapper function that builds the absolute url from
        the base url + sub-path and then passing it to the xhr GET request
        gis._con.get(<url>, <params>)

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path                   feeds | realtime | bigdata.
        ==================     ====================================================================

        :return: Endpoint response
        """
        url = f"{self._base_url}{path}"
        response = self._gis._con.get(url, self._params)

        return self._parse_response(response)

    def _put_request(self, task_type: str, id: str, payload: Dict = None) -> Dict:
        """
        Private wrapper function that builds the absolute url from
        the base url + sub-path and then passing it to the xhr PUT request
        gis._con.put(<url>, <params>, <payload>)

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ------------------     --------------------------------------------------------------------
        **Optional Argument**  **Description**
        ------------------     --------------------------------------------------------------------
        payload                post body
        ==================     ====================================================================

        :return: Endpoint response
        """
        path = f"{task_type}/{id}/"
        url = f'{self._base_url}{path}?{self._params.get("authorization")}'

        if payload is None:
            payload = {}

        params = {**self._params, "data": payload}
        response = self._gis._con.put(url, params, post_json=True, try_json=False)

        return self._parse_response(response)

    def _post_request(
        self,
        task_type: str,
        id: Optional[str] = None,
        payload: Dict = None,
        raise_error: bool = True,
    ) -> Dict:
        """
        Private wrapper function that builds the absolute url from
        the base url + sub-path and then passing it to the xhr POST request
        gis._con.post(<url>, <params>, <payload>)

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        task_type                  feeds | realtime | bigdata
        ----------------------     --------------------------------------------------------------------
        id                         Unique ID of a task.
        ----------------------     --------------------------------------------------------------------
        **Optional Argument**      **Description**
        ----------------------     --------------------------------------------------------------------
        payload                    post body
        ----------------------     --------------------------------------------------------------------
        raise_error                Default value is: True
        ======================     ====================================================================

        :return: Endpoint response
        """
        if id is not None:
            path = f"{task_type}/{id}/"
        else:
            path = f"{task_type}/"

        url = f'{self._base_url}{path}?{self._params.get("authorization")}'
        if payload is None:
            payload = {}

        params = {**self._params, "json": payload}

        response = self._gis._con.post(url, params, post_json=True, try_json=True)

        if raise_error is False:
            return response
        else:
            return self._parse_response(response)

    def _delete_request(self, path: str) -> bool:
        """
        Private wrapper function that builds the absolute url from
        the base url + sub-path and then passing it to the xhr DELETE reqest
        gis._con.delete(<url>, <params>)

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        path                   feeds | realtime | bigdata.
        ==================     ====================================================================

        :return: Endpoint response
        """
        url = f"{self._base_url}{path}"
        response = self._gis._con.delete(url, self._params)

        return self._parse_response(response, return_boolean_for_success=True)

    def _get(self, task_type: str, id: str) -> Dict:
        """
        Generic task operation to get item by id

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: Endpoint response
        """
        path = f"{task_type}/{id}"
        return self._get_request(path)

    def _start(self, task_type: str, id: str) -> Dict:
        """
        Generic start task operation

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: Endpoint response
        """
        path = f"{task_type}/{id}/start"
        return self._get_request(path)

    def _stop(self, task_type: str, id: str) -> Dict:
        """
        Generic stop task operation

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: boolean
        """
        path = f"{task_type}/{id}/stop"
        response = self._get_request(path)

        return response.get("status") == "success"

    def _status(self, task_type: str, id: str) -> Dict:
        """
        Generic get status task with possible task types: feed | realtime | bigdata

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: endpoint response for task status
        """
        path = f"{task_type}/{id}/status"
        return self._get_request(path)

    def _metrics(self, task_type, id: str) -> Dict:
        """
        Generic get metrics task with possible task types: feed | realtime | bigdata

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: endpoint response for task metrics
        """
        return self._post_request(task_type, id)

    def _delete(self, task_type: str, id: str) -> bool:
        """
        Generic task operation to delete item by id

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        task_type              feeds | realtime | bigdata
        ------------------     --------------------------------------------------------------------
        id                     Unique ID of a task.
        ==================     ====================================================================

        :return: A bool containing True (for success) or
         False (for failure) a dictionary with details is returned.
        """
        path = f"{task_type}/{id}"
        return self._delete_request(path)

    def _parse_response(
        self, response: str, return_boolean_for_success=False
    ) -> Union[bool, Dict]:
        """
        Generic task operation to get item by id

        ==========================      ====================================================================
        **Parameter**                    **Description**
        --------------------------      --------------------------------------------------------------------
        response                        Result object of an endpoint
        --------------------------      --------------------------------------------------------------------
        return_boolean_for_success      Default value is: False
        ==========================      ====================================================================

        :return: Result or raise exception if status has an 'error' attribute
        """
        if isinstance(response, dict) and response.get("status") == "error":
            raise Exception(response)
        elif isinstance(response, list):
            for item in response:
                if item.get("status") == "error":
                    raise Exception(item)
                else:
                    return response
        else:
            if return_boolean_for_success == True:
                return True
            else:
                return response

    def _validate_response(self, response: Dict) -> bool:
        if isinstance(response, dict) and response.get("status") == "error":
            return False
        elif isinstance(response, list):
            for item in response:
                if item.get("status") == "error":
                    return False
                else:
                    return True
        else:
            return True

    # ----------------------------------------------------------------------
    def sample_messages(self, input_type: str, payload: Dict = None) -> Dict:
        """
        Gets sample from a feed or source.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        input_type             "feed" | "sources"
        ------------------     --------------------------------------------------------------------
        payload                Payload for the sample message.
        ==================     ====================================================================

        :return: Sample message response including derived schema and raw samples
        """
        if payload is None:
            raise AttributeError("Post request payload is empty")

        path = f"{input_type}/sampleMessages"
        _response = self._post_request(path, id=None, payload=payload)
        return _response

    # ----------------------------------------------------------------------
    def test_connection(self, input_type: str, payload: Optional[Dict] = None) -> bool:
        """
        Tests connection to a feed, source, or output.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        input_type             "feed" | "sources" | "outputs"
        ------------------     --------------------------------------------------------------------
        payload                Payload for the feed.
        ==================     ====================================================================

        :return: True if test connection is successful else a dictionary with error details.
        """
        if payload is None:
            raise AttributeError("Post request payload is empty")

        path = f"{input_type}/testConnection"

        _response = self._post_request(
            path, id=None, payload=payload, raise_error=False
        )
        return self._validate_response(_response)

    def derive(self, sample_data: str, format_name: str = "Unknown") -> Dict:
        if not sample_data:
            raise AttributeError("sample_data should not be empty")

        post_body = {
            "content": sample_data,
            "formatName": format_name,
            "properties": {},
        }
        response = self._post_request(
            task_type="schema/derive", id=None, payload=post_body, raise_error=False
        )
        return response

    def is_valid(self, label: str) -> bool:
        pattern = "^[A-Za-z0-9_ ]*$"
        return bool(re.match(pattern, label))
