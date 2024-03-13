from requests.auth import AuthBase
from requests import PreparedRequest, Response


###########################################################################
class BaseEsriAuth(AuthBase):
    """
    A base class that developers can inherit from to create custom
    authentication handlers.

    For more information, please see:
    https://docs.python-requests.org/en/master/user/authentication/#new-forms-of-authentication
    """

    # ---------------------------------------------------------------------
    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        """
        Called Before PrepareRequest is Completed.

        The logic to attached preemptive authentication should go here.
        If you want to wait for response, register a response_hook within the call.

        :returns: PrepareRequest

        ```
        def __call__(self, r:PreparedRequest) -> PreparedRequest:
            r.register_hook(event="response", hook=self.response_hook)
            return r
        ```

        """
        raise NotImplementedError("Auth hooks must be callable.")

    # ---------------------------------------------------------------------
    def response_hook(self, response: Response, **kwargs) -> Response:
        """
        response hook logic

        :return: Response
        """
        return response

    # ---------------------------------------------------------------------
    @property
    def token(self) -> str:
        """
        returns a ArcGIS Token as a string

        :return: string
        """
        raise NotImplementedError("Token not implemented.")
