from __future__ import annotations

__all__ = ["ArcGISLoginError", "EsriHttpResponseError"]


class ArcGISLoginError(Exception):
    """Exception raised for login errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str = "Invalid username or password."):
        self.message: str = message
        super().__init__(self.message)


class EsriHttpResponseError(Exception):
    """Exception raised for http errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)
