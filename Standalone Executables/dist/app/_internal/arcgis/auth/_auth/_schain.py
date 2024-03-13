"""
Allows for requests authentication to be chained.

Inspired by: https://github.com/Colin-b/requests_auth

"""
from requests.auth import AuthBase


###########################################################################
class _MultiAuth(AuthBase):
    """Authentication using multiple authentication methods."""

    def __init__(self, *authentication_modes):
        self.authentication_modes = authentication_modes

    def __call__(self, r):
        for authentication_mode in self.authentication_modes:
            authentication_mode.__call__(r)
        return r

    def __add__(self, other):
        if isinstance(other, _MultiAuth):
            return _MultiAuth(*self.authentication_modes, *other.authentication_modes)
        return _MultiAuth(*self.authentication_modes, other)

    def __and__(self, other):
        if isinstance(other, _MultiAuth):
            return _MultiAuth(*self.authentication_modes, *other.authentication_modes)
        return _MultiAuth(*self.authentication_modes, other)


###########################################################################
class SupportMultiAuth:
    """Inherit from this class to be able to use your class with requests_auth provided authentication classes."""

    def __add__(self, other):
        if isinstance(other, _MultiAuth):
            return _MultiAuth(self, *other.authentication_modes)
        return _MultiAuth(self, other)

    def __and__(self, other):
        if isinstance(other, _MultiAuth):
            return _MultiAuth(self, *other.authentication_modes)
        return _MultiAuth(self, other)
