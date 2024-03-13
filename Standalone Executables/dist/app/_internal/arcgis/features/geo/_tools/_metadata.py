import json
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._isd import InsensitiveDict


class _Metadata(object):
    """
    Internal `Metadata` that stores information about the source data
    """

    _source = None
    _renderer = None

    # ----------------------------------------------------------------------
    def __init__(self):
        """initializer"""
        pass

    # ----------------------------------------------------------------------
    @property
    def source_type(self):
        if self.source is None:
            return None
        return self.source.__class__.__name__

    # ----------------------------------------------------------------------
    def __str__(self):
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    # ----------------------------------------------------------------------
    @property
    def source(self):
        """gets/sets the source data pointer"""
        return self._source

    # ----------------------------------------------------------------------
    @source.setter
    def source(self, value):
        """gets/sets the source data pointer"""
        self._source = value

    # ----------------------------------------------------------------------
    @property
    def renderer(self):
        """gets/sets the renderer"""
        return self._renderer

    # ----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, value):
        """gets/sets the renderer"""
        if isinstance(value, dict):
            value = InsensitiveDict.from_dict(value)
        elif isinstance(value, PropertyMap):
            value = InsensitiveDict.from_dict(dict(value))
        elif isinstance(value, InsensitiveDict):
            pass
        else:
            raise ValueError("Must be a a dictionary or InsensitiveDict")
        if value != self._renderer:
            self._renderer = value

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return self.__dict__
