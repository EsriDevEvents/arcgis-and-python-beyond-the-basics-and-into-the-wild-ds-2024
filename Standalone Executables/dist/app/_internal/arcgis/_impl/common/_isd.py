import json
import ujson as _ujson
from collections import OrderedDict
from collections.abc import MutableMapping, Mapping


###########################################################################
class InsensitiveDict(MutableMapping):
    """
    A case-insensitive ``dict`` like object used to update and alter JSON

    A varients of a case-less dictionary that allows for dot and bracket notation.
    """

    # ----------------------------------------------------------------------
    def __init__(self, data=None):
        self._store = OrderedDict()  #
        if data is None:
            data = {}

        self.update(data)
        self._to_isd(self)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return str(dict(self.items()))

    # ----------------------------------------------------------------------
    def __str__(self):
        return str(dict(self.items()))

    # ----------------------------------------------------------------------
    def __setitem__(self, key, value):
        if str(key).lower() in {"_store"}:
            super(InsensitiveDict, self).__setattr__(key, value)
        else:
            if isinstance(value, dict):
                self._store[str(key).lower()] = (key, InsensitiveDict(value))
            else:
                self._store[str(key).lower()] = (key, value)

    # ----------------------------------------------------------------------
    def __getitem__(self, key):
        return self._store[str(key).lower()][1]

    # ----------------------------------------------------------------------
    def __delitem__(self, key):
        del self._store[str(key).lower()]

    # ----------------------------------------------------------------------
    def __getattr__(self, key):
        if str(key).lower() in {"_store"}:
            return self._store
        else:
            return self._store[str(key).lower()][1]

    # ----------------------------------------------------------------------
    def __setattr__(self, key, value):
        if str(key).lower() in {"_store"}:
            super(InsensitiveDict, self).__setattr__(key, value)
        else:
            if isinstance(value, dict):
                self._store[str(key).lower()] = (key, InsensitiveDict(value))
            else:
                self._store[str(key).lower()] = (key, value)

    # ----------------------------------------------------------------------
    def __dir__(self):
        return self.keys()

    # ----------------------------------------------------------------------
    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self._store)

    # ----------------------------------------------------------------------
    def _lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return ((lowerkey, keyval[1]) for (lowerkey, keyval) in self._store.items())

    # ----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support"""
        self.__dict__.update(InsensitiveDict(d).__dict__)
        self = InsensitiveDict(d)

    # ----------------------------------------------------------------------
    def __getstate__(self):
        """pickle support"""
        return _ujson.loads(self.json)

    # ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, o):
        """Converts dict to a InsensitiveDict"""
        return cls(o)

    # ----------------------------------------------------------------------
    @classmethod
    def from_json(cls, o):
        """Converts JSON string to a InsensitiveDict"""
        if isinstance(o, str):
            o = _ujson.loads(o)
            return InsensitiveDict(o)
        return InsensitiveDict.from_dict(o)

    # ----------------------------------------------------------------------
    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = InsensitiveDict(other)
        else:
            return NotImplemented
        return dict(self._lower_items()) == dict(other._lower_items())

    # ----------------------------------------------------------------------
    def copy(self):
        return InsensitiveDict(self._store.values())

    # ---------------------------------------------------------------------
    def _to_isd(self, data):
        """converts a dictionary from InsensitiveDict to a dictionary"""
        for k, v in data.items():
            if isinstance(v, (dict, InsensitiveDict)):
                data[k] = self._to_isd(v)
            elif isinstance(v, (list, tuple)):
                l = []
                for i in v:
                    if isinstance(i, dict):
                        l.append(InsensitiveDict(i))
                    else:
                        l.append(i)
                if isinstance(v, tuple):
                    l = tuple(l)
                data[k] = l
        return data

    # ---------------------------------------------------------------------
    def _json(self):
        """converts an InsensitiveDict to a dictionary"""
        d = {}
        for k, v in self.items():
            if isinstance(v, InsensitiveDict):
                d[k] = v._json()
            elif type(v) in (list, tuple):
                l = []
                for i in v:
                    if isinstance(i, InsensitiveDict):
                        l.append(i._json())
                    else:
                        l.append(i)
                if type(v) is tuple:
                    v = tuple(l)
                else:
                    v = l
                d[k] = v
            else:
                d[k] = v  # not list, tuple, or InsensitiveDict
        return d

    # ----------------------------------------------------------------------
    @property
    def json(self):
        """returns the value as JSON String"""
        o = self._json()  # dict(self.copy())
        return _ujson.dumps(dict(o))
