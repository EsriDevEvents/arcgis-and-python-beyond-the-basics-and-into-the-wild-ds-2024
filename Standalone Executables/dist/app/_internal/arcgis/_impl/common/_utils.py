"""set of common utilities"""
import os
import sys
import time
import uuid
import zipfile
import datetime
from datetime import date
import tempfile
from contextlib import contextmanager
import logging
import decimal
import functools


@functools.lru_cache(maxsize=35)
def _validate_url(url: str, gis: "GIS", url_type: str = None) -> str:
    """calculates the service URL"""
    finders = [
        "naserver",
        "gpserver",
        "mapserver",
        "featureserver",
        "imageserver",
        "geoenrichmentserver",
    ]
    part = None
    if not any([url.lower().endswith(f) for f in finders]):
        part = os.path.basename(url)
        url = os.path.dirname(url)
    try:
        res = gis._private_service_url(url)
    except:
        if part:
            return f"{url}/{part}"
        return url
    return_url = None
    if gis._is_hosted_nb_home and "privateServiceUrl" in res:
        return_url = res["privateServiceUrl"]
    else:
        if url_type is None or str(url_type).lower() == "public":
            if "serviceUrl" in res:
                return_url = res["serviceUrl"]
            else:
                return_url = url
        else:
            if "privateServiceUrl" in res:
                return_url = res["privateServiceUrl"]
            elif "serviceUrl" in res:
                return_url = res["serviceUrl"]
            else:
                return_url = url
    if part:
        return f"{return_url}/{part}"
    return return_url


# ----------------------------------------------------------------------
def bytesto(size, to="m", bsize=1024):
    """convert bytes to megabytes, etc.
    sample code:
        print('mb= ' + str(bytesto(314575262000000, 'm')))

    sample output:
        mb= 300002347.946
    """

    a = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    r = float(size)
    for i in range(a[to]):
        r = r / bsize
    return r


# ----------------------------------------------------------------------
def create_uid():
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
def inspect_function_inputs(fn, **params):
    """
    Given any function and a set of key/value pairs, where the ```params``` is a dictionary,
    the code inspects the input method and then returns a list of accepted inputs as
    a new dictionary.  This method is used primarily to validate GP services and ensure
    that the parameters given are supported in the current version of the tool.

    :return: dictionary

    Example:

    >>> def add(x,y):
    >>>    return x+y

    >>> valid_inputs = inspect_function_inputs(fn=add, **{'x' : 1, 'y': 2, 'cat' : 3})
    >>> print(valid_inputs)
    {'x' : 1, 'y': 2}

    """
    import inspect

    try:
        args = list(inspect.signature(fn).parameters.keys()) + ["estimate"]
    except ValueError:
        args = inspect.getfullargspec(func=fn).args
    if "gis" in args:
        args.pop(args.index("gis"))
    valid = {}
    for key in params.keys():
        if key in args and params[key] is not None:
            valid[key] = params[key]

    return valid


# ----------------------------------------------------------------------
def _date_handler(obj):
    import numpy

    npversion = [int(i) for i in numpy.__version__.split(".")]
    if npversion < [1, 20, 0]:
        FLOAT_CHECKER = (numpy.float, numpy.float32, numpy.float64)
    else:
        FLOAT_CHECKER = (float, numpy.float32, numpy.float64)
    from ._mixins import PropertyMap

    if type(obj) is datetime.date:
        import datetime as _dt

        obj = _dt.datetime.combine(obj, _dt.datetime.min.time())
    if isinstance(obj, datetime.datetime) or isinstance(obj, date):
        try:
            return local_time_to_online(obj)
        except:
            diff = datetime.datetime(1970, 1, 1) - obj
            v = -diff.total_seconds() * 1000
            return int(v)
    elif isinstance(obj, (numpy.int32, numpy.int64)):
        return _date_handler(int(obj))
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, FLOAT_CHECKER):
        return float(obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    elif isinstance(obj, PropertyMap):
        return dict(obj)
    else:
        return obj


# ----------------------------------------------------------------------
def local_time_to_online(dt=None):
    """
    converts datetime object to a UTC timestamp for AGOL
    Inputs:
       dt - datetime object
    Output:
       Long value
    """

    if dt is None:
        dt = datetime.datetime.now()

    if sys.version_info.major == 3:
        return int(dt.timestamp() * 1000)
    elif isinstance(dt, datetime.datetime) and dt.tzinfo:
        dt = dt.astimezone()

    return int(time.mktime(dt.timetuple()) * 1000)


# ----------------------------------------------------------------------
def online_time_to_string(value, timeFormat):
    """
    Converts a timestamp to date/time string
    Inputs:
       value - timestamp as long
       timeFormat - output date/time format
    Output:
       string
    """
    return datetime.datetime.fromtimestamp(value / 1000).strftime(timeFormat)


# ----------------------------------------------------------------------
def timestamp_to_datetime(timestamp):
    """
    Converts a timestamp to a datetime object
    Inputs:
       timestamp - timestamp value as Long
    output:
       datetime object
    """
    return datetime.datetime.fromtimestamp(timestamp / 1000)


###########################################################################
class Error(Exception):
    pass


# --------------------------------------------------------------------------
@contextmanager
def _tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write((bytes(data, "UTF-8")))
    temp.close()
    yield temp.name
    os.unlink(temp.name)


# --------------------------------------------------------------------------
def _lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# --------------------------------------------------------------------------
def _is_shapefile(data):
    if zipfile.is_zipfile(data):
        zf = zipfile.ZipFile(data, "r")
        namelist = zf.namelist()
        for name in namelist:
            if name.endswith(".shp") or name.endswith(".SHP"):
                return True
    return False


# --------------------------------------------------------------------------
def rot13(s):
    result = ""

    # Loop over characters.
    for v in s:
        # Convert to number with ord.
        c = ord(v)

        # Shift number back or forward.
        if c >= ord("a") and c <= ord("z"):
            if c > ord("m"):
                c -= 13
            else:
                c += 13
        elif c >= ord("A") and c <= ord("Z"):
            if c > ord("M"):
                c -= 13
            else:
                c += 13

        # Append to result.
        result += chr(c)

    # Return transformation.
    return result


# --------------------------------------------------------------------------
def zipws(path, outfile, keep=True):
    """
    compress the contents of a folder
    Parameters:
     :path: folder or folder contents to compress as a zip file
     :outfile: output file and location
     :keep: boolean - if true, the folder structure is kept, else just the
      files
    Output:
     path to a compressed zip file.
    """
    zipobj = zipfile.ZipFile(outfile, "w", zipfile.ZIP_DEFLATED)
    path = os.path.normpath(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if not file.endswith(".lock") and not file.endswith(".zip"):
                try:
                    if keep:
                        zipobj.write(
                            os.path.join(dirpath, file),
                            os.path.join(
                                os.path.basename(path),
                                os.path.join(dirpath, file)[len(path) + len(os.sep) :],
                            ),
                        )
                    else:
                        zipobj.write(
                            os.path.join(dirpath, file),
                            os.path.join(dirpath[len(path) :], file),
                        )
                except Exception:
                    pass
    zipobj.close()
    return outfile


# --------------------------------------------------------------------------
def _to_utf8(data):
    """Converts strings and collections of strings from unicode to utf-8."""
    if isinstance(data, dict):
        return {
            _to_utf8(key): _to_utf8(value)
            for key, value in data.items()
            if value is not None
        }
    elif isinstance(data, list):
        return [_to_utf8(element) for element in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, str):
        return data.encode("utf-8")
    elif isinstance(data, (float, int)):
        return data
    else:
        return data


# --------------------------------------------------------------------------
class _DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


# --------------------------------------------------------------------------
def chunks(l, n):
    """yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


# --------------------------------------------------------------------------
def is_pdf_file(file_path):
    """check the file first bytes to match with pdf signature"""
    with open(file_path, "rb") as f:
        return f.read(4) == b"%PDF"
