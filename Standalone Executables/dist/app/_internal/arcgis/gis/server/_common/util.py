from __future__ import print_function
from __future__ import division
import os
import tempfile
from contextlib import contextmanager
from arcgis._impl.common._utils import create_uid, _date_handler
from arcgis._impl.common._utils import local_time_to_online, online_time_to_string
from arcgis._impl.common._utils import timestamp_to_datetime


@contextmanager
def _tempinput(data):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write((bytes(data, "UTF-8")))
    temp.close()
    yield temp.name
    os.unlink(temp.name)
