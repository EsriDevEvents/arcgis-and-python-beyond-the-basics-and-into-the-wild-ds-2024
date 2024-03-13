""" Defines util functions used internally by the workforce-python-api.
"""

import datetime


def from_arcgis_date(timestamp):
    """Converts an ArcGIS timestamp to a datetime.
    :param timestamp: An int representing an ArcGIS timestamp
    :return: An equivalent datetime.datetime.
    """
    if timestamp is not None:
        return datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone.utc)
    return None


def to_arcgis_date(datetime_obj):
    """Converts a datetime to an ArcGIS timestamp.
    :param datetime_obj: A datetime.datetime
    :return: An int expressing the same date/time as an ArcGIS timestamp.
    """
    if datetime_obj is not None:
        return datetime_obj.timestamp() * 1000
    return None
