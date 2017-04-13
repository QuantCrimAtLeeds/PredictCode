"""
sources.ukpolice
================

Reads a CSV file in the format (as of April 2017) of data available from:

- https://data.police.uk/data/

The default data is loaded from a file "uk_police.csv" which should be
downloaded from one of the above links.  Data from more than one month needs to
be manually joined.

The data is partly anonymous in that the address is a street name (or other
non-uniquely identifying location) and geocoding resolves to the centre of
streets.  Most importantly, all timestamps are only to a _monthly_ resolution.
"""

import csv
import os.path
import datetime
import numpy as np
from ..data import TimedPoints

_default_filename = os.path.join(os.path.split(__file__)[0],"uk_police.csv")
_DESCRIPTION_FIELD = 'Crime type'
_X_FIELD = 'Longitude'
_Y_FIELD = 'Latitude'
_TIME_FIELD = 'Month'

def _date_from_csv(date_string):
    return datetime.datetime.strptime(date_string, "%Y-%m")
    raise Exception("This: '{}'".format(date_string))

def _convert_header(header):
    lookup = dict()
    for field in [_DESCRIPTION_FIELD, _X_FIELD, _Y_FIELD, _TIME_FIELD]:
        if not field in header:
            raise Exception("No field '{}' found in header".format(field))
        lookup[field] = header.index(field)
    return lookup

def default_burglary_data():
    """Load the default data, if available.

    :return: An instance of :class:`open_cp.data.TimedPoints` or `None`.
    """
    try:
        return load(_default_filename, {"Burglary"})
    except Exception:
        return None

def load(filename, primary_description_names):
    """Load data from a CSV file in the expected format.

    :param filename: Name of the CSV file load.
    :param primary_description_names: Set of names to search for in the
      "primary description field". E.g. pass `{"Burglary"}` to return only the
      "burglary" crime type.

    :return: An instance of :class:`open_cp.data.TimedPoints` or `None`.
    """
    data = []

    with open(filename) as file:
        reader = csv.reader(file)
        lookup = _convert_header(next(reader))
        for row in reader:
            description = row[lookup[_DESCRIPTION_FIELD]].strip()
            if len(primary_description_names) > 0 and not description in primary_description_names:
                continue
            x = row[lookup[_X_FIELD]].strip()
            y = row[lookup[_Y_FIELD]].strip()
            t = row[lookup[_TIME_FIELD]].strip()
            if x != "" and y != "":
                data.append((_date_from_csv(t), float(x), float(y)))

    data.sort(key = lambda triple : triple[0])
    return TimedPoints.from_coords([t for t, _, _ in data],
        [x for _, x, _ in data], [y for _, _, y in data])