"""
sources.chicago
~~~~~~~

Reads a CSV file in the format (as of April 2017) of data available from:

- https://catalog.data.gov/dataset/crimes-one-year-prior-to-present-e171f
- https://catalog.data.gov/dataset/crimes-2001-to-present-398a4

The default data is loaded from a file "chicago.csv" which should be downloaded
from one of the above links.

The data is partly anonymous in that the address within a block is obscured,
and the geocoding always returns a coordinate in the middle of a block.
"""

import csv
import os.path
import datetime
import numpy as np
from ..data import TimedPoints

_default_filename = os.path.join(os.path.split(__file__)[0],"chicago.csv")
_DESCRIPTION_FIELD = ' PRIMARY DESCRIPTION'
_X_FIELD = 'X COORDINATE'
_Y_FIELD = 'Y COORDINATE'
_TIME_FIELD = 'DATE  OF OCCURRENCE'
_FEET_IN_METERS = 3.28084

def _date_from_csv(date_string):
    return datetime.datetime.strptime(date_string, "%m/%d/%Y %I:%M:%S %p")
    raise Exception("This: '{}'".format(date_string))

def _convert_header(header):
    lookup = dict()
    for field in [_DESCRIPTION_FIELD, _X_FIELD, _Y_FIELD, _TIME_FIELD]:
        if not field in header:
            raise Exception("No field '{}' found in header".format(field))
        lookup[field] = header.index(field)
    return lookup

def default_burglary_data():
    try:
        return load(_default_filename, {"THEFT"})
    except Exception:
        return None

def load(filename, primary_description_names, to_meters=True):
    data = []

    with open(filename) as file:
        reader = csv.reader(file)
        lookup = _convert_header(next(reader))
        for row in reader:
            description = row[lookup[_DESCRIPTION_FIELD]].strip()
            if not description in primary_description_names:
                continue
            x = row[lookup[_X_FIELD]].strip()
            y = row[lookup[_Y_FIELD]].strip()
            t = row[lookup[_TIME_FIELD]].strip()
            if x != "" and y != "":
                data.append((_date_from_csv(t), float(x), float(y)))

    data.sort(key = lambda triple : triple[0])
    xcoords = np.empty(len(data))
    ycoords = np.empty(len(data))
    for i, (_, x, y) in enumerate(data):
        xcoords[i], ycoords[i] = x, y
    times = [t for t, _, _ in data]
    if to_meters:
        xcoords /= _FEET_IN_METERS
        ycoords /= _FEET_IN_METERS
    return TimedPoints.from_coords(times, xcoords, ycoords)