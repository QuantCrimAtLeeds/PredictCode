"""
sources.chicago
===============

Reads a CSV file in the format (as of April 2017) of data available from:

- https://catalog.data.gov/dataset/crimes-one-year-prior-to-present-e171f
- https://catalog.data.gov/dataset/crimes-2001-to-present-398a4

The default data is loaded from a file "chicago.csv" which should be downloaded
from one of the above links.  The format of the data, frustratingly, differs
between the snapshot of last year, and the total.

The data is partly anonymous in that the address within a block is obscured,
while the geocoding seems complicated (work in progress to understand)...

The crime type "HOMICIDE" is reported multiple times in the dataset.
"""

import csv
import os.path
import datetime
import numpy as np
from ..data import TimedPoints

_default_filename = os.path.join(os.path.split(__file__)[0],"chicago.csv")

_FEET_IN_METERS = 3.28084

_FIELDS = {
    "snapshot" : {
        "_DESCRIPTION_FIELD" : ' PRIMARY DESCRIPTION',
        "_X_FIELD" : 'X COORDINATE',
        "_Y_FIELD" : 'Y COORDINATE',
        "_TIME_FIELD" : 'DATE  OF OCCURRENCE',

        "_GEOJSON_LOOKUP" : {"case": 'CASE#',
            "address": "BLOCK",
            "location": ' LOCATION DESCRIPTION',
            "crime": ' PRIMARY DESCRIPTION',
            "type": ' SECONDARY DESCRIPTION',
            "timestamp": 'DATE  OF OCCURRENCE'},
        "GEOJSON_COORDS" : ('LONGITUDE', 'LATITUDE')
    },
    "all" : {
        "_DESCRIPTION_FIELD" : 'Primary Type',
        "_X_FIELD" : 'X Coordinate',
        "_Y_FIELD" : 'Y Coordinate',
        "_TIME_FIELD" : 'Date',

        "_GEOJSON_LOOKUP" : {"case": 'Case Number',
            "address": "Block",
            "location": 'Location Description',
            "crime": 'Primary Type',
            "type": 'Description',
            "timestamp": 'Date'},
        "GEOJSON_COORDS" : ('Longitude', 'Latitude')
    }
}

def _date_from_csv(date_string):
    return datetime.datetime.strptime(date_string, "%m/%d/%Y %I:%M:%S %p")

def _date_from_iso(iso_string):
    return datetime.datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S")

def _convert_header(header, dic):
    lookup = dict()
    for field in [dic["_DESCRIPTION_FIELD"], dic["_X_FIELD"], dic["_Y_FIELD"], dic["_TIME_FIELD"]]:
        if not field in header:
            raise Exception("No field '{}' found in header".format(field))
        lookup[field] = header.index(field)
    return lookup

def default_burglary_data():
    """Load the default data, if available.

    :return: An instance of :class:`open_cp.data.TimedPoints` or `None`.
    """
    try:
        return load(_default_filename, {"THEFT"})
    except Exception:
        return None

def _get_dic(type):
    try:
        return _FIELDS[type]
    except KeyError:
        raise ValueError("Don't understand type {}".format(type))

def load(filename, primary_description_names, to_meters=True, type="snapshot"):
    """Load data from a CSV file in the expected format.

    :param filename: Name of the CSV file load.
    :param primary_description_names: Set of names to search for in the
      "primary description field". E.g. pass `{"THEFT"}` to return only the
      "theft" crime type.
    :param to_meters: Convert the coordinates to meters; True by default.
    :param type: Either "snapshot" or "all" depending on whether the data
      has headers conforming the the data "last year" or "2001 to present".

    :return: An instance of :class:`open_cp.data.TimedPoints` or `None`.
    """
    dic = _get_dic(type)
    data = []

    with open(filename) as file:
        reader = csv.reader(file)
        lookup = _convert_header(next(reader), dic)
        for row in reader:
            description = row[lookup[dic["_DESCRIPTION_FIELD"]]].strip()
            if not description in primary_description_names:
                continue
            x = row[lookup[dic["_X_FIELD"]]].strip()
            y = row[lookup[dic["_Y_FIELD"]]].strip()
            t = row[lookup[dic["_TIME_FIELD"]]].strip()
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

def _convert_header_for_geojson(header, dic):
    try:
        column_lookup = {}
        for key, col_head in dic["_GEOJSON_LOOKUP"].items():
            column_lookup[key] = header.index(col_head)
        coord_lookup = [header.index(chead) for chead in dic["GEOJSON_COORDS"]]
        return column_lookup, coord_lookup
    except KeyError as ex:
        raise ValueError("Header not in expected format: {} caused by {}/{}".format(
            header, type(ex), ex))

def _generate_GeoJSON_Features(file, dic):
    reader = csv.reader(file)
    column_lookup, coord_lookup = _convert_header_for_geojson(next(reader), dic)
    for row in reader:
        properties = {key : row[i] for key, i in column_lookup.items()}
        properties["timestamp"] = _date_from_csv(properties["timestamp"]).isoformat()
        if row[coord_lookup[0]] == "":
            geometry = None
        else:
            coordinates = [float(row[i]) for i in coord_lookup]
            geometry = {"type":"Point", "coordinates":coordinates}
        yield {"geometry": geometry, "properties": properties,
                "type": "Feature"}

def generate_GeoJSON_Features(file=_default_filename, type="snapshot"):
    """Generate a sequence of GeoJSON "features" from the CSV file.
    See :func:`load_to_GeoJSON`.
    
    :param file: Either a filename, or a file object.
    """
    dic = _get_dic(type)
    if isinstance(file, str):
        with open(file) as f:
            yield from _generate_GeoJSON_Features(f, dic)
    else:
        yield from _generate_GeoJSON_Features(file, dic)

def load_to_GeoJSON(filename=_default_filename, type="snapshot"):
    """Load the specified CSV file to a list of GeoJSON (see
    http://geojson.org/) features.  Events with no location data have `None`
    as the geometry.  Timestamps are converted to standard ISO string format.

    The returned "properties" have these keys:
    - "case" for the "CASE#" field
    - "crime" for the "PRIMARY DESCRIPTION" field
    - "type" for the "SECONDARY DESCRIPTION" field
    - "location" for the  "LOCATION DESCRIPTION" field
    - "timestamp" for the "DATE  OF OCCURRENCE" field
    - "address" for the "BLOCK" field

    :param filename: Filename of the CSV file to process
    :param type: Either "snapshot" or "all" depending on whether the data
      has headers conforming the the data "last year" or "2001 to present".

    :return: List of Python dictionaries in GeoJSON format.
    """
    return list(generate_GeoJSON_Features(filename, type))

try:
    import geopandas as gpd
    import shapely.geometry as _geometry
except:
    gpd = None
    _geometry = None

def convert_null_geometry_to_empty(frame):
    """Utility method.  Convert any geometry in the geoDataFrame which is
    "null" (`None` or empty) to a Point type geometry which is empty.  The
    returned geoDateFrame is suitable for projecting and other geometrical
    transformations.
    """
    def null_to_point(x):
        if x is None or x.is_empty:
            return _geometry.Point()
        return x
    newgeo = frame.geometry.map(null_to_point)
    return frame.set_geometry(newgeo)

def convert_null_geometry_to_none(frame):
    """Utility method.  Convert any geometry in the geoDataFrame which is
    "null" (`None` or empty) to `None`.  The returned geoDateFrame is suitable
    for saving.
    """
    def null_to_none(x):
        if x is None or x.is_empty:
            return None
        return x
    newgeo = frame.geometry.map(null_to_none)
    return frame.set_geometry(newgeo)

def load_to_geoDataFrame(filename=_default_filename, datetime_as_string=True,
                         type="snapshot", empty_geometry="none"):
    """Return the same data as :func:`load_to_GeoJSON` but as a geoPandas
    data-frame.

    :param filename: Filename of the CSV file to process
    :param datetime_as_string: Write the timestamp as an ISO formatted string.
      Defaults to True which is best for saving the dataframe as e.g. a shape
      file.  Set to False to get timestamps as python objects, which is best
      for using (geo)pandas to analyse the data.
    :param type: Either "snapshot" or "all" depending on whether the data
      has headers conforming the the data "last year" or "2001 to present".
    :param empty_geometry: Either "none" to return `None` as the geometry of
      crimes which have no location data in the CSV file (this is correct if
      you wish to save the data-frame); or "empty" to return an empty `Point`
      type (which is correct, for example, if you wish to re-project the
      data-frame).  Yes, GeoPandas appears to be annoying like this.

    """
    geo_data = load_to_GeoJSON(filename, type=type)
    if not datetime_as_string:
        for feature in geo_data:
            feature["properties"]["timestamp"] = _date_from_iso(feature["properties"]["timestamp"])
    frame = gpd.GeoDataFrame.from_features(geo_data)
    if empty_geometry == "none":
        pass
    elif empty_geometry == "empty":
        frame = convert_null_geometry_to_empty(frame)
    else:
        raise ValueError("Unknown `empty_geometry` parameter `{}`".format(empty_geometry))
    frame.crs = {"init":"EPSG:4326"}
    return frame