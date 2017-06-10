import collections
import dateutil.parser
import datetime
import logging
from . import funcs
from .common import CoordType
import json

InitialData = collections.namedtuple("InitialData", ["header", "firstrows", "rowcount", "filename"])


class ParseSettings():
    """A "model-like" object which stores all the settings we need to parse a file."""
    def __init__(self):
        self.coord_type = CoordType.LonLat
        self.meters_conversion = 1.0
        self.timestamp_field = -1
        self.xcoord_field = -1
        self.ycoord_field = -1
        self.crime_type_fields = []
        self.timestamp_format = ""

    @property
    def coord_type(self):
        """Lon/Lat or XY coords?

        :return: :class:`CoordType` enum
        """
        return self._coord_type

    @coord_type.setter
    def coord_type(self, value):
        if not isinstance(value, CoordType):
            raise ValueError("Must be instance of :class:`CoordType`")
        self._coord_type = value

    @property
    def meters_conversion(self):
        """If in XY coords, return the factor to multiple the values by to get
        to meters.
        """
        return self._proj_convert

    @meters_conversion.setter
    def meters_conversion(self, value):
        self._proj_convert = value

    @staticmethod
    def feet():
        """Conversion from feet to meters, see
        https://en.wikipedia.org/wiki/Foot_(unit)"""
        return 0.3048

    @property
    def timestamp_format(self):
        """The format to use to decode the timestamp.  Either "" to attempt
        auto-detecting, or a valid `strptime` format string.
        """
        return self._ts_format

    @timestamp_format.setter
    def timestamp_format(self, value):
        self._ts_format = value

    @property
    def timestamp_field(self):
        """Field from CSV file to use as the timestamp.  -1==None."""
        return self._ts_field

    @timestamp_field.setter
    def timestamp_field(self, value):
        self._ts_field = value

    @property
    def xcoord_field(self):
        """Field from CSV file to use as the X Coord / Longitude.  -1==None."""
        return self._x_field

    @xcoord_field.setter
    def xcoord_field(self, value):
        self._x_field = value

    @property
    def ycoord_field(self):
        """Field from CSV file to use as the Y Coord / Latitude.  -1==None."""
        return self._y_field

    @ycoord_field.setter
    def ycoord_field(self, value):
        self._y_field = value

    @property
    def crime_type_fields(self):
        """A list (possibly empty) of fields to use a crime time identifiers."""
        return self._ct_fields

    @crime_type_fields.setter
    def crime_type_fields(self, value):
        input = list(value)
        while len(input) > 0 and input[-1] == -1:
            del input[-1]
        self._ct_fields = input

    @property
    def coordinate_scaling(self):
        if self.coord_type == CoordType.XY:
            return self.meters_conversion
        else:
            return 1.0

    def to_dict(self):
        """Return a dictionary storing the settings."""
        return { "coord_type" : self.coord_type.name,
                "meters_conversion" : self.meters_conversion,
                "timestamp_field" : self.timestamp_field,
                "xcoord_field" : self.xcoord_field,
                "ycoord_field" : self.ycoord_field,
                "crime_type_fields" : self.crime_type_fields,
                "timestamp_format" : self.timestamp_format
            }

    @staticmethod
    def from_dict(data):
        out = ParseSettings()
        out.coord_type = CoordType[data["coord_type"]]
        out.meters_conversion = data["meters_conversion"]
        out.timestamp_field = data["timestamp_field"]
        out.xcoord_field = data["xcoord_field"]
        out.ycoord_field = data["ycoord_field"]
        out.crime_type_fields = data["crime_type_fields"]
        out.timestamp_format = data["timestamp_format"]
        return out


class ParseError(Exception):
    """Indicate a problem is parsing the input data.  Convert to string for a
    human readable reason."""


class Model():
    """The model.
    
    :param initial_data: An instance of :class:`InitialData`
    """
    def __init__(self, initial_data):
        self._initial_data = initial_data
        self._parsed = None

    @property
    def header(self):
        return self._initial_data.header

    @property
    def firstrows(self):
        return self._initial_data.firstrows
    
    @property
    def rowcount(self):
        return self._initial_data.rowcount

    @property
    def filename(self):
        return self._initial_data.filename
    
    @property
    def processed_data(self):
        """Returns `None` if parsing failed, or a triple of lists
        `(timestamps, xcoords, ycoords, crime_types)` where `crime_types`
        or is a list of tuples, each tuple being the crime types."""
        return self._parsed

    def try_parse(self, parse_settings):
        """Attempt to parse the initial data.

        :param parse_settings: An instance of :class:`ParseSettings` describing
          the parse settings to use.

        :return: An error message, or None for success (in which case the
          :attr:`processed_data` will be updated.)
        """
        self._parsed = None
        
        if parse_settings.timestamp_field == -1:
            return "Need to select a field for the timestamps"
        if parse_settings.xcoord_field == -1:
            return "Need to select a field for the X coordinates"
        if parse_settings.ycoord_field == -1:
            return "Need to select a field for the Y coordinates"
        if len(parse_settings.crime_type_fields) > 1:
            for i in range(1, len(parse_settings.crime_type_fields)+1):
                if parse_settings.crime_type_fields[-i] == -1:
                    return "Cannot specify a crime sub-type without the main crime type"

        logger = logging.getLogger(__name__)
        logger.debug("Attempting to parse the initial input data")
        tp = _TryParse(parse_settings, logger=logger)
        try:
            ts, xs, ys, typs = [], [], [], []
            for row in self._initial_data.firstrows:
                data = tp.try_parse(row)
                ts.append(data[0])
                xs.append(data[1])
                ys.append(data[2])
                if len(data) > 3:
                    typs.append(data[3:])
        except ParseErrorData as ex:
            if ex.reason == "time":
                return ("Cannot understand the data/time string '{}'.\n" +
                    "Make sure you have selected the correct field for the timestamps.  " +
                    "If necessary, try entering a specific timestamp format.").format(ex.data)
            else:
                return ("Cannot understand the {} coordinate string '{}'.\n" +
                    "Make sure you have selected the correct field for the {} coordinates."
                    ).format(ex.reason, ex.data, ex.reason)
        self._parsed = ts, xs, ys, typs

    @staticmethod
    def load_full_dataset(parse_settings):
        """A coroutine.  On error, yields the exception for that row."""
        if (parse_settings.timestamp_field == -1 or parse_settings.xcoord_field == -1
                or parse_settings.ycoord_field == -1):
            raise ValueError()
        if len(parse_settings.crime_type_fields) > 1:
            for i in range(1, len(parse_settings.crime_type_fields)+1):
                if parse_settings.crime_type_fields[-i] == -1:
                    raise ValueError()
        logger = logging.getLogger(__name__)
        logger.debug("Attempting to parse the whole data-set")
        tp = _TryParse(parse_settings, logger=logger)

        row = yield
        row_number = 0
        while True:
            row_number += 1
            try:
                data = tp.try_parse(row)
            except Exception as ex:
                data = (row_number, ex)
            row = yield data


class ParseErrorData(Exception):
    def __init__(self, reason, data):
        self.reason = reason
        self.data = data


class _TryParse():
    def __init__(self, parse_settings, logger=funcs.null_logger()):
        self.time_parser = self._time_parser(parse_settings.timestamp_format)
        self.time_field = parse_settings.timestamp_field
        self.x_field = parse_settings.xcoord_field
        self.y_field = parse_settings.ycoord_field
        self.crime_fields = parse_settings.crime_type_fields
        self.scale = parse_settings.coordinate_scaling
        self._logger = logger
            
    def try_parse(self, row):
        """Attempt to parse the row.  Raises :class:`ParseError` on error."""
        try:
            timestamp = self.time_parser(row[self.time_field])
        except Exception as ex:
            self._logger.debug("Timestamp parsing error was %s / %s", type(ex), ex)
            raise ParseErrorData("time", row[self.time_field])
        x = self._get_coord(row, self.x_field, "X") * self.scale
        y = self._get_coord(row, self.y_field, "Y") * self.scale
        data = [timestamp, x, y]
        for f in self.crime_fields:
            data.append( row[f] )
        return data

    def _time_parser(self, format):
        if format == "":
            parser = dateutil.parser.parse
        else:
            parser = lambda s : datetime.datetime.strptime(s, format)
        return parser

    def _get_coord(self, row, field, name):
        try:
            return float( row[field] )
        except:
            raise ParseErrorData(name, row[field])
