import collections
import dateutil.parser
import datetime
import logging
from . import funcs
from .common import CoordType

InitialData = collections.namedtuple("InitialData", ["header", "firstrows", "rowcount", "filename"])

#class Data():
#    def __init__(self):
#        pass


class ParseError(Exception):
    """Indicate a problem is parsing the input data.  Convert to string for a
    human readable reason."""
    pass


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

    @property
    def coordinate_scaling(self):
        if self.coord_type == CoordType.XY:
            return self.meters_conversion
        else:
            return 1.0

    @property
    def processed_data(self):
        """Returns `None` if parsing failed, or a triple of lists
        `(timestamps, xcoords, ycoords)`.  Applies any necessary scaling to the
        coordinates."""
        return self._parsed

    def try_parse(self, time_format, time_field, x_field, y_field):
        """Attempt to parse the initial data.

        :return: An error message, or None for success (in which case the
          :attr:`processed` will be updated.)
        """
        self._parsed = None
        
        if time_field is None:
            return "Need to select a field for the timestamps"
        if x_field is None:
            return "Need to select a field for the X coordinates"
        if y_field is None:
            return "Need to select a field for the Y coordinates"

        logger = logging.getLogger(__name__)
        logger.debug("Attempting to parse the initial input data")
        tp = _TryParse(time_format, time_field, x_field, y_field,
                       scale=self.coordinate_scaling, logger=logger)
        try:
            ts, xs, ys = [], [], []
            for row in self._initial_data.firstrows:
                t,x,y = tp.try_parse(row)
                ts.append(t)
                xs.append(x)
                ys.append(y)
        except ParseErrorData as ex:
            if ex.reason == "time":
                return ("Cannot understand the data/time string '{}'.\n" +
                    "Make sure you have selected the correct field for the timestamps.  " +
                    "If necessary, try entering a specific timestamp format.").format(ex.data)
            else:
                return ("Cannot understand the {} coordinate string '{}'.\n" +
                    "Make sure you have selected the correct field for the {} coordinates."
                    ).format(ex.reason, ex.data, ex.reason)
        self._parsed = ts, xs, ys

    @staticmethod
    def load_full_dataset(time_format, time_field, x_field, y_field, scaling):
        """A coroutine.  On error, yields the exception for that row."""
        if time_format is None or x_field is None or y_field is None:
            raise ValueError()
        logger = logging.getLogger(__name__)
        logger.debug("Attempting to parse the whole data-set")
        tp = _TryParse(time_format, time_field, x_field, y_field,
                       scale=scaling, logger=logger)

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
    def __init__(self, time_format, time_field, x_field, y_field, scale=1.0, logger=funcs.null_logger()):
        self.time_format = time_format
        self.time_field = time_field
        self.x_field = x_field
        self.y_field = y_field
        self._logger = logger
        self.scale = scale
            
    def try_parse(self, row):
        """Attempt to parse the row.  Raises :class:`ParseError` on error."""
        time_parser = self._time_parser()
        try:
            timestamp = time_parser(row[self.time_field])
        except Exception as ex:
            self._logger.debug("Timestamp parsing error was %s / %s", type(ex), ex)
            raise ParseErrorData("time", row[self.time_field])
        x = self._get_coord(row, self.x_field, "X")
        y = self._get_coord(row, self.y_field, "Y")
        return (timestamp, x * self.scale, y * self.scale)

    def _time_parser(self):
        if self.time_format == "":
            parser = dateutil.parser.parse
        else:
            parser = lambda s : datetime.datetime.strptime(s, self.time_format)
        return parser

    def _get_coord(self, row, field, name):
        try:
            return float( row[field] )
        except:
            raise ParseErrorData(name, row[field])
