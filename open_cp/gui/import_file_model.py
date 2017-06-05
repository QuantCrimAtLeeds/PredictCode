import collections
import open_cp.gui.enum as enum
import dateutil.parser
import datetime
import logging

InitialData = collections.namedtuple("InitialData", ["header", "firstrows", "rowcount", "filename"])
CoordType = enum.IntEnum("CoordType", "LonLat XY")

class Data():
    def __init__(self):
        pass


class ParseError(Exception):
    pass


class Model():
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
    def processed(self):
        """Returns `None` if parsing failed, or a triple of lists
        `(timestamps, xcoords, ycoords)`"""
        return self._parsed

    def try_parse(self, time_format, time_field, x_field, y_field):
        """Attempt to parse the initial data.

        :return: An error message, or None for success (in which case the
          :attr:`processed` will be updated.)
        """
        self._parsed = None
        from collections import namedtuple
        View = namedtuple("View", "time_format datetime_field xcoord_field ycoord_field")
        view = View(time_format, time_field, x_field, y_field)
        tp = TryParse(view, data=self._initial_data)
        try:
            tp.try_parse()
        except ParseError as ex:
            return str(ex)
        self._parsed = tp.parsed_data


class TryParse():
    def __init__(self, view, data=None, file=None):
        self.view = view
        self._logger = logging.getLogger(__name__+".TryParse")
        if data is not None:
            if file is not None:
                raise ValueError("Must specify one of `data` and `file`")
            self.data = data.firstrows
        else:
            if file is None:
                raise ValueError("Can only specify one of `data` and `file`")
            raise NotImplementedError

    def try_parse(self):
        """Attempt to parse the data.  Raises :class:`ParseError` on error."""
        self._parsed_data = None
        timestamps = self._try_parse_timestamps()
        xcoords = self._try_parse_coords(self.view.xcoord_field, "X")
        ycoords = self._try_parse_coords(self.view.ycoord_field, "Y")
        self._parsed_data = (timestamps, xcoords, ycoords)

    @property
    def parsed_data(self):
        """Returns `None` if parsing failed, or a triple of lists
        `(timestamps, xcoords, ycoords)`"""
        return self._parsed_data

    def _try_parse_coords(self, field, coord_name):
        if field is None:
            raise ParseError("Need to select a field for the {} coordinates".format(coord_name))
        coords = []
        for row in self.data:
            try:
                coords.append(float(row[field]))
            except:
                raise ParseError(("Cannot understand the {} coordinate string '{}'.\n" +
                    "Make sure you have selected the correct field for the {} coordinates.")
                    .format(coord_name, row[field], coord_name))
        return coords

    def _try_parse_timestamps(self):
        if self.view.time_format == "":
            parser = dateutil.parser.parse
        else:
            parser = lambda s : datetime.datetime.strptime(s, self.view.time_format)
        
        field = self.view.datetime_field
        if field is None:
            raise ParseError("Need to select a field for the timestamps")
        
        ts = []
        for row in self.data:
            try:
                ts.append(parser(row[field]))
            except Exception as ex:
                self._logger.debug("Timestamp parsing error was %s / %s", type(ex), ex)
                raise ParseError(("Cannot understand the data/time string '{}'.\n" +
                    "Make sure you have selected the correct field for the timestamps.  " +
                    "If necessary, try entering a specific timestamp format.").format(row[field]))
        return ts
