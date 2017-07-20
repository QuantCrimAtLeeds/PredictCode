"""
import_file_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.filedialog as tk_fd
import tkinter.ttk as ttk
import tkinter.messagebox
from . import util
from . import simplesheet
from . import tooltips
from .. import funcs
from ..import_file_model import CoordType
import functools

_text = {
    "getfile" : "Please select a CSV file to open",
    "cancel" : "Cancel",
    "continue" : "Continue",
    "inputfn" : "Input filename: ",
    "rows" : "Rows of data:",
    "out_cols" : ["Timestamp", "X Coord", "Y Coord"],
    "headings" : ["Timestamp heading:", "X Coordinate heading:", "Y Coordinate heading:"],
    "error_msgs" : "Error messages",
    "tsformat" : "Timestamp format:",
    "autodetect" : "Leave blank to attempt auto-detection",
    "lonlat" : "Longitude/Latitude",
    "proj" : "Projected coordinates",
    "meters" : "Meters",
    "feet" : "Feet",
    "scale" : "Scale to meters:",
    "input" : "Input file",
    "con" : "Conversion options",
    "proc" : "Processed data",
    "input_tt" : "The input comma separated file.",
    "rows_tt" : "The total number of rows in the input file.  We display only the first five.",
    "output_tt" : "The timestamps and coordinates as processed from the input file using the current settings.  Check that these appear to be correct.",
    "headers_tt" : ["Select the header which corresponds to the timestamp of the crime event",
                    "Select the header which corresponds to the X coordinate or longitude of the crime event",
                    "Select the header which corresponds to the Y coordinate or latitude of the crime event"],
    "tsformat_tt" : ("Specify the format of the timestamps, or leave blank to attempt auto-detecting.\n" +
                     "If you need to specify, then use the standard format.  For example, '%Y-%m' would specify" +
                     "that the input gives only the year and month as '2016-03' or '1978-11'.\n" +
                     "Standard identifiers are:\n" +
                     "%Y / %y for year as 2017 or 17\n"),
    "lonlat_tt" : "Select to show that the input data is in longitude/latitude format.",
    "proj_tt" :  "Select to show that the input data is in meters, feet, or some other length unit.",
    "meters_tt" : "The input data is in meters",
    "feet_tt" : "The input data is in feet (12 inches)",
    "proj_con_tt" : ("Specify a custom value.  The coordinates will be multiplied by this value to convert to meters.  " +
                     "For example, if the input data is in kilometers, enter '1000' here."),
    "error_tt" : "A message describing the current problem when trying to process the input data, along with a hint as to how to fix the problem.",
    "main_ct" : "Main crime type field:",
    "2nd_ct" : "Secondary crime type field:",
    "main_tt" : "Optionally, select the field which describes the crime type.",
    "2nd_tt" : "Optionally, select a second field which describes the crime sub-type.",
    "crime_type" : "Crime type "
}

_COORD_FORMAT = "{:0.2f}"
_LONLAT_FORMAT = "{:0.6f}"
_TIME_FORMAT = "%d %b %Y %H:%M:%S"

def get_file_name():
    return tk_fd.askopenfilename(defaultextension=".csv", title=_text["getfile"])


class LoadFileProgress(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid(sticky=tk.W+tk.E)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        bar = ttk.Progressbar(self, mode="indeterminate")
        bar.grid(sticky=util.NSEW)
        bar.start()


def display_error(message):
    tkinter.messagebox.showerror("Error", message)


class ImportFileView(tk.Frame):
    def __init__(self, parent, model, controller):
        super().__init__(parent)
        self.model = model
        self.controller = controller
        util.centre_window_percentage(self.master, 80, 60)
        self.master.minsize(400, 300)
        self.master.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grid(sticky=util.NSEW)
        util.stretchy_columns(self, [0])
        self._add_widgets()
        self.controller.notify_coord_format(CoordType.LonLat, initial=True)
        self.controller.notify_meters_conversion(1.0, initial=True)
        self.controller.notify_time_format("", initial=True)

    def _add_top_treeview(self, parent):
        text = funcs.string_ellipse(_text["inputfn"] + self.model.filename, 100)
        label = ttk.Label(parent, anchor=tk.W, text=text)
        label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        text = _text["rows"] + " {}".format(self.model.rowcount)
        label = ttk.Label(parent, anchor=tk.W, text=text)
        label.grid(row=0, column=1, sticky=tk.W)
        tooltips.ToolTipYellow(label, _text["rows_tt"])
        
        self.unprocessed = simplesheet.SimpleSheet(parent)
        self.unprocessed.grid(row=1, column=0, columnspan=3, sticky=util.NSEW, padx=5)
        self.unprocessed.set_columns(self.model.header)
        
        measurer = util.TextMeasurer()
        for c, _ in enumerate(self.model.header):
            width = measurer.measure(r[c] for r in self.model.firstrows)
            self.unprocessed.set_column_width(c, width)
        for r, row_data in enumerate(self.model.firstrows):
            self.unprocessed.add_row()
            for c, entry in enumerate(row_data):
                self.unprocessed.set_entry(r, c, entry)
        self.unprocessed.height = len(self.model.firstrows)
        sx = self.unprocessed.xscrollbar(parent)
        sx.grid(row=2, column=0, columnspan=3, sticky=(tk.E, tk.W), padx=5)
        tooltips.ToolTipYellow(self.unprocessed.widget, _text["input_tt"])

    def _add_bottom_treeview(self, parent):
        self.processed = simplesheet.SimpleSheet(parent)
        self.processed.grid(row=0, column=0, sticky=util.NSEW, padx=5)#, pady=5)
        self.processed.set_columns(_text["out_cols"])
        for c in range(3):
            self.processed.set_column_width(c, 70)
        self.new_parse_data()
        sx = self.processed.xscrollbar(parent)
        sx.grid(row=1, column=0, sticky=(tk.E, tk.W), padx=5)
        tooltips.ToolTipYellow(self.processed.widget, _text["output_tt"])

    def _widget_changed(self, e, our_index):
        w = e.widget
        self._widget_selections[our_index] = w.current()
        if our_index == 0:
            self.controller.notify_datetime_field(self.datetime_field)
        elif our_index == 1:
            self.controller.notify_xcoord_field(self.xcoord_field)
        elif our_index == 2:
            self.controller.notify_ycoord_field(self.ycoord_field)
        elif our_index == 3:
            self.controller.notify_crime_field(0, self.main_crime_field)
        elif our_index == 4:
            self.controller.notify_crime_field(1, self.second_crime_field)
        else:
            raise ValueError()

    def _add_options(self, parent):
        util.stretchy_columns(parent, [3])
        for r, t in enumerate(_text["headings"]):
            label = ttk.Label(parent, text=t)
            label.grid(row=r, column=0, sticky=tk.W, padx=5, pady=5)
        self._widget_selections = []
        for r, tttext in enumerate(_text["headers_tt"]):
            cbox = ttk.Combobox(parent, height=5, state="readonly")
            cbox["values"] = self.model.header
            cbox.bind("<<ComboboxSelected>>", functools.partial(self._widget_changed, our_index=r))
            self._widget_selections.append(None)
            cbox.grid(row=r, column=1, padx=5)
            tooltips.ToolTipYellow(cbox, tttext)

        frame = ttk.Frame(parent)
        frame.grid(row=0, column=2, sticky=tk.W)
        self._ts_format_options(frame)

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=2, sticky=tk.W)
        self._coord_options(frame)

        frame = ttk.Frame(parent)
        frame.grid(row=2, column=2, sticky=tk.W)
        self._proj_coord_options(frame)

        frame = ttk.LabelFrame(parent, text=_text["error_msgs"])
        frame.grid(row=0, column=3, rowspan=3, sticky=util.NSEW, padx=5, pady=5)
        tooltips.ToolTipYellow(frame, _text["error_tt"])
        util.stretchy_columns(frame, [0])
        self._error_message = ttk.Label(frame, text="", wraplength=250)
        self._error_message.grid(row=0, column=0, sticky=util.NSEW)
        util.auto_wrap_label(self._error_message, 5)

        frame = tk.Frame(parent)
        frame.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW)
        options = ["<<None>>"] + list(self.model.header)
        for i, (tt, tttext) in enumerate(zip([_text["main_ct"], _text["2nd_ct"]],
                    [_text["main_tt"], _text["2nd_tt"]])):
            ttk.Label(frame, text=tt).grid(row=0, column=i+i, sticky=tk.E, padx=5, pady=5)
            cbox = ttk.Combobox(frame, height=5, state="readonly")
            cbox["values"] = options
            cbox.bind("<<ComboboxSelected>>", functools.partial(self._widget_changed, our_index=i+3))
            cbox.grid(row=0, column=i+i+1, sticky=tk.W, padx=5)
            self._widget_selections.append(None)
            tooltips.ToolTipYellow(cbox, tttext)

    def _time_format_changed(self):
        self.controller.notify_time_format(self.time_format)

    def _ts_format_options(self, frame):
        label = ttk.Label(frame, text=_text["tsformat"])
        label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self._ts_format = tk.StringVar()
        tmft_entry = ttk.Entry(frame, width=20, textvariable=self._ts_format)
        util.Validator(tmft_entry, self._ts_format, callback=self._time_format_changed)
        tmft_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        tooltips.ToolTipYellow(tmft_entry, _text["tsformat_tt"])
        label = ttk.Label(frame, text=_text["autodetect"])
        label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

    def _coord_options(self, frame):
        self._coord_type = tk.IntVar()
        radio1 = ttk.Radiobutton(frame, text=_text["lonlat"], value=CoordType.LonLat.value, variable=self._coord_type, command=self._coord_type_cmd)
        radio1.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        tooltips.ToolTipYellow(radio1, _text["lonlat_tt"])
        radio2 = ttk.Radiobutton(frame, text=_text["proj"], value=CoordType.XY.value, variable=self._coord_type, command=self._coord_type_cmd)
        radio2.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        tooltips.ToolTipYellow(radio2, _text["proj_tt"])
        self._coord_type.set(CoordType.LonLat.value)

    def _to_meters(self):
        self._proj_convert.set("1.0")
        self._meters_conversion_changed()

    def _to_feet(self):
        # https://en.wikipedia.org/wiki/Foot_(unit)
        self._proj_convert.set("0.3048")
        self._meters_conversion_changed()

    def _meters_conversion_changed(self):
        self.controller.notify_meters_conversion(self.meters_conversion)

    def _proj_coord_options(self, frame):
        b = ttk.Button(frame, text=_text["meters"], command=self._to_meters)
        b.grid(row=0, column=0, padx=5, pady=5)
        tooltips.ToolTipYellow(b, _text["meters_tt"])
        self._coord_type_widgets = [b]
        b = ttk.Button(frame, text=_text["feet"], command=self._to_feet)
        b.grid(row=0, column=1, padx=5, pady=5)
        tooltips.ToolTipYellow(b, _text["feet_tt"])
        self._coord_type_widgets.append(b)
        label = ttk.Label(frame, text=_text["scale"])
        label.grid(row=0, column=2, padx=5, pady=5)
        self._coord_type_widgets.append(label)
        self._proj_convert = tk.StringVar()
        entry = ttk.Entry(frame, width=20, textvariable=self._proj_convert)
        tooltips.ToolTipYellow(entry, _text["proj_con_tt"])
        util.FloatValidator(entry, self._proj_convert, callback=self._meters_conversion_changed)
        entry.grid(row=0, column=3, pady=5)
        self._coord_type_widgets.append(entry)
        self._proj_convert.set("1.0")
        self._coord_type_state(tk.DISABLED)

    def _coord_type_state(self, state):
        for w in self._coord_type_widgets:
            w["state"] = state

    def _coord_type_cmd(self):
        value = self.coord_type
        if value == CoordType.LonLat:
            self._coord_type_state(tk.DISABLED)
        elif value == CoordType.XY:
            self._coord_type_state(tk.NORMAL)
        else:
            raise NotImplementedError()
        self.controller.notify_coord_format(value)

    def _add_widgets(self):
        frame = ttk.LabelFrame(self, text=_text["input"])
        frame.grid(row=0, column=0, columnspan=2, sticky=util.NSEW, padx=5, pady=5)
        util.stretchy_columns(frame, [0,1,2])
        self._add_top_treeview(frame)

        frame = ttk.LabelFrame(self, text=_text["con"])
        frame.grid(row=1, column=0, columnspan=2, sticky=util.NSEW, padx=5, pady=5)
        self._add_options(frame)

        frame = ttk.LabelFrame(self, text=_text["proc"])
        frame.grid(row=2, column=0, sticky=util.NSEW, padx=5, pady=5)
        self._add_bottom_treeview(frame)
        util.stretchy_columns(frame, [0])
        util.stretchy_rows(frame, [0])

        frame = ttk.Frame(self)
        frame.grid(row=2, column=1, padx=5, pady=5)
        self.okay_button = ttk.Button(frame, text=_text["continue"], command=self.controller.contin)
        self.allow_continue(False)
        self.okay_button.grid(row=0, padx=5, pady=15)
        self.cancel_button = ttk.Button(frame, text=_text["cancel"], command=self._cancel)
        self.cancel_button.grid(row=1, padx=5, pady=15)
        
    def _cancel(self):
        self.controller.cancel()
        
    def new_parse_data(self):
        current_num_columns = self.processed.column_count
        new_data = (self.processed.row_count == 0)
        if not new_data:
            self.processed.remove_rows()
        data = self.model.processed_data
        if data is None:
            self.processed.height = 2
        else:
            for i, ts in enumerate(data[0]):
                self.processed.add_row()
                self.processed.set_entry(i, 0, ts.strftime(_TIME_FORMAT))
            
            if self.coord_type == CoordType.LonLat:
                fmt = _LONLAT_FORMAT
            else:
                fmt = _COORD_FORMAT
            for i, (x, y) in enumerate(zip(data[1], data[2])):
                self.processed.set_entry(i, 1, fmt.format(x))
                self.processed.set_entry(i, 2, fmt.format(y))

            num_crime_type_cols = 0
            if len(data[3]) > 0:
                num_crime_type_cols = len(data[3][0])
                if 3 + num_crime_type_cols != current_num_columns:
                    cols = list(_text["out_cols"])
                    for i in range(num_crime_type_cols):
                        cols.append(_text["crime_type"] + str(i + 1))
                    self.processed.set_columns(cols)
                for row, ctypes in enumerate(data[3]):
                    for col, t in enumerate(ctypes):
                        self.processed.set_entry(row, col + 3, str(t))
            else:
                if current_num_columns != 3:
                    self.processed.set_columns(_text["out_cols"])

            if new_data or current_num_columns != 3 + num_crime_type_cols:
                measurer = util.TextMeasurer()
                for c, source in enumerate(data[:3]):
                    width = measurer.measure(source)
                    self.processed.set_column_width(c, width)
                if num_crime_type_cols > 0:
                    for c in range(num_crime_type_cols):
                        width = measurer.measure( x[c] for x in data[3] )
                        self.processed.set_column_width(c + 3, width)
            self.processed.height = len(data[0])

    def allow_continue(self, allow):
        """Should we allow the continue button to be pressed?"""
        if allow:
            self.okay_button["state"] = tk.NORMAL
        else:
            self.okay_button["state"] = tk.DISABLED

    @property
    def coord_type(self):
        """Lon/Lat or XY coords?

        :return: :class:`CoordType` enum
        """
        #return CoordType.fromvalue( self._coord_type.get() )
        return CoordType(self._coord_type.get())

    @property
    def meters_conversion(self):
        """If in XY coords, return the factor to multiple the values by to get
        to meters.
        """
        return float(self._proj_convert.get())

    @property
    def time_format(self):
        """Return the string to use for decoding date/times.  Can be blank,
        which indicates to use `dateutil.parse` (or similar automatic method).
        """
        return self._ts_format.get()

    @property
    def datetime_field(self):
        """Which field has the user selected for the date/time?  Maybe `None`."""
        return self._widget_selections[0]

    @property
    def xcoord_field(self):
        """Which field has the user selected for the X coord?  Maybe `None`."""
        return self._widget_selections[1]

    @property
    def ycoord_field(self):
        """Which field has the user selected for the Y coord?  Maybe `None`."""
        return self._widget_selections[2]

    @property
    def main_crime_field(self):
        """Which field has the user selected for the main crime type?  -1 == None"""
        sel = self._widget_selections[3]
        if sel is None:
            sel = 0
        return sel - 1

    @property
    def second_crime_field(self):
        """Which field has the user selected for the secondary crime type?  -1 == None"""
        sel = self._widget_selections[4]
        if sel is None:
            sel = 0
        return sel - 1

    def set_error(self, error):
        self._error_message["text"] = error