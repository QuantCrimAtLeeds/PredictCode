"""
import_file_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.font as tkfont
import tkinter.filedialog as tk_fd
import tkinter.ttk as ttk
from . import util
from . import simplesheet
from .. import funcs
from ..import_file_model import InitialData, CoordType
import functools


def get_file_name():
    return tk_fd.askopenfilename(defaultextension=".csv",
                                 title="Please select a CSV file to open")


class LoadFileProgress(tk.Frame):
    def __init__(self):
        super().__init__()
        self.grid()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        bar = ttk.Progressbar(self, mode="indeterminate")
        bar.grid(sticky=util.NSEW)
        bar.start()




class ImportFileView(tk.Frame):
    def __init__(self, model, controller):
        super().__init__()
        self.model = model
        self.controller = controller
        util.centre_window_percentage(self.master, 80, 60)
        self.master.minsize(400, 300)
        self.grid(sticky=util.NSEW)
        util.stretchy_columns(self, [0])
        self._add_widgets()
                
    def _add_top_treeview(self, parent):
        text = funcs.string_ellipse("Input filename: " + self.model.filename, 100)
        label = ttk.Label(parent, anchor=tk.W, text=text)
        label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        text = "Rows of data: {}".format(self.model.rowcount)
        label = ttk.Label(parent, anchor=tk.W, text=text)
        label.grid(row=0, column=1, sticky=tk.W)
        
        self.unprocessed = simplesheet.SimpleSheet(parent)
        self.unprocessed.grid(row=1, column=0, columnspan=3, sticky=util.NSEW, padx=5)
        self.unprocessed.set_columns(self.model.header)
        font = tkfont.Font()
        for c, _ in enumerate(self.model.header):
            width = max(font.measure(r[c]) for r in self.model.firstrows)
            width = int(width * 0.9)
            width = max(30, width)
            self.unprocessed.set_column_width(c, width)
        for r, row_data in enumerate(self.model.firstrows):
            self.unprocessed.add_row()
            for c, entry in enumerate(row_data):
                self.unprocessed.set_entry(r, c, entry)
        self.unprocessed.height = len(self.model.firstrows)
        sx = self.unprocessed.xscrollbar(parent)
        sx.grid(row=2, column=0, columnspan=3, sticky=(tk.E, tk.W), padx=5)

    def _add_bottom_treeview(self, parent):
        self.processed = simplesheet.SimpleSheet(parent)
        self.processed.grid(row=0, column=0, sticky=util.NSEW, padx=5)#, pady=5)
        self.processed.set_columns(["Timestamp", "X Coord", "Y Coord"])
        for c in range(3):
            self.processed.set_column_width(c, 70)
        self.new_parse_data()
        sx = self.processed.xscrollbar(parent)
        sx.grid(row=1, column=0, sticky=(tk.E, tk.W), padx=5)

    def _widget_changed(self, e, our_index):
        w = e.widget
        self._widget_selections[our_index] = w.current()
        if our_index == 0:
            self.controller.notify_datetime_field(self.datetime_field)
        elif our_index == 1:
            self.controller.notify_xcoord_field(self.xcoord_field)
        elif our_index == 2:
            self.controller.notify_ycoord_field(self.ycoord_field)
        else:
            raise ValueError()

    def _add_options(self, parent):
        util.stretchy_columns(parent, [3])
        for r, t in enumerate(["Timestamp heading:", "X Coordinate heading:", "Y Coordinate heading:"]):
            label = ttk.Label(parent, text=t)
            label.grid(row=r, column=0, sticky=tk.W, padx=5, pady=5)
        timestamp_header = tk.StringVar()
        xcoord_header = tk.StringVar()
        ycoord_header = tk.StringVar()
        self._widget_selections = []
        for r, t in enumerate([timestamp_header, xcoord_header, ycoord_header]):
            cbox = ttk.Combobox(parent, height=5, state="readonly", textvariable=t)
            cbox["values"] = self.model.header
            cbox.bind("<<ComboboxSelected>>", functools.partial(self._widget_changed, our_index=r))
            self._widget_selections.append(None)
            cbox.grid(row=r, column=1, padx=5)

        frame = ttk.Frame(parent)
        frame.grid(row=0, column=2, sticky=tk.W)
        self._ts_format_options(frame)

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=2, sticky=tk.W)
        self._coord_options(frame)

        frame = ttk.Frame(parent)
        frame.grid(row=2, column=2, sticky=tk.W)
        self._proj_coord_options(frame)

        frame = ttk.LabelFrame(parent, text="Error messages")
        frame.grid(row=0, column=3, rowspan=3, sticky=util.NSEW, padx=5, pady=5)
        util.stretchy_columns(frame, [0])
        self._error_message = ttk.Label(frame, text="", wraplength=250)
        self._error_message.grid(row=0, column=0, sticky=util.NSEW)
        util.auto_wrap_label(self._error_message, 5)

    def _time_format_changed(self):
        self.controller.notify_time_format(self.time_format)

    def _ts_format_options(self, frame):
        label = ttk.Label(frame, text="Timestamp format:")
        label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self._ts_format = tk.StringVar()
        tmft_entry = ttk.Entry(frame, width=20, textvariable=self._ts_format)
        util.Validator(tmft_entry, self._ts_format, callback=self._time_format_changed)
        tmft_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        text = "Leave blank to attempt auto-detection"
        label = ttk.Label(frame, text=text)
        label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

    def _coord_options(self, frame):
        self._coord_type = tk.IntVar()
        radio1 = ttk.Radiobutton(frame, text="Longitude/Latitude", value=CoordType.LonLat.value, variable=self._coord_type, command=self._coord_type_cmd)
        radio1.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        radio2 = ttk.Radiobutton(frame, text="Projected coordinates", value=CoordType.XY.value, variable=self._coord_type, command=self._coord_type_cmd)
        radio2.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
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
        b = ttk.Button(frame, text="Meters", command=self._to_meters)
        b.grid(row=0, column=0, padx=5, pady=5)
        self._coord_type_widgets = [b]
        b = ttk.Button(frame, text="Feet", command=self._to_feet)
        b.grid(row=0, column=1, padx=5, pady=5)
        self._coord_type_widgets.append(b)
        label = ttk.Label(frame, text="Scale to meters:")
        label.grid(row=0, column=2, padx=5, pady=5)
        self._coord_type_widgets.append(label)
        self._proj_convert = tk.StringVar()
        entry = ttk.Entry(frame, width=20, textvariable=self._proj_convert)
        util.FloatValidator(entry, self._proj_convert, callback=self._meters_conversion_changed)
        entry.grid(row=0, column=3, pady=5)
        self._coord_type_widgets.append(entry)
        self._proj_convert.set("1.0")

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
        frame = ttk.LabelFrame(self, text="Input file")
        frame.grid(row=0, column=0, columnspan=2, sticky=util.NSEW, padx=5, pady=5)
        util.stretchy_columns(frame, [0,1,2])
        self._add_top_treeview(frame)

        frame = ttk.LabelFrame(self, text="Conversion options")
        frame.grid(row=1, column=0, columnspan=2, sticky=util.NSEW, padx=5, pady=5)
        self._add_options(frame)

        frame = ttk.LabelFrame(self, text="Processed data")
        frame.grid(row=2, column=0, sticky=util.NSEW, padx=5, pady=5)
        self._add_bottom_treeview(frame)
        util.stretchy_columns(frame, [0])
        util.stretchy_rows(frame, [0])

        frame = ttk.Frame(self)
        frame.grid(row=2, column=1, padx=5, pady=5)
        self.okay_button = ttk.Button(frame, text="Continue")
        self.allow_continue(False)
        self.okay_button.grid(row=0, padx=5, pady=15)
        self.cancel_button = ttk.Button(frame, text="Cancel")
        self.cancel_button.grid(row=1, padx=5, pady=15)
        
    def new_parse_data(self):
        self.processed.remove_rows()
        if self.model.processed is None:
            self.processed.height = 2
        else:
            timestamps, xcoords, ycoords = self.model.processed
            font = tkfont.Font()
            for i, ts in enumerate(timestamps):
                self.processed.add_row()
                self.processed.set_entry(i, 0, ts)
            for i, (x, y) in enumerate(zip(xcoords, ycoords)):
                self.processed.set_entry(i, 1, x)
                self.processed.set_entry(i, 2, y)
            for c, source in enumerate([timestamps, xcoords, ycoords]):
                width = max(font.measure(s) for s in source)
                width = max(90, int(width * 0.9))
                self.processed.set_column_width(c, width)
            self.processed.height = len(timestamps)

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
        return CoordType.fromvalue( self._coord_type.get() )

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

    def set_error(self, error):
        self._error_message["text"] = error