"""
date_picker
~~~~~~~~~~~

The classic date picker widget, in pure Python / Tkinter.
"""

if __name__ == "__main__":
    # Allow runnig from root of project as a test/demo
    import os, sys
    sys.path.insert(0, os.path.abspath("."))

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import datetime

class DatePicker():
    """A implementation of the classic "date picker" widget.  The underlying
    widget, accessible as a property, is a :class:`tk.Frame`.  All the
    sub-widgets will scale with the size of this frame.

    Various properties can be changed to change the styling of the widget.

    The selected date can be retrieved from :attr:`selected_date` (this
    property can also be set).   Set the :attr:`command` to get a callback
    when the selection changes.

    :param parent: The parent `tk` widget of the frame.
    """
    def __init__(self, parent=None):
        self._command = None
        self._first_day = "Mon"
        now = datetime.date.today()
        self._month_year = (now.month, now.year)
        self._selected_date = now
        self._selected_colour = "#6666ff"

        self._view = _DatePickerView(parent, self)

    def day_to_text(self, day):
        """0 = Monday, 1 = Tuesday and so forth"""
        if day < 0 or day > 6:
            raise ValueError()
        d = datetime.date(year=2017, month=6, day=5+day)
        return d.strftime("%a")

    @property
    def command(self):
        """A callable with signature `selected(date)` to be called when a new
        date is selected.  Here `date` will be a :class:`datetime.date` object.
        """
        return self._command

    @command.setter
    def command(self, value):
        self._command = value

    @property
    def widget(self):
        """Get the :class:`tk.Frame` widget which contains the date picker."""
        return self._view

    @property
    def first_day_of_week(self):
        """The first day of the week, either "Mon" or "Sun"."""
        return self._first_day

    @first_day_of_week.setter
    def first_day_of_week(self, value):
        if value not in {"Mon", "Sun"}:
            raise ValueError("Should be 'Mon' or 'Sun'.")
        self._first_day = value
        self._view.make_day_labels()
        self._view.make_date_grid()

    @property
    def month_year(self):
        """The currently viewed page `(month, year)` as integers, with 1 = January,
        2 = February etc."""
        return self._month_year

    @month_year.setter
    def month_year(self, value):
        try:
            try:
                month, year = value
            except Exception:
                raise ValueError()
            month, year = int(month), int(year)
            if month < 1 or month > 12:
                raise ValueError
            self._month_year = (month, year)
            self._view.refresh_month_year()
        except ValueError:
            raise ValueError("Should be pair (month, year) of integers.")

    @property
    def selected_date(self):
        """The currently selected date."""
        return self._selected_date

    @selected_date.setter
    def selected_date(self, value):
        try:
            y, m, d = value.year, value.month, value.day
            value = datetime.date(year=y, month=m, day=d)
        except Exception:
            raise ValueError("Not a valid date")
        self._selected_date = value
        self._view.refresh_month_year()
        if self._command is not None:
            self._command(value)

    @property
    def selected_colour(self):
        """The colour of the button which indicates the selected date."""
        return self._selected_colour

    @selected_colour.setter
    def selected_colour(self, value):
        self._selected_colour = value
        self._view.make_date_grid()

    def show_selected_date(self):
        """Add a final row to the widget showing the currently selected date.
        Will rescale the height of the widget to accommodate the extra row.
        """
        self._view.show_date()

    def hide_selected_date(self):
        """Hide the selected date.  Will rescale the height of the widget to
        take account of the now missing bottom row."""
        self._view.hide_date()


class _DatePickerView(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self._model = model
        self._show = False
        util.stretchy_columns(self, range(7))
        util.stretchy_rows(self, range(7))
        self._add_widgets()
        self._make_square()

    def _make_square(self):
        self.update_idletasks()
        height = self.winfo_reqheight()
        self.grid_propagate(0)
        self["width"] = int(height * 1.3)
        self["height"] = height

    def show_date(self):
        if not self._show:
            self._show = True
            self._selected_label = ttk.Label(self, anchor=tk.CENTER)
            self._selected_label.grid(row=7, column=0, columnspan=7, sticky=tk.EW)
            util.stretchy_rows(self, range(8))
            self._update_date()
            self["height"] = int(self["height"] * 8/7)

    def _update_date(self):
        if self._show:
            fmt = "%A %d %B %Y"
            self._selected_label["text"] = self._model.selected_date.strftime(fmt)

    def hide_date(self):
        if self._show:
            self._selected_label.destroy()
            self._selected_label = None
            util.stretchy_rows(self, range(7))
            self["height"] = int(self["height"] * 7/8)
        self._show = False

    def _left(self):
        """Move one month into past."""
        m, y = self._model.month_year
        m -= 1
        if m == 0:
            y -= 1
            m = 12
        self._model.month_year = (m, y)

    def _right(self):
        """Move one month into future."""
        m, y = self._model.month_year
        m += 1
        if m == 13:
            y += 1
            m = 1
        self._model.month_year = (m, y)

    def make_day_labels(self):
        for label in self._day_labels:
            label.destroy()
        self._day_labels = []
        if self._model.first_day_of_week == "Sun":
            day = 6
        else:
            day = 0
        for c in range(7):
            label = ttk.Label(self, text=self._model.day_to_text(day), anchor=tk.CENTER)
            label.grid(row=1, column=c, sticky=tk.EW)
            self._day_labels.append(label)
            day = (day + 1) % 7

    def _day_to_column(self, day):
        """ValueError on day out of range."""
        m, y = self._model.month_year
        d = datetime.date(year = y, month = m, day = day)
        column = d.weekday()
        if self._model.first_day_of_week == "Sun":
            column = (column + 1) % 7
        return column

    def make_date_grid(self):
        for b in self._day_buttons:
            b.destroy()
        self._day_buttons = []

        done = False
        row = 2
        for day in range(1, 32):
            try:
                column = self._day_to_column(day)
            except ValueError:
                break
            if column == 6:
                done = True
            if column == 0 and done:
                done = False
                row += 1
                if row == 7:
                    row = 2
            # `d=day` makes sure we capture `day` by value, not reference
            cmd = lambda d=day : self._date_selected(d)
            b = tk.Button(self, text=str(day), width=20, relief="flat", command=cmd)
            b.grid(column=column, row=row, sticky=tk.NSEW)
            self._day_buttons.append(b)

        self._bg_colour = self._day_buttons[0]["bg"]
        if self._model.month_year == (self._model.selected_date.month, self._model.selected_date.year):
            b = self._day_buttons[ self._model.selected_date.day - 1 ]
            b["bg"] = self._model.selected_colour

        self._update_date()

    def _date_selected(self, day):
        m, y = self._model.month_year
        self._model.selected_date = datetime.date(year=y, month=m, day=day)

    def refresh_month_year(self):
        m, y = self._model.month_year
        d = datetime.date(year=y, month=m, day=1)
        self._month_label["text"] = d.strftime("%b %Y")
        self.make_date_grid()

    def _add_widgets(self):
        ttk.Button(self, text="<<", command=self._left).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(self, text=">>", command=self._right).grid(row=0, column=5, sticky=tk.EW)
        self._month_label = ttk.Label(self, anchor=tk.CENTER)
        self._month_label.grid(row=0, column=2, columnspan=3, sticky=tk.EW)
        
        self._day_labels = []
        self.make_day_labels()

        self._day_buttons = []
        self.refresh_month_year()


class PopUpDatePickerView(util.ModalWindow):
    """View for PopUpDatePicker"""
    def __init__(self, parent):
        super().__init__(parent, "", no_border=True)
        x, y = parent.winfo_pointerx() + 15, parent.winfo_pointery() + 5
        self.wm_geometry("+{}+{}".format(x, y))
        self.close_on_click_away()

    def add_widgets(self):
        frame = tk.Frame(self, bd=2, relief="ridge")
        frame.grid(sticky=tk.NSEW)
        self._dp = DatePicker(frame)
        self._dp.widget.grid(sticky=tk.NSEW)

    @property
    def date_picker(self):
        return self._dp


class PopUpDatePicker():
    """Display a date picker in a modal, popup dialog which closes immediately
    the user clicks away, or selects a date.

    :param parent: The parent window to be modal from
    :param widget: The widget to bind to: clicking in this widget will open
      the pop-up
    :param source_callable: A callable object which returns a `date` object
      which is the date the pop-up should initially display
    :param result_callable: A callable with signature `result_callable(date)`
      which is called if the user selects a date.  If the user clicks away,
      this is not called.
    """
    def __init__(self, parent, widget, source_callable, result_callable):
        self._window = None
        self._parent = parent
        self._widget = widget
        self._widget.bind("<Button-1>", self._show)
        self._source = source_callable
        self._sink = result_callable

    def _cmd(self, date):
        self._date = date
        self._dp_widget.destroy()

    def _show(self, e):
        self._dp_widget = PopUpDatePickerView(self._parent)
        self._date = None
        date = self._source()
        self._dp_widget.date_picker.selected_date = date
        self._dp_widget.date_picker.month_year =(date.month, date.year)
        self._dp_widget.date_picker.command = self._cmd
        self._parent.wait_window(self._dp_widget)
        if self._date is not None:
            self._sink(self._date)
        # Seems that I need to do this to stop the keyboard focus being lost
        # forever to the now hidden top window!
        self._widget.focus_force()


if __name__ == "__main__":
    root = tk.Tk()
    util.stretchy_columns(root, [0])
    util.stretchy_rows(root, [0])
    dp = DatePicker(root)
    dp.widget["borderwidth"] = 5
    dp.widget["relief"] = "groove"
    dp.widget.grid(sticky=tk.NSEW)
    frame = tk.Frame(root)
    frame.grid()

    def sunday():
        dp.first_day_of_week = "Sun"
    def monday():
        dp.first_day_of_week = "Mon"
    def go_month():
        dp.month_year = (1, 2016)
    def select_tomorrow():
        d = datetime.datetime.now() + datetime.timedelta(days=1)
        dp.selected_date = d
    def blue():
        dp.selected_colour = "#6666ff"
    def green():
        dp.selected_colour = "#66ff66"

    ttk.Button(frame, text="Quit", command=root.quit).grid(row=0, column=0)
    ttk.Button(frame, text="Sunday", command=sunday).grid(row=1, column=0)
    ttk.Button(frame, text="Monday", command=monday).grid(row=1, column=1)
    ttk.Button(frame, text="Jan 2016", command=go_month).grid(row=2, column=0)
    ttk.Button(frame, text="Tomorrow", command=select_tomorrow).grid(row=2, column=1)
    ttk.Button(frame, text="Blue", command=blue).grid(row=3, column=0)
    ttk.Button(frame, text="Green", command=green).grid(row=3, column=1)
    ttk.Button(frame, text="Show", command=dp.show_selected_date).grid(row=4, column=0)
    ttk.Button(frame, text="Hide", command=dp.hide_selected_date).grid(row=4, column=1)
    label = ttk.Label(frame)
    label.grid(row=5, column=0)
    def selected(d):
        label["text"] = str(d)
    dp.command = selected

    root.mainloop()