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

class DatePicker(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._first_day = "Mon"
        now = datetime.date.today()
        self._month_year = (now.month, now.year)
        self._selected_date = now
        
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

    def _left(self):
        """Move one month into past."""
        m, y = self.month_year
        m -= 1
        if m == 0:
            y -= 1
            m = 12
        self.month_year = (m, y)
        self._refresh_month_year()

    def _right(self):
        """Move one month into future."""
        m, y = self.month_year
        m += 1
        if m == 13:
            y += 1
            m = 1
        self.month_year = (m, y)
        self._refresh_month_year()

    def day_to_text(self, day):
        """0 = Monday, 1 = Tuesday and so forth"""
        if day < 0 or day > 6:
            raise ValueError()
        d = datetime.date(year=2017, month=6, day=5+day)
        return d.strftime("%a")

    def _make_day_labels(self):
        for label in self._day_labels:
            label.destroy()
        self._day_labels = []
        if self.first_day_of_week == "Sun":
            day = 6
        else:
            day = 0
        for c in range(7):
            label = ttk.Label(self, text=self.day_to_text(day), anchor=tk.CENTER)
            label.grid(row=1, column=c, sticky=tk.EW)
            self._day_labels.append(label)
            day = (day + 1) % 7

    def _day_to_column(self, day):
        """ValueError on day out of range."""
        m, y = self.month_year
        d = datetime.date(year = y, month = m, day = day)
        column = d.weekday()
        if self.first_day_of_week == "Sun":
            column = (column + 1) % 7
        return column

    def _make_date_grid(self):
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
        if self.month_year == (self.selected_date.month, self.selected_date.year):
            b = self._day_buttons[ self.selected_date.day - 1 ]
            b["bg"] = "#8888ff"

    def _date_selected(self, day):
        m, y = self.month_year
        self._selected_date = datetime.date(year=y, month=m, day=day)
        for i, b in enumerate(self._day_buttons):
            if i + 1 == day:
                b["bg"] = "#8888ff"
            else:
                b["bg"] = self._bg_colour

    def _refresh_month_year(self):
        m, y = self.month_year
        d = datetime.date(year=y, month=m, day=1)
        self._month_label["text"] = d.strftime("%b %Y")
        self._make_date_grid()

    def _add_widgets(self):
        ttk.Button(self, text="<<", command=self._left).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(self, text=">>", command=self._right).grid(row=0, column=5, sticky=tk.EW)
        self._month_label = ttk.Label(self, anchor=tk.CENTER)
        self._month_label.grid(row=0, column=2, columnspan=3, sticky=tk.EW)
        
        self._day_labels = []
        self._make_day_labels()

        self._day_buttons = []
        self._refresh_month_year()

    @property
    def first_day_of_week(self):
        """The first day of the week, either "Mon" or "Sun"."""
        return self._first_day

    @first_day_of_week.setter
    def first_day_of_week(self, value):
        if value not in {"Mon", "Sun"}:
            raise ValueError("Should be 'Mon' or 'Sun'.")
        self._first_day = value
        self._make_day_labels()
        self._make_date_grid()

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
            self._refresh_month_year()
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

if __name__ == "__main__":
    root = tk.Tk()
    util.stretchy_columns(root, [0])
    util.stretchy_rows(root, [0])
    dp = DatePicker(root)
    dp["borderwidth"] = 5
    dp["relief"] = "groove"
    dp.grid(sticky=tk.NSEW)
    frame = tk.Frame(root)
    frame.grid()

    def sunday():
        dp.first_day_of_week = "Sun"
    def monday():
        dp.first_day_of_week = "Mon"
    def go_month():
        dp.month_year = (1, 2016)

    ttk.Button(frame, text="Quit", command=root.quit).grid(row=0, column=0)
    ttk.Button(frame, text="Sunday", command=sunday).grid(row=1, column=0)
    ttk.Button(frame, text="Monday", command=monday).grid(row=1, column=1)
    ttk.Button(frame, text="Jan 2016", command=go_month).grid(row=2, column=0)

    root.mainloop()