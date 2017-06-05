"""
util
~~~~

Various utility routines for working with `tkinter`.
"""

import tkinter as tk

NSEW = tk.N + tk.S + tk.E + tk.W

def screen_size(root):
    """Returns (width, height).
    
    :param root: A valid window object
    """
    # https://stackoverflow.com/questions/3949844
    return (root.winfo_screenwidth(), root.winfo_screenheight())

def centre_window(window, width, height):
    """Set the window to be of the given size, centred on the screen."""
    w, h = screen_size(window)
    x = (w - width) // 2
    y = (h - height) // 2
    window.geometry("{}x{}+{}+{}".format(width, height, x, y))

def centre_window_percentage(window, width_percentage, height_percentage):
    """Set the window to be the given percentages of the total screen size,
    cented on the screen."""
    w, h = screen_size(window)
    centre_window(window, w * width_percentage // 100, h * height_percentage // 100)

def stretchy_columns(window, columns):
    """Set all the columns to have a "weight" of 1
    
    :param window: Window like object to call columnconfigure on
    :param columns: Iterable of columns to set
    """
    for i in columns:
        window.columnconfigure(i, weight=1)

def stretchy_rows(window, columns):
    """Set all the rows to have a "weight" of 1
    
    :param window: Window like object to call rowconfigure on
    :param columns: Iterable of rows to set
    """
    for i in columns:
        window.rowconfigure(i, weight=1)

class Validator():
    """Provide some user-friendly way to validate the contents of a
    `tkinter.Entry` widget.  By default, all entries are valid, so this class
    can also be used as an over-engineered way to get notification of a change.

    :param widget: The widget to bind to
    :param variable: The `tkinter.StringVar` which is bound to the widget.
    :param callback: Optional function-like object to call when the variable changes.
    """
    def __init__(self, widget, variable, callback=None):
        self._widget = widget
        self._variable = variable
        self._callback = callback
        self._old_value = ""
        cmd1 = self._widget.register(self._validate)
        self._widget["validatecommand"] = (cmd1, "%P", "%V")
        self._widget["validate"] = "focus"

    def _reset(self):
        self._variable.set(self._old_value)

    def _validate(self, val, why):
        if why == "focusin":
            self._old_value = self._variable.get()
        elif why == "focusout":
            if not self.validate(val):
                self._widget.after_idle(self._reset)
            elif self._callback is not None:
                self._widget.after_idle(self._callback)
        else:
            raise ValueError("Unexpected event")
        return True

    def validate(self, value):
        """Should check if the value is acceptable, or not.

        :param value: String of the value to check.

        :return: True if the value is acceptable; False otherwise.
        """
        return True


class FloatValidator(Validator):
    """A :class:`Validator` which only accepts values which are empty, or can
    parse to a python `float`.

    :param allow_empty: If True, allow "" as a value; otherwise not.
    """
    def __init__(self, widget, variable, callback=None, allow_empty=False):
        super().__init__(widget, variable, callback)
        self._allow_empty = allow_empty

    def validate(self, value):
        if value == "" and self._allow_empty:
            return True
        try:
            float(value)
        except:
            return False
        return True


def auto_wrap_label(label, padding=0):
    """Add a binding to a :class:`tk.Label` or :class:`ttk.Label` object so
    that when the label is resized, the text wrap length is automatically
    adjusted.

    :param label: The label object to bind to.
    :param padding: The padding to substract from the width; defaults to 0.
    """
    def callback(event):
        event.widget["wraplength"] = event.width - padding
    label.bind("<Configure>", callback)
