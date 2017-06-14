"""
tooltips
~~~~~~~~

Simple way to add tooltips to a widget.

Idea from https://stackoverflow.com/questions/3221956/what-is-the-simplest-way-to-make-tooltips-in-tkinter
"""

import datetime
import tkinter as tk
import tkinter.ttk as ttk

class ToolTip():
    """Class to display a tooltip over a widget.  Works by monitoring when the
    pointer moves over (or off) the widget.  If the pointer moves over a widget
    and stays over the widget for a timeout, then the tooltip is displayed.
    The tooltip will be removed once the pointer leaves the widget.
    
    Change settings by adjusting the attributes.
    
    The tooltip is displayed using a `ttk.Label` instance.  You may subclass
    and override :method:`configure_label` to change the appearance.
    
    :param widget: The widget to monitor.
    :param text: The text to display.
    """
    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self.timeout = 500
        self.width = 250

        self._future_id = None
        self._tool_tip_window = None
        self._widget.bind("<Enter>", self._enter, add=True)
        self._widget.bind("<Leave>", self._leave, add=True)
        self._widget.bind("<ButtonPress>", self._leave, add=True)
    
    def _enter(self, event=None):
        try:
            if str(self._widget["state"]) == tk.DISABLED:
                return
        except Exception as ex:
            pass
        self._start_timer()
        
    def _leave(self, event=None):
        self._cancel_timer()
        self._hide_text()

    def _start_timer(self):
        self._future_id = self._widget.after(self._timeout, self._show_text)
        
    def _cancel_timer(self):
        future = self._future_id
        self._future_id = None
        if future is not None:
            self._widget.after_cancel(future)

    def _hide_text(self):
        tw = self._tool_tip_window
        self._tool_tip_window = None
        if tw is not None:
            tw.destroy()
    
    def _show_text(self):
        self._hide_text()
        x, y = self._widget.winfo_pointerx() + 15, self._widget.winfo_pointery() + 5
        self._tool_tip_window = tk.Toplevel(self._widget)
        self._tool_tip_window.wm_overrideredirect(True)
        self._tool_tip_window.wm_geometry("+{}+{}".format(x, y))
        label = ttk.Label(self._tool_tip_window, text=self._text,
                          wraplength = self._width, anchor="center")
        self.configure_label(label)
        label.grid(ipadx=2, ipady=2)

    def configure_label(self, label):
        label["background"] = "#ffffff"
        label["relief"] = "solid"
        label["borderwidth"] = 1
        label["justify"] = "left"
        
    @property
    def timeout(self):
        """Time before the tooltip is displayed.  Returns a
        :class:`datetime.timedelta` instance.  May be set with such an instance
        or an integer number of milliseconds.
        """
        return datetime.timedelta(seconds = self._timeout / 1000)
    
    @timeout.setter
    def timeout(self, value):
        try:
            self._timeout = int(value.total_seconds() * 1000)
        except:
            self._timeout = value
            
    @property
    def width(self):
        """Maximum width of the tooltop before text is wrapped."""
        return self._width
    
    @width.setter
    def width(self, value):
        self._width = value
        
    @property
    def text(self):
        """The text which is displayed."""
        return self._text
    
    @text.setter
    def text(self, value):
        self._text = value


class ToolTipYellow(ToolTip):
    """As :class:`ToolTip` but with a yellow background."""
    def configure_label(self, label):
        super().configure_label(label)
        label["background"] = "#ffff99"

        
# A demo / test
if __name__ == "__main__":
    root = tk.Tk()
    #import tkinter.ttk as ttk
    button = ttk.Button(root, text="Button 1")
    button.grid(padx=10, pady=10)
    ToolTip(button, "Some text here")
    button = ttk.Button(root, text="Button 2")
    button.grid(padx=10, pady=10)
    ToolTip(button, "For instance, on the planet Earth, man had always assumed "
        +"that he was more intelligent than dolphins because he had achieved "
        +"so much—the wheel, New York, wars and so on—whilst all the dolphins "
        +"had ever done was muck about in the water having a good time. But "
        +"conversely, the dolphins had always believed that they were far more "
        +"intelligent than man—for precisely the same reasons.")
    entry = ttk.Entry(root)
    entry.grid(padx=10, pady=10)
    ToolTipYellow(entry, "Some more text")
    root.mainloop()
