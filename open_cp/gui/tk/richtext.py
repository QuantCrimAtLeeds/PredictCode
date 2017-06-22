"""
richtext
~~~~~~~~

Currently, a very thin wrapper around `ttk.Label`.

In the future, the plan is to add simple HTML or markdown (or both) formatting.
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import open_cp.gui.tk.util as util

class RichText(tk.Frame):
    """Currently, a simple wrapper around `tk.Text`.
    
    :param width: Width, in characters, of the panel.
    :param height: Height, in rows, of the panel.
    :param scroll: String: if contains "h" then add a horizontal scroll bar,
      and if contains "v" add a vertical scroll bar.
    """
    def __init__(self, parent, width=None, height=None, scroll=""):
        super().__init__(parent)
        util.stretchy_rows_cols(self, [0], [0])
        self._text = tk.Text(self, relief=tk.FLAT, borderwidth=0, highlightthickness=0, wrap=tk.WORD)
        if width is not None:
            self._text["width"] = width
        if height is not None:
            self._text["height"] = height
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        # Todo: set styles etc.
        self._text["state"] = tk.DISABLED
        font = tkfont.nametofont("TkTextFont")
        self._text.tag_config("t", font=font)
        self._xs_del, self._ys_del = None, None
        if "v" in scroll:
            self._yscroll = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
            self._yscroll.grid(row=0, column=1, sticky=tk.NS)
            self._ys_del = self._yscroll.set
        if "h" in scroll:
            self._xscroll = ttk.Scrollbar(self, orient="horizontal", command=self._text.xview)
            self._xscroll.grid(row=1, column=0, sticky=tk.EW)
            self._xs_del = self._xscroll.set
        self._text["xscrollcommand"] = self._xscrollcommand
        self._text["yscrollcommand"] = self._yscrollcommand
        
    def _xscrollcommand(self, lower, upper):
        if self._xs_del is not None:
            if float(upper) - float(lower) == 1.0:
                self._xscroll.grid_remove()
            else:
                self._xscroll.grid()
            self._xs_del(lower, upper)
        
    def _yscrollcommand(self, lower, upper):
        if self._ys_del is not None:
            if float(upper) - float(lower) == 1.0:
                self._yscroll.grid_remove()
            else:
                self._yscroll.grid()
            self._ys_del(lower, upper)

    def set_height_to_view(self, min_height = 1, max_height = 50):
        """Decreases and/or increases the `height` of the underlying text
        widget until the text exactly fits, or the passes bounds are met.
        
        This only works if the widget has been displayed in its final place
        already, and with all geometry set to resize.  That is, don't expect
        this always to work!
        """
        _SizeFinder(self._text, min_height, max_height)
        
    def add_text(self, text, extra_tags = None):
        """Add the unformatted text at the end."""
        self._text["state"] = tk.NORMAL
        tags = ["t"]
        if extra_tags is not None:
            tags.extend( list(extra_tags) )
        self._text.insert(tk.END, text, tuple(tags))
        self._text["state"] = tk.DISABLED
    
    def add_coloured_text(self, text, colour):
        if not hasattr(self, "_colours"):
            self._colours = dict()
        if colour not in self._colours:
            tag = "c{}".format(len(self._colours) + 1)
            self._colours[colour] = tag
            self._text.tag_config(tag, foreground=colour)
        tag = self._colours[colour]
        self.add_text(text, (tag, ))

    @property
    def widget(self):
        """The underlying `tk.Text` widget."""
        return self._text


class _SizeFinder():
    def __init__(self, text, min_height, max_height):
        self._text = text
        self._old_scroll = self._text["yscrollcommand"]
        def scrolled(lower, upper):
            self.length = float(upper) - float(lower)
        self._text["yscrollcommand"] = scrolled
        if self.decrease(min_height):
            self.increase(max_height)
        self._text["yscrollcommand"] = self._old_scroll

    def increase(self, max_height):
        while self._text["height"] < max_height and self.length < 1:
            self._text["height"] = self._text["height"] + 1
            self._text.update()
        
    def decrease(self, min_height):
        self._text.update()
        while self._text["height"] > min_height and self.length == 1:
            self._text["height"] = self._text["height"] - 1
            self._text.update()
        return self.length < 1
        