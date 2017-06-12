"""
error_list
~~~~~~~~~~

A simple modal dialog which displays list of error messages (or other messages...)
"""

import tkinter as tk
import tkinter.ttk as ttk
from . import util

_text = {
    "more" : "...and a further {} messages"
}

class ErrorList(util.ModalWindow):
    def __init__(self, parent, title, list_title, messages, button_text_list):
        self._list_title = list_title
        self._messages = messages
        self._button_text_list = button_text_list
        self.default_button = 0
        super().__init__(parent, title)
        util.centre_window(self)
        
    @property
    def default_button(self):
        """Which button should be consider "pressed" if the window is closed?"""
        return self._default_button
    
    @default_button.setter
    def default_button(self, value):
        self._default_button = value

    def run(self):
        self.wait_window(self)
        return self.result

    def add_widgets(self):
        frame = ttk.LabelFrame(self, text=self._list_title)
        frame.grid(row=0, column=0, sticky=util.NSEW, padx=5, pady=5)
        text = [e for _, e in zip(range(5), self._messages)]
        extras = len(self._messages) - 5
        if extras > 0:
            text.append(_text["more"].format(extras))
        text = "\n".join(text)
        ttk.Label(frame, text=text).grid(sticky=util.NSEW, padx=5, pady=5)

        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky=util.NSEW, padx=5, pady=5)
        for i, t in enumerate(self._button_text_list):
            b = ttk.Button(frame, text=t, command=lambda : self._pressed(i))
            b.grid(row=0, column=i, padx=3, pady=3)
            
    def _pressed(self, index):
        self.result = index
        self.destroy()

    def quit(self):
        self._pressed(self.default_button)