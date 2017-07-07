# Model dialog demo

import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import open_cp.gui.tk.util as util
import open_cp.gui.tk.date_picker as date_picker
import datetime
import tkinter as tk
import tkinter.ttk as ttk

import logging
logging.basicConfig(level=logging.DEBUG)

root = tk.Tk()

class OurDialog(util.ModalWindow):
    def __init__(self, parent):
        super().__init__(parent, "Our dialog box")
        util.centre_window_percentage(self, 20, 10)
        
    def add_widgets(self):
        ttk.Button(self, text="Cancel", command=self.cancel).grid(padx=5, pady=5, sticky=tk.NSEW)


class OurDialogNew(util.ModalWindow):
    def __init__(self, parent):
        super().__init__(parent, "Our dialog box")
        self.set_size_percentage(20, 10)
        
    def add_widgets(self):
        #frame = ttk.Frame(self)
        frame = util.ScrolledFrame(self, mode="v")
        frame.grid(sticky=tk.NSEW)
        frame = frame.frame
        
        ttk.Button(frame, text="Cancel", command=self.cancel).grid(padx=5, pady=5, sticky=tk.NSEW)


def open_dialog(parent):
    d = OurDialogNew(parent)
    root.wait_window(d)

util.centre_window_percentage(root, 30, 30)
util.stretchy_columns(root, [0])
util.stretchy_rows(root, [0])
b = ttk.Button(root, text="Click me...")
b.grid(padx=20, pady=20, sticky=tk.NSEW)
b["command"] = lambda : open_dialog(b)
b = ttk.Label(root, text="Date view...")
b.grid(padx=20, pady=20, sticky=tk.NSEW)
dp = date_picker.PopUpDatePicker(b, b,
    lambda : datetime.datetime.now(),
    lambda d : print(d))

root.mainloop()