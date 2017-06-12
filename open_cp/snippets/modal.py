# Model dialog demo

import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import open_cp.gui.tk.util as util
import open_cp.gui.tk.date_picker as date_picker
import datetime
import tkinter as tk

root = tk.Tk()

class OurDialog(util.ModalWindow):
    def __init__(self, parent):
        super().__init__(parent, "Our dialog box")
        util.centre_window_percentage(self, 20, 10)
        
    def add_widgets(self):
        tk.Button(self, text="Cancel", command=self.cancel).grid(padx=5, pady=5, sticky=tk.NSEW)

def open_dialog(parent):
    d = OurDialog(parent)
    d.wait_window(d)

util.centre_window_percentage(root, 30, 30)
util.stretchy_columns(root, [0])
util.stretchy_rows(root, [0])
b = tk.Button(root, text="Click me...")
b.grid(padx=20, pady=20, sticky=tk.NSEW)
b["command"] = lambda : open_dialog(b)
b = tk.Label(root, text="Date view...")
b.grid(padx=20, pady=20, sticky=tk.NSEW)
dp = date_picker.PopUpDatePicker(b, b,
    lambda : datetime.datetime.now(),
    lambda d : print(d))

root.mainloop()