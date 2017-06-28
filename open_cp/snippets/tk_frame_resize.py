import sys, os
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import tkinter as tk
import tkinter.ttk as ttk

import open_cp.gui.tk.util as util

root = tk.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# This nearly works, but you need to resize the main window
# to see the scrollbars.  Presumably this is because <Configure>
# isn't been triggered on the outer frame...??

frame1 = util.ScrolledFrame(root, mode="v")
frame1["width"] = 300
frame1["height"] = 250
frame1.grid_propagate(0)
frame1.grid(row=0, column=0)#, sticky=tk.NSEW)
frame1 = frame1.frame

frame2 = util.ScrolledFrame(root, mode="v")
frame2["width"] = 300
frame2["height"] = 250
frame2.grid_propagate(0)
frame2.grid(row=1, column=0)#, sticky=tk.NSEW)
frame2 = frame2.frame

def add_one(event=None):
    ttk.Button(frame1, text="Added").grid(sticky=tk.NSEW)

ttk.Button(frame1, text="Add", command=add_one).grid(sticky=tk.NSEW)

def add_two(event=None):
    ttk.Button(frame2, text="Added").grid(sticky=tk.NSEW)

ttk.Button(frame2, text="Add", command=add_two).grid(sticky=tk.NSEW)


root.mainloop()
