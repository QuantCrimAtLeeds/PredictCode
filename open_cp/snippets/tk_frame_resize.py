# Demos some manual height management of widgets.
#
# The default way of making rows "stretchy" means that only "excess" space is
#  evenly distributed between widgets.  In the example below, this means that
#  if the 1st frame becomes very large, it will completely dominate.
#
# Instead, we would like the scrollable frames to have a minimum size, or to
#  always be of equal size.  This can be accomplished by manually listening
#  to the enclosing window being resized and adjusting the heights.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import tkinter as tk
import tkinter.ttk as ttk

import open_cp.gui.tk.util as util

root = tk.Tk()

frame1 = util.ScrolledFrame(root, mode="v")
frame1["width"] = 300
frame1["height"] = 250
frame1.grid_propagate(0)
frame1.grid(row=0, column=0)#, sticky=tk.NSEW)

frame2 = util.ScrolledFrame(root, mode="v")
frame2["width"] = 300
frame2["height"] = 250
frame2.grid_propagate(0)
frame2.grid(row=1, column=0)#, sticky=tk.NSEW)

def add_one(event=None):
    ttk.Button(frame1.frame, text="Added").grid(sticky=tk.NSEW)

ttk.Button(frame1.frame, text="Add", command=add_one).grid(sticky=tk.NSEW)

def add_two(event=None):
    ttk.Button(frame2.frame, text="Added").grid(sticky=tk.NSEW)

ttk.Button(frame2.frame, text="Add", command=add_two).grid(sticky=tk.NSEW)

def same_size(event):
    # Binding to root binds to _all_ children!
    if event.widget == root:
        subheight = root.winfo_height() // 2
        frame1["height"] = subheight
        frame2["height"] = subheight

def min_height(event):
    minheight = 100
    if event.widget == root:
        height = root.winfo_height()
        if height <= minheight * 2:
            return same_size(event)
        want1 = frame1.frame.winfo_reqheight()
        want2 = frame2.frame.winfo_reqheight()
        scale = height / (want1 + want2)
        h1, h2 = want1 * scale, want2 * scale
        if h1 < minheight:
            h1 = minheight
            h2 = height - h1
        elif h2 < minheight:
            h2 = minheight
            h1 = height - h2
        frame1["height"] = h1
        frame2["height"] = h2
    
root.bind("<Configure>", min_height)

root.mainloop()
