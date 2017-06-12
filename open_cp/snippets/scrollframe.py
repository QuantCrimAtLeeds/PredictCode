import tkinter as tk
import tkinter.ttk as ttk

import os, sys
sys.path.insert(0, os.path.join("..", ".."))
import open_cp.gui.tk.util as util
            
root = tk.Tk()
util.centre_window_percentage(root, 50, 50)
util.stretchy_columns(root, [0])
util.stretchy_rows(root, [0])

sf = util.ScrolledFrame(root)
sf.grid()
ttk.Button(root, text="Some other button").grid(row=1,column=1)

def clicked(i):
    print("Clicked", i)

for i in range(5):
    b = ttk.Button(sf.frame, text="Click me {}".format(i), command=lambda n=i: clicked(n))
    b.grid(row=i, column=i)


root.mainloop()
