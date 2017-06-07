# Adapted from
# https://matplotlib.org/examples/user_interfaces/embedding_in_tk.html
# This gives the standard matplotlib interface, which for many applications would be
# great.  But I just want a static plot...
#  - Could just not use the `toolbar`


import matplotlib
print("Default backend appears to be:", matplotlib.get_backend())
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Embedding in TK")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)

a.plot(t, s)

# Loads of points...
x = np.random.random(size=100000)
y = np.random.random(size=100000)
a.scatter(x, y, marker="+", alpha=0.1, color="black")

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
f.set_size_inches(10, 10)
f.savefig("test.png", dpi=100,bbox_inches="tight")
#canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvas.get_tk_widget().grid(sticky=Tk.NSEW)

#toolbar = NavigationToolbar2TkAgg(canvas, root)
#toolbar.update()
#canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


#def on_key_event(event):
#    print('you pressed %s' % event.key)
#    key_press_handler(event, canvas, toolbar)

#canvas.mpl_connect('key_press_event', on_key_event)

def _quit():
    root.quit()     # stops mainloop
    #root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
#button.pack(side=Tk.BOTTOM)
button.grid()

Tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.