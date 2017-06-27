import tkinter as tk
import traceback
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Widget(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.var = tk.StringVar()
        #tk.Entry(self, textvariable=self.var).grid()
        self._thing = tk.Frame(self)
        def task():
            print("Running off thread on", threading.get_ident())
            fig = figure.Figure(figsize=(5,5))
            FigureCanvas(fig)
            fig.add_subplot(1,1,1)
            print("All done off thread...")
        threading.Thread(target=task).start()
        
    def __del__(self):
        print("Being deleted...", self.__repr__(), id(self))
        print("Thread is", threading.get_ident())
        traceback.print_stack()

root = tk.Tk()
frame = Widget(root)
frame.grid(row=1, column=0)

def click():
    global frame
    frame.destroy()
    frame = Widget(root)
    frame.grid(row=1, column=0)

tk.Button(root, text="Click me", command=click).grid(row=0, column=0)

root.mainloop()