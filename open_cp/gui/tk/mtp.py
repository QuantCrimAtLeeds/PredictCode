"""
mtp
~~~

Some utilities to work with `matplotlib`
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def figure_to_canvas(figure, root):
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.show()
    return canvas.get_tk_widget()