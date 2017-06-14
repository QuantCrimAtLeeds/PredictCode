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
    """Create a `tk` widget from a `matplotlib` figure.

    :param figure: The `matplotlib` figure to use
    :param root: The `tk` widget to be a child of

    :return: A new `tk` widget.
    """
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.show()
    return canvas.get_tk_widget()