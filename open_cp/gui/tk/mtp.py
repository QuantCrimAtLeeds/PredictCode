"""
mtp
~~~

Some utilities to work with `matplotlib`
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import threading as _threading
import open_cp.gui.locator as _locator
import open_cp.gui.tk.threads as _threads
import io as _io
import PIL.ImageTk, PIL.Image

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


class _MakeFigTask(_threads.OffThreadTask):
    def __init__(self, canvas_figure, fig_dpi):
        super().__init__()
        self._canvas_figure = canvas_figure
        self._fig_dpi = fig_dpi
    
    def __call__(self):
        fig, dpi = self._fig_dpi
        if self.cancelled:
            fig.clf()
            return
        file = _io.BytesIO()
        fig.savefig(file, dpi=dpi)
        # Do this to stop memory leak...
        plt.close(fig)
        fig.clf() # This line is the important one...
        if self.cancelled:
            return
        # This works...
        #import base64
        #file = base64.b64encode(file.getvalue())
        #image = tk.PhotoImage(data=file)
        image = PIL.Image.open(file)
        self._canvas_figure.set_image(image)

    def on_gui_thread(self, value):
        pass


class _RescaleTask(_threads.OffThreadTask):
    def __init__(self, canvas_figure, image):
        super().__init__()
        self._canvas_figure = canvas_figure
        self._image = image

    def _correct_aspect_ratio(self, image_width, image_height):
        width, height = self._canvas_figure.size
        awidth, aheight = image_width, image_height
        if aheight == 0 or awidth == 0:
            return
        aspect = awidth / aheight
        if aspect * height > width:
            height = int(width / aspect)
        elif width / aspect >  height:
            width = int(aspect * height)
        return width, height

    def __call__(self):
        if self.cancelled:
            return
        width, height = self._correct_aspect_ratio(*self._image.size)
        image = self._image.resize((width, height), resample=PIL.Image.LANCZOS)
        if self.cancelled:
            return
        photo = PIL.ImageTk.PhotoImage(image)
        return photo

    def on_gui_thread(self, value):
        if value is None:
            return
        self._canvas_figure.set_photo(value)


class CanvasFigure(tk.Frame):
    """A subclass of :class:`tk.Frame` which contains a :class:`PIL.Image`
    image which is automatically rescaled to fit.  Also allows the automatic
    setting of this image to be generated from a `matplotlib` figure (this
    being the primary use case).

    :param width: Optionally set the initial width of the canvas object
    :param height: Optionally set the initial width of the canvas object
    """
    def __init__(self, parent, width=None, height=None):
        super().__init__(parent)
        util.stretchy_rows_cols(self, [0], [0])
        self._canvas = tk.Canvas(self, borderwidth=0, selectborderwidth=0, highlightthickness=0)
        if width is not None:
            self._canvas["width"] = width
        if height is not None:
            self._canvas["height"] = height
        self._canvas.grid(sticky=tk.NSEW, row=0, column=0)
        self._canvas_image = None
        self._image = None
        self._size = (1,1)
        self._rlock = _threading.RLock()
        self.bind("<Configure>", self._resize)

    def _resize(self, event=None):
        if event is None:
            width = self.winfo_width()
            height = self.winfo_height()
        else:
            width, height = event.width, event.height
        # NOTE: The canvas size itself does not change...
        with self._rlock:
            self._size = (width, height)
        self._draw()

    def _draw(self):
        with self._rlock:
            if self._image is None:
                return
            rescale_task = _RescaleTask(self, self._image)
            _locator.get("pool").submit(rescale_task)
            self._canvas["cursor"] = "watch"

    @property
    def size(self):
        """Get the current size of the image needed."""
        with self._rlock:
            return self._size

    def set_image(self, image):
        """Set the :class:`PIL.Image` to display."""
        with self._rlock:
            self._image = image
            self._draw()

    def set_photo(self, photo):
        """Set the canvas photo; library use only."""
        with self._rlock:
            if self._canvas_image is None:
                self._canvas_image = self._canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            else:
                self._canvas.itemconfig(self._canvas_image, image=photo)
            # Don't GC me
            self._photo = photo
            self._canvas["cursor"] = "arrow"

    def set_figure(self, figure, dpi=150):
        """Set the figure to use."""
        with self._rlock:
            fig_task = _MakeFigTask(self, (figure, dpi))
            _locator.get("pool").submit(fig_task)
            self._canvas["cursor"] = "watch"

    def set_figure_task(self, task, dpi=150):
        """Pass a task which generates a figure.  This will be run off-thread
        and then displayed."""
        def on_gui_thread(fig):
            self.set_figure(fig, dpi)
        _locator.get("pool").submit(task, on_gui_thread)
        self._canvas["cursor"] = "watch"

    def destroy(self):
        self._photo = None
        self._image = None
        super().destroy()
