"""
util
~~~~

Various utility routines for working with `tkinter`.
"""

def screen_size(root):
    """Returns (width, height).
    
    :param root: A valid window object
    """
    # https://stackoverflow.com/questions/3949844
    return (root.winfo_screenwidth(), root.winfo_screenheight())

def centre_window(window, width, height):
    """Set the window to be of the given size, centred on the screen."""
    w, h = screen_size(window)
    x = (w - width) // 2
    y = (h - height) // 2
    window.geometry("{}x{}+{}+{}".format(width, height, x, y))
