import pytest

import open_cp.gui.tk as tk

def test_screen_size():
    root = tk.tk.Tk()
    w, h = tk.screen_size(root)
    assert((w,h) == (1280,1024))