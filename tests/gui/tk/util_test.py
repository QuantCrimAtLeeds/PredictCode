import pytest

import open_cp.gui.tk as tk

@pytest.fixture
def root():
    try:
        root = tk.tk.Tk()
        return root
    except tk.tk.TclError:
        pass

def test_screen_size(root):
    if root is None: return
    w, h = tk.screen_size(root)
    assert((w,h) == (1280,1024))