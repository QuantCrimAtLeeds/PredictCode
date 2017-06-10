import pytest

import open_cp.gui.tk as tk

@pytest.fixture
def root():
    try:
        root = tk.tk.Tk()
        return root
    except tk.tk.TclError:
        pass
