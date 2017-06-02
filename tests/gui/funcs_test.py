import open_cp.gui.funcs as funcs

def test_string_ellipse():
    assert funcs.string_ellipse("ahdsgs", 10) == "ahdsgs"
    assert funcs.string_ellipse("sasagdjgha", 7) == "... gha"