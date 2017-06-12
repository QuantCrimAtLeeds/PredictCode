import pytest

import open_cp.gui.analysis as analysis
import datetime

@pytest.fixture
def model():
    ts = [datetime.datetime.now() for _ in range(10)]
    xcs = [0 for _ in range(10)]
    ycs = [0 for _ in range(10)]
    ctypes = [["One"], ["Two"], ["Three"], ["Four"], ["One"], ["Two"], ["Three"], ["Four"], ["One"], ["Two"]]
    data = (ts, xcs, ycs, ctypes)
    return analysis.Model(None, data, None)

def test_crime_types(model):
    assert model.num_crime_type_levels == 1
    assert model.unique_crime_types == [ ("Four",), ("One",), ("Three",), ("Two",) ]

@pytest.fixture
def model2():
    ts = [datetime.datetime.now() for _ in range(10)]
    xcs = [0 for _ in range(10)]
    ycs = [0 for _ in range(10)]
    ctypes = [["One", "A"], ["Two", "B"], ["Three", "C"], ["Four", "D"],
        ["One", "B"], ["Two", "C"], ["Three", "D"], ["Four", "C"], ["One", "A"], ["Two", "B"]]
    data = (ts, xcs, ycs, ctypes)
    return analysis.Model(None, data, None)

def test_crime_types_2_levels(model2):
    assert model2.num_crime_type_levels == 2
    assert model2.unique_crime_types == [ ("Four", "C"), ("Four", "D"), ("One", "A"),
        ("One", "B"), ("Three", "C"), ("Three", "D"), ("Two", "B"), ("Two", "C") ]

def test_crime_type_selection(model):
    assert model.selected_crime_types == set()

    model.selected_crime_types = [1, 2, 3]

    with pytest.raises(ValueError):
        model.selected_crime_types = [-1]

    with pytest.raises(ValueError):
        model.selected_crime_types = [4]
        
def test_counts_by_crime_type(model):
    assert model.counts_by_crime_type() == 0

    model.selected_crime_types = [0]
    assert model.counts_by_crime_type() == 2

    model.selected_crime_types = [1]
    assert model.counts_by_crime_type() == 3

    model.selected_crime_types = [0, 1]
    assert model.counts_by_crime_type() == 5
    