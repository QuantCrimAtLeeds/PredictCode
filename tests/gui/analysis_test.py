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
    assert model.unique_crime_types(None) == [ "Four", "One", "Three", "Two" ]

def test_crime_types_2_levels(model):
    model.crime_types = [["One", "A"], ["Two", "B"], ["Three", "C"], ["Four", "D"],
        ["One", "B"], ["Two", "C"], ["Three", "D"], ["Four", "C"], ["One", "A"], ["Two", "B"]]

    assert model.num_crime_type_levels == 2
    assert model.unique_crime_types(None) == [ "Four", "One", "Three", "Two" ]
    assert model.unique_crime_types(["One"]) == [ "A", "B" ]
    assert model.unique_crime_types(["Two"]) == [ "B", "C" ]
    assert model.unique_crime_types(["Three"]) == [ "C", "D" ]
    assert model.unique_crime_types(["Four"]) == [ "C", "D" ]
    assert model.unique_crime_types({"One", "Four"}) == [ "A", "B", "C", "D" ]
    assert model.unique_crime_types({"Two", "One"}) == [ "A", "B", "C" ]

def test_crime_types_3_levels(model):
    model.crime_types = [["One", "A", "a"], ["Two", "B", "b"], ["Three", "C", "a"],
        ["Four", "D", "c"], ["One", "B", "a"], ["Two", "C", "d"], ["Three", "D", "f"],
        ["Four", "C", "a"], ["One", "A", "b"], ["Two", "B", "a"]]

    assert model.num_crime_type_levels == 3
    assert model.unique_crime_types(None) == [ "Four", "One", "Three", "Two" ]
    assert model.unique_crime_types(["One"]) == [ "A", "B" ]
    assert model.unique_crime_types({("One", "A")}) == [ "a", "b" ]
    
def test_crime_type_selection(model):
    assert model.selected_crime_types == []

    model.selected_crime_types = [["a"], ["b"], ["c"], ["a"]]
    assert model.selected_crime_types == { ("a",), ("b",), ("c",) }

    with pytest.raises(ValueError):
        model.selected_crime_types = ["a", "b"]

    with pytest.raises(ValueError):
        model.selected_crime_types = [["a"], ["b", "c"]]
        
def test_counts_by_crime_type(model):
    assert model.counts_by_crime_type() == 0

    model.selected_crime_types = [("One",)]
    assert model.counts_by_crime_type() == 3

    model.selected_crime_types = [("Three",)]
    assert model.counts_by_crime_type() == 2

    model.selected_crime_types = [("Three",), ("One",)]
    assert model.counts_by_crime_type() == 5
    