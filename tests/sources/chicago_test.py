import pytest
import unittest.mock as mock
from tests.helpers import MockOpen
import os.path
import numpy as np

import open_cp.sources.chicago as chicago

def test_set_data_dir():
    chicago.set_data_directory("..")
    assert chicago.get_default_filename() == os.path.join("..", "chicago.csv")

@pytest.fixture
def string_data_snap():
    dic = chicago._FIELDS["snapshot"]
    return "\n".join([
    ",".join([dic["_DESCRIPTION_FIELD"], dic["_X_FIELD"], dic["_Y_FIELD"],
        "other", dic["_TIME_FIELD"]]),
    "THEFT, 789, 1012, ahgd, 01/01/2017 10:30:23 PM",
    "ASSAULT, 12, 34, dgs, sgjhg",
    "THEFT, 123, 456, as, 03/13/2016 02:53:30 AM"
    ])

def test_load_data(string_data_snap):
    with mock.patch("builtins.open", MockOpen(string_data_snap)) as open_mock:
        points = chicago.load("filename", {"THEFT"})
        assert( open_mock.calls[0][0] == ("filename",) )

        assert( len(points.timestamps) == 2 )
        assert( points.timestamps[0] == np.datetime64("2016-03-13T02:53:30") )
        assert( points.timestamps[1] == np.datetime64("2017-01-01T22:30:23") )
        np.testing.assert_allclose( points.coords[:,0], np.array([123, 456]) * 1200 / 3937 )
        np.testing.assert_allclose( points.coords[:,1], np.array([789, 1012]) * 1200 / 3937 )

def test_load_data_keep_in_feet(string_data_snap):
    with mock.patch("builtins.open", MockOpen(string_data_snap)) as open_mock:
        points = chicago.load("filename", {"THEFT"}, to_meters=False)
        np.testing.assert_allclose( points.coords[:,0], [123, 456] )
        np.testing.assert_allclose( points.coords[:,1], [789, 1012] )
