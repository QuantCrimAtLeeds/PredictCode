import pytest
import unittest.mock as mock
from tests.helpers import MockOpen
import os.path
import numpy as np

import open_cp.sources.ukpolice as ukpolice

def test_load_default_filename():
    with mock.patch("builtins.open", MockOpen(None)) as open_mock:
        assert( ukpolice.default_burglary_data() == None )
        filename = open_mock.calls[0][0][0]
        assert( os.path.split(filename)[1] == "uk_police.csv" )

string_data = "\n".join([
    ",".join([ukpolice._DESCRIPTION_FIELD, ukpolice._X_FIELD, ukpolice._Y_FIELD,
        "other", ukpolice._TIME_FIELD]),
    "Burglary, 789, 1012, ahgd, 2017-01",
    "ASSAULT, 12, 34, dgs, 2015-05",
    "Burglary, 123, 456, as, 2016-03"
    ])

def test_load_data():
    with mock.patch("builtins.open", MockOpen(string_data)) as open_mock:
        points = ukpolice.load("filename", {"Burglary"})
        assert( open_mock.calls[0][0] == ("filename",) )

        assert( len(points.timestamps) == 2 )
        assert( points.timestamps[0] == np.datetime64("2016-03") )
        assert( points.timestamps[1] == np.datetime64("2017-01") )
        np.testing.assert_allclose( points.coords[:,0], np.array([123, 456]) )
        np.testing.assert_allclose( points.coords[:,1], np.array([789, 1012]) )
