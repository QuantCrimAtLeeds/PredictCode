import pytest
import unittest.mock as mock
import io
import os.path
import numpy as np

import open_cp.sources.chicago as chicago

# ARGH: https://github.com/pytest-dev/pytest/issues/2180

class OurOpen():
    _original_open = open
    def __call__(self, filename, **kwargs):
        return _original_open(filename, **kwargs)

@mock.patch("builtins.open", OurOpen())
def test_load_default_filename(open_mock):
    #open_mock.return_value = None
    assert( chicago.default_burglary_data() == None )
    #filename = open_mock.call_args[0][0]
    #assert( os.path.split(filename)[1] == "chicago.csv" )

@mock.patch("builtins.open")
def test_load_data(open_mock):
    string_data = "\n".join([
        ",".join([chicago._DESCRIPTION_FIELD, chicago._X_FIELD, chicago._Y_FIELD,
            "other", chicago._TIME_FIELD]),
        "THEFT, 123, 456, as, 03/13/2016 02:53:30 AM",
        "ASSAULT, 12, 34, dgs, sgjhg",
        "THEFT, 789, 1012, ahgd, 01/01/2017 10:30:23 PM" ])

    #open_mock.return_value = io.StringIO(string_data, newline="\n")
    points = chicago.load("filename", {"THEFT"})
    #assert( open_mock.call_args[0][0] == "filename" )

    #assert( points.timestamps[0] == np.DateTime64("2016-03-13T02:53:30"))