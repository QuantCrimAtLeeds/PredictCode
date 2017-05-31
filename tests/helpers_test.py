import pytest
import tests.helpers as helpers
from unittest.mock import patch
import io

import numpy as np

@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_single():
    assert( np.random.random() == 0.1 )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_1D():
    np.testing.assert_allclose( np.random.random(size=5), [.1,.2,.3,.4,.1] )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_2D():
    np.testing.assert_allclose( np.random.random(size=(2,3)),
        np.array([[.1,.2,.3],[.4,.1,.2]]) )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_3D():
    np.testing.assert_allclose( np.random.random(size=(2,1,3)),
        np.array([[[.1,.2,.3]],[[.4,.1,.2]]]) )
    

@patch("builtins.open", helpers.MockOpen("fish"))
def test_MockOpen():
    with open("file.txt") as file:
        assert( next(file) == "fish" )
        with pytest.raises(StopIteration):
            next(file)
            
    with pytest.raises(FileNotFoundError):
        with open("doesntexists.txt") as file:
            pass

def test_MockOpen_captureOutput():
    capture = helpers.BytesIOWrapper()
    with patch("builtins.open", helpers.MockOpen(capture)):
        with open("somefile", "wb") as file:
            file.write(b"test data")

    assert capture.data == b"test data"

def test_MockOpen_casesFilter():
    with patch("builtins.open", helpers.MockOpen("ahsgs")) as mock_open:
        mock_open.filter = helpers.ExactlyTheseFilter([2])

        with pytest.raises(FileNotFoundError):
            with open("doesntexists.txt") as file:
                pass

        with open("doesntexists.txt") as file:
            pass

def test_MockOpen_filenameFilter():
    with patch("builtins.open", helpers.MockOpen("ahsgs")) as mock_open:
        mock_open.filter = helpers.FilenameFilter("exists.txt")

        with pytest.raises(FileNotFoundError):
            with open("doesntexists_.txt") as file:
                pass

        with open("doesntexists.txt") as file:
            pass
    