import pytest

import open_cp.gui.settings as settings
import tests.helpers as helpers
from unittest.mock import patch

import json, logging, io

@pytest.fixture
def setts():
    return settings.Settings()

def test_add_get(setts):
    setts["key"] = "value1"
    setts["key5"] = "value2"
    
    assert( setts["key"] == "value1" )
    assert( setts["key5"] == "value2" )
    assert( "key" in setts )
    assert( set(setts.values()) == {"value1", "value2"} )
    
def test_logging():
    logger = logging.getLogger("open_cp.gui.settings")
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    ch = logging.StreamHandler(stream)
    logger.addHandler(ch)
    
    settings.Settings()
    
    log = stream.getvalue().split("\n")
    assert log[0].startswith("Using filename '")
    assert log[0].endswith("open_cp_ui_settings.json'")

def test_save_settings():
    capture = helpers.StrIOWrapper()
    with patch("builtins.open", helpers.MockOpen(capture)) as open_mock:
        open_mock.filter = helpers.ExactlyTheseFilter([2])
        
        sett = settings.Settings()
        sett["name"] = "matt"
        sett.save()
    
    assert json.loads(capture.data) == {"name":"matt"}

def test_load_settings():
    with patch("builtins.open", helpers.MockOpen("{\"stuff\":\"value00\"}")):
        
        sett = settings.Settings()
        assert sett["stuff"] == "value00"
        assert set(sett.keys()) == {"stuff"}

def test_context():
    capture = helpers.StrIOWrapper("{\"stuff\":\"value00\"}")
    with patch("builtins.open", helpers.MockOpen(capture)) as open_mock:
        open_mock.filter = helpers.FilenameFilter("test.json")
        with settings.Settings("test.json") as sett:
            assert set(sett.keys()) == {"stuff"}
            assert sett["stuff"] == "value00"
            
            sett["stuff"] = "value"
            sett["name"] = "matt"
            
    assert json.loads(capture.data) == {"name":"matt", "stuff":"value"}
