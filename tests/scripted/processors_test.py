import pytest
from unittest.mock import patch

import open_cp.scripted.processors as processors
import open_cp.scripted.evaluators as evaluators
from .. import helpers

import io

@pytest.fixture
def outfile():
    with io.StringIO() as f:
        yield f

@pytest.fixture
def hit_rate_save(outfile):
    hrs = processors.HitRateSave(outfile, [10,15,20,100])
    hrs.init()
    return hrs

def test_HitRateSave_header(hit_rate_save, outfile):
    hit_rate_save.done()
    assert outfile.getvalue().strip() == "Predictor,Start time,End time,10%,15%,20%,100%"

def test_HitRateSave_header_filename():
    capture = helpers.StrIOWrapper()
    with patch("builtins.open", helpers.MockOpen(capture)):
        hrs = processors.HitRateSave("out.csv", [10, 20])
        hrs.init()
        hrs.done()
    
    assert capture.data.strip() == "Predictor,Start time,End time,10%,20%"

def test_HitRateSave(hit_rate_save, outfile):
    hit_rate_save.process("predname", evaluators.HitRateEvaluator(), [{10:12, 15:20, 20:100, 100:100}], [("absa", "ahjsdjh")])
    hit_rate_save.process("dave", 6, None, None)
    hit_rate_save.done()
    rows = [x.strip() for x in outfile.getvalue().split("\n")]
    assert rows[0] == "Predictor,Start time,End time,10%,15%,20%,100%"
    assert rows[1] == "predname,absa,ahjsdjh,12,20,100,100"

@pytest.fixture
def hit_count_save(outfile):
    hcs = processors.HitCountSave(outfile, [10,15,20,100])
    hcs.init()
    return hcs
    
def test_HitCountSave_header(hit_count_save, outfile):
    hit_count_save.done()
    assert outfile.getvalue().strip() == "Predictor,Start time,End time,Number events,10%,15%,20%,100%"

def test_HitCountSave(hit_count_save, outfile):
    hit_count_save.process("pn", evaluators.HitCountEvaluator(), [{10:(5,12), 15:(6,12), 20:(8,12), 100:(12,12)}], [("absa", "ahjsdjh")])
    hit_count_save.process("dave", 6, None, None)
    hit_count_save.done()
    rows = [x.strip() for x in outfile.getvalue().split("\n")]
    assert rows[1] == "pn,absa,ahjsdjh,12,5,6,8,12"
