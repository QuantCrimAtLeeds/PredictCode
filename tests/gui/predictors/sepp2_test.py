from .helper import *

import open_cp.gui.predictors.sepp2 as sepp2
#import datetime

@pytest.fixture
def analysis_model2(analysis_model):
    analysis_model.time_range = (datetime.datetime(2017,5,21,12,30),
                                 datetime.datetime(2017,5,21,13,30), None, None)
    return analysis_model

def test_SEPP(model, project_task, analysis_model2, grid_task):
    provider = sepp2.SEPP(model)
    assert provider.name == "Grid based SEPP"
    assert provider.settings_string is None
    standard_calls(provider, project_task, analysis_model2, grid_task)

def test_serialise(model, project_task, analysis_model2, grid_task):
    serialise( sepp2.SEPP(model) )


# TODO: Test no data raising...