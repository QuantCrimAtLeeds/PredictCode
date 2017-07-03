from .helper import *

import open_cp.gui.predictors.stscan as stscan
import datetime

def test_STScan(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    assert provider.settings_string == "geo(50%/3000m) time(50%/60days)"
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_ProHotspot_serialise(model, project_task, analysis_model, grid_task):
    serialise( stscan.STScan(model) )
