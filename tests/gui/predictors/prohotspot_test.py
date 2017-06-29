from .helper import *

import open_cp.gui.predictors.prohotspot as prohotspot
import datetime

def test_time_unit():
    assert prohotspot.ProHotspotView.timeunit(datetime.timedelta(hours=5)) == (5,2)
    assert prohotspot.ProHotspotView.timeunit(datetime.timedelta(hours=25)) == (25,2)
    assert prohotspot.ProHotspotView.timeunit(datetime.timedelta(hours=48)) == (2,1)
    assert prohotspot.ProHotspotView.timeunit(datetime.timedelta(days=8)) == (8,1)
    assert prohotspot.ProHotspotView.timeunit(datetime.timedelta(days=7)) == (1,0)
    with pytest.raises(ValueError):
        prohotspot.ProHotspotView.timeunit(datetime.timedelta(minutes=119))

def test_ProHotspot(model, project_task, analysis_model, grid_task):
    provider = prohotspot.ProHotspot(model)
    assert provider.settings_string == "DiagsSame / Classic(8x8) @ 1 Week(s)"
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_ProHotspot_serialise(model, project_task, analysis_model, grid_task):
    serialise( prohotspot.ProHotspot(model) )

@mock.patch("open_cp.prohotspot.ProspectiveHotSpot")
def test_RetroHotspot_subtask(model, project_task, analysis_model, grid_task):
    provider = prohotspot.ProHotspot(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.prohotspot.ProspectiveHotSpot.call_args_list
    assert len(l) == 1
    grid = l[0][1]["grid"]
    assert grid.xsize == 50
    assert grid.ysize == 50
    assert grid.xoffset == 25
    assert grid.yoffset == 35
    assert grid.xextent == 15
    assert grid.yextent == 10
    timeunit = l[0][1]["time_unit"]
    assert timeunit == np.timedelta64(7, "D")

    pred = open_cp.prohotspot.ProspectiveHotSpot.return_value
    assert isinstance(pred.weight, open_cp.prohotspot.ClassicWeight)
    assert isinstance(pred.distance, open_cp.prohotspot.DistanceDiagonalsSame)
    np.testing.assert_array_equal(pred.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred.data.ycoords, [10,20,0])
    
    predict_date = np.datetime64(datetime.datetime(2017,6,7))
    pred.predict.assert_called_with(predict_date, predict_date)
    
    assert prediction is pred.predict.return_value

@mock.patch("open_cp.prohotspot.ProspectiveHotSpot")
def test_RetroHotspot_window_length(model, project_task, analysis_model, grid_task):
    provider = prohotspot.ProHotspot(model)
    provider.time_window_length = datetime.timedelta(days=4)
    assert provider.settings_string == "DiagsSame / Classic(8x8) @ 4 Day(s)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.prohotspot.ProspectiveHotSpot.call_args_list
    timeunit = l[0][1]["time_unit"]
    assert timeunit == np.timedelta64(4 * 24, "h")

@mock.patch("open_cp.prohotspot.ProspectiveHotSpot")
def test_RetroHotspot_kernel(model, project_task, analysis_model, grid_task):
    provider = prohotspot.ProHotspot(model)
    provider.weight_model.space_bandwidth = 5
    provider.weight_model.time_bandwidth = 9
    assert provider.settings_string == "DiagsSame / Classic(5x9) @ 1 Week(s)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.prohotspot.ProspectiveHotSpot.return_value
    assert pred.weight.space_bandwidth == 5
    assert pred.weight.time_bandwidth == 9

@mock.patch("open_cp.prohotspot.ProspectiveHotSpot")
def test_RetroHotspot_distance(model, project_task, analysis_model, grid_task):
    provider = prohotspot.ProHotspot(model)
    provider.distance_type = 1
    assert provider.settings_string == "DiagsDiff / Classic(8x8) @ 1 Week(s)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.prohotspot.ProspectiveHotSpot.return_value
    assert isinstance(pred.distance, open_cp.prohotspot.DistanceDiagonalsDifferent)

