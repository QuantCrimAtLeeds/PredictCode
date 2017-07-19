from .helper import *

import open_cp.gui.predictors.prohotspotcts as prohotspotcts
import datetime

def test_ProHotspot(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    assert provider.settings_string == "DiagsSame / Classic(8x8) @ 50m, 168h"
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_ProHotspot_serialise(model, project_task, analysis_model, grid_task):
    serialise( prohotspotcts.ProHotspotCts(model) )


class OurContinuousPrediction(open_cp.predictors.ContinuousPrediction):
    def risk(self, x, y):
        return 1.0 + np.zeros_like(x)


@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_subtask(model, project_task, analysis_model, grid_task):
    open_cp.prohotspot.ProspectiveHotSpotContinuous.return_value.predict.return_value = OurContinuousPrediction(20, 30, 10, 20)

    provider = prohotspotcts.ProHotspotCts(model)
    provider.space_length = 75
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.prohotspot.ProspectiveHotSpotContinuous.call_args_list
    assert len(l) == 1
    assert l[0][1]["grid_size"] == 75
    timeunit = l[0][1]["time_unit"]
    assert timeunit == np.timedelta64(7, "D")

    pred = open_cp.prohotspot.ProspectiveHotSpotContinuous.return_value
    assert isinstance(pred.weight, open_cp.prohotspot.ClassicWeight)
    assert isinstance(pred.distance, open_cp.prohotspot.DistanceDiagonalsSame)
    np.testing.assert_array_equal(pred.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred.data.ycoords, [10,20,0])
    
    predict_date = np.datetime64(datetime.datetime(2017,6,7))
    pred.predict.assert_called_with(predict_date, predict_date)
    
    assert prediction.xsize == 50
    assert prediction.ysize == 50
    assert prediction.xoffset == 25
    assert prediction.yoffset == 35
    assert prediction.xextent == 15
    assert prediction.yextent == 10
    assert prediction.intensity_matrix[0,0] == 1

@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_window_length(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    provider.time_window_length = datetime.timedelta(days=4)
    assert provider.settings_string == "DiagsSame / Classic(8x8) @ 50m, 96h"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.prohotspot.ProspectiveHotSpotContinuous.call_args_list
    timeunit = l[0][1]["time_unit"]
    assert timeunit == np.timedelta64(4 * 24, "h")

@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_space_length(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    provider.space_length = 23
    assert provider.settings_string == "DiagsSame / Classic(8x8) @ 23m, 168h"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.prohotspot.ProspectiveHotSpotContinuous.call_args_list
    assert l[0][1]["grid_size"] == 23

@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_kernel(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    provider.weight_model.space_bandwidth = 5
    provider.weight_model.time_bandwidth = 9
    assert provider.settings_string == "DiagsSame / Classic(5x9) @ 50m, 168h"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.prohotspot.ProspectiveHotSpotContinuous.return_value
    assert pred.weight.space_bandwidth == 5
    assert pred.weight.time_bandwidth == 9

@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_distance(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    provider.distance_type = 1
    assert provider.settings_string == "DiagsDiff / Classic(8x8) @ 50m, 168h"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.prohotspot.ProspectiveHotSpotContinuous.return_value
    assert isinstance(pred.distance, open_cp.prohotspot.DistanceDiagonalsDifferent)

@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspot_distance2(model, project_task, analysis_model, grid_task):
    provider = prohotspotcts.ProHotspotCts(model)
    provider.distance_type = 2
    assert provider.settings_string == "DiagsCircle / Classic(8x8) @ 50m, 168h"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.prohotspot.ProspectiveHotSpotContinuous.return_value
    assert isinstance(pred.distance, open_cp.prohotspot.DistanceCircle)
