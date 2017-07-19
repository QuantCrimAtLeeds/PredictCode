from .helper import *

import open_cp.gui.predictors.retro as retro
import open_cp.retrohotspot

def test_RetroHotspot(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspot(model)
    assert provider.settings_string == "60 Days, Quartic(200m)"
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_RetroHotspot_serialise(model, project_task, analysis_model, grid_task):
    serialise( retro.RetroHotspot(model) )

@mock.patch("open_cp.retrohotspot.RetroHotSpotGrid")
def test_RetroHotspot_subtask(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspot(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    l = open_cp.retrohotspot.RetroHotSpotGrid.call_args_list
    assert len(l) == 1
    grid = l[0][1]["grid"]
    assert grid.xsize == 50
    assert grid.ysize == 50
    assert grid.xoffset == 25
    assert grid.yoffset == 35
    assert grid.xextent == 15
    assert grid.yextent == 10

    pred = open_cp.retrohotspot.RetroHotSpotGrid.return_value
    assert isinstance(pred.weight, open_cp.retrohotspot.Quartic)
    np.testing.assert_array_equal(pred.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred.data.ycoords, [10,20,0])
    
    s = datetime.datetime(2017,6,7) - datetime.timedelta(days=60)
    e = datetime.datetime(2017,6,7)
    pred.predict.assert_called_with(start_time=np.datetime64(s), end_time=np.datetime64(e))
    
    assert prediction is pred.predict.return_value
    
@mock.patch("open_cp.retrohotspot.RetroHotSpotGrid")
def test_RetroHotspot_window_length(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspot(model)
    provider.window_length = 5
    assert provider.settings_string == "5 Days, Quartic(200m)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.retrohotspot.RetroHotSpotGrid.return_value
    s = datetime.datetime(2017,6,7) - datetime.timedelta(days=5)
    e = datetime.datetime(2017,6,7)
    pred.predict.assert_called_with(start_time=np.datetime64(s), end_time=np.datetime64(e))
    
@mock.patch("open_cp.retrohotspot.RetroHotSpotGrid")
def test_RetroHotspot_Quartic_options(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspot(model)
    provider.kernel_model().bandwidth=45
    assert provider.settings_string == "60 Days, Quartic(45m)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))

    pred = open_cp.retrohotspot.RetroHotSpotGrid.return_value
    assert pred.weight(10, 0) == pytest.approx((1 - 100 / (45*45))**2)
    assert pred.weight(45, 0) == 0.0
    assert pred.weight(50, 0) == 0.0
    
@mock.patch("open_cp.retrohotspot.RetroHotSpotGrid")
def test_RetroHotspot_Gaussian_options(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspot(model)
    provider.kernel = 1
    provider.kernel_model().bandwidth=30
    provider.kernel_model().std_devs=2
    assert provider.settings_string == "60 Days, Gaussian(30m, 2sds)"
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    subtask(datetime.datetime(2017,6,7))
    
    pred = open_cp.retrohotspot.RetroHotSpotGrid.return_value
    x = np.exp( - 100 / 900 * 4 / 2 )
    assert pred.weight(10, 0) == pytest.approx(x)
    assert pred.weight(30.1, 0) == 0.0

class ConstCtsPred(open_cp.predictors.ContinuousPrediction):
    def risk(self, x, y):
        return 1.5 + np.zeros_like(x)
    
@mock.patch("open_cp.retrohotspot.RetroHotSpot")
def test_RetroHotspotCtsGrid_subtask(model, project_task, analysis_model, grid_task):
    provider = retro.RetroHotspotCtsGrid(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    pred = open_cp.retrohotspot.RetroHotSpot.return_value
    pred.predict.return_value = ConstCtsPred()
    open_cp.retrohotspot.RetroHotSpot.assert_called_with()

    prediction = subtask(datetime.datetime(2017,6,7))

    assert isinstance(pred.weight, open_cp.retrohotspot.Quartic)
    np.testing.assert_array_equal(pred.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred.data.ycoords, [10,20,0])
    
    s = datetime.datetime(2017,6,7) - datetime.timedelta(days=60)
    e = datetime.datetime(2017,6,7)
    pred.predict.assert_called_with(start_time=np.datetime64(s), end_time=np.datetime64(e))
    
    print(type(prediction))
    assert prediction.xsize == 50
    assert prediction.ysize == 50
    assert prediction.xoffset == 25
    assert prediction.yoffset == 35
    assert prediction.xextent == 15
    assert prediction.yextent == 10
