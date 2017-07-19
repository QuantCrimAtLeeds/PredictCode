from .helper import *

import open_cp.gui.predictors.sepp as sepp
import open_cp.gui.predictors.predictor
import datetime
import numpy as np

@pytest.fixture
def analysis_model2(analysis_model):
    analysis_model.time_range = (datetime.datetime(2017,5,21,12,30),
                                 datetime.datetime(2017,5,21,13,30), None, None)
    return analysis_model

@mock.patch("open_cp.sepp")
def test_SEPP(seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    assert provider.name == "KDE and SEPP (kt=100, ks=15)"
    assert provider.settings_string == "iters=40 initial=0.10/50.0 cutoff=120/500"
    standard_calls(provider, project_task, analysis_model2, grid_task)

def test_serialise(model, project_task, analysis_model2, grid_task):
    serialise( sepp.SEPP(model) )

def test_no_data(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,5,20,12,30),
                                 datetime.datetime(2017,5,20,13,30), None, None)
    provider = sepp.SEPP(model)
    with pytest.raises(open_cp.gui.predictors.predictor.PredictionError):
        standard_calls(provider, project_task, analysis_model, grid_task)

@mock.patch("open_cp.sepp")
@mock.patch("open_cp.predictors.GridPredictionArray")
def test_training_usage(gpmock, seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    subtask = standard_calls(provider, project_task, analysis_model2, grid_task)
    
    seppmock.SEPPTrainer.assert_called_with(k_time=100, k_space=15)
    trainer_mock = seppmock.SEPPTrainer.return_value
    np.testing.assert_allclose(trainer_mock.data.xcoords, [0, 10])
    np.testing.assert_allclose(trainer_mock.data.ycoords, [10, 20])
    time_diffs = ( (trainer_mock.data.timestamps -
        [np.datetime64("2017-05-21T12:30"), np.datetime64("2017-05-21T13:00")])
        / np.timedelta64(1,"ms") )
    np.testing.assert_allclose(time_diffs, [0,0])
    trainer_mock.train.assert_called_with(iterations=40)
    
    pred = trainer_mock.train.return_value
    np.testing.assert_allclose(pred.data.xcoords, [0, 10, 20])
    np.testing.assert_allclose(pred.data.ycoords, [10, 20, 0])
    
    train_date = datetime.datetime(2017,5,22,5,35)
    prediction = subtask(train_date)
    pred.predict.assert_called_with(train_date)
    gpmock.from_continuous_prediction_grid.assert_called_with(pred.predict.return_value, grid_task.return_value)
    assert prediction == gpmock.from_continuous_prediction_grid.return_value

@mock.patch("open_cp.sepp")
@mock.patch("open_cp.predictors.GridPredictionArray")
def test_bandwidth(gpmock, seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    provider.kspace = 25
    provider.ktime = 33
    
    serialise( provider )
    assert provider.name == "KDE and SEPP (kt=33, ks=25)"
    
    standard_calls(provider, project_task, analysis_model2, grid_task)
    seppmock.SEPPTrainer.assert_called_with(k_time=33, k_space=25)

@mock.patch("open_cp.sepp")
@mock.patch("open_cp.predictors.GridPredictionArray")
def test_iterations(gpmock, seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    provider.iterations = 56

    serialise( provider )
    assert provider.settings_string == "iters=56 initial=0.10/50.0 cutoff=120/500"
    
    standard_calls(provider, project_task, analysis_model2, grid_task)
    
    seppmock.SEPPTrainer.assert_called_with(k_time=100, k_space=15)
    trainer_mock = seppmock.SEPPTrainer.return_value
    np.testing.assert_allclose(trainer_mock.data.xcoords, [0, 10])
    np.testing.assert_allclose(trainer_mock.data.ycoords, [10, 20])
    time_diffs = ( (trainer_mock.data.timestamps -
        [np.datetime64("2017-05-21T12:30"), np.datetime64("2017-05-21T13:00")])
        / np.timedelta64(1,"ms") )
    np.testing.assert_allclose(time_diffs, [0,0])
    trainer_mock.train.assert_called_with(iterations=56)

@mock.patch("open_cp.sepp")
@mock.patch("open_cp.predictors.GridPredictionArray")
def test_initial_bandwidths(gpmock, seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    provider.ibtime = 0.3
    assert provider.ibtime == pytest.approx(0.3)
    
    provider.ibtime = "0.2"
    assert provider.ibtime == pytest.approx(0.2)

    provider.ibtime = datetime.timedelta(hours=4)
    assert provider.ibtime == pytest.approx(4/24)
    
    with pytest.raises(ValueError):
        provider.ibtime = np.timedelta64(4, "h")

    provider.ibspace = 25.4
    
    serialise( provider )
    assert provider.settings_string == "iters=40 initial=0.17/25.4 cutoff=120/500"
    
    standard_calls(provider, project_task, analysis_model2, grid_task)
    trainer = seppmock.SEPPTrainer.return_value
    assert trainer.initial_time_bandwidth == datetime.timedelta(hours=4)
    assert trainer.initial_space_bandwidth == pytest.approx(25.4)

@mock.patch("open_cp.sepp")
@mock.patch("open_cp.predictors.GridPredictionArray")
def test_cutoffs(gpmock, seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp.SEPP(model)
    provider.cttime = 30.2
    provider.ctspace = 123
    
    serialise( provider )
    assert provider.settings_string == "iters=40 initial=0.10/50.0 cutoff=30/123"
    
    standard_calls(provider, project_task, analysis_model2, grid_task)
    trainer = seppmock.SEPPTrainer.return_value
    assert trainer.time_cutoff == datetime.timedelta(days=30.2)
    assert trainer.space_cutoff == 123
