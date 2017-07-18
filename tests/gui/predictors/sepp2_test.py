from .helper import *

import open_cp.gui.predictors.sepp2 as sepp2
import open_cp.gui.predictors.predictor
#import datetime

@pytest.fixture
def analysis_model2(analysis_model):
    analysis_model.time_range = (datetime.datetime(2017,5,21,12,30),
                                 datetime.datetime(2017,5,21,13,30), None, None)
    return analysis_model

@mock.patch("open_cp.seppexp")
def test_SEPP(seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp2.SEPP(model)
    assert provider.name == "Grid based SEPP"
    assert provider.settings_string is None
    standard_calls(provider, project_task, analysis_model2, grid_task)

def test_serialise(model, project_task, analysis_model2, grid_task):
    serialise( sepp2.SEPP(model) )

def test_no_data(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,5,20,12,30),
                                 datetime.datetime(2017,5,20,13,30), None, None)
    provider = sepp2.SEPP(model)
    with pytest.raises(open_cp.gui.predictors.predictor.PredictionError):
        standard_calls(provider, project_task, analysis_model, grid_task)

@mock.patch("open_cp.seppexp")
def test_training_usage(seppmock, model, project_task, analysis_model2, grid_task):
    provider = sepp2.SEPP(model)
    subtask = standard_calls(provider, project_task, analysis_model2, grid_task)
    
    seppmock.SEPPTrainer.assert_called_with(grid=grid_task.return_value)
    trainer_mock = seppmock.SEPPTrainer.return_value
    np.testing.assert_allclose(trainer_mock.data.xcoords, [0, 10])
    np.testing.assert_allclose(trainer_mock.data.ycoords, [10, 20])
    time_diffs = ( (trainer_mock.data.timestamps -
        [np.datetime64("2017-05-21T12:30"), np.datetime64("2017-05-21T13:00")])
        / np.timedelta64(1,"ms") )
    np.testing.assert_allclose(time_diffs, [0,0])
    trainer_mock.train.assert_called_with(iterations=40, use_corrected=True)
    
    pred = trainer_mock.train.return_value
    np.testing.assert_allclose(pred.data.xcoords, [0, 10, 20])
    np.testing.assert_allclose(pred.data.ycoords, [10, 20, 0])
    
    train_date = datetime.datetime(2017,5,22,5,35)
    prediction = subtask(train_date)
    assert prediction == pred.predict.return_value
    pred.predict.assert_called_with(train_date)
