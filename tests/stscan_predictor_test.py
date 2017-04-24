import numpy as np
import pytest
import unittest.mock as mock
import datetime

import open_cp.stscan as testmod
import open_cp

def an_STSResult():
    region = open_cp.RectangularRegion(xmin=0,xmax=100,ymin=50,ymax=100)
    clusters = [testmod.Cluster([10,60],10), testmod.Cluster([50,50], 20)]
    return testmod.STSResult(region, clusters)

def test_STSContinuousPrediction():
    pred = an_STSResult().continuous_prediction()
    assert(pred.risk(0,0) == 0)
    assert(pred.risk(10,60) == 2.0)
    assert(pred.risk(50,50) == 1.0)
    d = np.sqrt(50) / 20
    assert(pred.risk(55,55) == pytest.approx((1-d*d)**2))
    
def an_STSResult_overlapping_clusters():
    region = open_cp.RectangularRegion(xmin=0,xmax=100,ymin=50,ymax=50)
    clusters = [testmod.Cluster([10,60],10), testmod.Cluster([15,60], 15)]
    return testmod.STSResult(region, clusters)
    
def test_STSContinuousPrediction_overlapping():
    pred = an_STSResult_overlapping_clusters().continuous_prediction()
    d1 = 2 / 10
    r1 = (1 - d1**2)**2
    d2 = 3 / 15
    r2 = (1 - d2**2)**2
    assert(pred.risk(12,60) == pytest.approx(r1+r2+1))
    
def test_STSContinuousPrediction_customweight():
    pred = an_STSResult_overlapping_clusters().continuous_prediction()
    pred.weight = lambda t : 1
    assert(pred.risk(12,60) == 3)
    
def an_STSResult_nonaligned_centres():
    region = open_cp.RectangularRegion(xmin=0,xmax=100,ymin=50,ymax=100)
    clusters = [testmod.Cluster([9,59.5],10), testmod.Cluster([49,50.5], 12)]
    return testmod.STSResult(region, clusters)

def test_STSResult():
    pred = an_STSResult_nonaligned_centres().grid_prediction(10)
    assert((pred.xsize, pred.ysize) == (10, 5))
    assert(pred.intensity_matrix.shape == (5,10))
    # 1st row has centre points (5,55), (15,55), (25,55) etc.
    expected = np.zeros((5,10))
    expected[0][0] = 2.0
    expected[0][1] = 1.5
    expected[1][0] = 1.75
    expected[1][1] = 1.25
    
    expected[0][4] = 1.0
    expected[0][5] = 0.5
    
    np.testing.assert_allclose(pred.intensity_matrix, expected)
    
def test_STSTrainer_properties():
    trainer = testmod.STSTrainer()
    assert(trainer.geographic_population_limit == pytest.approx(0.5))
    with pytest.raises(ValueError):
        trainer.geographic_population_limit = 50
    trainer.geographic_population_limit = 0.2
    assert(trainer.geographic_population_limit == pytest.approx(0.2))
    
    assert(trainer.geographic_radius_limit == 3000)
    trainer.geographic_radius_limit = 1000
    assert(trainer.geographic_radius_limit == 1000)

    assert(trainer.time_population_limit == pytest.approx(0.5))
    with pytest.raises(ValueError):
        trainer.time_population_limit = -2
    trainer.time_population_limit = 0.3
    assert(trainer.time_population_limit == pytest.approx(0.3))
    
    assert(trainer.time_max_interval / np.timedelta64(1,"W") == 12)
    trainer.time_max_interval = datetime.timedelta(days=5)
    assert(trainer.time_max_interval / np.timedelta64(1,"D") == 5)

def a_custom_trainer():
    trainer = testmod.STSTrainer()
    trainer.geographic_population_limit = 0.2
    trainer.geographic_radius_limit = 1000
    trainer.time_population_limit = 0.3
    trainer.time_max_interval = datetime.timedelta(days=5)
    
    times = [datetime.datetime(2017,4,20) + i * datetime.timedelta(hours=2)
        for i in range(100)]
    xcoords = list(range(100))
    ycoords = [200 - i for i in range(100)]
    trainer.data = open_cp.TimedPoints.from_coords(times, xcoords, ycoords)
    
    return trainer    
    
def test_STSTrainer_bin_timestamps():
    trainer = a_custom_trainer()
    new = trainer.bin_timestamps(datetime.datetime(2017,4,20,0,35),
                           datetime.timedelta(hours=3))
    assert(new.data.timestamps[0] == np.datetime64("2017-04-19T21:35"))
    assert(new.data.timestamps[1] == np.datetime64("2017-04-20T00:35"))
    assert(new.data.timestamps[2] == np.datetime64("2017-04-20T03:35"))
    assert(new.data.timestamps[3] == np.datetime64("2017-04-20T03:35"))
    
def test_STSTrainer_grid_coords():
    trainer = a_custom_trainer()
    region = open_cp.RectangularRegion(xmin=5.2, ymin=4.5, xmax=0, ymax=0)
    new = trainer.grid_coords(region, 2)
    # Centres of grid will have offset of (6.2, 5.5) and each width/height is 2
    assert(new.data.xcoords[0] == pytest.approx(0.2))
    assert(new.data.xcoords[1] == pytest.approx(0.2))
    assert(new.data.xcoords[2] == pytest.approx(2.2))
    assert(new.data.ycoords[0] == pytest.approx(199.5))
    assert(new.data.ycoords[1] == pytest.approx(199.5))
    assert(new.data.ycoords[2] == pytest.approx(197.5))
    
def test__possible_start_times():
    timestamps = np.datetime64("2017-04-20") + np.array([
            i * np.timedelta64(1,"h") for i in range(10)])
    times = testmod._possible_start_times(timestamps, np.timedelta64(5,"h"),
                                  np.datetime64("2017-04-20T07:00"))
    assert(times[0] == np.datetime64("2017-04-20T02:00"))
    assert(times[1] == np.datetime64("2017-04-20T03:00"))
    assert(len(times) == 6)
    
    timestamps = np.datetime64("2017-04-20") + np.array([
            0 * np.timedelta64(1,"h") for i in range(10)])
    times = testmod._possible_start_times(timestamps, np.timedelta64(5,"h"),
                                  np.datetime64("2017-04-20T07:00"))
    assert(len(times) == 0)    
    
    times = testmod._possible_start_times(timestamps, np.timedelta64(5,"h"),
                                  np.datetime64("2017-04-20T04:00"))
    assert(times[0] == np.datetime64("2017-04-20"))
    assert(len(times) == 1)

    timestamps = np.array([np.datetime64("2017-04-20"),
        np.datetime64("2017-04-20"), np.datetime64("2017-04-20T01:00"),
        np.datetime64("2017-04-20T02:00"), np.datetime64("2017-04-20T02:00"),
        np.datetime64("2017-04-20T02:00"), np.datetime64("2017-04-20T03:00") ])
    times = testmod._possible_start_times(timestamps, np.timedelta64(5,"h"),
                                  np.datetime64("2017-04-20T03:00"))
    assert(times[0] == np.datetime64("2017-04-20"))
    assert(times[1] == np.datetime64("2017-04-20T01:00"))
    assert(times[2] == np.datetime64("2017-04-20T02:00"))
    assert(times[3] == np.datetime64("2017-04-20T03:00"))
    assert(len(times) == 4)
                                      
def resulting_sets(points, discs):
    out = []
    for disc in discs:
        mask = np.sum((points - disc.centre[:,None])**2, axis=0) <= disc.radius**2
        out.append(frozenset(i for i in range(len(mask)) if mask[i]))
    return out
    
def test__possible_space_clusters():
    points = np.array([[0,0],[1,0],[2,0]]).T
    discs = testmod._possible_space_clusters(points)
    sets = resulting_sets(points, discs)
    expected = { frozenset([0]), frozenset([1]), frozenset([2]),
                frozenset([0,1]), frozenset([1,2]), frozenset([0,1,2]) }
    assert(len(sets) == len(expected))
    assert(set(sets) == expected)

def test__possible_space_clusters2():
    points = np.array([[0,0],[1,0],[2,0],[1,2]]).T
    discs = testmod._possible_space_clusters(points)
    sets = resulting_sets(points, discs)
    expected = { frozenset([0]), frozenset([1]), frozenset([2]),
                frozenset([3]), frozenset([0,1]), frozenset([1,2]),
                frozenset([0,1,2,3]), frozenset([0,1,2]), frozenset([1,3]) }
    assert(len(sets) == len(expected))
    assert(set(sets) == expected)
    