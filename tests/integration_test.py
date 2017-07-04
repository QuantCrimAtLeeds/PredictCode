# These are longer tests running on data.

import numpy as np
import pytest
import os

import open_cp
import open_cp.stscan

def read_test_file(basename):
    coords = []
    with open(basename + ".geo") as geofile:
        for line in geofile:
            i, x, y = line.split()
            coords.append( (float(x), float(y)) )

    timestamps = []
    with open(basename + ".cas") as casefile:
        for line in casefile:
            i, count, t = line.split()
            timestamps.append(np.datetime64(t))
            
    return np.asarray(timestamps), np.asarray(coords).T

def test_stscan_slow():
    timestamps, points = read_test_file(os.path.join("tests", "sts_test_data"))
    trainer = open_cp.stscan.STSTrainerSlow()
    trainer.data = open_cp.TimedPoints.from_coords(timestamps, points[0], points[1])
    result = trainer.predict()

    assert(result.clusters[0].centre[0] == pytest.approx(0.31681704))
    assert(result.clusters[0].centre[1] == pytest.approx(0.26506492))
    assert(result.clusters[0].radius == pytest.approx(0.1075727))
    assert(result.statistics[0] == pytest.approx(1.8503078))
    assert(result.time_ranges[0][0] == np.datetime64("2017-01-10"))
    assert(result.time_ranges[0][1] == np.datetime64("2017-01-10"))

    assert(result.clusters[1].centre[0] == pytest.approx(0.25221791))
    assert(result.clusters[1].centre[1] == pytest.approx(0.9878925))
    assert(result.clusters[1].radius == pytest.approx(0.1579045))
    assert(result.statistics[1] == pytest.approx(1.2139669))
    assert(result.time_ranges[1][0] == np.datetime64("2017-01-09"))
    assert(result.time_ranges[1][1] == np.datetime64("2017-01-10"))

    assert(result.clusters[2].centre[0] == pytest.approx(0.77643025))
    assert(result.clusters[2].centre[1] == pytest.approx(0.80196054))
    assert(result.clusters[2].radius == pytest.approx(0.268662))
    assert(result.statistics[2] == pytest.approx(0.81554083))
    assert(result.time_ranges[2][0] == np.datetime64("2017-01-09"))
    assert(result.time_ranges[2][1] == np.datetime64("2017-01-10"))
    
def test_stscan():
    timestamps, points = read_test_file(os.path.join("tests", "sts_test_data"))
    trainer = open_cp.stscan.STSTrainer()
    trainer.data = open_cp.TimedPoints.from_coords(timestamps, points[0], points[1])
    result = trainer.predict()

    assert(result.statistics[0] == pytest.approx(1.8503078))
    #assert(result.clusters[0].centre[0] == pytest.approx(0.31681704))
    #assert(result.clusters[0].centre[1] == pytest.approx(0.26506492))
    #assert(result.clusters[0].radius == pytest.approx(0.1075716))
    assert(result.clusters[0].centre[0] == pytest.approx(0.31267453))
    assert(result.clusters[0].centre[1] == pytest.approx(0.15757309))
    assert(result.clusters[0].radius == pytest.approx(0.201690059))
    assert(result.time_ranges[0][0] == np.datetime64("2017-01-10"))
    assert(result.time_ranges[0][1] == np.datetime64("2017-01-10"))

    assert(result.statistics[1] == pytest.approx(1.2139669))
    assert(result.clusters[1].centre[0] == pytest.approx(0.25221791))
    assert(result.clusters[1].centre[1] == pytest.approx(0.9878925))
    assert(result.clusters[1].radius == pytest.approx(0.1579029))
    assert(result.time_ranges[1][0] == np.datetime64("2017-01-09"))
    assert(result.time_ranges[1][1] == np.datetime64("2017-01-10"))

    assert(result.statistics[2] == pytest.approx(0.81554083))
    assert(result.clusters[2].centre[0] == pytest.approx(0.77643025))
    assert(result.clusters[2].centre[1] == pytest.approx(0.80196054))
    assert(result.clusters[2].radius == pytest.approx(0.2686593))
    assert(result.time_ranges[2][0] == np.datetime64("2017-01-09"))
    assert(result.time_ranges[2][1] == np.datetime64("2017-01-10"))