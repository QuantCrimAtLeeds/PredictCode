import pytest
import unittest.mock as mock

import open_cp.gui.predictors.lonlat as lonlat
import numpy as np
import scipy.spatial.distance as distance

def compare_projs(lon, lat):
    # UK+Ireland is about -11 <= lon <= 2,  50 <= lat <= 61
    xs = np.random.random(size=50) + lon
    ys = np.random.random(size=50) + lat

    projs = [ lonlat.Builtin(ys), lonlat.ViaUTM(xs), lonlat.BritishNationalGrid() ]
    dists = []
    for p in projs:
        x, y = p(xs, ys)
        dists.append( distance.pdist(np.asarray([x,y]).T) )
    a = np.abs(dists[1] - dists[2])
    assert np.average(a) < 250

    dists = []
    for p in projs:
        x, y = p(lon + 0.5, lat + 0.5)
        x1, y1 = p(lon + 0.501, lat + 0.501)
        dists.append(np.sqrt((x-x1)**2 + (y-y1)**2))

    assert max(dists) - min(dists) < 1
        

def test_projs():
    lon = np.random.uniform(low=-11, high=2, size=100)
    lat = np.random.uniform(low=50, high=61, size=100)
    for x, y in zip(lon, lat):
        compare_projs(x, y)

@pytest.fixture
def model():
    import datetime
    times = [datetime.datetime.now() for _ in range(4)]
    xcs = [0, -1, 0, 1]
    ycs = [54, 50, 55, 52]
    import collections
    Model = collections.namedtuple("Model", "times xcoords ycoords coord_type")
    import open_cp.gui.import_file_model as import_file_model
    return Model(times, xcs, ycs, import_file_model.CoordType.XY)

def test_passthrough(model):
    pt = lonlat.PassThrough(model)
    tasks = pt.make_tasks()
    assert len(tasks) == 1
    assert tasks[0].order == 0
    assert tasks[0].off_process == False
    x, y = tasks[0](model.xcoords, model.ycoords)
    np.testing.assert_allclose(x, [0,-1,0,1])
    np.testing.assert_allclose(y, [54, 50, 55, 52])

def test_lonlat_tasks(model):
    con = lonlat.LonLatConverter(model)
    tasks = con.make_tasks()
    assert len(tasks) == 1
    assert tasks[0].order == 0
    assert tasks[0].off_process == False
    
    x, y = tasks[0](model.xcoords, model.ycoords)
    xcs = [0, -1, 0, 1]
    ycs = [54, 50, 55, 52]
    expect = lonlat.Builtin(ycs)
    xe, ye = expect(xcs, ycs)
    np.testing.assert_allclose(x, xe)
    np.testing.assert_allclose(y, ye)

    con.selected = 1
    tasks = con.make_tasks()
    x, y = tasks[0](model.xcoords, model.ycoords)
    expect = lonlat.ViaUTM(xcs)
    xe, ye = expect(xcs, ycs)
    np.testing.assert_allclose(x, xe)
    np.testing.assert_allclose(y, ye)
    
    con.selected = 2
    tasks = con.make_tasks()
    x, y = tasks[0](model.xcoords, model.ycoords)
    expect = lonlat.BritishNationalGrid()
    xe, ye = expect(xcs, ycs)
    np.testing.assert_allclose(x, xe)
    np.testing.assert_allclose(y, ye)

    con.set_epsg(7405)
    tasks = con.make_tasks()
    x, y = tasks[0](model.xcoords, model.ycoords)
    expect = lonlat.EPSG(7405)
    xe, ye = expect(xcs, ycs)
    np.testing.assert_allclose(x, xe)
    np.testing.assert_allclose(y, ye)
