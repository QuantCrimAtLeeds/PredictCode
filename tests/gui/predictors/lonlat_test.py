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
    assert np.average(a) < 200

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
