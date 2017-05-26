import pytest
import numpy as np

import open_cp.stscan2 as stscan

def test_sort():
    times = [10,25,0]
    coords = [[1,2,3],[6,7,8]]
    s = stscan.AbstractSTScan(coords, times)
    
    np.testing.assert_array_equal(s.timestamps, [0,10,25])
    np.testing.assert_array_equal(s.coords, [[3,1,2],[8,6,7]])
    
def test_same_size():
    times = list(range(5))
    coords = np.random.random(size=(2,4))
    with pytest.raises(ValueError):
        stscan.AbstractSTScan(coords, times)
    
def test_times_into_past():
    times = list(range(50))
    coords = np.random.random(size=(2,50))
    s = stscan.AbstractSTScan(coords, times)
    expected = list(range(25))
    np.testing.assert_array_equal(s.allowed_times_into_past(), expected)
    
    times = [0,0,0,0,1,1,2,5,10]
    coords = np.random.random(size=(2,len(times)))
    s = stscan.AbstractSTScan(coords, times)
    expected = [0]
    np.testing.assert_array_equal(s.allowed_times_into_past(), expected)
    
    times = [0,0,29,29,30,31]
    coords = np.random.random(size=(2,len(times)))
    s = stscan.AbstractSTScan(coords, times)
    expected = [0]
    np.testing.assert_array_equal(s.allowed_times_into_past(), expected)
    
    times = [0]*4
    coords = np.random.random(size=(2,len(times)))
    s = stscan.AbstractSTScan(coords, times)
    assert len(s.allowed_times_into_past()) == 0
    
    times = [30]*4
    s = stscan.AbstractSTScan(coords, times)
    assert len(s.allowed_times_into_past()) == 0

    times = [0,0,0,2]
    s = stscan.AbstractSTScan(coords, times)
    assert len(s.allowed_times_into_past()) == 0

    times = [0,1,1,2]
    s = stscan.AbstractSTScan(coords, times)
    np.testing.assert_array_equal(s.allowed_times_into_past(), [0])

def test_cutoff_times():
    times = [0,0,0,0,1,1,2,5,10]
    coords = np.random.random(size=(2,len(times)))
    s = stscan.AbstractSTScan(coords, times)
    s.time_population_limit = 1
    times, cutoff = s.build_times_cutoff()
    np.testing.assert_array_equal(times, [0,1,2,5,10])
    np.testing.assert_array_equal(cutoff, [4,6,7,8,9])

def test_unique_points():
    coords = np.array([[0,0], [1,0], [1,1]]).T
    s = stscan.AbstractSTScan(coords, [0,0,0])
    got = { (x,y) for x,y in s._unique_points.T }
    expected = { (0,0), (1,0), (1,1) }
    assert( got == expected )
    assert( s._unique_points.shape == (2,3) )

    coords = np.array([[0,0], [1,2], [1,2]]).T
    s = stscan.AbstractSTScan(coords, [0,0,0])
    got = { (x,y) for x,y in s._unique_points.T }
    expected = { (0,0), (1,2) }
    assert( got == expected )
    assert( s._unique_points.shape == (2,2) )

def _to_binary(got):
    return set(tuple(1 if t else 0 for t in x[1]) for x in got)

def test_all_discs_around():
    times = [0,1,2]
    coords = np.array([[0,0], [1,0], [1,1]]).T
    s = stscan.AbstractSTScan(coords, times)
    assert len(list(s.all_discs_around([0,0]))) == 0
    
    times = [0,1,2,3]
    coords = np.array([[0,0], [0,0], [1,0], [2,2]]).T
    s = stscan.AbstractSTScan(coords, times)
    assert _to_binary(s.all_discs_around([0,0])) == {(1,1,0,0)}
    
    s.geographic_population_limit = 0.75
    assert _to_binary(s.all_discs_around([0,0])) == {(1,1,0,0), (1,1,1,0)}
    
    s.geographic_population_limit = 1
    assert _to_binary(s.all_discs_around([0,0])) == {(1,1,0,0), (1,1,1,0), (1,1,1,1)}

    s.geographic_population_limit = 1
    s.geographic_radius_limit = 1.0
    assert _to_binary(s.all_discs_around([0,0])) == {(1,1,0,0), (1,1,1,0)}

def test_all_discs_around_boundary():
    times = [0,1,2]
    coords = np.array([[0,0], [1,0], [0,1]]).T
    s = stscan.AbstractSTScan(coords, times)
    s.geographic_population_limit = 1
    assert _to_binary( s.all_discs_around([0,0]) ) == {(1,1,0), (1,0,1), (1,1,1)}

    times = [0,1,2,3]
    coords = np.array([[0,0], [1,0], [0,1], [1,1]]).T
    s = stscan.AbstractSTScan(coords, times)
    s.geographic_population_limit = 1
    assert _to_binary( s.all_discs_around([0,0]) ) == {(1,1,0,0), (1,0,1,0),
                     (1,1,1,0), (1,1,1,1)}
    
    coords = np.array([[0,0], [1,0], [0,1], [-1,0]]).T
    s = stscan.AbstractSTScan(coords, times)
    s.geographic_population_limit = 1
    assert _to_binary( s.all_discs_around([0,0]) ) == {(1,1,0,0), (1,0,1,0),
                     (1,0,0,1), (1,1,1,0), (1,1,0,1), (1,0,1,1), (1,1,1,1)}

def _disc_to_binary(got):
    return set(tuple(1 if t else 0 for t in x.mask) for x in got)

def _disc_to_tuples(got):
    return set( (x.centre[0], x.centre[1], x.radius_sq) for x in got)

def test_all_discs():
    times = [0,1,2]
    coords = np.array([[0,0], [1,0], [1,1]]).T
    s = stscan.AbstractSTScan(coords, times)
    assert len(list(s.all_discs())) == 0
    
    s.geographic_population_limit = 1
    assert _disc_to_binary(s.all_discs()) == {(1,1,0), (0,1,1), (1,1,1)}

def test_score_clusters():
    times = [0,0,1,1]
    coords = np.array([[0,0], [1,0], [1,1], [0,1]]).T
    s = stscan.AbstractSTScan(coords, times)
    s.time_population_limit = 1
    result = list(s.score_clusters())
    assert len(result) == 1
    assert s._statistic(2,1,4) == result[0][2]

def _masks_to_binary(m):
    return [ tuple(1 if x else 0 for x in y) for y in m.T ]

def test_STScanNumpy_time_ranges():
    times = [0,0,0,1,2,3]
    coords = np.random.random(size=(2, len(times)))
    s = stscan.STScanNumpy(coords, times)
    
    masks, counts, times = s.make_time_ranges()
    assert _masks_to_binary(masks) == [ (1,1,1,0,0,0) ]
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_array_equal(times, [0])
    
    s.time_population_limit = 0.8
    masks, counts, times = s.make_time_ranges()
    assert _masks_to_binary(masks) == [ (1,1,1,0,0,0), (1,1,1,1,0,0) ]
    np.testing.assert_array_equal(counts, [3, 4])
    np.testing.assert_array_equal(times, [0, 1])
    
    s.time_max_interval = 0
    masks, counts, times = s.make_time_ranges()
    assert _masks_to_binary(masks) == [ (1,1,1,0,0,0) ]
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_array_equal(times, [0])

def test_STScanNumpy_find_discs():
    coords = np.array([[0,0], [0,0], [1,0], [1,1]]).T
    times = np.random.random(coords.shape[1])
    s = stscan.STScanNumpy(coords, times)
    
    masks, counts, rds = s.find_discs([0,0])
    assert _masks_to_binary(masks) == [ (1,1,0,0) ]
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(rds, [0])

    s.geographic_population_limit = .75
    masks, counts, rds = s.find_discs([0,0])
    assert _masks_to_binary(masks) == [ (1,1,0,0), (1,1,1,0) ]
    np.testing.assert_array_equal(counts, [2,3])
    np.testing.assert_array_equal(rds, [0,1])
    
    s.geographic_population_limit = 1
    masks, counts, rds = s.find_discs([0,0])
    assert _masks_to_binary(masks) == [ (1,1,0,0), (1,1,1,0), (1,1,1,1) ]
    np.testing.assert_array_equal(counts, [2,3,4])
    np.testing.assert_array_equal(rds, [0,1,2])

    s.geographic_radius_limit = 0.5
    masks, counts, rds = s.find_discs([0,0])
    assert _masks_to_binary(masks) == [ (1,1,0,0) ]
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(rds, [0])

def test_STScanNumpy_find():
    coords = np.array([[0,0], [0,0], [1,0]]).T
    times = [0,1,2]
    s = stscan.STScanNumpy(coords, times)
    s.geographic_population_limit = 1
    s.time_population_limit = 1
    
    scores = list(s.find_all_clusters())
    assert len(scores) == 1
    np.testing.assert_array_equal(scores[0][0], [0,0])
    assert scores[0][1] == 0
    assert scores[0][2] == 1
    assert scores[0][3] == pytest.approx(0.3001045924)
