import open_cp.sources.random as testmod

import open_cp
from datetime import datetime
import numpy as np

def test_uniform_random():
    region = open_cp.RectangularRegion(xmin=0, xmax=100, ymin=-20, ymax=-10)
    start = datetime(2017, 3, 10, 0)
    end = datetime(2017, 3, 20, 0)
    points = testmod.uniform_random(region, start, end, 100)
    
    for p in points.coords:
        x, y = p
        assert( x >= 0 and x <= 100 )
        assert( y >= -20 and y <= -10 )
    for t in points.timestamps:
        assert( t >= start and t <= end )
        
def check_uniform_bins(data, num_bins, conf):
    size = max(data) / num_bins
    for i in range(num_bins):
        bin_start = i * size
        bin_end = bin_start + size
        count = len(data[ (data >= bin_start) & (data <= bin_end) ])
        count -= len(data) / num_bins
        assert( abs(count) < conf )
        
def test_uniform_random_statistical_properties():
    region = open_cp.RectangularRegion(xmin=0, xmax=100, ymin=-20, ymax=-10)
    start = datetime(2017, 3, 10, 0)
    end = datetime(2017, 3, 20, 0)

    num_points = []
    times, xcs, ycs = [], [], []
    samples = 1000
    for _ in range(samples):
        points = testmod.uniform_random(region, start, end, 100)
        num_points.append( len(points.timestamps) )
        times.extend(points.timestamps)
        xcs.extend(points.coords[:,0])
        ycs.extend(points.coords[:,0])
    
    num_points_delta = np.mean(num_points) - 100
    assert( num_points_delta >= -5 and num_points_delta <= 5 )
    
    times = (np.array(times) - np.datetime64(start)) / np.timedelta64(1, "s")
    # Guessing at confidence intervals
    check_uniform_bins(times, 10, samples * 0.3)
    check_uniform_bins(np.array(xcs), 10, samples * 0.3)
    ycs = np.array(ycs)
    check_uniform_bins(ycs - min(ycs), 10, samples * 0.3)