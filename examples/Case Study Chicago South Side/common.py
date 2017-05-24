# Allow us to load `open_cp` without installing
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import matplotlib.pyplot as plt
import matplotlib
import descartes
import os
import numpy as np
import descartes
import pandas as pd

import open_cp.sources.chicago as chicago
import open_cp.predictors
import open_cp.naive
import open_cp.geometry
import open_cp.plot
import open_cp.evaluation

def load_data(datadir):
    """Load the data: Burglary, in the South side only, limit to events happening
    on the days 2011-03-01 to 2012-01-06 inclusive."""
    global points
    global south_side
    chicago.set_data_directory(datadir)
    south_side = chicago.get_side("South")
    points = chicago.load(os.path.join(datadir, "chicago_two.csv"), {"BURGLARY"}, type="all_other")
    start = np.datetime64("2011-03-01")
    end = np.datetime64("2012-01-07")
    points = points[(points.timestamps >= start) & (points.timestamps <= end)]
    points = open_cp.geometry.intersect_timed_points(points, south_side)
    return south_side, points

_cdict = {'red':   [(0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],
         'green': [(0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)],
         'blue':  [(0.0,  0.2, 0.2),
                   (1.0,  0.2, 0.2)]}
yellow_to_red = matplotlib.colors.LinearSegmentedColormap("yellow_to_red", _cdict)

def grid_for_south_side(xoffset=0, yoffset=0):
    """Generated a masked grid for the South side geometry.
    
    :param xoffset: How much to move the left side by
    :param yoffset: How much to move the bottom side by
    """
    grid = open_cp.data.Grid(xsize=250, ysize=250, xoffset=0, yoffset=0)
    global south_side
    return open_cp.geometry.mask_grid_by_intersection(south_side, grid)

def time_range():
    """28th September 2011 – 6th January 2012"""
    return open_cp.evaluation.HitRateEvaluator.time_range(np.datetime64("2011-09-28"),
            np.datetime64("2012-01-06"), np.timedelta64(1, "D"))

def to_dataframe(rates):
    frame = pd.DataFrame(rates).T
    frame.index.name = "Prediction Date"
    frame.columns.name = "% Coverage"
    return frame
    