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

if not "GDAL_DATA" in os.environ:
    try:
        home = os.path.join(os.path.expanduser("~"), "Anaconda3", "Library", "share", "gdal")
        if "gcs.csv" in os.listdir(home):
            os.environ["GDAL_DATA"] = home
        else:
            print("GDAL_DATA not set and failed to find suitable location...")
    except:
        print("GDAL_DATA not set and failed to find suitable location...  This is probably not a problem on linux.")

import open_cp.sources.chicago as chicago
import open_cp.predictors
import open_cp.naive
import open_cp.geometry
import open_cp.plot
import open_cp.pool
import open_cp.evaluation

import open_cp.retrohotspot as retro

side_choices = {"Far North", "Northwest", "North", "West", "Central",
        "South", "Southwest", "Far Southwest", "Far Southeast"}

def get_side(side="South"):
    return chicago.get_side(side)

def load_data(datadir, side="South"):
    """Load the data: Burglary, in the South side only, limit to events happening
    on the days 2011-03-01 to 2012-01-06 inclusive.
    
    :return: Pair of `(geometry, points)`
    """
    chicago.set_data_directory(datadir)
    points = chicago.load(os.path.join(datadir, "chicago_two.csv"), {"BURGLARY"}, type="all_other")
    #points = chicago.load(os.path.join(datadir, "chicago_all_old.csv"), {"BURGLARY"}, type="all")
    # Limit time range
    start = np.datetime64("2011-03-01")
    end = np.datetime64("2012-01-07")
    points = points[(points.timestamps >= start) & (points.timestamps <= end)]
    
    geo = get_side(side)
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return geo, points

_cdict = {'red':   [(0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],
         'green': [(0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)],
         'blue':  [(0.0,  0.2, 0.2),
                   (1.0,  0.2, 0.2)]}
yellow_to_red = matplotlib.colors.LinearSegmentedColormap("yellow_to_red", _cdict)

def grid_for_side(xoffset=0, yoffset=0, xsize=250, ysize=250, side="South"):
    """Generated a masked grid for the passed side value.
    
    :param xoffset: How much to move the left side by
    :param yoffset: How much to move the bottom side by
    """
    grid = open_cp.data.Grid(xsize=xsize, ysize=ysize, xoffset=xoffset, yoffset=yoffset)
    return open_cp.geometry.mask_grid_by_intersection(get_side(side), grid)

def grid_for_south_side(xoffset=0, yoffset=0, xsize=250, ysize=250):
    """Generated a masked grid for the South side geometry.
    
    :param xoffset: How much to move the left side by
    :param yoffset: How much to move the bottom side by
    """
    return grid_for_side(xoffset, yoffset, xsize, ysize, "South")
    
def time_range():
    """28th September 2011 – 6th January 2012"""
    return open_cp.evaluation.HitRateEvaluator.time_range(np.datetime64("2011-09-28"),
            np.datetime64("2012-01-06"), np.timedelta64(1, "D"))

def to_dataframe(rates):
    frame = pd.DataFrame(rates).T
    frame.index.name = "Prediction Date"
    frame.columns.name = "% Coverage"
    return frame
    
# ---------------------------------------------------------------------------
# Retro hotspot stuff
# Defining here allows us to use multi-process code in a notebook

class RetroHotSpotEval(open_cp.evaluation.PredictionProvider):
    def __init__(self, masked_grid, points, time_window_length = np.timedelta64(56, "D")):
        self.time_window_length = time_window_length
        self.masked_grid = masked_grid
        self.points = points
    
    def predict(self, time):
        grid_pred = retro.RetroHotSpotGrid(grid=self.masked_grid)
        grid_pred.data = self.points
        grid_pred.weight = retro.Quartic(bandwidth = 1000)
        grid_risk = grid_pred.predict(start_time = time - self.time_window_length, end_time = time)
        grid_risk.mask_with(self.masked_grid)
        return grid_risk

class RHS_Eval_Task(open_cp.pool.Task):
    def __init__(self, masked_grid, points, key):
        super().__init__(key)
        self.evaluator = open_cp.evaluation.HitRateEvaluator(RetroHotSpotEval(masked_grid, points))
        self.evaluator.data = points
        
    def __call__(self):
        return self.evaluator.run(time_range(), range(0,51))
