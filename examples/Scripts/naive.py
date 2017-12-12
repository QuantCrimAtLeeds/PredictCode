"""
naive.py
~~~~~~~~

Example of making predictions using the two `naive` predictors.
"""

# Allow running without installing
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# What we always need
import open_cp.scripted as scripted


# Custom code to load Chicago data

import open_cp.sources.chicago
import lzma

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
datadir = os.path.join("/media", "disk", "Data")

def load_points():
    """Load Chicago data for 2016"""
    filename = os.path.join(datadir, "chicago_all.csv.xz")
    with lzma.open(filename, "rt") as f:
        return open_cp.sources.chicago.load(f, "BURGLARY", type="all")

def load_geometry():
    """Load the geometry for Chicago; we'll use Southside, as ever..."""
    open_cp.sources.chicago.set_data_directory(datadir)
    return open_cp.sources.chicago.get_side("South")



## Run

import datetime

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    state.add_prediction(scripted.NaiveProvider, time_range)
    state.add_prediction(scripted.ScipyKDEProvider, time_range)

    state.score(scripted.HitRateEvaluator)
    state.score(scripted.HitCountEvaluator)

    state.save_predictions("naive_preds.pic.xz")

    state.process(scripted.HitRateSave("naive.csv"))
    state.process(scripted.HitCountSave("naive_counts.csv"))
