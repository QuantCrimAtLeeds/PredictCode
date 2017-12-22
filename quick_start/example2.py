# Import libraries we need
import csv
import open_cp.data
import geopandas as gpd
import pyproj
import dateutil.parser
import open_cp.scripted as scripted
import datetime


# Load the input data; see `scripted_intro.md`

def row_to_datetime(row):
    datetime_string = row[1]
    return dateutil.parser.parse(datetime_string)

proj = pyproj.Proj({"init" : "epsg:2790"})

def row_to_coords(row):
    x = float(row[5])
    y = float(row[4])
    return proj(x, y)

def load_points():
    with open("example.csv") as file:
        reader = csv.reader(file)
        header = next(reader)
        # Assume the header is correct
        times = []
        xcoords = []
        ycoords = []
        for row in reader:
            times.append(row_to_datetime(row))
            x, y = row_to_coords(row)
            xcoords.append(x)
            ycoords.append(y)
      
    # Maybe `times` is not sorted.
    times, xcoords, ycoords = open_cp.data.order_by_time(times, xcoords, ycoords)
      
    return open_cp.data.TimedPoints.from_coords(times, xcoords, ycoords)

def load_geometry():
    frame = gpd.read_file("SouthSide")
    return frame.geometry[0]


# Perform the predictions; see `scripted_intro.md`

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:

    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))

    state.add_prediction(scripted.NaiveProvider, time_range)

    state.score(scripted.HitRateEvaluator)
    state.score(scripted.HitCountEvaluator)

    state.process(scripted.HitRateSave("rates.csv"))
    state.process(scripted.HitCountSave("counts.csv"))
