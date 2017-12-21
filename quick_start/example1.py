import csv
import open_cp.data
import geopandas as gpd
import pyproj
import dateutil.parser

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


geo = load_geometry()
print(geo.geom_type)
data = load_points()
print(data, data.number_data_points)
print(data.timestamps[0])
