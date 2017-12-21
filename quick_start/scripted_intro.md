# `scripted` module quick start

Follow the instructions to install the `open_cp` library.

We will consider the problem of loading in some data, running two different predictions algorithms on it, and then saving the result and viewing it in a spread-sheet.


## The input data

This data has been extracted from [the Chicago Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2).


Case Number | Date | Block | Primary Type | Latitude | Longitude
---  | ---  | ---  | ---  | --- | ---
HM263692 | 03/31/2006 08:15:00 AM | 014XX E 71ST ST | BATTERY | 41.766015201 | -87.590845351
HM269325 | 04/03/2006 10:30:00 AM | 042XX W IRVING PARK RD | THEFT | 41.953570492 | -87.734518833

All we need for the predictions algorithms to work are times and x and y coordinates.

- Times should not be text, but should be converted to `datetime` objects (from the Python standard library)
- Coordinates should be projected to be in meters, and not Latitude and Longitude

Optionally, you might want to specify some base geometry: an outline of your study area.  If you _don't_ do this, then the predictions will be on a rectangular grid.  If you specify some outline, then only the grid cells which intersect with the outline will be used.  In our example, we use the south side of Chicago, downloaded from [Chicago Data](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6) and processed using a [Jupyter notebook](Generate%20example%20dataset.ipynb).


### Writing the script

We need to provide two Python functions:

1. The input data (times and locations) using the container `open_cp.data.TimedPoints`
2. The geometry, in the form of a `shapely` polygon.


### Loading the data

If you are comfortable with parsing a CSV file in Python, then you just need to know that you can construct the container by:

    return open_cp.data.TimedPoints.from_coords(times, xcoords, ycoords)
    
Where `times` is a list of `datetime.datetime` objects, and `xcoords` and `ycoords` are lists of floats.  `times` needs to be ordered to be increasing.

If you don't know how to do this in Python, read on...

Firstly we need to **read the csv file** for which we use the [`csv`](https://docs.python.org/3.6/library/csv.html) module from the standard library.

    import csv
    
    with open("example.csv") as file:
        reader = csv.reader(file)
        header = next(reader)

Now `header` will be the list `["Case Number", "Date", ...]`.  We need to extract the datetime and the coordinates.

For the **datetime** you could use the [`datetime`](https://docs.python.org/3.6/library/datetime.html) standard library module directly, using the [`strptime`](https://docs.python.org/3.6/library/datetime.html#strftime-strptime-behavior) method, noting that the timestamps appear to be in US format.  An easier alternative is to use the [`python-dateutil`](https://dateutil.readthedocs.io/en/stable/) library which will have been installed with `open_cp`.  Our method is then

    import dateutil.parser
    
    def row_to_datetime(row):
        datetime_string = row[1]
        return dateutil.parser.parse(datetime_string)
        
We use `row[1]` as the timestamp is in column 1 of the csv file (counting from 0 of course).

For the **coordinates**, if they were already projected with a unit of meters, then we could simply read them as:

    def row_to_coords(row):
        x = float(row[5])
        y = float(row[4])
        return x, y

In our case, however, we need to project from longitude and latitude.  The correct Python package to use is [`pyproj`](https://pypi.python.org/pypi/pyproj).  We use [EPSG:2790](http://spatialreference.org/ref/epsg/2790/) for Chicago, and so the code becomes

    import pyproj
    
    proj = pyproj.Proj({"init" : "epsg:2790"})
    
    def row_to_coords(row):
        x = float(row[5])
        y = float(row[4])
        return proj(x, y)

Combining these, we get the required function

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


### Loading the geometry

For the geometry we have provided a sample [Shapefile](https://en.wikipedia.org/wiki/Shapefile).  You can use [`fiona`](https://pypi.python.org/pypi/Fiona) to load this, but it's not very beginner-friendly.  Instead, I recommend [`geopandas`](http://geopandas.org/), which we installed above.  It is now a simple matter of

    import geopandas as gpd
    
    def load_geometry():
        frame = gpd.read_file("SouthSide")
        return frame.geometry[0]

See [`example1.py`](example1.py) for a script putting all this together.



### The predictors












