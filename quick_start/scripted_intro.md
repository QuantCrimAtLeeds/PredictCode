# `scripted` module quick start

Follow the [instructions to install](install.md) the `open_cp` library.

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


## Structure of a script

A script will now have this structure:

    import open_cp.scripted as scripted
    import datetime

    ### Functions to load data and geometry as above

    with scripted.Data(load_points, load_geometry,
            start=datetime.datetime(2016,1,1)) as state:

        time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
                datetime.datetime(2017,1,1), datetime.timedelta(days=1))

        # Add predictors

        # Add evaluation

        # Add saving the results

You can add the predictors, evaluators etc. in any order.  (Under the hood, the `with`
command shows that we're using a [context manager](https://jeffknupp.com/blog/2016/03/07/python-with-context-managers/) and once inside 
the context, I use lazy evaluation to just store the requests.  At the end of the context, all the tasks are actually run.  I partly copied this style from `pymc3`.)

The `start=datetime.datetime(2016,1,1)` parameter to `scripted.Data` means that we will only load data which has a timestamp on or after this date.  You can also specify an `end` date, but this is rarely needed, as the predictors also limit data to an end point.

The `time_range` specifies the range of dates we will produce and score evaluations on.  Our setting of

        time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
                datetime.datetime(2017,1,1), datetime.timedelta(days=1))

means that we will make prediction for every day between 1st October 2016 (inclusive) and 1st January 2017 (exclusive), and that we will evaluate each prediction on the next day's worth of events.

### The predictors

We give a full list below, but for now we'll just use the `naive` predictor,

    state.add_prediction(scripted.NaiveProvider, time_range)

We pass the _class_ `scripted.NaiveProvider` as the predictor, and the `time_range` to specify which predictions to make.

### The evaluators

We use two variants of the hit-rate

    state.score(scripted.HitRateEvaluator)
    state.score(scripted.HitCountEvaluator)

The `HitRateEvaluator` computes the hit rate for each prediction.  The `HitCountEvaluator` computes the _count_ of captured crimes, and the total count, but does not perform the division to give the hit rate (we prefer to fit the data later to a simple statistical model).

### Saving the results

To make use of these evaluations, we need to save them:

    state.process(scripted.HitRateSave("rates.csv"))
    state.process(scripted.HitCountSave("counts.csv"))

This will save, in a user-friendly format, to CSV files.

Optionally, we can save the actual predictions:

    state.save_predictions("naive_preds.pic.xz")

This will use `pickle` compressed with `lzma`.  The use is to allow loading into a Jupyter notebook (or other Python script) later, see [an example](../examples/Scripts/Reload%20naive%20predictions.ipynb).  


## Putting it all together

See [`examples2.py`](example2.py) for a fully working example.  This can be run as

    python example2.py

You should see useful logs displayed, showing progress.  Once predictors are running, we'll only log progress once every minute; the "time left" is only a rough estimate.

Examine `rates.csv` and `counts.csv` in your favourite spread-sheet.  For the brave(r) we provide some tools for use in Jupyter notebooks; compare [an example](../examples/Scripts/Reload%20naive%20predictions.ipynb).


### Specifying a grid

Currently all methods are "grid based".  The default grid is 150m squared.  You can specify a different grid size through:

    import open_cp.data
    grid = open_cp.data.Grid(xsize=100, ysize=100, xoffset=50, yoffset=0)

    with scripted.Data(load_points, load_geometry,
            start=datetime.datetime(2016,1,1), grid=grid) as state:

Here we specify a grid of 100m square (rectangular grid cells are _almost_ universally
supported).  We also specify an "offset"; in this example we shift the grid 50m in the x
direction.  The script [`example3.py`](example3.py) shows this in action.  Run it, and check the log line showing the new grid setting.


# All (current) evaluators

[For the brave(r) these are defined in [`open_cp.evaluation`](../open_cp/evaluation.py) but are loaded automatically by the `scripted` module.)

### Naive predictors

From [`open_cp.naive`](../open_cp/naive.py):

- `NaiveProvider` which uses `open_cp.naive.CountingGridKernel`
- `ScipyKDEProvider` which uses `open_cp.naive.ScipyKDE`

Neither have any settings.


### Retrospective hotspotting

From [`open_cp.retrohotspot`](../open_cp/retrohotspot.py):

- `RetroHotspotProvider` which uses `open_cp.retrohotspot.RetroHotSpotGrid`.  You need to specify a "weight" to use:

        import open_cp.retrohotspot

        weight = open_cp.retrohotspot.Quartic(150)
        state.add_prediction(scripted.RetroHotspotProvider(weight), time_range)

Here we specify the bandwidth of the `Quartic` weight as 150m.

- `RetroHotspotCtsProvider` which uses `open_cp.retrohotspot.RetroHotSpot`.
Again, you need to specify a "weight" to use:

        import open_cp.retrohotspot

        weight = open_cp.retrohotspot.Quartic(120)
        state.add_prediction(scripted.RetroHotspotCtsProvider(weight), time_range)


### Prospective hotspotting

From [`open_cp.prohotspot`](../open_cp/prohotspot.py):

- `ProHotspotProvider` which uses `open_cp.prohotspot.ProspectiveHotSpot`.  You need to specify the "weight", the "distance" which specifies how to compute distances between grid cells, and a "time unit" which specifies how to measure time.

        import open_cp.prohotspot

        weight = open_cp.prohotspot.ClassicWeight(time_bandwidth=2, space_bandwidth=4)
        distance = open_cp.prohotspot.DistanceCircle()
        state.add_prediction(scripted.ProHotspotProvider(weight, distance,
                datetime.timedelta(days=7)), time_range)

- `ProHotspotCtsProvider` which uses `open_cp.prohotspot.ProspectiveHotSpotContinuous`.  You need to specify the weight, and a "distance unit".  The time unit is always one week, so the distance unit needs to be compatible with this.

        import open_cp.prohotspot

        weight = open_cp.prohotspot.ClassicWeight(time_bandwidth=8, space_bandwidth=3)
        state.add_prediction(scripted.ProHotspotCtsProvider(weight, 150), time_range)

Here we use a distance of 150m.


### KDE

From [`open_cp.kde`](../open_cp/kde.py):

- `KDEProvider` which uses `open_cp.kde.KDE`.  You need to specify a time kernel and a space kernel, from choices in `open_cp.kde` (or, for the brave, a user-defined shape).

        import open_cp.kde

        tk = open_cp.kde.ExponentialTimeKernel(10)
        sk = open_cp.kde.GaussianFixedBandwidthProvider(50)
        state.add_prediction(scripted.KDEProvider(tk, sk), time_range)

Here we use an exponentially decaying time kernel, with a scale of 10 (days), and a Gaussian space kernel with a bandwidth of 50m.


### STScan

From [`open_cp.stscan`](../open_cp/stscan.py):

- `STScanProvider` which uses `open_cp.stscan.STSTrainer`.  You need to specify the maximum size of "cylinders" to consider in the scan statistic.  

        provider = scripted.STScanProvider(500, datetime.timedelta(days=21), False)
        state.add_prediction(provider, time_range)

Here we use 500m and 21 days.


### SEPP

Coming soon...

