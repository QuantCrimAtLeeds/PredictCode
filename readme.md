# Predictive algorithms for crime


## Data input

Input data is encapsulated in a `TimedPoints` class, which is a wrapper around a two dimensional numpy array, each row of which is a timestamp (type `numpy.datetime64`) and a pair of x-y coordinates.  We assume that these are "homogeneous", in that they represent a flat geometry.  Data in units of meters, feet, miles etc. are allowed; data as longitude / latitude is not.  The class exists mainly to provide convenience methods.

We provide converters from various other formats:

- Using `pyproj` we support conversion from longitude / latitude data given a map projection.


## Prediction

Prediction algorithms seek to give an estimate of "risk intensity", which we can think of as:

> Pick a point (x,y) and a small region A around the point with area |A|.  The risk intensity f(x,y) is the expected number of events to occur A in the next time window, divided by the area |A| times the size of the time window.

(Mathematically, f(x,y) is the _limit_ as both |A| and the time window become very small.)  Point process algorithms may estimate f as a continuously varying kernel, while grid based techniques will divide space into a grid, and estimate f as a constant value in each grid cell.






### Dependencies

For the moment, I will try to only use:

- Python 3
- numpy

For some of the "plugins" I may use some the geopandas stack.