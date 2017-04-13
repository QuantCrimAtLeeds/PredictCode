Quick start
===========

As a work in progress, the `open_cp` module cannot be installed, but can simply be imported.  This is a Python 3 module.
It relies upon `numpy` only, for core functionality.

1. The recommended Python distribution is `Anaconda <https://www.continuum.io/downloads>`_
2. Fork or download `the repository <https://github.com/QuantCrimAtLeeds/PredictCode>`_ from github.
3. Unpack on your computer and in the main directory run `jupyter notebook`.
4. Navigate to the `examples <https://github.com/QuantCrimAtLeeds/PredictCode/tree/master/examples>`_ directory
   and look at the example notebooks.


Data input
----------

Input data is encapsulated in a :any:`TimedPoints` class.
We assume that these are "homogeneous", in that they represent a flat geometry.  That is, data in units of meters, feet, miles etc. is allowed;
data as longitude / latitude won't function correctly.

We provide converters from various other formats:

- Using `pyproj <https://github.com/jswhit/pyproj>`_ we support conversion from longitude / latitude data given a map projection.

See the :any:`open_cp.data` module.


Prediction
----------

Prediction algorithms seek to give an estimate of "risk intensity", which we can think of as:

  Pick a point `(x,y)` and a small region `A` around the point with area `|A|`.  The risk intensity `f(x,y)` is the expected number of events to occur `A` in the next time window,
  divided by the area `A` times the size of the time window.

(Mathematically, :math:`f(x,y)` is the *limit* as both :math:`|A|` and the time window become very small.)  Point process algorithms may estimate :math:`f`
as a continuously varying kernel, while grid based techniques will divide space into a grid, and estimate :math:`f` as a constant value in each grid cell.

Once the relative risk intensity has been estimated, it is common to sample the risk intensity to a grid (if the prediction method was continuous)
and then to select the top 1%, 5%, 10% etc. grid cells by risk.  These can then form the basis for targeted "guardian" action.  For historic data, when
"scoring" the prediction method, it is common to calculate the fraction of actual crime which occurred in these highest risk grid cells to get a "hit rate".

The module :any:`open_cp.predictors` contains base classes and utility methods.


Data sources
------------

The ability to load some external open data sources is provided (with instructions for downloading the actual data!) and some, currently simple,
"synthetic data" producers are provided.  See the :doc:`open_cp.sources` sub-package.


Algorithms
----------

See the `examples <https://github.com/QuantCrimAtLeeds/PredictCode/tree/master/examples>`_ notebooks for reviews of
the literature, implementation details, and examples, for each of these algorithms.

- The :doc:`retrohotspot` module implements the "retrospective hotspotting" algorithm.
- The :doc:`prohotspot` module implements the "prospective hotspotting" algorithm.
- The :doc:`sepp` module implements the "epidemic-type aftershock model" (ETAS) algorithm, with a variable bandwidth kernel density estimator.
- The :doc:`seppexp` module implements a grid based parametric ETAS model.


Other modules
-------------

- The :any:`open_cp.kernels` module provides base classes for "kernels" (probability density functions, or risk intensity functions)
  along with some kernel density estimation algorithms.


Testing
-------

The code has fairly good unit test coverage.  See the `tests <https://github.com/QuantCrimAtLeeds/PredictCode/tree/master/tests>`_ directory which
contains `pytest <https://docs.pytest.org/en/latest/>`_ tests.