"""
naive
~~~~~

Implements some very "naive" prediction techniques, mainly for baseline
comparisons.
"""

from . import predictors
import numpy as _np
try:
    import scipy.stats as _stats
except Exception:
    import sys
    print("Failed to load scipy.stats", file=sys.stderr)
    _stats = None


class CountingGridKernel(predictors.DataTrainer):
    """Makes "predictions" by simply laying down a grid, and then counting the
    number of events in each grid cell to generate a relative risk.

    This can also be used to produce plots of the actual events which occurred:
    essentially a two-dimensional histogram.
    
    :param grid_width: The width of each grid cell.
    :param grid_height: The height of each grid cell, if None, then the same as
      `width`.
    :param region: Optionally, the :class:`RectangularRegion` to base the grid
      on.  If not specified, this will be the bounding box of the data.
    """
    def __init__(self, grid_width, grid_height = None, region = None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.region = region
        
    def predict(self):
        """Produces an instance of :class:`GridPredictionArray` based upon the
        set :attrib:`region` (defaulting to the bounding box of the input
        data).  Each entry of the "risk intensity matrix" will simply be the
        count of events in that grid cell.

        Changing the "region" may be important, as it will affect exactly which
        grid cell an event falls into.  Events are always clipped to the region
        before being assigned to cells.  (This is potentially important if the
        region is not an exact multiple of the grid size.)
        """
        if self.region is None:
            region = self.data.bounding_box
        else:
            region = self.region
        xsize, ysize = region.grid_size(self.grid_width, self.grid_height)
        height = self.grid_width if self.grid_height is None else self.grid_height

        matrix = _np.zeros((ysize, xsize))
        mask = ( (self.data.xcoords >= region.xmin) & (self.data.xcoords <= region.xmax)
                & (self.data.ycoords >= region.ymin) & (self.data.ycoords <= region.ymax) )
        xc, yc = self.data.xcoords[mask], self.data.ycoords[mask]
        xg = _np.floor((xc - region.xmin) / self.grid_width).astype(_np.int)
        yg = _np.floor((yc - region.ymin) / height).astype(_np.int)
        for x, y in zip(xg, yg):
            matrix[y][x] += 1

        return predictors.GridPredictionArray(self.grid_width, height,
            matrix, region.xmin, region.ymin)


class ScipyKDE(predictors.DataTrainer):
    """A light wrapper around the `scipy` Gaussian KDE.  Uses just the space
    coordinates of the events to estimate a risk density.
    """
    def __init__(self):
        pass

    def predict(self, bw_method = None):
        """Produces an instance of :class:`KernelRiskPredictor` wrapping the
        result of the call to `scipy.stats.kde.gaussian_kde()`.

        :param bw_method: The bandwidth estimation method, to be passed to
          `scipy`.  Defaults to None (currently the "scott" method).
        """
        kernel = _stats.kde.gaussian_kde(self.data.coords, bw_method)
        return predictors.KernelRiskPredictor(kernel)

    def grid_predict(self, grid_size, bw_method = None):
        """Produces an instance of :class:`GridPredictionArray` wrapping the
        result of the call to `scipy.stats.kde.gaussian_kde()`.  The region
        used is the bounding box of the input data.  For more control, use the
        :method:`predict` and set the offset and grid size to sample down to a
        custom grid.

        :param grid_size: The width and height of each grid cell.
        :param bw_method: The bandwidth estimation method, to be passed to
          `scipy`.  Defaults to None (currently the "scott" method).
        """
        kernel = _stats.kde.gaussian_kde(self.data.coords, bw_method)
        region = self.data.bounding_box
        return predictors.grid_prediction_from_kernel(kernel, region, grid_size)
