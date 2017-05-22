"""
naive
~~~~~

Implements some very "naive" prediction techniques, mainly for baseline
comparisons.
"""

from . import predictors
import numpy as _np

class CountingGridKernel(predictors.DataTrainer):
    """Makes "predictions" by simply laying down a grid, and then counting the
    number of events in each grid cell to generate a relative risk.
    
    :param grid_size: The width and height of each grid cell.
    :param region: Optionally, the :class:`RectangularRegion` to base the grid
      on.  If not specified, this will be the bounding box of the data.
    """
    def __init__(self, grid_size, region = None):
        self.grid_size = grid_size
        self.region = region
        
    def predict(self):
        if self.region is None:
            region = self.data.bounding_box
        xsize, ysize = region.grid_size(self.grid_size)

        matrix = _np.zeros((ysize, xsize))
        xg = _np.floor((self.data.xcoords - region.xmin) / self.grid_size).astype(_np.int)
        yg = _np.floor((self.data.ycoords - region.ymin) / self.grid_size).astype(_np.int)
        for x, y in zip(xg, yg):
            matrix[y][x] += 1

        return predictors.GridPredictionArray(self.grid_size, self.grid_size,
            matrix, region.xmin, region.ymin)
