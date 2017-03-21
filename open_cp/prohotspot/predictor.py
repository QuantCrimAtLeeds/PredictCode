from .. import predictors
from .. import data

import abc as _abc
import numpy as _np

class Weight(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def weight(self, cell, timestamp, x, y):
        pass

class ClassicDiagonalsSame(Weight):
    def __init__(self):
        self.space_bandwidth = 400
        self.time_bandwith = 8
        self.time_unit = _np.timedelta64(1, "W")

    def _gridsize(self, cell):
        gridsize = cell.xmax - cell.xmin
        if cell.ymax - cell.ymin != gridsize:
            raise ValueError("Expect cells to be square.")
        return gridsize

    def _cell(self, x, y, gridsize):
        return _np.rint(x / gridsize), _np.rint(y / gridsize)

    def distance(self, x1, y1, x2, y2):
        """Distance in the grid.  Diagonal distances are one, so (1,1) and (2,2) are adjacent points.
        This equates to using an \ell^\infty norm"""
        return max(abs(x1 - x2), abs(y1 - y2))

    def weight(self, cell, time_into_past, x, y):
        time_delta = _np.floor(time_into_past / self.time_unit + 0.0001) + 1
        if time_delta >= self.time_bandwith:
            return 0
        
        gridsize = self._gridsize(cell)
        cellx, celly = self._cell((cell.xmin + cell.xmax) / 2, (cell.ymin + cell.ymax) / 2)
        gx, gy = self._cell(x, y, gridsize)
        space_delta = self._distance(cellx, celly, gx, gy) + 1
        space_cutoff = _np.rint(self.space_bandwidth / gridsize)
        if space_delta >= space_cutoff:
            return 0

        return 1 / (space_delta * time_delta)


class ClassicDiagonalsDifferent(ClassicDiagonalsSame):
    def distance(self, x1, y1, x2, y2):
        """Distance in the grid.  Now diagonal distances are two, so (1,1) and (2,2) are two grid
        cells apart.  This equates to using an \ell^1 norm."""
        return abs(x1 - x2) + abs(y1 - y2)


class ProspectiveHotSpot(predictors.Predictor):
    def __init__(self, region):
        self.grid = 50
        self.region = region
        self.weight = ClassicDiagonalsSame()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, data.TimedPoints):
            raise TypeError("data should be of class TimedPoints")
        self._data = value

    def predict(self, cutoff_time, predict_time):
        if not cutoff_time <= predict_time:
            raise ValueError("Data cutoff point should be before prediction time")
        mask = self._data.timestamps <= cutoff_time
        time_deltas = _np.datetime64(predict_time) - self._data.timestamps[mask]
        xcoords = self._data.coords[mask][:,0]
        ycoords = self._data.coords[mask][:,1]
        width = int(_np.rint((self.region.xmax - self.region.xmin) / self.grid))
        height = int(_np.rint((self.region.ymax - self.region.ymin) / self.grid))
        matrix = _np.empty((height, width))
        cell_outline = data.RectangluarRegion(0, self.grid, 0, self.grid)
        for x in range(width):
            for y in range(height):
                cell = cell_outline + self.region.min
                data_gen = ( self.weight.weight(cell, time_deltas[i], xcoords[i], ycoords[i])
                    for i in range(len(time_deltas)) )
                matrix[y][x] = sum(data_gen)
        return predictors.GridPredictionArray(self.grid, self.grid, matrix, self.region.xmin, self.region.ymin)