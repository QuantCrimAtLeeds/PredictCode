"""
predictors
~~~~~~~~~~

Contains base classes and utility functions for classes which make predictions,
and classes which encapsulate a given prediction.


"""

import numpy as _np
from . import data

class DataTrainer():
    """Base class for most "trainers": classes which take data and "train"
    themselves (fit a statistical model, etc.) to the data.  Can also be used
    as a base for classes which can directly return a "prediction".
    """
    @property
    def data(self):
        """An instance of :class:`TimedPoints` giving the data to be trained
        on.
        """
        return self._data

    @data.setter
    def data(self, value):
        if value is not None and not isinstance(value, data.TimedPoints):
            raise TypeError("data should be of class TimedPoints")
        self._data = value
        

class GridPrediction(data.BoundedGrid):
    """A prediction based on a grid.  The risk is always computed by finding
    the grid cell the coordinates contained, and then deferring to the abstract
    `grid_risk` method.  Notice also that the "extent" of the prediction is not
    (meaningfully) defined.
    
    :param xsize: The width of each grid cell.
    :param ysize: The height of each grid cell.
    :param xoffset: How much to offset the input x coordinate by; default 0.
    :param yoffset: How much to offset the input y coordinate by; default 0.
    """
    
    def __init__(self, xsize, ysize, xoffset = 0, yoffset = 0):
        super().__init__(xsize, ysize, xoffset, yoffset)
    
    def risk(self, x, y):
        """The risk at coordinate `(x,y)`."""
        return self.grid_risk(*self.grid_coord(x, y))

    def grid_risk(self, gridx, gridy):
        raise NotImplementedError()

    @property
    def xextent(self):
        return 0

    @property
    def yextent(self):
        return 0

    def is_valid(self, gx, gy):
        """Is the grid cell included in the possibly masked grid?  If False
        then this cell should be ignored for computations.  Is *not*
        guaranteed to return False merely because the grid coordinates are out
        of range of the "extent".
        """
        return True

    @property
    def intensity_matrix(self):
        """Generate, or get, a matrix representing the risk.  May be
        implemented by a lookup, or may repeatedly call :method:`grid_risk`.
        """
        intensity = _np.empty((self.yextent, self.xextent))
        mask = _np.empty((self.yextent, self.xextent), dtype=_np.bool)
        for y in range(self.yextent):
            for x in range(self.xextent):
                intensity[y][x] = self.grid_risk(x, y)
                mask[y][x] = not self.is_valid(x, y)
        if not _np.any(mask):
            return intensity
        return _np.ma.masked_array(intensity, mask)

    def __repr__(self):
        return "GridPrediction(offset=({},{}), size={}x{})".format(self.xoffset,
                self.yoffset, self.xsize, self.ysize)


class GridPredictionArray(GridPrediction):
    """A :class:`GridPrediction` backed by a numpy array (or other
    two-dimensional list-like object).

    :param xsize: The width of each grid cell.
    :param ysize: The height of each grid cell.
    :param matrix: A two dimensional numpy array (or other object with a
      `shape` attribute and allowing indexing as `matrix[y][x]`).
    :param xoffset: How much to offset the input x coordinate by; default 0.
    :param yoffset: How much to offset the input y coordinate by; default 0.
    """
    def __init__(self, xsize, ysize, matrix, xoffset = 0, yoffset = 0):
        super().__init__(xsize, ysize, xoffset, yoffset)
        self._matrix = matrix

    def clone(self):
        """Return a new instance with the same data."""
        matrix = _np.ma.masked_array(self._matrix, copy=True)
        copy = GridPredictionArray(xsize=self.xsize, ysize=self.ysize, xoffset=self.xoffset,
            yoffset=self.yoffset, matrix=matrix)
        return copy
    
    def new_extent(self, xoffset, yoffset, xextent, yextent):
        """Return a new instance with a different offset and x/y extent.  The
        intensity matrix is clipped appropriated, and new cells will have
        intensity zero.  You may change the x and y offsets, but only by
        multiples of the grid size.  Does _not_ preserve any mask.
        """
        if (xoffset - self.xoffset) % self.xsize != 0:
            raise ValueError("Must change x offset by multiple of x size")
        if (yoffset - self.yoffset) % self.ysize != 0:
            raise ValueError("Must change y offset by multiple of y size")
        xmove = (xoffset - self.xoffset) // self.xsize
        ymove = (yoffset - self.yoffset) // self.ysize
        xlookup = _np.arange(xextent) + xmove
        xlookup = _np.broadcast_to(xlookup[None,:], (yextent, xextent))
        ylookup = _np.arange(yextent) + ymove
        ylookup = _np.broadcast_to(ylookup[:,None], (yextent, xextent))
        mask = ~((xlookup < 0) | (xlookup >= self.xextent) | (ylookup < 0) | (ylookup >= self.yextent))
        new_matrix = self.intensity_matrix[ylookup * mask, xlookup * mask] * mask
        return GridPredictionArray(xsize=self.xsize, ysize=self.ysize,
            xoffset=xoffset, yoffset=yoffset, matrix=new_matrix)

    def zero_to_constant(self):
        """Adjust in place the intensity matrix to be constant (non-zero!) if
        it is currently constantly zero."""
        norm = _np.ma.sum(self._matrix)
        if norm <= 0:
            try:
                mat = _np.ma.zeros(self._matrix.shape) + 1
                mat.mask = _np.array(self._matrix.mask)
            except AttributeError:
                mat = _np.zeros(self._matrix.shape) + 1
            self._matrix = mat

    def renormalise(self):
        """Return a new instance with the intensity matrix scaled so that it
        sums to 1, taking account of any mask."""
        norm = _np.ma.sum(self._matrix)
        if norm <= 0:
            raise ValueError("Cannot renormalise a zero matrix!")
        mat = _np.ma.array(self._matrix) / norm
        return GridPredictionArray(xsize=self.xsize, ysize=self.ysize,
            xoffset=self.xoffset, yoffset=self.yoffset, matrix=mat)        

    def break_ties(self):
        """If there are (non-masked) grid cells with identical intensity
        values, then add a small amount of noise to break ties, without
        otherwise changing the _ordering_ of intensities.  Because of finite
        floating-point accuracy, this is not always possible.
        
        :return: A new instance with noise added.
        """
        x = _np.sort(self._matrix.flatten())
        x = x[1:] - x[:-1]
        if not _np.all(x > 0):
            epsilon = min(1e-9, _np.min(x[x>0]) / 100)
            mat = self._matrix + _np.random.random(size=self._matrix.shape) * 2 * epsilon - epsilon
        else:
            mat = _np.ma.array(self._matrix)
        return GridPredictionArray(xsize=self.xsize, ysize=self.ysize,
            xoffset=self.xoffset, yoffset=self.yoffset, matrix=mat)        

    def grid_risk(self, gx, gy):
        """Find the risk in a grid cell.

        :param gx: x coordinate of the cell
        :param gy: y coordinate of the cell

        :return: The risk in the cell, or 0 if the cell is outside the range
          of the data we have.
        """
        ylim, xlim = self._matrix.shape
        gx = _np.asarray(gx)
        gy = _np.asarray(gy)
        mask = (gx >= 0) & (gy >= 0) & (gx < xlim) & (gy < ylim)
        out = _np.empty(gx.shape)
        out[~mask] = 0
        out[mask] = self._matrix[gy[mask], gx[mask]]
        return out
        if gx < 0 or gy < 0 or gx >= xlim or gy >= ylim:
            return 0
        return self._matrix[gy][gx]

    @staticmethod
    def from_continuous_prediction(prediction, width, height):
        """Construct an instance from an instance of
        :class:`ContinuousPrediction` using the grid size and offset specified
        in that instance.  This is more efficient as we sample each grid cell
        once and then store the result.

        :param prediction: An instance of ContinuousPrediction to sample from
        :param width: Width of the grid, in number of cells
        :param height: Height of the grid, in number of cells
        """
        matrix = prediction.to_matrix(width, height)
        return GridPredictionArray(prediction.cell_width, prediction.cell_height,
            matrix, prediction.xoffset, prediction.yoffset)

    @staticmethod
    def from_continuous_prediction_region(prediction, region, cell_width, cell_height=None):
        """Construct an instance from an instance of
        :class:`ContinuousPrediction` using the region and passed cell sizes.

        :param prediction: An instance of :class:`ContinuousPrediction` to
          sample from
        :param region: The :class:`RectangularRegion` the grid
        :param cell_width: Width of each cell in the resulting grid
        :param cell_height: Optional; height of each cell in the resulting
          grid; defaults to `cell_width`
        """
        if cell_height is None:
            cell_height = cell_width
        width = int(_np.rint((region.xmax - region.xmin) / cell_width))
        height = int(_np.rint((region.ymax - region.ymin) / cell_height))
        newpred = prediction.rebase(cell_width, cell_height, region.xmin, region.ymin)
        return GridPredictionArray.from_continuous_prediction(newpred, width, height)

    @staticmethod
    def from_continuous_prediction_grid(prediction, grid):
        """Construct an instance from an instance of
        :class:`ContinuousPrediction` and an :class:`BoundedGrid` instance.

        :param prediction: An instance of :class:`ContinuousPrediction` to
          sample from
        :param grid: An instance of :class:`BoundedGrid` to base the grid on.
        """
        newpred = prediction.rebase(grid.xsize, grid.ysize, grid.xoffset, grid.yoffset)
        return GridPredictionArray.from_continuous_prediction(newpred, grid.xextent, grid.yextent)

    @property
    def intensity_matrix(self):
        """Get the matrix containing data which we use"""
        return self._matrix

    @property
    def xextent(self):
        return self._matrix.shape[1]

    @property
    def yextent(self):
        return self._matrix.shape[0]

    def is_valid(self, gx, gy):
        if not hasattr(self._matrix, "mask"):
            return True
        ylim, xlim = self._matrix.shape
        if gx < 0 or gy < 0 or gx >= xlim or gy >= ylim:
            return True
        return not self._matrix.mask[gy][gx]

    def mesh_data(self):
        """Returns a pair (xcoords, ycoords) which when paired with
        :meth:`intensity_matrix` is suitable for passing to `matplotlib.pcolor`
        or `pcolormesh`.  That is, `intensity_matrix[i][j]` is the risk intensity
        in the rectangular cell with diagonally opposite vertices
        `(xcoords[j], ycoords[i])`, `(xcoords[j+1], ycoords[i+1])`.
        """
        xcoords = _np.arange(self._matrix.shape[1] + 1) * self.xsize + self.xoffset
        ycoords = _np.arange(self._matrix.shape[0] + 1) * self.ysize + self.yoffset
        return (xcoords, ycoords)
    
    def percentile_matrix(self):
        """Returns a matrix of the same shape as :meth:`intensity_matrix` but
        with float values giving the percentile of risk, normalised to [0,1].
        So the cell with the highest risk is assigned 1.0.  Ties are rounded up,
        so if three cells share the highest risk, they are all assigned 1.0.
        """
        data = self._matrix.ravel().copy()
        data.sort()
        return _np.searchsorted(data, self._matrix, side="right") / len(data)

    def mask_with(self, mask):
        """Mask the intensity matrix with the given instance of
        :class:`MaskedGrid`."""
        if self.xsize != mask.xsize or self.ysize != mask.ysize:
            raise ValueError("Grid cell sizes differ")
        if self.xoffset != mask.xoffset or self.yoffset != mask.yoffset:
            raise ValueError("Grid offsets differ")
        if self.intensity_matrix.shape != mask.mask.shape:
            raise ValueError("Extent of the grids differ")
        self._matrix = mask.mask_matrix(self.intensity_matrix)

    def __repr__(self):
        return "GridPredictionArray(offset=({},{}), size={}x{}, risk intensity size={}x{})".format(
                self.xoffset, self.yoffset, self.xsize, self.ysize,
                self.xextent, self.yextent)


class ContinuousPrediction():
    """A prediction which allows the "risk" to be calculated at any point in a
    continuous fashion.  Allows monte-carlo sampling to produce a grid risk.
    
    :param cell_width: Width of cells to use in producing a grid risk.
    :param cell_height: Height of cells to use in producing a grid risk.
    :param xoffset: The x coordinate of the start of the grid.
    :param yoffset: The y coordinate of the start of the grid.
    :param samples: The number of samples to use when computing the risk in a
      grid cell.  Set to `None` to use a fixed density.  Set to a negative
      number to sample on a regular pattern.
    """
    def __init__(self, cell_width=50, cell_height=50, xoffset=0, yoffset=0, samples=None):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.samples = samples
    
    @property
    def samples(self):
        """The number of samples to use per cell.
        
        Set to `None` to use a _fixed density_, currently 1 sample per 200
        units of area.
        
        Set to a negative number, say `-x`, to sample on a subgrid of size
        `x * x` per grid cell.  This is more reproducible.
        """
        return self._samples
    
    @samples.setter
    def samples(self, v):
        self.__samples = v
        if v is None:
            # Use a fixed _density_
            v = int(self.cell_width * self.cell_height / 200)
            if v < 2:
                v = 2
        self._samples = v
    
    def grid_risk(self, gx, gy):
        """Return an estimate of the average risk in the grid cell"""
        x = (gx + _np.random.random(self.samples)) * self.cell_width + self.xoffset
        y = (gy + _np.random.random(self.samples)) * self.cell_height + self.yoffset
        return _np.mean(self.risk(x, y))

    def _risk_array(self, x, y):
        # Like `return self.risk(x,y)` but do in blocks of at most 50 to avoid
        # excessive memory usage
        assert len(x.shape) == 1
        out = _np.empty_like(x)
        offset = 0
        length = x.shape[0]
        while offset < length:
            end = min(offset + 50, length)
            xx, yy = x[offset : end], y[offset : end]
            out[offset : end] = self.risk(xx, yy)
            offset = end
        return out
    
    def to_matrix(self, width, height):
        """Sample the risk at each grid point from `(0, 0)` to
        `(width-1, height-1)` inclusive.  Optimised."""
        if self.samples < 0:
            return self._to_matrix_grid(width, height)
        matrix = _np.empty((height, width))
        for gy in range(height):
            y = (gy + _np.random.random(size=self.samples * width)) * self.cell_height + self.yoffset
            # 0,1,...,width-1, 0,1,...,width-1  with the block repeated self.sample times
            gx = _np.broadcast_to(_np.arange(width), (self.samples, width)).ravel()
            x = (gx + _np.random.random(self.samples * width)) * self.cell_width + self.xoffset
            matrix[gy] = _np.mean(_np.reshape(self._risk_array(x, y), (self.samples, width)), axis=0)
        return matrix

    def _sub_grid_mesh(self):
        s = -self.samples
        pat = (_np.arange(s) * 2 + 1) / (s + s)
        xx, yy = _np.meshgrid(pat, pat)
        return xx.ravel(), yy.ravel()
        
    def _to_matrix_grid(self, width, height):
        matrix = _np.empty((height, width))
        xx, yy = self._sub_grid_mesh()
        for gy in range(height):
            y = (yy + gy) * self.cell_height + self.yoffset
            for gx in range(width):
                x = (xx + gx) * self.cell_width + self.xoffset
                matrix[gy][gx] = _np.mean(self.risk(x, y))
        return matrix

    def to_matrix_from_masked_grid(self, masked_grid):
        """Sample the risk at each "valid" grid point from `masked_grid`.
        Takes grid geometry from `masked_grid` and not from own settings.
        Useful for when the kernel cannot be evaluated at certain points."""
        if self.samples < 0:
            return self._to_matrix_from_masked_grid_regular(masked_grid)
        locations = []
        to_cell = []
        size = [masked_grid.xsize, masked_grid.ysize]
        offset = [masked_grid.xoffset, masked_grid.yoffset]
        for gy in range(masked_grid.yextent):
            for gx in range(masked_grid.xextent):
                if masked_grid.is_valid(gx, gy):
                    to_cell.append((gx, gy))
                    parts = (_np.random.random(size=(self.samples,2)) + [gx,gy]) * size + offset
                    locations.extend(parts)
        locations = _np.asarray(locations)
        values = self._risk_array(*locations.T)
        values = _np.mean(_np.reshape(values, (values.shape[0] // self.samples, self.samples)), axis=1)
        matrix = _np.zeros((masked_grid.yextent, masked_grid.xextent))
        for v, (gx, gy) in zip(values, to_cell):
            matrix[gy, gx] = v
        return matrix

    def _to_matrix_from_masked_grid_regular(self, masked_grid):
        locations = []
        to_cell = []
        size = [masked_grid.xsize, masked_grid.ysize]
        offset = [masked_grid.xoffset, masked_grid.yoffset]
        xx, yy = self._sub_grid_mesh()
        for gy in range(masked_grid.yextent):
            for gx in range(masked_grid.xextent):
                if masked_grid.is_valid(gx, gy):
                    to_cell.append((gx, gy))
                    parts = _np.vstack([xx + gx, yy + gy]).T * size + offset
                    locations.extend(parts)
        locations = _np.asarray(locations)
        values = self._risk_array(*locations.T)
        ss = self.samples * self.samples
        values = _np.mean(_np.reshape(values, (values.shape[0] // ss, ss)), axis=1)
        matrix = _np.zeros((masked_grid.yextent, masked_grid.xextent))
        for v, (gx, gy) in zip(values, to_cell):
            matrix[gy, gx] = v
        return matrix


    def to_kernel(self):
        """Returns a callable object which when called at `point` gives the
        risk at (point[0], point[1]).  `point` may be an array."""
        def kernel(point):
            return self.risk(point[0], point[1])
        return kernel

    def rebase(self, cell_width, cell_height, xoffset, yoffset, samples=None):
        """Returns a new instance using the same risk but with a different grid
        size and offset"""
        if samples is None:
            samples = self.__samples
        instance = ContinuousPrediction(cell_width, cell_height, xoffset,
            yoffset, samples)
        # Monkey-patch a delegation
        instance.risk = self.risk
        return instance

    def risk(self, x, y):
        """Return the risk at (a) coordinate(s).

        :param x: The x coordinate to evaluate the risk at.  May be a scalar
          or a one-dimensional numpy array.
        :param y: The y coordinate to evaluate the risk at.  Should match `x`
          in being a scalar or a one-dimensional numpy array.

        :return: A scalar or numpy array as appropriate.
        """
        raise NotImplementedError()


class KernelRiskPredictor(ContinuousPrediction):
    """Wraps a kernel object so as to make a :class:`ContinuousPrediction`
    instance
    
    :param kernel: A callable object with signature `kernel(points)` where
      points may be an array of size 2, for a single point, or an array of shape
      `(2,N)` for `N` points to be computed at once.
    :param kwards: Any constructor arguments which :class:`ContinuousPrediction`
      takes.
    """
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self._kernel = kernel
    
    def risk(self, x, y):
        """The risk given by the kernel."""
        return self._kernel(_np.vstack([x,y]))


def grid_prediction_from_kernel(kernel, region, grid_size, samples=None):
    """Utility function to convert a space kernel into a grid based prediction.
    
    :param kernel: A kernel object taking an array of shape (2,N) of N lots
      of spatial coordinates, and returning an array of shape (N).
    :param region: An instance of :class RectangularRegion: giving the
      region to use.
    :param grid_size: The size of grid to use.
    :param samples: As :class:`ContinuousPrediction`
    
    :return: An instance of :class GridPredictionArray:
    """
    width, height = region.grid_size(grid_size)
    cts_predictor = KernelRiskPredictor(kernel, xoffset=region.xmin,
            yoffset=region.ymin, cell_width=grid_size, cell_height=grid_size,
            samples=samples)
    return GridPredictionArray.from_continuous_prediction(cts_predictor,
            width, height)

def grid_prediction_from_kernel_and_masked_grid(kernel, masked_grid, samples=None):
    """Utility function to convert a space kernel into a grid based prediction.
    
    :param kernel: A kernel object taking an array of shape (2,N) of N lots
      of spatial coordinates, and returning an array of shape (N).
    :param masked_grid: An instance of :class:`MaskedGrid` to both base the
      grid geometry on, and to select which grid cells to sample.
    :param samples: As :class:`ContinuousPrediction`
    
    :return: An instance of :class GridPredictionArray:
    """
    cts_predictor = KernelRiskPredictor(kernel,
        xoffset=masked_grid.xoffset, yoffset=masked_grid.yoffset,
        cell_width=masked_grid.xsize, cell_height=masked_grid.ysize,
        samples=samples)
    intensity_matrix = cts_predictor.to_matrix_from_masked_grid(masked_grid)
    pred = GridPredictionArray(masked_grid.xsize, masked_grid.ysize,
        intensity_matrix, masked_grid.xoffset, masked_grid.yoffset)
    try:
        pred.mask_with(masked_grid)
    except:
        pass
    return pred

def grid_prediction(continuous_prediction, grid):
    """Utility function to convert a continuous prediction to a grid based
    prediction.

    :param continuous_prediction: An instance of :class:`ContinuousPrediction`
      or a kernel.
    :param grid: An instance of :class:`BoundedGrid`, which may be masked.
    """
    try:
        kernel = continuous_prediction.to_kernel()
    except:
        kernel = continuous_prediction
    
    prediction = KernelRiskPredictor(kernel)
    risk = GridPredictionArray.from_continuous_prediction_grid(prediction, grid)
    
    try:
        risk.mask_with(grid)
    except:
        pass
    
    return risk