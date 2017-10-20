import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import open_cp.sources.chicago


def load_geometry(datadir):
    open_cp.sources.chicago.set_data_directory(os.path.join("/media", "disk", "Data"))
    return open_cp.sources.chicago.get_side("North")

def make_grid(chicago):
    grid = open_cp.data.Grid(150, 150, 0, 0)
    return open_cp.geometry.mask_grid_by_intersection(northside, grid)

def load(datadir):
    """Load and compute initial data.

    :return: `(geometry, grid)`
    """
    chicago = load_geometry(datadir)
    return chicago, make_grid(chicago)


class BaseModel():
    def __init__(self, grid):
        self._grid = grid
    
    def prob(self, x, y):
        want, pt = self._in_grid(x, y)
        if want:
            return self._probs[pt[1], pt[0]]
        return 0.0
        
    def max_prob(self):
        return np.max(np.ma.array(self._probs, mask=self._grid.mask))
        
    @property
    def grid(self):
        return self._grid
        
    def to_grid_coords(self, x, y):
        pt = np.asarray([x,y]) - [self.grid.xoffset, self.grid.yoffset]
        pt = np.floor_divide(pt, [self.grid.xsize, self.grid.ysize]).astype(np.int)
        return pt

    def in_grid(self, x, y):
        pt = self.to_grid_coords(x, y)
        if np.all((pt >= [0, 0]) & (pt < [self.grid.xextent, self.grid.yextent])):
            return self.grid.is_valid(*pt)
        return False
    
    def to_prediction(self):
        mat = np.array(self._probs)
        pred = open_cp.predictors.GridPredictionArray(self.grid.xsize, self.grid.ysize,
                                        mat, self.grid.xoffset, self.grid.yoffset)
        pred.mask_with(self.grid)
        return pred
    
    def map_to_grid(self, x, y):
        """Map a point in :math:`[0,1]^2` to the bounding box of the grid"""
        pt = np.asarray([x,y])
        pt = pt * [self._grid.xsize * self._grid.xextent, self._grid.ysize * self._grid.yextent]
        return pt + [self._grid.xoffset, self._grid.yoffset]
    
    def _in_grid(self, x, y):
        pt = self.to_grid_coords(x, y)
        if np.all((pt >= [0, 0]) & (pt < [self.grid.xextent, self.grid.yextent])):
            return self.grid.is_valid(*pt), pt
        return False, pt
    
    def _set_probs(self, probs):
        probs = self.grid.mask_matrix(probs)
        self._probs = probs / np.sum(probs)


class Model1(BaseModel):
    """Homogeneous poisson process."""
    def __init__(self, grid):
        super().__init__(grid)
        probs = np.zeros((grid.yextent, grid.xextent)) + 1
        self._set_probs(probs)
        
    def to_randomised_prediction(self):
        mat = np.array(self._probs)
        mat += np.random.random(mat.shape) * 1e-7
        pred = open_cp.predictors.GridPredictionArray(self.grid.xsize, self.grid.ysize,
                                        mat, self.grid.xoffset, self.grid.yoffset)
        pred.mask_with(self.grid)
        pred = pred.renormalise()
        return pred
    
    def __str__(self):
        return "Model1"


class Model2(BaseModel):
    """Inhomogeneous Poisson process, linearly increases left to right across
    the grid."""
    def __init__(self, grid):
        super().__init__(grid)
        probs = np.linspace(0, 1, grid.xextent)
        probs = probs[None,:] + np.zeros(grid.yextent)[:,None]
        self._set_probs(probs)
        
    def __str__(self):
        return "Model2"


class Model3(BaseModel):
    """Inhomogeneous Poisson process, on `[0,1]^2` has intensity

      :math:`2 \exp(-20(x-y)^2) + y`

    then spread out to the grid.
    """
    def __init__(self, grid):
        super().__init__(grid)
        probs = np.empty((grid.yextent, grid.xextent))
        for x in range(grid.xextent):
            for y in range(grid.yextent):
                probs[y,x] = self.cts_prob((x+0.5) / grid.xextent, (y+0.5) / grid.yextent)
        self._set_probs(probs)
        
    def cts_prob(self, x, y):
        return 2 * np.exp(-(x-y)**2 * 20) + y
    
    def __str__(self):
        return "Model3"


def sample(model, size):
    """Return points from the model.

    :param model: The model to use to sample from.
    :param size: Number of points to return.

    :return: Array of shape `(size, 2)`.
    """
    out = []
    renorm = model.max_prob()
    while len(out) < size:
        pt = model.map_to_grid(*np.random.random(2))
        if model.in_grid(*pt):
            if model.prob(*pt) > np.random.random() * renorm:
                out.append(pt)
    return np.asarray(out)

def sample_to_timed_points(model, size):
    """As :func:`sample` but return in :class:`open_cp.data.TimedPoints`.
    """
    t = [datetime.datetime(2017,1,1)] * size
    pts = sample(model, size)
    assert pts.shape == (size, 2)
    return open_cp.data.TimedPoints.from_coords(t, *pts.T)

def generate_data_preds(num_trials=1000, base_intensity=10):
    """Yield triples `(key, pred, tps)` where `key` is a pair
    `(source_model_name, pred_model_name)`; `pred` is the prediction as a
    :class:`GridPredictionArray` instance; `tps` is a :class:`TimedPoints`
    instance.

    :param num_trials: The number of trials to run.  Will not yield empty point
      collections, so may return fewer than this many tuples.
    :param base_intensity: Each trial has `n` events where `n` is distributed
      as a Poisson with mean `base_intensity`.
    """
    for SourceModel in [Model1, Model2, Model3]:
        source_model = SourceModel(grid)
        for PredModel in [Model1, Model2, Model3]:
            pred_model = PredModel(grid)
            key = (str(source_model), str(pred_model))
            pred = pred_model.to_prediction()
            for trial in range(num_trials):
                num_pts = np.random.poisson(base_intensity)
                if num_pts == 0:
                    continue
                tps = sample_to_timed_points(source_model, num_pts)
                try:
                    pred = pred_model.to_randomised_prediction()
                except:
                    pass
                yield key, pred, tps

def plot_models(data, func):
    """Helper method to plot a 3 by 3 grid of results.

    :param data: A dictionary from `key` to object.  Each `key` is a tuple
      `(k1, k2, *)` where `k1` is the source name (which model generated the
      points) and `k2` is the prediction name (which model generated the
      prediction).
    :param func: A function object with signature  `func(result, ax, key)`
      where `result` is a list of the objects passed in `data` which have
      appropriate `k1` and `k2`; `ax` is the `matplotlib` `axis` object to
      plot to; `key` is `(k1,k2)`.  This function should process the object
      appropriately, and draw to the axis.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,12))
    labels = ["Model1", "Model2", "Model3"]
    for axe, k1 in zip(axes, labels):
        for ax, k2 in zip(axe, labels):
            ax.set_title("{}/{}".format(k1,k2))
            result = [data[k] for k in data if k[:2] == (k1, k2)]
            func(result, ax, (k1, k2))
            None

    fig.tight_layout()
    
def plot_three(data, func):
    """As :func:`plot_models`, but will pass a _dictionary_ `results` as the
    first argument, where this is map from `k2` (the prediction model) to a 
    list of objects.  Allows plotting all predictions (for one given
    generation model) on a single axis object.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))
    labels = ["Model1", "Model2", "Model3"]
    for ax, k1 in zip(axes, labels):
        ax.set_title("Data from {}".format(k1))
        results = {k2 : [data[k] for k in data if k[:2] == (k1, k2)] for k2 in labels}
        func(results, ax, k1)

    fig.tight_layout()
