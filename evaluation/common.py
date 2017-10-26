import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import descartes
import os, datetime, collections
import open_cp.sources.chicago
import open_cp.geometry
import open_cp.plot
import open_cp.predictors
import open_cp.evaluation
import open_cp.kernels
import pandas as pd
import scipy.stats


def load_geometry(datadir):
    open_cp.sources.chicago.set_data_directory(datadir)
    return open_cp.sources.chicago.get_side("North")

def make_grid(geo):
    grid = open_cp.data.Grid(150, 150, 0, 0)
    return open_cp.geometry.mask_grid_by_intersection(geo, grid)

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


def kde_scorer(grid):
    """Use a "plugin" bandwidth estimator to form a probability surface from
    the actual events.  Then return the integral of :math:`| f(x) - p(x)|^2`.
    
    :param grid: The grid to conform the KDE to.
    
    :return: A function object with with signature `func(pred, tps)`.
    """
    def score_kde(pred, tps):
        points = np.asarray([tps.xcoords, tps.ycoords])
        if tps.number_data_points <= 2:
            raise ValueError("Need at least 3 events.")
        kernel = open_cp.kernels.GaussianEdgeCorrectGrid(points, grid)
        kde_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
        kde_pred = kde_pred.renormalise()
        return np.sum((pred.intensity_matrix - kde_pred.intensity_matrix)**2) * grid.xsize * grid.ysize
    return score_kde

def kde_scorer_fixed_bandwidth(grid, bandwidth):
    """As :func:`kde_scorer` but with a fixed bandwidth.
    
    :param grid: The grid to conform the KDE to.
    :param bandwidth: The bandwith to use.
    
    :return: A function object with with signature `func(pred, tps)`.
    """
    def score_kde(pred, tps):
        points = np.asarray([tps.xcoords, tps.ycoords])
        if tps.number_data_points <= 2:
            raise ValueError("Need at least 3 events.")
        kernel = open_cp.kernels.GaussianEdgeCorrectGrid(points, grid)
        kernel.covariance_matrix = [[1,0],[0,1]]
        kernel.bandwidth = bandwidth
        kde_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
        kde_pred = kde_pred.renormalise()
        return np.sum((pred.intensity_matrix - kde_pred.intensity_matrix)**2) * grid.xsize * grid.ysize
    return score_kde


def multiscale_brier(pred, tps):
    """Score the prediction using multiscale Brier."""
    maxsize = min(pred.xextent, pred.yextent)
    return {s : open_cp.evaluation.multiscale_brier_score(pred, tps,s) for s in range(1, maxsize+1)}

def multiscale_kl(pred, tps):
    """Score the prediction using multiscale Kullback-Leibler."""
    maxsize = min(pred.xextent, pred.yextent)
    return {s : open_cp.evaluation.multiscale_kl_score(pred, tps,s) for s in range(1, maxsize+1)}

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

def generate_data_preds(grid, num_trials=1000, base_intensity=10):
    """Yield triples `(key, pred, tps)` where `key` is a pair
    `(source_model_name, pred_model_name)`; `pred` is the prediction as a
    :class:`GridPredictionArray` instance; `tps` is a :class:`TimedPoints`
    instance.

    :param grid: The masked grid to base everything on.
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

def make_data_preds(grid, num_trials=1000, base_intensity=10):
    """Make the data in one go.

    :param grid: The masked grid to base everything on.
    :param num_trials: The number of trials to run.  Will not yield empty point
      collections, so may return fewer than this many tuples.
    :param base_intensity: Each trial has `n` events where `n` is distributed
      as a Poisson with mean `base_intensity`.

    :return: A dictionary from `key` to a list of pairs `(pred, tps)`.  Here
      `key` is the pair `(source_model_name, pred_model_name)`, `pred` is the
      prediction as a :class:`GridPredictionArray` instance, and `tps` is a
      :class:`TimedPoints` instance.
    """
    predictions = collections.defaultdict(list)
    for key, pred, tps in generate_data_preds(grid, num_trials):
        predictions[key].append((pred, tps))
    return predictions

def process(all_data, func):
    """`func` should be a function object with signature `func(pred, tps)`, and
    should return some object.

    :param all_data: The output of :func:`make_data_preds`.
    
    :return: A dictionary from `key` to `object` where each key will be `(k1,
      k2, i)` with `k1,k2` as before, and `i` a counter.  `object` is the
      return value of func.
    """
    return { (key) + (i,) : func(pred, tps) for key in all_data
            for i, (pred, tps) in enumerate(all_data[key]) }

def constrain_to_number_events(all_data, minimum_event_count):
    """Remove trials with too few events.

    :param all_data: The output of :func:`make_data_preds`; or data conforming
      to this dictionary style.
    :param minimum_event_count: The minimum number of events we want in a
      trial.
      
    :return: The same format of data, but each "actual occurrance" will have at
      least `minimum_event_count` events.
    """
    return {k:[(g,tps) for g,tps in v if tps.number_data_points >= minimum_event_count]
               for k,v in all_data.items()}

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

def data_by_source_model(data):
    """Assuming `data` is as in the format returned from :func:`process`,
    yield data in the same format, for from source model 1, then model 2,
    then model 3 (with the key now being `(prediction_model, trial_number)`).
    """
    for key in ["Model1", "Model2", "Model3"]:
        yield {k[1:] : v for k, v in data.items() if k[0] == key}
        
PairedItem = collections.namedtuple("PairedItem", "key one two")
        
def paired_data(data_for_one_source):
    """Assuming `data_for_one_source` is as returned by
    :func:`data_by_source_model`, yield a list of :class:`PairedItem` objects,
    in the order of "models 1/2 compared" then "1/3" and then "2/3".
    """
    for k1, k2 in [(1,2), (1,3), (2,3)]:
        k1 = "Model{}".format(k1)
        k2 = "Model{}".format(k2)
        keys = {k[1:] for k in data_for_one_source if k[0] == k1}
        keys.intersection_update({k[1:] for k in data_for_one_source if k[0] == k2})
        keys = list(keys)
        keys.sort()
        yield [PairedItem(k, data_for_one_source[(k1,*k)], data_for_one_source[(k2,*k)]) for k in keys]

def plot_paired_data(data, plot_func):
    """Plot a 3x3 array of plots, each row being data from one source model,
    and the columns being "models 1/2 compared" then "1/3" and then "2/3".
    
    :param data: As before, return of :func:`process`.
    :param plot_func:` Should have signature `plot_func(ax, paired_data)`
      where `paired_data` is a list of :class:`PairedItem` objects.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,12))
    models = ["Model1", "Model2", "Model3"]
    comparisons = [(1,2), (1,3), (2,3)]
    for axe, source_name, row_data in zip(axes, models, data_by_source_model(data)):
        for ax, key, d in zip(axe, comparisons, paired_data(row_data)):
            plot_func(ax, d)
            ax.set_title("{} -> {} vs {}".format(source_name, *key))
    fig.tight_layout()
    return fig, axes

def comparison_uni_paired(lower_bound, upper_bound):
    """Returns a `plot_func` which is suitable for passing to
    `plot_paired_data`.  Assumes that the "objects" returned from
    :func:`process` are just numbers, and plots the difference between "one"
    and "two" in the :class:`PairedItem` instance.  Plots an estimate of the
    cumulative density.
    
    :param lower_bound:
    :param upper_bound: `x` axis range.
    """
    def plot_func(ax, data):
        diff = np.asarray([d.one - d.two for d in data])
    
        y = np.sum(diff <= 0) / len(diff)
        lc = matplotlib.collections.LineCollection([[[lower_bound,y],[0,y]], [[0,0],[0,y]]], color="black", linewidth=1)
        ax.add_collection(lc)
    
        x = np.linspace(lower_bound, upper_bound, 100)
        cumulative = []
        for cutoff in x:
            cumulative.append( np.sum(diff <= cutoff) )
        ax.plot(x, np.asarray(cumulative) / len(diff))
    return plot_func

def scatter_uni_paired_plot_func(ax, data):
    """A `plot_func` which is suitable for passing to `plot_paired_data`.
    Assumes that the "objects" returned from :func:`process` are just numbers,
    and plots a scatter diagram."""
    x = [d.one for d in data]
    y = [d.two for d in data]
    ax.scatter(x, y)
    start = min(min(x), min(y))
    end = max(max(x), max(y))
    d = (end - start) * 0.1
    ax.plot([start-d, end+d], [start-d, end+d], color="red", linewidth=1)

def label_scatter_uni_paired(fig, axes):
    """Suitable for calling after using :func:`scatter_uni_paired_plot_func`.
    Sets labels on each axis, and tightly lays out the figure."""
    for axes_row in axes:
        for ax, (x,y) in zip(axes_row, [(1,2), (1,3), (2,3)]):
            ax.set(xlabel="Model{}".format(x), ylabel="Model{}".format(y))
    fig.tight_layout()


#############################################################################
# More tentative, data vis stuff.
#############################################################################

def hitrate_inverse_to_hitrate(inv_dict, coverages):
    """Convert the "inverse hitrate dictionary" to a more traditional
    lookup, using `coverages`."""
    out = dict()
    for cov in coverages:
        choices = [k for k,v in inv_dict.items() if v <= cov]
        out[cov] = 0 if len(choices) == 0 else max(choices)
    return out

def plot_hit_rate(data):
    """Draws a 3x3 grid; plots the mean and 25%, 75% percentiles of coverage
    against hit rate."""
    coverages = list(range(0,102,2))

    def to_by_coverage(result):
        by_cov = {cov : [] for cov in coverages}
        for row in result:
            out = hitrate_inverse_to_hitrate(row, coverages)
            for c in out:
                by_cov[c].append(out[c])
        return by_cov

    def plot_func(result, ax, key):
        frame = pd.DataFrame(to_by_coverage(result)).describe().T
        ax.plot(frame["mean"], label="mean")
        ax.plot(frame["25%"], label="25% percentile")
        ax.plot(frame["75%"], label="75% percentile")
        ax.set(xlabel="Coverage (%)", ylabel="Hit rate (%)")
        ax.set(xlim=[-5,105], ylim=[-5,105])
        ax.legend()
    
    plot_models(data, plot_func)

def plot_likelihood(data, grid):
    """Draws a 3x3 grid; plots an estimated distribution.  For model1, we have
    a delta function, which is drawn as a red line on the other plots for
    comparison."""
    def plot_func(result, ax, key):
        kernel = scipy.stats.kde.gaussian_kde(result)
        xx = -np.log(np.sum(~grid.mask))
        if key[1] == "Model1":
            d = 0.0002
            x = np.linspace(xx - d, xx + d, 100)
        else:
            x = np.linspace(-9, -6, 100)
        ax.plot(x, kernel(x))
        line = [[xx,0], [xx,100000]]
        ax.add_collection(matplotlib.collections.LineCollection([line], color="red"))
        
    plot_models(data, plot_func)
    
