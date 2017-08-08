"""
evaluation
~~~~~~~~~~

Contains routines and classes to help with evaluation of predictions.
"""

import numpy as _np
import collections as _collections
from . import naive as _naive
from . import predictors as _predictors
from . import network as _network
from . import geometry as _geometry

def _top_slice_one_dim(risk, fraction):
    data = risk.compressed().copy()
    data.sort()
    N = len(data)
    n = int(_np.floor(N * fraction))
    n = min(max(0, n), N)
    if n == N:
        ret = _np.zeros(risk.shape) + 1
        return (ret * (~risk.mask)).astype(_np.bool)
    if n == 0:
        return _np.zeros(risk.shape, dtype=_np.bool)
    mask = (risk >= data[-n])
    mask = mask.data & (~risk.mask)
    have = _np.sum(mask)
    if have == n:
        return mask
    
    top = _np.ma.min(_np.ma.masked_where(~mask, risk))
    for i in range(len(risk)):
        if risk[i] == top:
            mask[i] = False
            have -= 1
            if have == n:
                return mask
    raise Exception("Failed to sufficient cells")
    
def top_slice(risk, fraction):
    """Returns a boolean array of the same shape as `risk` where there are
    exactly `n` True entries.  If `risk` has `N` entries, `n` is the greatest
    integer less than or equal to `N * fraction`.  The returned cells are True
    for the `n` greatest cells in `risk`.  If there are ties, then returns the
    first (in the natual ordering) cells.

    The input array may be a "masked array" (see `numpy.ma`), in which case
    only the "valid" entries will be used in the computation.  The output is
    always a normal boolean array, where all invalid entries will not be
    selected.  For example, if half of the input array is masked, and
    `fraction==0.5`, then the returned array will have 1/4 of its entries as
    True.
    
    :param risk: Array of values.
    :param fraction: Between 0 and 1.

    :return: A boolean array, of the same shape as `risk`, where True indicates
      that cell is in the slice.
    """
    risk = _np.ma.asarray(risk)
    if len(risk.shape) == 1:
        return _top_slice_one_dim(risk, fraction)
    mask = _top_slice_one_dim(risk.ravel(), fraction)
    return _np.reshape(mask, risk.shape)

def grid_risk_to_graph(grid_pred, graph, percentage_coverage, intersection_cutoff=None):
    """Find the given coverage for the grid prediction, and then intersect with
    the graph.
    
    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.
    :param graph: An instance of :class:`network.PlanarGraph`
    :param percentage_coverage: An iterable of percentage coverages to test.
    :param intersection_cutoff: If `None` then return any edge in the graph
      which intersects a grid cell.  Otherwise a value between 0 and 1
      specifying the minimum intersection amount (based on length).
    
    :return: A new graph with only those edges which intersect.
    """
    builder = _network.PlanarGraphBuilder()
    builder.vertices.update(graph.vertices)
    covered = top_slice(grid_pred.intensity_matrix, percentage_coverage / 100)
    for gy in range(covered.shape[0]):
        for gx in range(covered.shape[1]):
            if covered[gy][gx]:
                _add_intersections(grid_pred.bounding_box_of_cell(gx, gy),
                                   graph, builder, intersection_cutoff)
    builder.remove_unused_vertices()
    return builder.build()

def _add_intersections(bbox, graph, builder, intersection_cutoff):
    bbox = (*bbox.min, *bbox.max)
    for edge in graph.edges:
        start, end = graph.vertices[edge[0]], graph.vertices[edge[1]]
        tt = _geometry.intersect_line_box(start, end, bbox)
        if tt is not None:
            if intersection_cutoff is None or tt[1] - tt[0] >= intersection_cutoff:
                builder.edges.append(edge)

def network_hit_rate(graph, timed_network_points, source_graph=None):
    """Computes the "hit rate" for the given prediction for the passed
    collection of events.  We compute the fraction of events which fall in the
    graph.

    :param graph: An instance of :class:`network.PlanarGraph` describing the
      valid edges.
    :param timed_network_points: An instance of :class:`TimedNetworkPoints`
      to get events from.  We assume that the vertex keys used are the same
      as in `graph`.
    :param source_graph: If not `None` then this is assumed to be the orignal
      graph associated with `timed_network_points` and we perform a check to
      see that the vertex keys agree.

    :return: The hit rate, a value between 0 and 1.
      If there were no events in the `timed_points`, we return -1.
    """
    if len(timed_network_points.distances) == 0:
        return -1
    if source_graph is not None:
        keys = set(timed_network_points.start_keys)
        keys.update(timed_network_points.end_keys)
        for key in keys:
            if key not in graph.vertices:
                continue
            x, y = graph.vertices[key]
            xx, yy = source_graph.vertices[key]
            if ((x-xx)**2 + (y-yy)**2) > 1e-10:
                raise ValueError("Graphs appear to differ")
    
    edges = set(graph.edges)
    edges.update((b,a) for a,b in graph.edges)
    hits = 0
    for start, end in zip(timed_network_points.start_keys, timed_network_points.end_keys):
        if (start, end) in edges:
            hits += 1
    return hits / len(timed_network_points.distances)

def hit_rates(grid_pred, timed_points, percentage_coverage):
    """Computes the "hit rate" for the given prediction for the passed
    collection of events.  For each percent, we top slice that percentage of
    cells from the grid prediction, and compute the fraction of events which
    fall in those cells.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.
    :param percentage_coverage: An iterable of percentage coverages to test.

    :return: A dictionary from percentage coverage to hit rate percentage.
      If there were no events in the `timed_points`, we return -1.
    """
    if len(timed_points.xcoords) == 0:
        return {cov : -1.0 for cov in percentage_coverage}
    risk = grid_pred.intensity_matrix
    out = dict()
    for coverage in percentage_coverage:
        covered = top_slice(risk, coverage / 100)
        
        gx, gy = grid_pred.grid_coord(timed_points.xcoords, timed_points.ycoords)
        gx, gy = gx.astype(_np.int), gy.astype(_np.int)
        mask = (gx < 0) | (gx >= covered.shape[1]) | (gy < 0) | (gy >= covered.shape[0])
        gx, gy = gx[~mask], gy[~mask]
        count = _np.sum(covered[(gy,gx)])

        out[coverage] = count / len(timed_points.xcoords)
    return out

def maximum_hit_rate(grid, timed_points, percentage_coverage):
    """For the given collection of points, and given percentage coverages,
    compute the maximum possible hit rate: that is, if the coverage gives `n`
    grid cells, find the `n` cells with the most events in, and report the
    percentage of all events this is.

    :param grid: A :class:`BoundedGrid` defining the grid to use.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.
    :param percentage_coverage: An iterable of percentage coverages to test.

    :return: A dictionary from percentage coverage to hit rate percentage.
    """
    pred = _naive.CountingGridKernel(grid.xsize, grid.ysize, grid.region())
    pred.data = timed_points
    risk = pred.predict()
    try:
        risk.mask_with(grid)
    except:
        # Oh well, couldn't do that
        pass
    return hit_rates(risk, timed_points, percentage_coverage)


class PredictionProvider():
    def predict(self, time):
        """Produce a prediction at this time."""
        raise NotImplementedError()


HitRateDetail = _collections.namedtuple("HitRateDetail",
    ["total_cell_count", "prediction"])


class HitRateResult():
    def __init__(self, rates, details):
        self._rates = rates
        self._details = details
        
    @property
    def rates(self):
        """Dictionary from `start` to a dictionary from "coverage
        percentage level" to "fractional hit rate".
        """
        return self._rates
    
    @property
    def details(self):
        return self._details


class HitRateEvaluator(_predictors.DataTrainer):
    """Abstracts the task of running a "trainer" and/or "predictor" over a set
    of data, producing a prediction, and then comparing this prediction against
    reality at various coverage levels, and then repeating for all dates in a
    range.
    """
    def __init__(self, provider):
        self._provider = provider
        
    def _points(self, start, end):
        mask = (self.data.timestamps >= start) & (self.data.timestamps < end)
        return self.data[mask]
        
    @staticmethod
    def time_range(start, end, length):
        """Helper method to generate an iterable of (start, end)
        
        :param start: Start time
        :param end: End time, inclusive
        :param length: Length of time for interval
        """
        s = start
        while s <= end:
            yield (s, s + length)
            s += length
    
    def run(self, times, coverage_levels):
        """Run tests.
        
        :param times: Iterable of (start, end) times.  A prediction will be
          made for the time `start` and then evaluated across the range `start`
          to `end`.
        :param coverage_levels: Iterable of *percentage* coverage levels to
          test the hit rate for.
          
        :return: Instance of :class:`HitRateResult`
        """
        coverage_levels = list(coverage_levels)
        details = dict()
        out = dict()
        for start, end in times:
            pred = self._provider.predict(start)
            points = self._points(start, end)
            if points.number_data_points == 0:
                continue
            out[start] = hit_rates(pred, points, coverage_levels)
            details[start] = HitRateDetail(
                total_cell_count=_np.ma.sum(~pred.intensity_matrix.mask),
                prediction = pred
                )
        return HitRateResult(out, details)
