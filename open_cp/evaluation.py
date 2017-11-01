"""
evaluation
~~~~~~~~~~

Contains routines and classes to help with evaluation of predictions.
"""

import numpy as _np
import scipy.special as _special
import collections as _collections
import logging as _logging
from . import naive as _naive
from . import predictors as _predictors
from . import network as _network
from . import geometry as _geometry
from . import data as _data

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

def top_slice_prediction(prediction, fraction):
    """As :func:`top_slice`.  Returns a new grid based prediction masked with
    just the selected coverage.
    
    :param prediction: Grid based prediction.
    :param fraction: Between 0 and 1.

    :return: A new grid based prediction.
    """
    covered = top_slice(prediction.intensity_matrix, fraction)
    hotspots = _data.MaskedGrid.from_grid(prediction, ~covered)
    grid_pred = prediction.clone()
    grid_pred.mask_with(hotspots)
    return grid_pred

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
        pass
    return hit_rates(risk, timed_points, percentage_coverage)

def inverse_hit_rates(grid_pred, timed_points):
    """For the given prediction and the coordinates of events, find the
    coverage level needed to achieve every possible hit-rate.  One problem is
    how to break ties: that is, what if the prediction assigns the same
    probability to multiple cells.  At present, we take a "maximal" approach,
    and round coverage levels up to include all cells of the same probability.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.

    :return: A dictionary from hit rates to minimal required coverage levels.
      Or empty dictionary if `timed_points` is empty.
    """
    if len(timed_points.xcoords) == 0:
        return dict()
    risk = grid_pred.intensity_matrix

    gx, gy = grid_pred.grid_coord(timed_points.xcoords, timed_points.ycoords)
    gx, gy = gx.astype(_np.int), gy.astype(_np.int)
    mask = (gx < 0) | (gx >= risk.shape[1]) | (gy < 0) | (gy >= risk.shape[0])
    gx, gy = gx[~mask], gy[~mask]
    
    indices = []
    for x, y in zip(gx, gy):
        indices.append( x + y * risk.shape[1] )
    indices = _np.asarray(indices)
    risk = risk.flatten()
    mask = _np.ma.getmaskarray(risk)
    indices = indices[~mask[indices]]
    
    reorder = _np.argsort(-risk)
    risk = risk[reorder] # Decreasing with mask values at end
    reordered_indices = []
    for newi, oldi in enumerate(reorder):
        reordered_indices.extend([newi] * _np.sum(indices == oldi))
    reordered_indices = _np.asarray(reordered_indices)
    risk = _np.asarray(risk[~mask])

    total_points = len(timed_points.xcoords)
    out = dict()
    index = 0
    while index < risk.shape[0]:
        while index < risk.shape[0] - 1 and risk[index] == risk[index+1]:
            index += 1
        coverage = 100 * (index + 1) / risk.shape[0]
        hitrate = 100 * _np.sum(reordered_indices <= index) / total_points
        if hitrate > 0 and hitrate not in out:
            out[hitrate] = coverage
        index += 1
    return out

def _timed_points_to_grid(grid_pred, timed_points):
    risk = grid_pred.intensity_matrix

    gx, gy = grid_pred.grid_coord(timed_points.xcoords, timed_points.ycoords)
    gx, gy = gx.astype(_np.int), gy.astype(_np.int)
    mask = (gx < 0) | (gx >= risk.shape[1]) | (gy < 0) | (gy >= risk.shape[0])
    if _np.any(mask):
        raise ValueError("All points need to be inside the grid.")
    mask = _np.ma.getmaskarray(risk)
    if _np.any(mask[gy,gx]):
        raise ValueError("All points need to be inside the non-masked area of the grid.")

    return gx, gy

def likelihood(grid_pred, timed_points, minimum=1e-9):
    """Compute the normalised log likelihood,
    
    :math:`\frac{1}{N} \sum_{i=1}^N \log f(x_i)`
    
    where the prediction gives the probability density function :math:`f`.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.
    :param minimum: Adjust 0 probabilities to this value, defaults to `1e-9`

    :return: A number, the average log likelihood.
    """
    if len(timed_points.xcoords) == 0:
        return 0.0
    gx, gy = _timed_points_to_grid(grid_pred, timed_points)
    risk = grid_pred.intensity_matrix
    pts = risk[gy,gx]
    pts[pts<=0] = minimum
    return _np.mean(_np.log(pts))
    

def _brier_setup(grid_pred, timed_points):
    if len(timed_points.xcoords) == 0:
        raise ValueError("Need non-empty timed points")
    gx, gy = _timed_points_to_grid(grid_pred, timed_points)
    risk = grid_pred.intensity_matrix
    
    u = _np.zeros_like(risk)
    for x, y in zip(gx, gy):
        u[y,x] += 1
    u = u / _np.sum(u)
    
    try:
        u = _np.ma.array(u, mask=risk.mask)
    except AttributeError:
        pass

    return risk, u    

def brier_score(grid_pred, timed_points):
    """Compute the brier score,
    
      :math:`\frac{1}{A} \sum_i (p_i - u_i)^2`
      
    where `A` is the area of the (masked) grid, :math:`p_i` is the grid
    prediction in cell `i`, and :math:`u_i` is the fraction of events which
    occur in cell `i`.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.

    :return: `(score, skill)` where `score` is as above, and `skill` is
      :math:`\frac{2\sum_i p_iu_i}{\sum_i u_i^2 + \sum_i p_i^2}`.
    """
    risk, u = _brier_setup(grid_pred, timed_points)
    try:
        num_cells = _np.sum(~risk.mask)
    except AttributeError:
        num_cells = risk.shape[0] * risk.shape[1]
    area = num_cells * grid_pred.xsize * grid_pred.ysize
    score = _np.sum((u - risk)**2) / area
    skill = 2 * _np.sum(u * risk) / (_np.sum(u * u + risk * risk))
    return score, skill

def _kl_log_func(x, y):
    score = 0
    x, y = _np.asarray(x), _np.asarray(y)
    m = (x>0) & (y<=0)
    if _np.any(m):
        score += _np.sum(x[m] * (_np.log(x[m]) + 20))
    m = (x>0) & (y>0)
    if _np.any(m):
        score += _np.sum(x[m] * (_np.log(x[m]) - _np.log(y[m])))
    return score

def kl_score(grid_pred, timed_points):
    """Compute the (ad hoc) Kullback-Leibler divergance score,
    
      :math:`\frac{1}{A} \sum_i u_i\log(u_i/p_i) + (1-u_i)\log((1-u_i)/(1-p_i))
      
    where `A` is the area of the (masked) grid, :math:`p_i` is the grid
    prediction in cell `i`, and :math:`u_i` is the fraction of events which
    occur in cell `i`.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.

    :return: The score
    """
    risk, u = _brier_setup(grid_pred, timed_points)
    mask = _np.ma.getmaskarray(risk)
    num_cells = _np.sum(~mask)
    area = num_cells * grid_pred.xsize * grid_pred.ysize

    x, y = u.flatten(), risk.flatten()
    score = _kl_log_func(x, y) + _kl_log_func(1-x, 1-y)
    return score / area

def _to_array_and_norm(a):
    a = _np.asarray(a)
    return a / _np.sum(a)

def multiscale_brier_score(grid_pred, timed_points, size=1):
    """Compute the brier score,
    
      :math:`\frac{1}{A} \sum_i (p_i - u_i)^2`
      
    where `A` is the area of the (masked) grid, :math:`p_i` is the grid
    prediction in cell `i`, and :math:`u_i` is the fraction of events which
    occur in cell `i`.  This version is slower, but allows an "aggregation
    level" whereby we use a "moving window" to group together cells of a
    square shape of a certain size.  Takes account of the mask sensibly.

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.
    :param size: The "aggregation level", an integer `>=1`.

    :return: `(score, skill)` where `score` is as above, and `skill` is
      :math:`\frac{2\sum_i p_iu_i}{\sum_i u_i^2 + \sum_i p_i^2}`.
    """
    risk, u = _brier_setup(grid_pred, timed_points)
    cell_area = grid_pred.xsize * grid_pred.ysize

    agg_risk, agg_u, cell_sizes = [], [], []
    for (s1, c1), (s2, c2) in zip(generate_aggregated_cells(risk, size),
                         generate_aggregated_cells(u, size)):
        if c1 > 0:
            cell_sizes.append(c1)
            agg_risk.append(s1)
            agg_u.append(s2)

    agg_risk = _to_array_and_norm(agg_risk)
    agg_u = _to_array_and_norm(agg_u)
    cell_sizes = _to_array_and_norm(cell_sizes)

    score = _np.sum( cell_sizes * (agg_risk - agg_u)**2 )
    score_worst = _np.sum( cell_sizes * (agg_risk**2 + agg_u**2) )
    skill = 1 - score / score_worst
    return score / cell_area, skill
    
def _kl_log_func_weighted(x, y, w):
    score = 0
    x, y = _np.asarray(x), _np.asarray(y)
    m = (x>0) & (y<=0)
    if _np.any(m):
        score += _np.sum(w[m] * x[m] * (_np.log(x[m]) + 20))
    m = (x>0) & (y>0)
    if _np.any(m):
        score += _np.sum(w[m] * x[m] * (_np.log(x[m]) - _np.log(y[m])))
    return score

def multiscale_kl_score(grid_pred, timed_points, size=1):
    """As :func:`kl_score` but with aggregation."""
    risk, u = _brier_setup(grid_pred, timed_points)
    cell_area = grid_pred.xsize * grid_pred.ysize

    agg_risk, agg_u, cell_sizes = [], [], []
    for (s1, c1), (s2, c2) in zip(generate_aggregated_cells(risk, size),
                         generate_aggregated_cells(u, size)):
        if c1 > 0:
            cell_sizes.append(c1)
            agg_risk.append(s1)
            agg_u.append(s2)

    agg_risk = _to_array_and_norm(agg_risk)
    agg_u = _to_array_and_norm(agg_u)
    cell_sizes = _to_array_and_norm(cell_sizes)

    score = ( _kl_log_func_weighted(agg_u, agg_risk, cell_sizes)
        + _kl_log_func_weighted(1 - agg_u, 1 - agg_risk, cell_sizes) )
    return score / cell_area

def generate_aggregated_cells(matrix, size):
    """Working left to right, top to bottom, aggregate the values of the grid
    into larger grid squares of size `size` by `size`.  Also computes the
    fraction of grid cells not masked, if approrpriate.
    
    :param matrix: A (optionally masked) matrix of data.
    :param size: The "aggregation level".
    
    :return: Generates pairs `(value, valid_cells)` where `value` is the sum
      of the un-masked cells, and `valid_cells` is a count.  If the input
      grid has size `X` by `Y` then returns `(Y+1-size) * (X+1-size)` pairs.
    """
    risk = matrix
    have_mask = False
    try:
        risk.mask
        have_mask = True
    except AttributeError:
        pass
    
    if have_mask:
        m = risk.mask
        for y in range(risk.shape[0] + 1 - size):
            for x in range(risk.shape[1] + 1 - size):
                s = _np.ma.sum(risk[y:y+size,x:x+size])
                if s is _np.ma.masked:
                    s = 0
                yield s, _np.sum(~m[y:y+size,x:x+size])
    else:
        for y in range(risk.shape[0] + 1 - size):
            for x in range(risk.shape[1] + 1 - size):
                yield _np.sum(risk[y:y+size,x:x+size]), size * size
    
def _bayesian_prep(grid_pred, timed_points, bias, lower_bound):
    if len(timed_points.xcoords) == 0:
        raise ValueError("Need non-empty timed points")

    try:
        alpha = _np.ma.array(grid_pred.intensity_matrix, mask=grid_pred.intensity_matrix.mask)
        alpha[alpha <= 0] = lower_bound
    except AttributeError:
        alpha = _np.array(grid_pred.intensity_matrix, dtype=_np.float)
        alpha[alpha <= 0] = lower_bound
    alpha = alpha / _np.sum(alpha) * bias

    gx, gy = _timed_points_to_grid(grid_pred, timed_points)
    counts = _np.zeros_like(alpha)
    for x, y in zip(gx, gy):
        counts[y,x] += 1
    try:
        counts = _np.ma.array(counts, mask=alpha.mask)
    except AttributeError:
        pass

    return alpha.flatten(), counts.flatten()

def bayesian_dirichlet_prior(grid_pred, timed_points, bias=10, lower_bound=1e-10):
    """Compute the Kullback-Leibler diveregence between a Dirichlet prior and
    the posterior given the data in `timed_points`.
    
    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.
    :param bias: How much to scale the "prediction" by.
    :param lower_bound: Set zero probabilities in the prediction to this,
      before applying the `bias`.
    """
    alpha, counts = _bayesian_prep(grid_pred, timed_points, bias, lower_bound)
    count = _np.sum(counts)
    
    score = _np.sum(_np.log(_np.arange(bias, bias + count)))
    m = counts > 0
    for a, c in zip(alpha[m], counts[m]):
        score -= _np.sum(_np.log(_np.arange(a, a+c)))
    
    score += _np.sum(_special.digamma(alpha[m] + counts[m]) * counts[m])
    score -= count * _special.digamma(bias + count)
    
    return score

def bayesian_predictive(grid_pred, timed_points, bias=10, lower_bound=1e-10):
    """Compute the Kullback-Leibler diveregence between the prior and posterior
    _predictive_ distributions, given a Dirichlet prior.
    
    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.
    :param bias: How much to scale the "prediction" by.
    :param lower_bound: Set zero probabilities in the prediction to this,
      before applying the `bias`.
    """
    alpha, counts = _bayesian_prep(grid_pred, timed_points, bias, lower_bound)
    count = _np.sum(counts)

    w = (alpha + counts) / (bias + count)
    return _np.sum(w * (_np.log(w) + _np.log(bias) - _np.log(alpha)))

def convert_to_precentiles(intensity):
    """Helper method.  Converts the (possibly masked) intensity array into a 
    ranking" array, whereby the `i`th entry is the fraction of entries in
    `intensity` which are less than or equal to `intensity[i]`.
    
    :param intensity: A possibly masked array
    
    :return: A "ranking" array of the same shape and masking as `intensity`.
    """
    flat = intensity.flatten()
    ranking = _np.sum(flat[:,None] <= flat[None,:], axis=0)
    mask = _np.ma.getmaskarray(ranking)
    ranking = ranking / _np.sum(~mask)
    return ranking.reshape(intensity.shape)
    
def ranking_score(grid_pred, timed_points):
    """Convert the `timed_points` into a "ranking".  First the intensity matrix
    of the prediction is converted to rank order (see
    :func:`convert_to_precentiles`) and then each point is evaluated on the
    rank.
    
    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.  Should be normalised.
    :param timed_points: An instance of :class:`TimedPoints` from which to look
      at the :attr:`coords`.  All the points should fall inside the non-masked
      area of the prediction.  Raises `ValueError` is not.
      
    :return: Array of rankings, same length as `timed_points`.
    """
    if len(timed_points.xcoords) == 0:
        raise ValueError("Need non-empty timed points")
    gx, gy = _timed_points_to_grid(grid_pred, timed_points)
    ranking = convert_to_precentiles(grid_pred.intensity_matrix)
    return ranking[gy,gx]





#############################################################################
# Network stuff
#############################################################################

def grid_risk_coverage_to_graph(grid_pred, graph, percentage_coverage, intersection_cutoff=None):
    """Find the given coverage for the grid prediction, and then intersect with
    the graph.
    
    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.
    :param graph: An instance of :class:`network.PlanarGraph`
    :param percentage_coverage: The percentage coverage to apply.
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

def grid_risk_to_graph(grid_pred, graph, strategy="most"):
    """Transfer the grid_prediction to a graph risk prediction.  For each grid
    cell, assigns the risk in the cell to each edge of the network which
    intersects that cell.  The parameter `strategy` determines exactly how this
    is done:
    
    - "most" means that for each network edge, find the cell which most
      overlaps it, and use that cell's risk
    - "subdivide" means that we should generate a new graph by chopping each
      edge into parts so that every edge in the new graph intersects exactly
      one grid cell

    :param grid_pred: An instance of :class:`GridPrediction` to give a
      prediction.
    :param graph: An instance of :class:`network.PlanarGraph`
    :param strategy: "most" or "subdivide"
    
    :return: `(graph, lookup, risks)` where `graph` is a possible new graph,
      and `risks` is an array of risks, correpsonding to the edges in the
      graph.  If we built a new graph, then `lookup` will be a dictionary from
      edge index in the new graph to edge index in the old graph (in general a
      one-to-many mapping).
    """
    if strategy == "most":
        return graph, None, _grid_risk_to_graph_most(grid_pred, graph)
    elif strategy == "subdivide":
        return _grid_risk_to_graph_subdivide(grid_pred, graph)
    else:
        raise ValueError()
        
def _grid_risk_to_graph_subdivide(grid_pred, graph):
    risks = []
    lookup = dict()
    builder = _network.PlanarGraphBuilder()
    builder.vertices.update(graph.vertices)
    for edge_index, edge in enumerate(graph.edges):
        line = (graph.vertices[edge[0]], graph.vertices[edge[1]])
        segments, intervals = _geometry.full_intersect_line_grid(line, grid_pred)
        
        start_key = edge[0]
        for i in range(len(segments) - 1):
            key = builder.add_vertex(*segments[i][1])
            lookup[ len(builder.edges) ] = edge_index
            builder.add_edge(start_key, key)
            start_key = key
            gx, gy, _, _ = intervals[i]
            risks.append(grid_pred.grid_risk(gx, gy))
        lookup[ len(builder.edges) ] = edge_index
        builder.add_edge(start_key, edge[1])
        gx, gy, _, _ = intervals[-1]
        risks.append(grid_pred.grid_risk(gx, gy))
            
    return builder.build(), lookup, _np.asarray(risks)
        
def _grid_risk_to_graph_most(grid_pred, graph):
    risks = _np.empty(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        line = (graph.vertices[edge[0]], graph.vertices[edge[1]])
        gx, gy = _geometry.intersect_line_grid_most(line, grid_pred)
        if grid_pred.is_valid(gx, gy):
            risks[i] = grid_pred.grid_risk(gx, gy)
        else:
            risks[i] = 0
    return risks

def network_coverage(graph, risks, fraction):
    """For the given graph and risks for each edge, find the top fraction
    of the network by length.
    
    :param graph: An instance of :class:`network.PlanarGraph`
    :param risks: Array of risks the same length as the number of edges in
      `graph`.
    :param fraction: Between 0 and 1.
    
    :return: Boolean array of length the same length as the number of edges in
      `graph`, with `True` meaning that the edge should be included.
    """
    target_length = _np.sum(graph.lengths) * fraction
    indices = _np.argsort(risks)
    included = _np.zeros_like(indices, dtype=_np.bool)
    length = 0.0
    for index in indices[::-1]:
        length += graph.length(index)
        if length > target_length:
            break
        included[index] = True
    return included

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

def network_hit_rates_from_coverage(graph, risks, timed_network_points, percentage_coverages):
    """Computes the "hit rate" for the given prediction for the passed
    collection of events.  For each percent, we top slice that percentage of
    edges from the `risks`, and compute the fraction of events which fall in
    those edges.
    
    :param graph: The :class:`network.PlanarGraph` used to construct the
      prediction.
    :param risks: An array of risks of each edge, same length as `graph.edges`.
    :param timed_network_points: An instance of :class:`TimedNetworkPoints`
      to get events from.  We assume that the vertex keys used are the same
      as in `graph`.
    :param percentage_coverages: An iterable of percentage coverages to test.

    :return: A dictionary from percentage coverage to hit rate percentage.
      If there were no events in the `timed_points`, we return -1.
    """
    if len(timed_network_points.start_keys) == 0:
        return {cov : -1.0 for cov in percentage_coverage}
    edges = []
    for st, en in zip(timed_network_points.start_keys, timed_network_points.end_keys):
        e, _ = graph.find_edge(st, en)
        edges.append(e)
    out = dict()
    for coverage in percentage_coverages:
        mask = network_coverage(graph, risks, coverage / 100)
        out[coverage] = sum(mask[e] for e in edges) * 100 / len(timed_network_points.start_keys)
    return out





####  Below is some _tentative_ code...

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
        self._logger = _logging.getLogger(__name__)
        
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
    
    def _process(self, pred, points, coverage_levels):
        out = hit_rates(pred, points, coverage_levels)
        details = HitRateDetail(
            total_cell_count=_np.ma.sum(~pred.intensity_matrix.mask),
            prediction = pred
            )
        return out, details

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
            self._logger.debug("Making prediction using %s for %s--%s", self._provider, start, end)
            points = self._points(start, end)
            if points.number_data_points == 0:
                continue
            preds = self._provider.predict(start)
            try:
                outs, ds = [], []
                for pred in preds:
                    ou, d = self._process(pred, points, coverage_levels)
                    outs.append(ou)
                    ds.append(d)
                out[start] = outs
                details[start] = ds
            except:
                out[start] = hit_rates(preds, points, coverage_levels)
                details[start] = HitRateDetail(
                    total_cell_count=_np.ma.sum(~preds.intensity_matrix.mask),
                    prediction = preds
                    )
        return HitRateResult(out, details)
