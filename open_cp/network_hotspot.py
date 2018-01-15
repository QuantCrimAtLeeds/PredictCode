"""
network_hotspot
~~~~~~~~~~~~~~~

Implements, to a good approximation, the "network prospective hotspot"
algorithm as described in:

- Rosser et al. "Predictive Crime Mapping: Arbitrary Grids or Street Networks?"
  Journal of Quantitative Criminology 33 (2017) 569--594,
  DOI: 10.1007/s10940-016-9321-x

By setting the time kernel to a constant, or a window function, you can also
perform a "network retrospective hotspot".
"""

from . import predictors
from . import network
from . import logger as _logger_mod
import math as _math
import numpy as _np
from scipy import ndimage as _ndimage
import logging as _logging
import datetime as _datetime

_logger = _logging.getLogger(__name__)

class Trainer(predictors.DataTrainer):
    """Handles "data preparation" tasks:
      - Specifying the geometry
      - Adjusting the geometry
      - Projecting each event to the network
    """
    def __init__(self, graph=None, maximum_edge_length=None):
        self.graph = graph
        self.maximum_edge_length = maximum_edge_length

    @property
    def graph(self):
        """The instance of :class:`network.PlanarGraph` which specifies
        the network."""
        return self._graph

    @graph.setter
    def graph(self, v):
        self._graph = v

    @property
    def maximum_edge_length(self):
        """The maximum length allowed for edges.  Edges longer than this will
        have vertices added to split the edge.  Or `None` to not make any
        changes.
        """
        return self._max_edge_len

    @maximum_edge_length.setter
    def maximum_edge_length(self, v):
        self._max_edge_len = v

    def compile(self):
        """Run the specified tasks.

        :return: An instance of :class:`Predictor` which can perform
        predictions.
        """
        if self.maximum_edge_length is None:
            graph = self.graph
        else:
            _logger.debug("Mutating the graph so that maximum edge length is %s", self.maximum_edge_length)
            graph = _GraphSplitter(self.graph, self.maximum_edge_length).split()
        _logger.debug("Projecting %s events to the network", self.data.number_data_points)
        tnp = network.TimedNetworkPoints.project_timed_points(self.data, graph)
        return Predictor(tnp, graph)


class _GraphSplitter():
    """Convert edge lengths to a maximum."""
    def __init__(self, graph, maximum_edge_length):
        self.graph = graph
        self.maximum_edge_length = maximum_edge_length
        self.builder = network.PlanarGraphBuilder()
        self.builder.vertices.update(self.graph.vertices)

    def _split_edge(self, i, key1, key2):
        nodes_to_add = _math.ceil(self.graph.length(i) / self.maximum_edge_length) - 1
        start_key = key1
        for i in range(nodes_to_add):
            t = (i + 1) / (nodes_to_add + 1)
            x, y = self.graph.edge_to_coords(key1, key2, t)
            key = self.builder.add_vertex(x, y)
            self.builder.add_edge(start_key, key)
            start_key = key
        self.builder.add_edge(start_key, key2)

    def split(self):
        for i, (key1, key2) in enumerate(self.graph.edges):
            self._split_edge(i, key1, key2)
        return self.builder.build()


from . import kde as _kde

ConstantTimeKernel = _kde.ConstantTimeKernel

class ExponentialTimeKernel(_kde.ExponentialTimeKernel):
    """An exponentially decaying kernel,
    :math:`f(x) = \beta^{-1} \exp(-x/\beta)`
    where :math:`beta` is the "scale".
    """
    def __init__(self, scale):
        super().__init__(scale)
    
    def __repr__(self):
        return "ExponentialTimeKernel({})".format(self._scale)


class QuadDecayTimeKernel(_kde.QuadDecayTimeKernel):
    """A quadratically decaying kernel, :math:`f(x) = \frac{2}{\pi\beta}
    (1 + (x/\beta)^2)^{-1]}` where :math:`beta` is the "scale".
    """
    def __init__(self, scale):
        super().__init__(scale)

    def __repr__(self):
        return "QuadDecayTimeKernel({})".format(self._scale)


class NetworkKernel():
    """Abstract base class for a "kernel" for the spatial component.
    This should be a _normalised_ kernel defined for values `s >= 0`;
    by "normalised" we mean that the total integral should be 1/2
    (so that reflecting about 0 gives a probability kernel).
    """
    def __call__(self, x):
        """Should accept a scalar or array."""
        raise NotImplementedError()

    def integrate(self, a, b):
        """Compute the integral of the kernel between `0<=a<=b`.
        The default implementation uses the value at the mid-point 
        times the length.  If it is analytically tractable, override
        and supply an exact answer.
        
        You should (carefully) support `a` and `b` being either scalars or
        numpy arrays.
        """
        a, b = _np.asarray(a), _np.asarray(b)
        return (b - a) * self.__call__((a + b) / 2)

    @property
    def cutoff(self):
        """The maxmimum value at which the kernel is non-zero."""
        raise NotImplementedError()


class TriangleKernel(NetworkKernel):
    """From the paper of Rosser et al.  :math:`f(s) = (h - s) / h^2`
    where `h` is the bandwidth."""
    def __init__(self, bandwidth):
        self._h = bandwidth

    def __call__(self, x):
        x = _np.asarray(x)
        y = (self._h - x) / (self._h * self._h)
        if len(y.shape) == 0:
            if x >= self._h:
                return 0
            return y
        else:
            return _np.maximum(0, y)

    def integrate(self, a, b):
        a, b = _np.asarray(a), _np.asarray(b)
        if len(a.shape) == 0:
            if a >= self._h:
                return 0
            b = min(b, self._h)
            return (b - a) * (self._h + self._h - a - b) / (self._h * self._h) * 0.5
        mask = a >= self._h
        b = _np.minimum(b, self._h)
        c = (b - a) * (self._h + self._h - a - b) / (self._h * self._h) * 0.5
        c[mask] = 0
        return c

    @property
    def cutoff(self):
        return self._h
    
    def __repr__(self):
        return "TriangleKernel({})".format(self._h)


class Predictor():
    """The class which can make predictions.  Should be constructed by using an
    instance of :class:`Trainer`.
    """
    def __init__(self, network_timed_points, graph):
        self._network_timed_points = network_timed_points
        self._graph = graph
        self.time_kernel_unit = _np.timedelta64(1, "D")
        self.time_kernel = None
        self.kernel = None

    @property
    def network_timed_points(self):
        return self._network_timed_points

    @property
    def graph(self):
        return self._graph

    @property
    def time_kernel(self):
        """The time kernel in use."""
        return self._time_kernel

    @time_kernel.setter
    def time_kernel(self, v):
        self._time_kernel = v

    @property
    def time_kernel_unit(self):
        """The time-unit, instance of :class:`numpy.timedelta64` in use.
        Think hard before changing!"""
        return self._time_unit
    
    @time_kernel_unit.setter
    def time_kernel_unit(self, v):
        self._time_unit = v

    @property
    def kernel(self):
        """The spatial / network kernel"""
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        self._kernel = v
        
    def add(self, risks, edge, orient, offset, time_weight=1):
        """Internal use: add to the risks from all paths.

        :param risks: Array of risks to add to.
        :param edge: Index of the edge we are starting at
        :param orient: 1 or -1 for which way to initially walk to edge
        :param offset: How far along the edge we are (between 0 and 1)
        :param time_weight: How much to scale by
        """
        start_key, avoid_key = self.graph.edges[edge]
        if orient == 1:
            start_key, avoid_key = avoid_key, start_key
        offset = (1.0 - offset) * self.graph.length(edge)
        for index, start_length, end_length, degree in self.graph.walk_with_degrees(start_key, avoid_key, self.kernel.cutoff, 20000):
            if index is None or index == -1:
                a, b = 0, offset
                index = edge
            else:
                a, b = start_length + offset, end_length + offset
            if a >= self.kernel.cutoff:
                continue
            risks[index] += self.kernel.integrate(a, b) * time_weight / degree

    def add_edge(self, risks, edge_index, dist, tw):
        """Internal use.  Add both contributions to an edge.

        :param risks: Array of risks to add to.
        :param edge_index: Index of the edge we are starting at
        :param dist: How far along the edge we are (between 0 and 1)
        :param tw: How much to scale by
        """
        self.add(risks, edge_index, 1, dist, tw)
        self.add(risks, edge_index, -1, 1.0 - dist, tw)

    def predict(self, predict_time=None, cutoff_time=None):
        """Make a prediction.

        :param predict_time: Use only events before this time, and treat this
          as the time 0 point for the time kernel.  If `None` then use the
          last time-stamp in the input data.
        :param cutoff_time: Use only events after this time.  If `None` then use
          all events from the start of the input data.
        """
        if cutoff_time is None:
            cutoff_time = self.network_timed_points.time_range[0]
        cutoff_time = _np.datetime64(cutoff_time)
        mask = (self.network_timed_points.timestamps >= cutoff_time)
        if predict_time is not None:
            predict_time = _np.datetime64(predict_time)
            mask &= (self.network_timed_points.timestamps < predict_time)
        else:
            predict_time = self.network_timed_points.time_range[1]
        data = self.network_timed_points[mask]

        times = (predict_time - data.timestamps) / self.time_kernel_unit
        time_weights = self.time_kernel(times)
        risks = _np.zeros(len(self.graph.edges))
        _logger.debug("Making prediction with %s events using %s/%s", len(times), self.kernel, self.time_kernel)
        progress = _logger_mod.ProgressLogger(len(times), _datetime.timedelta(minutes=2), _logger)
        for tw, key1, key2, dist in zip(time_weights, data.start_keys,
                data.end_keys, data.distances):
            edge_index, orient = self.graph.find_edge(key1, key2)
            if orient == -1:
                dist = 1.0 - dist
            self.add_edge(risks, edge_index, dist, tw)
            progress.increase_count()
        risks /= self.graph.lengths

        return Result(self.graph, risks)


class FastPredictor(Predictor):
    """A version of :class:`Predictor` which needs to be "compiled", and then
    can quickly perform predictors.  It performs lazy initialisation, so making
    the first prediction will be slow, but subsequent calls should be fast(er).
    Will also cache intermediate results so that you can change the _time_
    kernel (but not the _space_ kernel) and quickly recompute a result.
    
    Unfortunately is very memory hungry.
    
    :param predictor: An :class:`Predictor` to initialise from.
    :param max_length: The maximum "support" length which any (spatial) kernel
      will be able to have.
    """
    def __init__(self, predictor, max_length):
        super().__init__(predictor.network_timed_points, predictor.graph)
        self.time_kernel_unit = predictor.time_kernel_unit
        self.time_kernel = predictor.time_kernel
        self.kernel = predictor.kernel
        self._max_length = max_length
        self._cache = dict()
        self._idx_cache = _np.array([])
        self._add_cache = dict()
        
    def _get(self, edge, orient):
        key = (edge, orient)
        if key not in self._cache:
            _logger.debug("Populating cache for %s", key)
            start_key, avoid_key = self.graph.edges[edge]
            if orient == 1:
                start_key, avoid_key = avoid_key, start_key
            data = list(self.graph.walk_with_degrees(start_key, avoid_key, self._max_length, 20000))
            index = _np.empty(len(data), dtype=_np.int)
            start = _np.empty(len(data))
            end = _np.empty(len(data))
            degree = _np.empty(len(data))
            for i, (idx, st, en, deg) in enumerate(data):
                if i==0:
                    idx = edge
                index[i] = idx
                start[i] = st
                end[i] = en
                degree[i] = 1.0 / deg
            self._cache[key] = (index, start, end, degree)
        return self._cache[key]

    def add(self, risks, edge, orient, offset, time_weight=1):
        """Internal use: add to the risks from all paths.

        :param risks: Array of risks to add to.
        :param edge: Index of the edge we are starting at
        :param orient: 1 or -1 for which way to initially walk to edge
        :param offset: How far along the edge we are (between 0 and 1)
        :param time_weight: How much to scale by
        """
        key = (edge, orient, offset)
        if key not in self._add_cache:
            if self.kernel.cutoff > self._max_length:
                raise ValueError("Build from maximum length {}".format(self._max_length))
            offset = (1.0 - offset) * self.graph.length(edge)
            index, start, end, degree = self._get(edge, orient)
            start = _np.array(start) + offset
            end = _np.array(end) + offset
            start[0] = 0
            mask = start < self.kernel.cutoff
            index, start, end, degree = index[mask], start[mask], end[mask], degree[mask]
            to_add = self.kernel.integrate(start, end) * degree
            if len(self._idx_cache) != len(risks):
                self._idx_cache = _np.arange(len(risks))
            self._add_cache[key] = _ndimage.sum(to_add, labels=index, index=self._idx_cache)
        risks += self._add_cache[key] * time_weight

    @property
    def kernel(self):
        """The spatial / network kernel"""
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        self._add_cache = dict()
        self._kernel = v


class ApproxPredictor(Predictor):
    """Uses an approximation to the KDE method: we compute the minimal distance
    between the edge of the event and the edge we're interested in, compute
    the kernel, and then weight by the degrees of the vertices in the path.

    This is definitely an approximation; but it seems fairly accurate in
    real-world networks.

    :param predictor: An :class:`Predictor` to initialise from.
    """
    def __init__(self, predictor):
        super().__init__(predictor.network_timed_points, predictor.graph)
        self.time_kernel_unit = predictor.time_kernel_unit
        self.time_kernel = predictor.time_kernel
        self.kernel = predictor.kernel

    def add_edge(self, risks, edge_index, dist, tw):
        """Internal use.  Add both contributions to an edge.

        :param risks: Array of risks to add to.
        :param edge_index: Index of the edge we are starting at
        :param dist: How far along the edge we are (between 0 and 1).
          We ignore and set to 0.5
        :param tw: How much to scale by
        """
        dists, prevs = network.shortest_edge_paths(self.graph, edge_index, 0.5)
        to_add = _np.zeros(self.graph.number_edges, dtype=_np.float)
        for i in range(self.graph.number_edges):
            if i == edge_index:
                to_add[i] = self.kernel(0)
                continue
            k1, k2 = self.graph.edges[i]
            le = self.graph.length(i) * 0.5
            if k1 not in dists:
                continue
            le1, le2 = dists[k1], dists[k2]
            if le1 < le2:
                target = k1
                weight = self.kernel(le1 + le)
            else:
                target = k2
                weight = self.kernel(le2 + le)
            if weight == 0:
                continue
            cum_deg = max(1, self.graph.degree(target) - 1)
            while True:
                next_target = prevs[target]
                if next_target == target:
                    break
                target = next_target
                cum_deg *= max(1, self.graph.degree(target) - 1)
            to_add[i] = weight / cum_deg

        risks += to_add * self.graph.lengths * tw


class ApproxPredictorCaching(ApproxPredictor):
    """As :class:`ApproxPredictor` but caches data.  Uses the same strategy
    as :class:`FastPredictor` and also caches spatial kernel data.

    :param predictor: An :class:`Predictor` to initialise from.
    """
    def __init__(self, predictor):
        super().__init__(predictor)
        self._cache = dict()
        self._add_cache = dict()

    def _get_data(self, edge_index):
        _logger.debug("ApproxPredictorCaching: Calculating for %s", edge_index)
        return network.shortest_edge_paths_with_degrees(self.graph, edge_index)

    def add_edge(self, risks, edge_index, dist, tw):
        """Internal use.  Add both contributions to an edge.

        :param risks: Array of risks to add to.
        :param edge_index: Index of the edge we are starting at
        :param dist: How far along the edge we are (between 0 and 1).
          We ignore and set to 0.5
        :param tw: How much to scale by
        """
        if edge_index not in self._add_cache:
            if edge_index not in self._cache:
                self._cache[edge_index] = self._get_data(edge_index)
            kernel_dists, cumulative_degrees = self._cache[edge_index]
            mask = kernel_dists > -1
            self._add_cache[edge_index] = mask, self.kernel(kernel_dists[mask]) / cumulative_degrees[mask] * self.graph.lengths[mask]
        mask, toadd = self._add_cache[edge_index]
        risks[mask] += toadd * tw

    @property
    def kernel(self):
        """The spatial / network kernel"""
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        self._add_cache = dict()
        self._kernel = v


class Result():
    """The result of a prediction.

    :param graph: The network
    :param risks: An array of risk intensities, corresponding to the edges in
      `graph`.
    """
    def __init__(self, graph, risks):
        self._graph = graph
        self._risks = _np.asarray(risks)

    @property
    def graph(self):
        return self._graph

    @property
    def risks(self):
        return self._risks
    
    def coverage(self, percentage):
        """Return a new instance with only the edges forming the top
        `percentage` of edges, by total length, weighted by the risk.
        
        :param percentage: Between 0 and 100.
        
        :return: New instance of :class:`Result`.
        """
        total_length = sum(self._graph.length(i) for i in range(self._graph.number_edges))
        target_length = total_length * percentage / 100
        length = 0.0
        risks = []
        builder = network.PlanarGraphBuilder()
        builder.vertices.update(self._graph.vertices)
        for index in _np.argsort(-self._risks):
            risks.append( self._risks[index] )
            builder.edges.append( self._graph.edges[index] )
            length += self._graph.length(index)
            if length >= target_length:
                break
        builder.remove_unused_vertices()
        return Result(builder.build(), risks)


