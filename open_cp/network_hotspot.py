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
import math as _math
import numpy as _np

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
            graph = _GraphSplitter(self.graph, self.maximum_edge_length).split()
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
    
    def __call__(self, x):
        return _np.exp( - _np.asarray(x) / self._scale ) / self._scale


class QuadDecayTimeKernel(_kde.QuadDecayTimeKernel):
    """A quadratically decaying kernel, :math:`f(x) = \frac{2}{\pi\beta}
    (1 + (x/\beta)^2)^{-1]}` where :math:`beta` is the "scale".
    """
    def __init__(self, scale):
        super().__init__(scale)

    def __call__(self, x):
        y = _kde.QuadDecayTimeKernel.__call__(self, x)
        return 2 * y / (_np.pi * self._scale)


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
        and supply an exact answer."""
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
            y[x >= self._h] = 0
            return y

    def integrate(self, a, b):
        b = min(b, self._h)
        return (b - a) * (self._h + self._h - a - b) / (self._h * self._h) * 0.5

    @property
    def cutoff(self):
        return self._h


class Predictor():
    """The class which can make predictions.  Should be constructed by using an
    instance of :class:`Trainer`.
    """
    def __init__(self, network_timed_points, graph):
        self._network_timed_points = network_timed_points
        self._graph = graph

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
    def kernel(self):
        """The spatial / network kernel"""
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        self._kernel = v
    
    def add(self, risks, edge, orient, offset):
        """Internal use: add to the risks all paths.

        :param risks: Array of risks to add to.
        :param edge: Index of the edge we are starting at
        :param orient: 1 or -1 for which way to initially walk to edge
        :param offset: How far along the edge we are
        """
        key1, key2 = self.graph.edges[edge]
        initial_dist = self.graph.length(edge) * (1.0 - offset)
        if orient == 1:
            todo = ([edge], [key2], 0, initial_dist, 1)
        else:
            todo = ([edge], [key1], 0, initial_dist, 1)
        todo = [todo]
        while len(todo) > 0:
            current_path, current_vertices, old_length, current_length, cumulative_degree = todo.pop()
            if old_length >= self.kernel.cutoff:
                continue
            risks[current_path[-1]] += self.kernel.integrate(old_length, current_length) * cumulative_degree
            for e in self.graph.neighbourhood_edges(current_vertices[-1]):
                if e == edge:
                    continue
                key1, key2 = self.graph.edges[e]
                if key2 == current_vertices[-1]:
                    key1, key2 = key2, key1
                if key2 in current_vertices:
                    continue
                vertices = current_vertices + [key2]
                path = current_path + [e]
                new_length = current_length + self.graph.length(e)
                new_degree = cumulative_degree / (self.graph.degree(current_vertices[-1]) - 1)
                todo.append((path, vertices, current_length, new_length, new_degree))

    def predict(self, predict_time=None, cutoff_time=None):
        """Make a prediction.

        :param predict_time: Use only events before this time, and treat this
          as the time 0 point for the time kernel.  If `None` then use the
          last time-stamp in the input data.
        :param cutoff_time: Use only events after this time.  If `None` then use
          all events from the start of the input data.
        """
        if predict_time is None:
            predict_time = self.network_timed_points.time_range[1]
        if cutoff_time is None:
            cutoff_time = self.network_timed_points.time_range[0]
        
        mask = ( (self.network_timed_points.timestamps >= cutoff_time) &
            (self.network_timed_points.timestamps <= predict_time) )
        data = self.network_timed_points[mask]
        
        risks = _np.zeros(len(self.graph.edges))
        time_weights = self.time_kernel(data.timestamps)
        for tw, key1, key2, dist in zip(time_weights, data.start_keys,
                data.end_keys, data.distances):
            edge_index, orient = self.graph.find_edge(key1, key2)
            if orient == -1:
                key1, key2, dist = key2, key1, 1.0 - dist
            self.add(risks, edge_index, 1, dist)
            self.add(risks, edge_index, -1, 1.0 - dist)

        return Result(self.graph, risks)


class Result():
    """The result of a prediction.

    :param graph: The network
    :param risks: An array of risk intensities, corresponding to the edges in
      `graph`.
    """
    def __init__(self, graph, risks):
        self._graph = graph
        self._risks = risks

    @property
    def graph(self):
        return self._graph

    @property
    def risks(self):
        return self._risks
