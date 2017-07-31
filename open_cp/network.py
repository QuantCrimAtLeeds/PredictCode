"""
network
~~~~~~~

Some basic "network" / graph theory tools.

We roll our own graph class as we don't need much, and we are particularly
interested in "embedded planar" graphs, that is, the exact way our graph is
presented in the plane is vital.
"""

import numpy as _np
#from . import data as _data
import logging as _logging

_logger = _logging.getLogger(__name__)


class PlanarGraphGeoBuilder():
    """TODO
    """
    def __init__(self):
        self._nodes = dict()
        self._current_key = 0
        self._edges = []
    
    def _new_key(self, x, y):
        key = self._current_key
        self._current_key += 1
        if (x, y) not in self._nodes:
            self._nodes[(x, y)] = list()
        self._nodes[(x, y)].append(key)
        return key

    def _key(self, x, y):
        if (x,y) in self._nodes:
            return self._nodes[(x, y)][0]
        else:
            return self._new_key(x, y)
    
    def add_path(self, path):
        """Add a new "path" to the graph.  A "path" has a start and end node,
        and possibly nodes in the middle.
        
        The start and end nodes are assumed to possibly correspond to existing
        nodes: if the coordinates _exactly_ match any existing start/end nodes,
        we assume the nodes are the same.
        
        Interior nodes are _not_ assumed to correspond to existing nodes: even
        if the (x,y) coordinates match, we generate a new vertex.
        
        This is compatible with the UK Ordnance Survey data where nodes can be
        shared between e.g. under and over passes.  We do not wish to treat
        a node as being a valid path in the graph.
        
        :param path: A list of coordinates `(x,y)` (or possibly `(x,y,z)` with
          the `z` to be ignored).  This is compatible with the `shapely`
          format.
        """
        path = list(path)
        x,y,*z = path[0]
        key1 = self._key(x,y)
        for (x,y,*z) in path[1:-1]:
            key = self._new_key(x, y)
            self._edges.append((key1, key))
            key1 = key
        x,y,*z = path[-1]
        key2 = self._key(x, y)
        self._edges.append((key1, key2))

    @property
    def coord_nodes(self):
        """Dictionary (do not mutate) from coordinates to node keys"""
        return self._nodes
    
    @property
    def edges(self):
        """List (do not mutate) of edges, as unordered pairs of keys."""
        return self._edges
    
    def build(self):
        vertices = []
        for (x,y) in self._nodes:
            for key in self._nodes[(x,y)]:
                vertices.append( (key, x, y) )
        return PlanarGraph(vertices, self.edges)


class PlanarGraph():
    """A simple graph class.
    
    - "Nodes" or "vertices" are (x,y) coordinates in the plane, but are also
      keyed by any hashable Python object (typically, integers).
    - "Edges" are undirected links between two vertices.
    
    We assume that the graph is "simple" (between two vertices there is at
    most one edge, and an edge is always between _distinct_ vertices).
    
    This class is immutable (at least by design: do not mutate the underlying
    dictionaries!)  See the static constructors and the builder class for ways
    to construct a graph.    
    
    :param vertices: An iterables of triples `(key, x, y)`.
    :param edges: An iterable of (unordered) pairs `(key1, key2)`.
    """
    def __init__(self, vertices, edges):
        self._vertices = dict()
        for key, x, y in vertices:
            if key in self._vertices:
                raise ValueError("Keys of vertices should be unique; but {} is repeated".format(key))
            self._vertices[key] = (x,y)
        self._edges = list()
        for key1, key2 in edges:
            if key1 == key2:
                raise ValueError("Cannot have an edge from vertex {} to itself".format(key1))
            self._edges.append((key1, key2))
            
    @property
    def vertices(self):
        """A dictionary (do not mutate!) from `key` to planar coordinates
        `(x,y)`.
        """
        return self._vertices
    
    @property
    def edges(self):
        """A list of unordered edges `(key1, key2)`."""
        return self._edges
    
    def as_quads(self):
        """Returns a numpy array of shape `(N,4)` where `N` is the number of
        edges in the graph.  Each entry is `(x1,y1,x2,y1)` giving the
        coordinates of the "start" and "end" of the edge."""
        out = []
        for k1, k2 in self._edges:
            x1, y1 = self._vertices[k1]
            x2, y2 = self._vertices[k2]
            out.append((x1,y1,x2,y2))
        return _np.asarray(out)
    
    def as_lines(self):
        """Returns a list of "lines" where each "line" has the format
        `[(x1,y1), (x2,y2)]`.  Suitable for passing into a
        :class:`matplotlib.collections.LineCollection` for example."""
        out = []
        for k1, k2 in self._edges:
            x1, y1 = self._vertices[k1]
            x2, y2 = self._vertices[k2]
            out.append(((x1,y1),(x2,y2)))
        return out

    def edge_to_coords(self, key1, key2, t):
        """Return the coordinate of the point which is `t` distant along
        the straight line from `key1` to `key2`.
        
        :param key1: Key of the start vertex.
        :param key2: Key of the end vertex.
        :param t: For `0<=t<=1` the distance along the line.
        
        :return: `(x,y)`
        """
        xs, ys = self._vertices[key1]
        xe, ye = self._vertices[key2]
        return (xs * (1-t) + xe * t, ys * (1-t) + ye * t)

    def project_point_to_graph(self, x, y):
        """Projects a point to the nearest edge in the graph.
        
        Uses a `numpy` O(N) algorithm which is not great, but is acceptable,
        and "just works".
        
        :param x:
        :param y: The coordinates of the point
        
        :return: `(edge, t)` where `edge` is a pair `(key1, key2)` of the edge,
          and `0 <= t <= 1` is the distance from the node `key1` to the node
          `key2` where the point is projected.
        """
        if not hasattr(self, "_projector"):
            self._projector = PointProjector(self.as_quads())
        index, t = self._projector.project_point(x, y)
        return self._edges[index], t
        
        
        lines = self.as_quads().T
        point = _np.array((x,y))
        v = lines[2:4, :] - lines[0:2, :]
        x = point[:,None] - lines[0:2, :]
        t = (x[0]*v[0] + x[1]*v[1]) / (v[0]*v[0] + v[1]*v[1])
        t[t < 0] = 0
        t[t > 1] = 1
        proj = lines[0:2, :] + t[None, :] * v
        distsq = _np.sum((point[:,None] - proj)**2, axis=0)
        index = _np.argmin(distsq)
        return self._edges[index], t[index]


try:
    import rtree as _rtree
except:
    _logger.error("Failed to import `rtree`.")
    _rtree = None
    
class PointProjector():
    def __init__(self, quads):
        self._quads = _np.asarray(quads)
        if _rtree is None:
            self.project_point = self._project_point
        else:
            def gen():
                for i, line in enumerate(self._quads):
                    bds = self._bounds(*line)
                    yield i, bds, None
            self._idx = _rtree.index.Index(gen())
            self.project_point = self._project_point_rtree

    @staticmethod
    def _bounds(x1, y1, x2, y2):
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return [xmin, ymin, xmax, ymax]

    def _project_point(self, x, y, quads=None):
        if quads is None:
            lines = self._quads.T
        else:
            lines = _np.asarray(quads).T
        point = _np.array((x, y))
        v = lines[2:4, :] - lines[0:2, :]
        x = point[:,None] - lines[0:2, :]
        t = (x[0]*v[0] + x[1]*v[1]) / (v[0]*v[0] + v[1]*v[1])
        t[t < 0] = 0
        t[t > 1] = 1
        proj = lines[0:2, :] + t[None, :] * v
        distsq = _np.sum((point[:,None] - proj)**2, axis=0)
        index = _np.argmin(distsq)
        return index, t[index]

    def _project_point_rtree(self, x, y):
        point = _np.asarray((x, y))
        h = 10
        while True:
            xmin, xmax = point[0] - h, point[0] + h
            ymin, ymax = point[1] - h, point[1] + h
            indices = list(self._idx.intersection((xmin,ymin,xmax,ymax)))
            if len(indices) > 0:
                choices = [self._quads[i] for i in indices]
                index, t = self._project_point(x, y, choices)
                x1, y1, x2, y2 = choices[index]
                xx, yy = x1 * (1-t) + x2 * t, y1 * (1-t) + y2 * t
                distsq = (x-xx)*(x-xx) + (y-yy)*(y-yy)
                if distsq <= h*h:
                    return indices[index], t
            h += h
        
        