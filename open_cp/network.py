"""
network
~~~~~~~

Some basic "network" / graph theory tools.

We roll our own graph class as we don't need much, and we are particularly
interested in "embedded planar" graphs, that is, the exact way our graph is
presented in the plane is vital.
"""

import numpy as _np
import scipy.spatial as _spatial
from . import data as _data
import logging as _logging
import bz2 as _bz2
import io as _io
import base64 as _base64
import json as _json
import collections as _collections

_logger = _logging.getLogger(__name__)


class PlanarGraphGeoBuilder():
    """Construct a :class:`PlanarGraph` instance from a series of "paths".
    A path is formed from one or more contiguous line segments.  We only allow
    paths to intersect at their end points (geometrically, paths are allowed
    to intersect, but this will never be reflected in the generate graph.  This
    allows over and under passes, for example).  These assumptions are
    satisfied by the UK Ordnance Survey data.
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


class PlanarGraphNodeOneShot():
    """Like :class:`PlanarGraphNodeBuilder` but much faster, at the cost of
    needing all possible nodes to be set in the constructor.
    
    :param nodes: An iterable of pairs `(x,y)` of coordinates.  (Also allowed
      is `(x,y,z)` but the `z` will be ignored.)
    :param tolerance: The cut-off distance at which nodes will be merged.
    """
    def __init__(self, nodes, tolerance = 0.1):
        self._edges = []
        #all_nodes = list(set(((x,y) for x,y in points)))
        all_nodes = [(x,y) for (x,y,*z) in nodes]
        tree = _spatial.cKDTree(all_nodes)
        
        self._nodes = []
        allnodes_to_nodes_lookup = dict()
        
        for i, pt in enumerate(all_nodes):
            if i in allnodes_to_nodes_lookup:
                continue
            index = len(self._nodes)
            self._nodes.append(pt)
            for x in tree.query_ball_point(pt, tolerance):
                if x not in allnodes_to_nodes_lookup:
                    allnodes_to_nodes_lookup[x] = index
                    
        self._lookup = {pt:allnodes_to_nodes_lookup[i] for i, pt in enumerate(all_nodes)}

    def _add_node(self, x, y):
        return self._lookup[(x,y)]
        
    def add_path(self, path):
        """Add a new "path" to the graph.  A "path" has a start and end node,
        and possibly nodes in the middle.

        :param path: A list of coordinates `(x,y)` (or possibly `(x,y,z)` with
          the `z` to be ignored).  This is compatible with the `shapely`
          format.
          
        :return: List of edges added, each in format `(key1, key2)` where these
          are keys to vertices.
        """
        path = list(path)
        edges = []
        for i in range(len(path) - 1):
            x1,y1,*z = path[i]
            x2,y2,*z = path[i + 1]
            edges.append( self.add_edge(x1, y1, x2, y2) )
        return edges

    def add_edge(self, x1, y1, x2, y2):
        """Add an edge from `(x1, y1)` to `(x2, y2)`.
        
        :return: The edge added, in format `(key1, key2)` where these are keys
          to vertices.
        """
        key1 = self._add_node(x1, y1)
        key2 = self._add_node(x2, y2)
        e = (key1, key2)
        self._edges.append(e)
        return e

    def remove_duplicate_edges(self):
        """A neccessary evil.  Also removes loops."""
        edges = set()
        index = 0
        while index < len(self._edges):
            edge = frozenset(self._edges[index])
            if edge in edges or len(edge)==1:
                del self._edges[index]
            else:
                edges.add(edge)
                index += 1
        
    def build(self):
        vertices = [ (key, x, y) for key, (x,y) in enumerate(self._nodes) ]
        return PlanarGraph(vertices, self._edges)

        
class PlanarGraphNodeBuilder():
    """Construct a :class:`PlanarGraph` instance from a series of "paths".
    A path is formed from one or more contiguous line segments.  The start and
    end of each line segment is converted to being a node, and nodes which have
    _almost_ (subject to a certain tolerance) are merged.  As such, for example,
    if an over-pass and an under-pass share a node, there will be a "path" from
    one to the other in the generated graph.

    This class is slow when the graph is large.  See
    :class`:PlanarGraphNodeOneShot` as well.
    
    These (weaker) assumptions are suitable for the US TIGER/Lines data, for
    example.
    """
    def __init__(self):
        self._nodes = []
        self._edges = []
        self._tolerance = 0.1
        
    @property
    def tolerance(self):
        """Nodes which are within this distance of one another will be merged.
        """
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, v):
        self._tolerance = v
    
    @property
    def coord_nodes(self):
        """A list (do not mutate!) of coordinates `(x,y)`"""
        return self._nodes
    
    @property
    def edges(self):
        """A list of unordered edges `(key1, key2)`."""
        return self._edges
    
    def _add_node(self, x, y):
        if len(self._nodes) == 0:
            self._nodes.append((x, y))
            return 0
        n = _np.asarray(self._nodes).T
        distsq = (n[0] - x)**2 + (n[1] - y)**2
        index = _np.argmin(distsq)
        if distsq[index] < self._tolerance * self._tolerance:
            return index
        self._nodes.append((x, y))
        return len(self._nodes) - 1
        
    def add_path(self, path):
        """Add a new "path" to the graph.  A "path" has a start and end node,
        and possibly nodes in the middle.

        :param path: A list of coordinates `(x,y)` (or possibly `(x,y,z)` with
          the `z` to be ignored).  This is compatible with the `shapely`
          format.
        """
        path = list(path)
        for i in range(len(path) - 1):
            x1,y1,*z = path[i]
            x2,y2,*z = path[i + 1]
            self.add_edge(x1, y1, x2, y2)

    def add_edge(self, x1, y1, x2, y2):
        """Add an edge from `(x1, y1)` to `(x2, y2)`."""
        key1 = self._add_node(x1, y1)
        key2 = self._add_node(x2, y2)
        self._edges.append((key1, key2))
    
    def build(self):
        vertices = [ (key, x, y) for key, (x,y) in enumerate(self._nodes) ]
        return PlanarGraph(vertices, self.edges)


class PlanarGraphBuilder():
    """General purpose builder class.  Can be constructed from a
    :class:`PlanarGraph` instance; is designed for mutating a
    :class:`PlanarGraph` instance.

    :param graph: If not `None`, construct from the network in this instance.
    """
    def __init__(self, graph=None):
        if graph is None:
            self._vertices = dict()
            self._edges = []
        else:
            self._vertices = dict(graph.vertices)
            self._edges = list(graph.edges)

    @property
    def vertices(self):
        """Dictionary from key to coordinates `(x,y)`.
        Mutate to change the vertices."""
        return self._vertices

    @property
    def edges(self):
        """List of unordered edges `(key1, key2)`.  Mutate to change."""
        return self._edges

    def add_edge(self, key1, key2):
        self._edges.append((key1, key2))

    def set_vertex(self, key, x, y):
        self._vertices[key] = (x,y)

    def add_vertex(self, x, y):
        """Assumes that vertices are keyed by integers.
        
        :return: The key added.
        """
        if len(self.vertices) == 0:
            key = 0
        else:
            key = max(self.vertices.keys()) + 1
        self.set_vertex(key, x, y)
        return key
    
    def remove_unused_vertices(self):
        """Remove any vertex which is not part of an edge."""
        used_keys = set(k for k,_ in self._edges)
        used_keys.update(k for _,k in self._edges)
        for k in list(self._vertices.keys()):
            if k not in used_keys:
                del self._vertices[k]

    def build(self):
        verts = ((key,x,y) for key,(x,y) in self.vertices.items())
        return PlanarGraph(verts, self.edges)


class GraphBuilder():
    """General purpose builder class.  Can be constructed from a
    :class:`Graph` instance; is designed for mutating a
    :class:`Graph` instance.

    :param graph: If not `None`, construct from the network in this instance.
    """
    def __init__(self, graph=None):
        if graph is None:
            self._vertices = set()
            self._edges = []
            self._lengths = None
        else:
            self._vertices = set(graph.vertices)
            self._edges = list(graph.edges)
            if graph.lengths is None:
                self._length = None
            else:
                self._lengths = list(graph.lengths)

    @property
    def vertices(self):
        """Set of vertices.  Mutate to add a vertex."""
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    @property
    def edges(self):
        """List of unordered edges `(key1, key2)`.  Mutate to change."""
        return self._edges

    @edges.setter
    def edges(self, v):
        self._edges = v

    @property
    def lengths(self):
        """List of lengths of edges.  Or `None`."""
        return self._lengths
    
    @lengths.setter
    def lengths(self, v):
        self._lengths = v

    def add_edge(self, key1, key2):
        """Adds a edge.  Vertices automatically added."""
        self._vertices.update([key1, key2])
        self._edges.append((key1, key2))
        return self
    
    def remove_unused_vertices(self):
        """Remove any vertex which is not part of an edge."""
        used_keys = set(k for k,_ in self._edges)
        used_keys.update(k for _,k in self._edges)
        self._vertices.intersection_update(used_keys)

    def remove_duplicate_edges(self):
        if self.lengths is not None:
            raise ValueError("Cannot process if we have lengths")
        edges = set(frozenset(e) for e in self._edges)
        self._edges = [tuple(e) for e in edges]

    def build(self):
        if self.lengths is not None and len(self.edges) != len(self.lengths):
            raise ValueError("Lengths of edges and lengths disagree.")
        return Graph(self.vertices, self.edges, self.lengths)


class Graph():
    """A simple graph abstract class.
    
    - "Nodes" or "vertices" are simply keys given by any hashable Python object
      (typically, integers).
    - "Edges" are undirected links between two vertices.
    - Each edge can, optionally, have a "length" associated with it.
    
    We assume that the graph is "simple" (between two vertices there is at
    most one edge, and an edge is always between _distinct_ vertices).
    
    This class is immutable (at least by design: do not mutate the underlying
    dictionaries!)  See the static constructors and the builder classes for
    ways to construct a graph.    
    
    :param vertices: An iterables of of keys.
    :param edges: An iterable of (unordered) pairs `(key1, key2)`.
    """
    def __init__(self, vertices, edges, lengths=None):
        self._vertices = set()
        for key in vertices:
            if key in self._vertices:
                raise ValueError("Keys of vertices should be unique; but {} is repeated".format(key))
            self._vertices.add(key)
        self._edges = list()
        edges_set = set()
        for key1, key2 in edges:
            if key1 == key2:
                raise ValueError("Cannot have an edge from vertex {} to itself".format(key1))
            e = frozenset((key1, key2))
            if e in edges_set:
                raise ValueError("Trying to add a 2nd edge from {} to {}".format(key1, key2))
            edges_set.add(e)
            self._edges.append((key1, key2))
        if lengths is None:
            self._lengths = None
        else:
            self._lengths = _np.asarray(lengths)
            if len(self._lengths) != len(self._edges):
                raise ValueError("Should be as many lengths as edges.")
        self._precompute()
    
    def _precompute(self):
        self._neighbours = {k:set() for k in self._vertices}
        for key1, key2 in self.edges:
            self._neighbours[key1].add(key2)
            self._neighbours[key2].add(key1)
        for k in list(self._neighbours.keys()):
            self._neighbours[k] = list(self._neighbours[k])
            self._neighbours[k].sort()
    
        self._neighbourhood_edges = dict()
        for index, edge in enumerate(self.edges):
            for key in edge:
                if key not in self._neighbourhood_edges:
                    self._neighbourhood_edges[key] = []
                self._neighbourhood_edges[key].append(index)
        for k in list(self._neighbourhood_edges.keys()):
            self._neighbourhood_edges[k].sort()
            
        self._edges_inverse = dict()
        for index, edge in enumerate(self._edges):
            self._edges_inverse[edge] = (index, 1)
            e = (edge[1], edge[0])
            self._edges_inverse[e] = (index, -1)
    
    @property
    def vertices(self):
        """A set (do not mutate!) of `key`s."""
        return self._vertices

    @property
    def edges(self):
        """A list of unordered edges `(key1, key2)`."""
        return self._edges
    
    @property
    def number_edges(self):
        return len(self._edges)

    def length(self, edge_index):
        """If we have lengths, the length of this edge."""
        if self._lengths is None:
            raise ValueError("No lengths.")
        return self._lengths[edge_index]
    
    @property
    def lengths(self):
        """Array of lengths of each edge, or None."""
        return self._lengths
        
    def find_edge(self, key1, key2):
        """Find the edge in the list of edges.  Raises `KeyError` on failure
        to find.

        :return: `(index, order)` where `index` is into :attr:`edges` and
          `order==1` if `self.edges[index] == (key1, key2)` while
          `order==-1` if `self.edges[index] == (key2, key1)`.
        """
        return self._edges_inverse[(key1, key2)]

    def neighbours(self, vertex_key):
        """A list of all the neighbours of the given vertex"""
        return self._neighbours[vertex_key]
    
    def neighbourhood_edges(self, vertex_key):
        """A list of all the edges (as indicies into `self.edges`) incident
        with the given vertex."""
        return self._neighbourhood_edges[vertex_key]

    def degree(self, vertex_key):
        """The degree (number of neighbours) of the vertex."""
        return len(self.neighbourhood_edges(vertex_key))
    
    def paths_between(self, key_start, key_end, max_length=None):
        """Iterable yielding all paths which start and end at the given
        vertices, and which are of length at most `max_length`.  A path will
        never be cyclic.
        
        :return: Iterable yielding lists of vertices which start and end at
          the prescribed vertices, and do not feature repeats.
        """
        yield from self.paths_between_avoiding(key_start, key_end, [], max_length)
                    
    def edge_paths_between(self, edge_start, edge_end, max_length=None):
        """Iterable yielding paths which start in the middle of the starting
        edge, and end in the middle of the ending edge.  We only report the
        vertices visiting in the path.  The length calculation will ignore the
        starting ending edges.
        
        :param edge_start: Unordered pair `(key1, key2)` giving the starting
          edge.  Does not actually have to be an edge in the graph.
        :param edge_end: Unordered pair `(key1, key2)` giving the ending edge.
          Does not actually have to be an edge in the graph.

        :return: Iterable yielding lists of vertices which start and end at
          the prescribed edges, and do not feature repeats.
        """
        # Same as above, but need to not use both the start and end edges
        to_avoid = [edge_start, edge_end]
        yield from self.paths_between_avoiding(edge_start[0], edge_end[0], to_avoid, max_length)
        yield from self.paths_between_avoiding(edge_start[0], edge_end[1], to_avoid, max_length)
        yield from self.paths_between_avoiding(edge_start[1], edge_end[0], to_avoid, max_length)
        yield from self.paths_between_avoiding(edge_start[1], edge_end[1], to_avoid, max_length)

    def paths_between_avoiding(self, key_start, key_end, edges_to_avoid, max_length=None):
        """Iterable yielding all paths which start and end at the given
        vertices, and which are of length at most `max_length`.  A path will
        never be cyclic.  Furthermore, we avoid walking along any "edge" (in
        either direction) in the specified collection.
        
        :param edges_to_avoid: Iterable of pairs `(key1, key2)` giving edges
          to avoid.

        :return: Iterable yielding lists of vertices which start and end at
          the prescribed vertices, and do not feature repeats.
        """
        # Depth-first search of all partial paths
        todo = [ ([key_start], 0.0) ]
        to_avoid = list(edges_to_avoid)
        to_avoid.extend((a,b) for b,a in list(to_avoid))
        while len(todo) > 0:
            partial_path, current_length = todo.pop()
            end_key = partial_path[-1]
            if end_key == key_end:
                yield partial_path
                continue
            for edge_index in self.neighbourhood_edges(end_key):
                key1, key2 = self.edges[edge_index]
                if (key1, key2) in to_avoid:
                    continue
                if key2 == end_key:
                    key1, key2 = key2, key1
                new_length = current_length + self.length(edge_index)
                if max_length is not None and new_length > max_length:
                    continue
                if key2 not in partial_path:
                    todo.append((partial_path + [key2], new_length))

    def walk_from(self, key1, key2):
        """Start from the edge `(key1, key2)`, walk to `key1` and then search
        outwards where we do not visit the same vertex twice, and we do not
        use the edge `(key1, key2)`.  Depth-first search.

        Implemented as a coroutine:
        
            search = graph.walk_from(key1, key2)
            assert next(search) == ([key1], 0.0)
        
        To continue exploring this branch, use `next_path, length =
        search.send(True)` or to cancel that branch of the search tree, use
        `next_path, length = search.send(False)`.

        The ordering is that we walk the edge which appears _last_ in the
        :attr:`edges` list first.
        """
        todo = [ ([key1], 0.0) ]
        while len(todo) > 0:
            partial_path, current_length = todo.pop()
            okay = yield partial_path, current_length
            if not okay:
                continue
            end_key = partial_path[-1]
            for edge_index in self.neighbourhood_edges(end_key):
                k1, k2 = self.edges[edge_index]
                if (k1, k2) == (key1, key2) or (k2, k1) == (key1, key2):
                    continue
                if k2 == end_key:
                    k1, k2 = k2, k1
                new_length = current_length + self.length(edge_index)
                if k2 not in partial_path:
                    todo.append((partial_path + [k2], new_length))

    def walk_with_degrees(self, start_key, initial_avoid_key, max_length, max_degree):
        """Find all paths which start at `start_key`, do _not_ immediately
        go to `initial_avoid_key` (but they may later visit that vertex),
        do not self-intersect, and are of maximum length.
        
        Yields `(current_edge_index, previous_length, current_length,
        total_degree)` where `current_edge_index` is the edge we have just
        walked,
        `previous_length, current_length` gives the length of the walk up to
        the previous vertex, and up to the current vertex (i.e. the difference
        is the length of the edge we just walked) and total_degree is the
        product of (1 - degree) of all the previous vertices.
        
        Initially yields `(None, 0, 0, 1)`.
        
        Will never yield a path where `previous_length >= max_length` but will
        yield paths with `previous_length < max_length` and
        `current_length >= max_length`.
        
        We will never yield a path where `total_degree > max_degree`.
        """
        yield (None, 0, 0, 1)
        
        todo = []
        for i in self.neighbourhood_edges(start_key):
            k1, k2 = self.edges[i]
            if k2 == start_key:
                k1, k2 = k2, k1
            if k2 == initial_avoid_key:
                continue
            todo.append((start_key, i, 0, 1, {start_key}))
        
        while len(todo) > 0:
            old_vertex, edge_index, old_length, total_degree, vertices_in_path = todo.pop()
            k1, k2 = self.edges[edge_index]
            if k2 == old_vertex:
                k1, k2 = k2, k1
            new_vertex = k2
            new_length = old_length + self.length(edge_index)
            fact = self.degree(old_vertex) - 1
            new_total_degree = total_degree
            if fact > 0:
                new_total_degree *= fact
            new_vertices_in_path = set(vertices_in_path)
            new_vertices_in_path.add(new_vertex)
            if new_total_degree > max_degree:
                continue
            yield edge_index, old_length, new_length, new_total_degree
            if new_length >= max_length:
                continue
            for i in self.neighbourhood_edges(new_vertex):
                k1, k2 = self.edges[i]
                if k2 == new_vertex:
                    k1, k2 = k2, k1
                if k2 in new_vertices_in_path:
                    continue
                todo.append((new_vertex, i, new_length, new_total_degree, new_vertices_in_path))
            
    def partition_by_segments(self):
        """A "segment" is a maximal path `(k1,k2,...,kn)` where the degree of
        every vertex excepting `k1, kn` is 2.  This is well-defined, for if
        `kn` has degree 2, then we can add the unique neighbour which is not
        `k(n-1)` and get a longer segment.
        
        Returns a maximal list of maximal segments which must cover all edges.
        Any simple segment `(k1,k2)` will be returned in the same order as it
        appears in `edges`.
        
        Note that such a decomposition is not unique (for some graphs).
        
        :return: Yields segments
        """
        edges = set(self._edges)
        def remove(a, b):
            edges.discard((a,b))
            edges.discard((b,a))
        while len(edges) > 0:
            segment = edges.pop()
            while segment is not None:
                if self.degree(segment[0]) == 2:
                    nhood = set(self.neighbours(segment[0]))
                    assert len(nhood) == 2
                    nhood.discard(segment[1])
                    key = nhood.pop()
                    remove(key, segment[0])
                    segment = (key, ) + segment
                elif self.degree(segment[-1]) == 2:
                    nhood = set(self.neighbours(segment[-1]))
                    if not len(nhood) == 2:
                        raise AssertionError(segment[-1])
                    nhood.discard(segment[-2])
                    key = nhood.pop()
                    remove(key, segment[-1])
                    segment = segment + (key, )
                else:
                    yield segment
                    segment = None
                    

class PlanarGraph(Graph):
    """A simple graph class.
    
    - "Nodes" or "vertices" are (x,y) coordinates in the plane, but are also
      keyed by any hashable Python object (typically, integers).
    - "Edges" are undirected links between two vertices.
    
    We assume that the graph is "simple" (between two vertices there is at
    most one edge, and an edge is always between _distinct_ vertices).
    
    This class is immutable (at least by design: do not mutate the underlying
    dictionaries!)  See the static constructors and the builder classes for
    ways to construct a graph.    
    
    :param vertices: An iterables of triples `(key, x, y)`.
    :param edges: An iterable of (unordered) pairs `(key1, key2)`.
    """
    def __init__(self, vertices, edges):
        verts = dict()
        for key, x, y in vertices:
            if key in verts:
                raise ValueError("Keys of vertices should be unique; but {} is repeated".format(key))
            verts[key] = (x,y)
        super().__init__(verts.keys(), edges)
        self._vertices = verts
        quads = self.as_quads().T
        if len(quads) == 0:
            self._lengths = _np.asarray([])
        else:
            self._lengths = _np.sqrt((quads[0] - quads[2])**2 + (quads[1] - quads[3])**2)
            
    def project(self, projector):
        """Returns a new instance with the geometry projected according to
        `projector`.

        :param projector: Function object of signature `projector(x, y)` returning
          a pair of coordinates.  For example, a :class:`pyproj.Proj` object.

        :return: A new instance of :class:`PlanarGraph`.
        """
        vertices = [(k, *projector(*pt)) for k, pt in self._vertices.items()]
        return PlanarGraph(vertices, list(self._edges))

    def dump_bytes(self):
        """Write data to a `bytes` object.  The vertices are saved using the
        `numpy.save` method (which is portable and won't leave to floating
        point errors) and then `base64` encoded.  This data and other settings
        are written to a JSON payload.  This is compressed using `bz2` and
        returned.
        
        The keys need to be integers.
        """
        return _bz2.compress(self.dump_json().encode("UTF8"))
        
    def dump_json(self):
        """As :meth:`dump_bytes` but returns the JSON payload."""
        keys, xcs, ycs, edges = self._to_arrays()
        out = dict()
        for name, array in [("keys", keys), ("xcoords", xcs), ("ycoords", ycs),
                            ("edges", edges)]:
            with _io.BytesIO() as file:
                _np.save(file, array, allow_pickle=False)
                b = file.getvalue()
                out[name] = _base64.b64encode(b).decode("UTF8")
        return _json.dumps(out)
        
    @staticmethod
    def from_json(json):
        in_dict = _json.loads(json)
        keys = PlanarGraph._load_numpy_array(in_dict["keys"])
        xcs = PlanarGraph._load_numpy_array(in_dict["xcoords"])
        ycs = PlanarGraph._load_numpy_array(in_dict["ycoords"])
        edges = PlanarGraph._load_numpy_array(in_dict["edges"])
        return PlanarGraph(zip(keys, xcs, ycs), edges)

    @staticmethod
    def from_bytes(data):
        json = _bz2.decompress(data).decode("UTF8")
        return PlanarGraph.from_json(json)
        
    @staticmethod
    def _load_numpy_array(b64data):
        if isinstance(b64data, str):
            b64data = b64data.encode("UTF8")
        b = _base64.b64decode(b64data)
        with _io.BytesIO(b) as file:
            return _np.load(file)
    
    def _to_arrays(self):
        try:
            for x in self._vertices.keys():
                assert x == int(x)
        except:
            raise ValueError("Vertex keys need to be integers.")
        
        number_vertices = len(self._vertices)
        keys = _np.empty(number_vertices, dtype=_np.int)
        xcs = _np.empty(number_vertices)
        ycs = _np.empty(number_vertices)
        for i, (key, (x,y)) in enumerate(self._vertices.items()):
            keys[i] = key
            xcs[i] = x
            ycs[i] = y

        edges = _np.asarray(self._edges, dtype=_np.int)
        assert edges.shape == (len(self._edges), 2)

        return keys, xcs, ycs, edges
            
    @property
    def vertices(self):
        """A dictionary (do not mutate!) from `key` to planar coordinates
        `(x,y)`.
        """
        return self._vertices

    @property
    def bounds(self):
        """A bounding box of vertex locations, as `(xmin, ymin, xmax, ymax)`"""
        xcs = [x for x, _ in self._vertices.values()]
        ycs = [y for _, y in self._vertices.values()]
        return min(xcs), min(ycs), max(xcs), max(ycs)

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
        and "just works".  If installed, `rtree` will be used to accelerate
        the search.
        
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


class TimedNetworkPoints(_data.TimeStamps):
    """A variant of :class:`Data.TimedPoints` where each event has a location
    given by reference to a graph.

    :param timestamps: An array of timestamps (must be convertible to
      :class:`numpy.datetime64`).
    :param locations: An iterable of pairs `(edge, t)` where `edge` is a pair
      `(key1, key2)` describing an edge in a graph, and `0 <= t <= 1` gives the
      location along the edge.
    """
    def __init__(self, timestamps, locations):
        super().__init__(timestamps)
        self._start_keys = []
        self._end_keys = []
        distances = []
        for ((key1, key2), t) in locations:
            self._start_keys.append(key1)
            self._end_keys.append(key2)
            distances.append(float(t))
        self._start_keys = _np.asarray(self._start_keys)
        self._end_keys = _np.asarray(self._end_keys)
        self._distances = _np.asarray(distances)
        if len(distances) != len(self.timestamps):
            raise ValueError("Number of locations should match the number of timestamps")

    @staticmethod
    def project_timed_points(timed_points, graph):
        """Construct a new instance by projecting coordinates onto a graph.

        :param timed_points: Instance of :class:`data.TimedPoints`
        :param graph: Instance of :class:`PlanarGraph`
        """
        def projector():
            for x, y in zip(timed_points.xcoords, timed_points.ycoords):
                yield graph.project_point_to_graph(x, y)
        return TimedNetworkPoints(timed_points.timestamps, projector())

    @property
    def distances(self):
        """Array of distances along each edge."""
        return self._distances

    @property
    def start_keys(self):
        """List of vertices which form the start of the edges"""
        return self._start_keys

    @property
    def end_keys(self):
        """List of vertices which form the end of the edges"""
        return self._end_keys

    def __getitem__(self, index):
        if isinstance(index, int):
            return [self.timestamps[index], self.start_keys[index], self.end_keys[index], self.distances[index]]
        # Assume slice-like object
        data = list(zip(self.timestamps[index], self.start_keys[index],
                self.end_keys[index], self.distances[index]))
        data.sort(key = lambda tup : tup[0])
        new_times = [t for t,_,_,_ in data]
        locations = [((k1,k2),t) for _,k1,k2,t in data]
        return TimedNetworkPoints(new_times, locations)

    def to_timed_points(self, graph):
        """Use the graph object to convert to absolute coordinates.list

        :param graph: Instance of :class:`PlanarGraph` to use.

        :return: Instance of :class:`data.TimedPoints`
        """
        xcs, ycs = [], []
        for key1, key2, t in zip(self.start_keys, self.end_keys, self.distances):
            x, y = graph.edge_to_coords(key1, key2, t)
            xcs.append(x)
            ycs.append(y)
        return _data.TimedPoints.from_coords(self.timestamps, xcs, ycs)


def approximately_equal(graph1, graph2, tolerance=0.1):
    """Do the two graphs represent the same edges, where nodes are allowed to
    vary by tolerance?  Applies a greedy algorithm, so will be falsely
    negative in rare edge cases."""
    lines1 = list(graph1.as_quads())
    lines2 = list(graph2.as_quads())
    if len(lines1) != len(lines2):
        return False
    
    cutoff = 2 * tolerance * tolerance
    for li in lines1:
        lines = _np.asarray(lines2).T
        distsq = ((lines[0] - li[0])**2 + (lines[1] - li[1])**2 +
                  (lines[2] - li[2])**2 + (lines[3] - li[3])**2)
        index = _np.argmin(distsq)
        if distsq[index] < cutoff:
            del lines2[index]
        else:
            return False
    return True

def simple_reduce_graph(graph):
    """Build a new graph where we have deleted all vertices of degree 2, while
    maintaining our condition that the graph has to be simple (so we do not
    delete a degree 2 vertex if that would lead to a double edge).

    :return: The new graph.
    """
    neighbours = _collections.defaultdict(set)
    for k1, k2 in graph.edges:
        neighbours[k1].add(k2)
        neighbours[k2].add(k1)
    for key in list(neighbours.keys()):
        nhood = list(neighbours[key])
        if len(nhood) == 2 and nhood[1] not in neighbours[nhood[0]]:
            del neighbours[key]
            neighbours[nhood[0]].remove(key)
            neighbours[nhood[0]].add(nhood[1])
            neighbours[nhood[1]].remove(key)
            neighbours[nhood[1]].add(nhood[0])

    builder = GraphBuilder()
    builder.vertices.update(neighbours.keys())
    for key in list(neighbours.keys()):
        for x in neighbours[key]:
            builder.add_edge(key, x)
            neighbours[x].discard(key)
    return builder.build()

def to_derived_graph(graph, use_edge_indicies=False):
    """Construct the "derived graph" of `graph`.  Each vertex is an edge from
    the original graph, and two vertices are adjacent if the original edges
    shared a vertex in the original graph.  If the original graph had lengths
    then we calculate lengths in the new graph by computing the distance from
    mid-points of original edges (i.e. take the average of the lengths of the
    edges involved).

    :param graph: Instance of :class:`Graph` or a subclass
    :param use_edge_indicies: If `True` then label the old edges by their
      index; default if `False` which is to label the old edges by the vertices
      they are between.

    :return: Instance of :class:`Graph`
    """
    builder = GraphBuilder()
    if use_edge_indicies:
        builder.vertices.update(range(len(graph.edges)))
    else:
        builder.vertices.update(graph.edges)
    lengths = []
    for i, edge1 in enumerate(graph.edges):
        for key in edge1:
            for j in graph.neighbourhood_edges(key):
                if j > i:
                    edge2 = graph.edges[j]
                    if use_edge_indicies:
                        builder.add_edge(i, j)
                    else:
                        builder.add_edge(edge1, edge2)
                    if graph.lengths is not None:
                        lengths.append((graph.lengths[i] + graph.lengths[j])/2)
    if graph.lengths is not None:
        builder.lengths = lengths
    return builder.build()

def shortest_paths(graph, vertex_key):
    """Uses Dijkstra's algorithm to find the shortest path from
    `vertex_key` to all other vertices.  If we have no lengths, then each
    edge has length 1.
    
    :return: `(lengths, prevs)` where `lengths` is a dictionary from key
        to length.  A length of -1 means that the vertex is not connected to
        `vertex_key`.  `prevs` is a dictionary from key to key, giving for
        each vertex the previous vertex in the path from `vertex_key` to that
        vertex.  Working backwards, you can hence construct all shortest
        paths.
    """
    shortest_length = { k : -1 for k in graph.vertices }
    shortest_length[vertex_key] = 0
    candidates = {vertex_key}
    done = set()
    prevs = {vertex_key:vertex_key}
    while len(candidates) > 0:
        next_vertex, min_dist = None, -1
        for v in candidates:
            dist = shortest_length[v]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                next_vertex = v
        candidates.discard(next_vertex)
        done.add(next_vertex)

        for v in graph.neighbours(next_vertex):
            edge_index, _ = graph.find_edge(next_vertex, v)
            dist = min_dist + graph.length(edge_index)
            current_dist = shortest_length[v]
            if current_dist == -1 or current_dist > dist:
                shortest_length[v] = dist
                prevs[v] = next_vertex
            if v not in done:
                candidates.add(v)

    return shortest_length, prevs

def shortest_edge_paths(graph, edge_index, position=0.5):
    """Find the shortest path from the edge given
    by `edge_index`.  If we have no lengths, then each edge has length 1.
    This could be achieved by using the "derived graph", but our use will also
    require knowing the _vertex_ degree of the path.

    We use a simple modification of Dijkstra's algorithm whereby the initial
    distance to 
    
    :param graph: :class:`Graph` to use
    :param edge_index: The edge to start on
    :param position: `0 <= t <= 1` along the edge to start at.  Defaults
      to the midpoint.

    :return: `(lengths, prevs)` where `lengths` is a dictionary from key
        to length.  If a key is not present, it means that vertex is not
        connected to `vertex_key`.  `prevs` is a dictionary from key to key,
        giving for each vertex the previous vertex in the path from
        `vertex_key` to that vertex.  Working backwards, you can hence
        construct all shortest paths.  These paths will end at either vertex
        of the initial edge.
    """
    shortest_length = dict()
    v1, v2 = graph.edges[edge_index]
    shortest_length[v1] = graph.length(edge_index) * position
    shortest_length[v2] = graph.length(edge_index) * (1 - position)
    candidates = {v1, v2}
    done = set()
    prevs = {v1:v1, v2:v2}
    while len(candidates) > 0:
        next_vertex, min_dist = None, -1
        for v in candidates:
            dist = shortest_length[v]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                next_vertex = v
        candidates.discard(next_vertex)
        done.add(next_vertex)

        for v in graph.neighbours(next_vertex):
            edge_index, _ = graph.find_edge(next_vertex, v)
            dist = min_dist + graph.length(edge_index)
            if v not in shortest_length or shortest_length[v] > dist:
                shortest_length[v] = dist
                prevs[v] = next_vertex
            if v not in done:
                candidates.add(v)

    return shortest_length, prevs

def shortest_edge_paths_with_degrees(graph, edge_index):
    """Find the shortest paths between the middle of the `edge_index` to each
    other edge.  Also computes the "cumulative degree" of each path: that is,
    the product of `max(1, degree - 1)` over each vertex in the path.

    :return: `(distances, degrees)` where `distances` is an array corresponding
      to `graph.edges`, as is `degrees`.  Edges which are disconnected from
      `edge_index` will have `degress == 1` and `distances == -1`.
    """
    shortest_length = dict()
    v1, v2 = graph.edges[edge_index]
    shortest_length[v1] = graph.length(edge_index) * 0.5
    shortest_length[v2] = shortest_length[v1]
    candidates = {v1, v2}
    done = set()
    degrees = dict()
    degrees[v1] = max(1, graph.degree(v1) - 1)
    degrees[v2] = max(1, graph.degree(v2) - 1)
    while len(candidates) > 0:
        next_vertex, min_dist = None, -1
        for v in candidates:
            dist = shortest_length[v]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                next_vertex = v
        candidates.discard(next_vertex)
        done.add(next_vertex)

        for v in graph.neighbours(next_vertex):
            i, _ = graph.find_edge(next_vertex, v)
            dist = min_dist + graph.length(i)
            if v not in shortest_length or shortest_length[v] > dist:
                shortest_length[v] = dist
                degrees[v] = degrees[next_vertex] * max(1, graph.degree(v) - 1)
            if v not in done:
                candidates.add(v)

    distances = _np.empty(graph.number_edges)
    cum_degrees = _np.empty(graph.number_edges)
    for i in range(graph.number_edges):
        if i == edge_index:
            distances[i] = 0
            cum_degrees[i] = 1
            continue
        k1, k2 = graph.edges[i]
        le = graph.length(i) * 0.5
        if k1 not in shortest_length:
            distances[i] = -1
            cum_degrees[i] = 1
            continue
        le1, le2 = shortest_length[k1], shortest_length[k2]
        if le1 < le2:
            distances[i] = le1 + le
            cum_degrees[i] = degrees[k1]
        else:
            target = k2
            distances[i] = le2 + le
            cum_degrees[i] = degrees[k2]
    return distances, cum_degrees

def segment_graph(graph):
    """Partition the edges of a graph into "segments", where a segment is a
    maximal path where each "internal" vertex has degree two.  That is, a
    segment is an equivalence class for the equivalence relation defined by
    saying that two edges are equivalent if there is a path between them with
    each vertex in the path having degree 2.
    
    :return: A generator of segments, where each segment is a set of indicies
      into `graph.edges`.
    """
    edges_done = set()
    for edge_index, (v1, v2) in enumerate(graph.edges):
        if edge_index in edges_done:
            continue
        segment = {edge_index}
        todo = {v1, v2}
        vertices_visited = set()
        while len(todo) > 0:
            v = todo.pop()
            vertices_visited.add(v)
            edges = graph.neighbourhood_edges(v)
            if len(edges) == 2:
                for e in edges:
                    segment.add(e)
                    for u in graph.edges[e]:
                        if u not in vertices_visited:
                            todo.add(u)
        edges_done.update(segment)
        yield segment

def edge_segment_to_vertices(graph, segment):
    """Turn a set of edges which form a "segment" (e.g. an output of
    :func:`segment_graph`) into an ordered list of vertices giving the
    same segment.

    :return: A list of vertex keys, giving the path formed by the segment.
      Will include the start/end vertices of the path (i.e. vertices not of
      degree 2); if the segment is a cycle, then the start and end vertices
      will be the same.
    """
    seg = set(segment)
    v1, v2 = graph.edges[seg.pop()]
    ordered_vertices = [v1, v2]
    while len(seg) > 0:
        added = False
        choices = [x for x in graph.neighbourhood_edges(ordered_vertices[0]) if x in seg]
        if len(choices) > 0:
            seg.remove(choices[0])
            v1, v2 = graph.edges[choices[0]]
            v = v1 if ordered_vertices[0] == v2 else v2
            ordered_vertices.insert(0, v)
            added = True
        choices = [x for x in graph.neighbourhood_edges(ordered_vertices[-1]) if x in seg]
        if len(choices) > 0:
            seg.remove(choices[0])
            v1, v2 = graph.edges[choices[0]]
            v = v1 if ordered_vertices[-1] == v2 else v2
            ordered_vertices.append(v)
            added = True
    return ordered_vertices

def ordered_segment_graph(graph):
    """As :func:`segment_graph` but yields actual paths, in the same order.
    
    :return: A generator of segments, where is segment is a list of vertex
      keys, giving the path formed by the segment.  Will include the start/end
      vertices of the path (i.e. vertices not of degree 2); if the segment is a
      cycle, then the start and end vertices will be the same.
    """
    for segment in segment_graph(graph):
        yield edge_segment_to_vertices(graph, segment)

def connected_components(graph):
    """Find all connected components (and isolated vertices) in the graph.
    
    :return: Generated sets of maximally connected vertices.
    """
    done = set()
    for v in graph.vertices:
        if v in done:
            continue
        com = set()
        todo = [v]
        while len(todo) > 0:
            next_v = todo.pop()
            if next_v in com:
                continue
            com.add(next_v)
            todo.extend(graph.neighbours(next_v))
        done.update(com)
        yield com
