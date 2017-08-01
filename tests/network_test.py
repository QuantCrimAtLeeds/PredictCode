import pytest
import unittest.mock as mock

import open_cp.network as network
import numpy as np

@pytest.fixture
def planar_graph_geo_builder():
    b = network.PlanarGraphGeoBuilder()
    b.add_path([(0,0),(1,1),(5.1,1.2)])
    b.add_path([(2,0),(1,1),(0,5),(5.1,1.2)])
    b.add_path([(0,0),(0,5)])
    return b    

def test_PlanarGraphGeoBuilder(planar_graph_geo_builder):
    b = planar_graph_geo_builder
    assert b.coord_nodes == {(0,0):[0], (1,1):[1,4], (5.1,1.2):[2],
                             (2,0):[3], (0,5):[5]}
    assert b.edges == [(0,1), (1,2), (3,4), (4,5), (5,2), (0,5)]

def test_PlanarGraphGeoBuilder_builds(planar_graph_geo_builder):
    g = planar_graph_geo_builder.build()

    assert g.vertices == {0:(0,0), 1:(1,1), 2:(5.1,1.2), 3:(2,0), 4:(1,1), 5:(0,5)}        
    assert g.edges == [(0,1), (1,2), (3,4), (4,5), (5,2), (0,5)]

@pytest.fixture
def planar_graph_node_builder():
    b = network.PlanarGraphNodeBuilder()
    b.add_path([(0,0),(1,1),(5.1,1.2)])
    b.add_edge(0,0,2,2)
    b.add_path([(1,0),(1,1),(2,2)])
    return b

def test_PlanarGraphNodeBuilder(planar_graph_node_builder):
    b = planar_graph_node_builder
    assert b.coord_nodes == [(0,0), (1,1), (5.1,1.2), (2,2), (1,0)]
    assert b.edges == [(0,1), (1,2), (0,3), (4,1), (1,3)]

def test_PlanarGraphNodeBuilder_builds(planar_graph_node_builder):
    g = planar_graph_node_builder.build()
    
    assert g.vertices == {0:(0,0), 1:(1,1), 2:(5.1,1.2), 3:(2,2), 4:(1,0)}
    assert g.edges == [(0,1), (1,2), (0,3), (4,1), (1,3)]

def test_PlanarGraphNodeBuilder_tolerance():
    b = network.PlanarGraphNodeBuilder()
    b.tolerance = 0.2
    assert b.tolerance == pytest.approx(0.2)
    
    b.add_path([(0,0),(1,1),(5.1,1.2)])
    b.add_edge(0.1,0.01,2,2)
    
    assert b.coord_nodes == [(0,0), (1,1), (5.1,1.2), (2,2)]
    assert b.edges == [(0,1), (1,2), (0,3)]

def test_PlanarGraphNodeOneShot():
    nodes = [(0,0), (1,1), (5.1,1.2), (0.1,0.01), (2,2)]
    b = network.PlanarGraphNodeOneShot(nodes, 0.2)
    
    b.add_path([(0,0),(1,1),(5.1,1.2)])
    b.add_edge(0.1,0.01,2,2)
    
    g = b.build()
    assert set(g.vertices.values()) == {(0,0), (1,1), (5.1,1.2), (2,2)}
    assert len(g.edges) == 3
    assert [g.vertices[x] for x in g.edges[0]] == [(0,0), (1,1)]
    assert [g.vertices[x] for x in g.edges[1]] == [(1,1), (5.1,1.2)]
    assert [g.vertices[x] for x in g.edges[2]] == [(0,0), (2,2)]

def test_PlanarGraph_constructs():
    with pytest.raises(ValueError):
        network.PlanarGraph([(0,1,2), (0,2,3)], [])
        
    with pytest.raises(ValueError):
        network.PlanarGraph([(0,1,2), (1,2,3)], [(0,0)])
        
    g = network.PlanarGraph([(0,1,2), (1,2,3)], [(0,1)])
    assert g.vertices == {0:(1,2), 1:(2,3)}
    assert g.edges == [(0,1)]

@pytest.fixture
def graph1():
    b = network.PlanarGraphGeoBuilder()
    b.add_path([(0,0), (10,0)])
    b.add_path([(0,1), (5,5), (9,1)])
    return b.build()

def test_PlanarGraph_lengths(graph1):
    assert graph1.length(0) == pytest.approx(10)
    assert graph1.length(1) == pytest.approx(np.sqrt(25+16))
    assert graph1.length(2) == pytest.approx(np.sqrt(32))
    
def test_PlanarGraph_as_quads(graph1):
    exp = [ (0,0,10,0), (0,1,5,5), (5,5,9,1) ]
    x = graph1.as_quads()
    np.testing.assert_allclose(x, exp)
    
def test_PlanarGraph_as_lines(graph1):
    exp = [ ((0,0),(10,0)), ((0,1),(5,5)), ((5,5),(9,1)) ]
    x = graph1.as_lines()
    np.testing.assert_allclose(x, exp)

def test_PlanarGraph_project(graph1):
    edge, t = graph1.project_point_to_graph(5,1) 
    assert edge == (0, 1)
    assert t == pytest.approx(0.5)

    edge, t = graph1.project_point_to_graph(-0.5, -0.5)
    assert edge == (0, 1)
    assert t == 0
    
    edge, t = graph1.project_point_to_graph(-0.1, 1)
    assert edge == (2, 3)
    assert t == 0
    
    edge, t = graph1.project_point_to_graph(5, 5.2)
    assert (edge, t) == ((2,3), 1) or (edge, t) == ((3,4), 0)

    edge, t = graph1.project_point_to_graph(9, .4)
    assert edge == (0, 1)
    assert t == pytest.approx(0.9)

    edge, t = graph1.project_point_to_graph(9, .6)
    assert edge == (3, 4)
    assert t == 1

    edge, t = graph1.project_point_to_graph(2.5, 2)
    assert edge == (2, 3)
    assert t == pytest.approx(0.402439024)

@pytest.fixture
def graph2():
    b = network.PlanarGraphGeoBuilder()
    b.add_path([(0,10), (1,10)])
    b.add_path([(1,10), (2,11), (3, 11), (4,10)])
    b.add_path([(1,10), (2,9), (3, 9), (4,10)])
    b.add_path([(2,9), (2,11)])
    b.add_path([(4,10), (5,10)])
    return b.build()

def test_graph2(graph2):
    assert graph2.vertices == {0:(0,10), 1:(1,10), 2:(2,11), 3:(3,11), 4:(4,10),
                               5:(2,9), 6:(3,9), 7:(5,10)}
    assert graph2.edges == [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,4), (5,2), (4,7)]

def test_PlanarGraph_neighbours(graph2):
    assert graph2.neighbours(0) == [1]
    assert graph2.neighbours(1) == [0,2,5]
    assert graph2.neighbours(2) == [1,3,5]
    assert graph2.neighbours(3) == [2,4]
    assert graph2.neighbours(4) == [3,6,7]
    assert graph2.neighbours(5) == [1,2,6]
    assert graph2.neighbours(6) == [4,5]
    assert graph2.neighbours(7) == [4]
    
def test_PlanarGraph_neighbourhood_edges(graph2):
    assert graph2.neighbourhood_edges(0) == [0]
    assert graph2.neighbourhood_edges(1) == [0,1,4]
    assert graph2.neighbourhood_edges(2) == [1,2,7]
    assert graph2.neighbourhood_edges(3) == [2,3]
    assert graph2.neighbourhood_edges(4) == [3,6,8]
    assert graph2.neighbourhood_edges(5) == [4,5,7]
    assert graph2.neighbourhood_edges(6) == [5,6]
    assert graph2.neighbourhood_edges(7) == [8]
    
def test_PlanarGraph_neighbourhood_paths_between(graph2):
    assert list(graph2.paths_between(0,1,10000)) == [[0,1]]
    
    out = [ tuple(x) for x in graph2.paths_between(0,2,10000) ]
    assert len(set(out)) == len(out)
    assert set(out) == {(0,1,2), (0,1,5,2), (0,1,5,6,4,3,2)}
    
    out = [ tuple(x) for x in graph2.paths_between(0,3,10000) ]
    assert len(set(out)) == len(out)
    assert set(out) == {(0,1,2,3), (0,1,5,2,3), (0,1,5,6,4,3), (0,1,2,5,6,4,3)}
    
    out = [ tuple(x) for x in graph2.paths_between(0,7,10000) ]
    assert len(set(out)) == len(out)
    assert set(out) == {(0,1,2,3,4,7), (0,1,2,5,6,4,7), (0,1,5,6,4,7), (0,1,5,2,3,4,7)}
    
    # TODO: With length bounds...
    