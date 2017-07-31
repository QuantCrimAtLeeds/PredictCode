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
