import pytest
import unittest.mock as mock

import open_cp.network_hotspot as network_hotspot
import open_cp.network
import open_cp.data
import datetime
import numpy as np

@pytest.fixture
def graph():
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(0,0)
    b.add_vertex(1,0)
    b.add_vertex(1,1)
    b.add_vertex(0,1)
    b.add_edge(0,1)
    b.add_edge(1,2)
    b.add_edge(2,3)
    b.add_edge(3,0)
    return b.build()

def test__GraphSplitter(graph):
    g = network_hotspot._GraphSplitter(graph, 10).split()
    open_cp.network.approximately_equal(g, graph)

    g = network_hotspot._GraphSplitter(graph, 1).split()
    open_cp.network.approximately_equal(g, graph)

    g = network_hotspot._GraphSplitter(graph, 0.6).split()
    assert g.vertices == {0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1),
        4:(0.5,0), 5:(1,0.5), 6:(0.5,1), 7:(0,0.5)}
    assert g.edges == [(0,4), (4,1), (1,5), (5,2), (2,6), (6,3), (3,7), (7,0)]

    g1 = network_hotspot._GraphSplitter(graph, 0.5).split()
    open_cp.network.approximately_equal(g, g1)

    g2 = network_hotspot._GraphSplitter(graph, 0.4).split()
    assert len(g2.edges) == 3 * 4

@pytest.fixture
def points():
    times = [datetime.datetime(2017,8,7,11,30), datetime.datetime(2017,8,7,12,30)]
    xcs = [0.2, 0.9]
    ycs = [0.1, 0.6]
    return open_cp.data.TimedPoints.from_coords(times, xcs, ycs)

def test_Trainer(graph, points):
    trainer = network_hotspot.Trainer(graph, 10)
    trainer.data = points
    predictor = trainer.compile()

    open_cp.network.approximately_equal(predictor.graph, graph)
    np.testing.assert_allclose([0,0],
        (points.timestamps - predictor.network_timed_points.timestamps) / np.timedelta64(1,"s"))
    np.testing.assert_allclose(predictor.network_timed_points.start_keys, [0,1])
    np.testing.assert_allclose(predictor.network_timed_points.end_keys, [1,2])
    np.testing.assert_allclose([0.2, 0.6], predictor.network_timed_points.distances)

def test_ConstantTimeKernel():
    tk = network_hotspot.ConstantTimeKernel()
    assert tk(5) == pytest.approx(1)
    assert tk(7) == pytest.approx(1)
    np.testing.assert_allclose(tk([5,7]), [1,1])

def test_ExponentialTimeKernel():
    tk = network_hotspot.ExponentialTimeKernel(1)
    assert tk(5) == pytest.approx(np.exp(-5))
    assert tk(1) == pytest.approx(np.exp(-1))
    np.testing.assert_allclose(tk([5,1]), np.exp(-np.array([5,1])))

    tk = network_hotspot.ExponentialTimeKernel(2)
    assert tk(5) == pytest.approx(np.exp(-2.5)/2)
    assert tk(1) == pytest.approx(np.exp(-0.5)/2)
    np.testing.assert_allclose(tk([5,1]), np.exp(-np.array([5,1])/2)/2)
    
def test_QuadDecayTimeKernel():
    tk = network_hotspot.QuadDecayTimeKernel(1)
    assert tk(1) == pytest.approx(1/(1+1) * 2 / np.pi)
    assert tk(3) == pytest.approx(1/(1+9) * 2 / np.pi)
    np.testing.assert_allclose(tk([1,3]), 1 / np.asarray([2,10]) * 2 / np.pi)

    tk = network_hotspot.QuadDecayTimeKernel(2)
    assert tk(1) == pytest.approx(1/(1+0.25) / np.pi)
    assert tk(3) == pytest.approx(1/(1+2.25) / np.pi)

def test_NetworkKernel():
    class TestNetworkKernel(network_hotspot.NetworkKernel):
        def __call__(self, x):
            x = np.asarray(x)
            if len(x.shape) == 0:
                if x <= 1:
                    return 1
                return 0
            mask = (x <= 1)
            return mask.astype(np.float)
        
    kernel = TestNetworkKernel()
    assert kernel(0.5) == 1
    assert kernel(1.0) == 1
    assert kernel(1.1) == 0
    np.testing.assert_allclose(kernel([0.5,1.0,1.1]), [1,1,0])
    
    assert kernel.integrate(0.2, 0.5) == pytest.approx(0.3)
    assert kernel.integrate(0, 1) == pytest.approx(1)
    assert kernel.integrate(0, 2) == pytest.approx(2)
    assert kernel.integrate(0, 3) == pytest.approx(0)
    np.testing.assert_allclose(kernel.integrate([0.2, 0, 0, 0], [0.5, 1, 2, 3]), [0.3, 1, 2, 0])

def test_TriangleKernel():
    kernel = network_hotspot.TriangleKernel(1)
    assert kernel.cutoff == pytest.approx(1)
    assert kernel(1) == 0
    assert kernel(2) == 0
    assert kernel(0.5) == pytest.approx(0.5)
    assert kernel(0) == pytest.approx(1)
    np.testing.assert_allclose(kernel([1,2,0.5,0]), [0,0,0.5,1])

    kernel = network_hotspot.TriangleKernel(5)
    assert kernel.cutoff == pytest.approx(5)
    assert kernel(10) == 0
    assert kernel(4) == pytest.approx(1/25)
    assert kernel(1) == pytest.approx(4/25)
    assert kernel(0) == pytest.approx(1/5)
    np.testing.assert_allclose(kernel([10,4,1,0]), [0, 1/25, 4/25, 1/5])

    assert kernel.integrate(0, 5) == pytest.approx(0.5)
    assert kernel.integrate(0, 6) == pytest.approx(0.5)
    assert kernel.integrate(1, 2) == pytest.approx(7/50)
    assert kernel.integrate(5, 10) == pytest.approx(0)
    
    np.testing.assert_allclose(kernel.integrate([0], [5]), [0.5])
    np.testing.assert_allclose(kernel.integrate([0,0,1,5], [5,6,2,10]), [0.5,0.5,7/50,0])

@pytest.fixture
def netpoints():
    times = [datetime.datetime(2017,8,7,11,30), datetime.datetime(2017,8,7,12,30)]
    locations = [ ((0,1), 0.2), ((2,1), 0.3) ]
    return open_cp.network.TimedNetworkPoints(times, locations)

def test_Predictor_add(graph):
    pred = network_hotspot.Predictor(None, graph)
    pred.kernel = network_hotspot.TriangleKernel(5)

    risks = [0,0,0,0]
    pred.add(risks, 0, 1, 0.2)
    assert risks[0] == pytest.approx(0.4 * (0.4 - 0.8/25))
    assert risks[1] == pytest.approx(pred.kernel.integrate(0.8, 1.8))
    assert risks[2] == pytest.approx(pred.kernel.integrate(1.8, 2.8))
    assert risks[3] == pytest.approx(pred.kernel.integrate(2.8, 3.8))

    risks = [0,0,0,0]
    pred.add(risks, 0, -1, 0.8)
    assert risks[0] == pytest.approx(pred.kernel.integrate(0, 0.2))
    assert risks[3] == pytest.approx(pred.kernel.integrate(0.2, 1.2))
    assert risks[2] == pytest.approx(pred.kernel.integrate(1.2, 2.2))
    assert risks[1] == pytest.approx(pred.kernel.integrate(2.2, 3.2))

def test_Predictor(graph, netpoints):
    pred = network_hotspot.Predictor(netpoints, graph)
    pred.kernel = network_hotspot.TriangleKernel(0.2)
    pred.time_kernel = network_hotspot.ConstantTimeKernel()
    result = pred.predict()

    assert result.graph is pred.graph
    assert result.risks[0] == pytest.approx(1)
    assert result.risks[1] == pytest.approx(1)

def test_Predictor_with_time_kernel(graph, netpoints):
    pred = network_hotspot.Predictor(netpoints, graph)
    pred.kernel = network_hotspot.TriangleKernel(0.2)
    pred.time_kernel = network_hotspot.ExponentialTimeKernel(1)
    result = pred.predict()

    assert result.graph is pred.graph
    assert result.risks[0] == pytest.approx(1 * np.exp(-1/24))
    assert result.risks[1] == pytest.approx(1)

def test_FastPredictor(graph, netpoints):
    pred = network_hotspot.Predictor(netpoints, graph)
    pred.kernel = network_hotspot.TriangleKernel(0.2)
    pred.time_kernel = network_hotspot.ExponentialTimeKernel(1)
    fast_pred = network_hotspot.FastPredictor(pred, 100)
    result = fast_pred.predict()
    
    assert result.graph is pred.graph
    assert result.risks[0] == pytest.approx(1 * np.exp(-1/24))
    assert result.risks[1] == pytest.approx(1)

@pytest.fixture
def graph2():
    b = open_cp.network.PlanarGraphGeoBuilder()
    b.add_path([(0,10), (1,10)])
    b.add_path([(1,10), (2,11), (3, 11), (4,10)])
    b.add_path([(1,10), (2,9), (3, 9), (4,10)])
    b.add_path([(2,9), (2,11)])
    b.add_path([(4,10), (5,10)])
    return b.build()

def test_Predictor_add_split(graph2):
    pred = network_hotspot.Predictor(None, graph2)
    pred.kernel = mock.Mock()
    pred.kernel.integrate.return_value = 1.0
    pred.kernel.cutoff = 3

    risks = np.zeros(9)
    pred.add(risks, 0, 1, 0)
    np.testing.assert_allclose(risks, [1, 0.5, 0.25, 0, 0.5, 0.25, 0, 0.5, 0])

    risks = np.zeros(9)
    pred.add(risks, 7, -1, 0.5)
    np.testing.assert_allclose(risks, [0.25, 0.25, 0, 0, 0.5, 0.5, 0.5, 1, 0])

def test_FastPredictor_add_split(graph2):
    pred = network_hotspot.Predictor(None, graph2)
    pred.kernel = mock.Mock()
    pred.kernel.integrate.return_value = 1.0
    pred.kernel.cutoff = 3
    pred = network_hotspot.FastPredictor(pred, 100)

    risks = np.zeros(9)
    pred.add(risks, 0, 1, 0)
    np.testing.assert_allclose(risks, [1, 0.5, 0.25, 0, 0.5, 0.25, 0, 0.5, 0])

    risks = np.zeros(9)
    pred.add(risks, 7, -1, 0.5)
    np.testing.assert_allclose(risks, [0.25, 0.25, 0, 0, 0.5, 0.5, 0.5, 1, 0])

def test_Result_coverage(graph2):
    risks = [0,5,4,3,2,6,7,1,8]
    result = network_hotspot.Result(graph2, risks)

    r1 = result.coverage(10)
    assert r1.graph.number_edges == 2
    assert r1.graph.edges[0] == (4,7)
    assert r1.graph.edges[1] == (6,4)
    np.testing.assert_allclose(r1.risks, [8,7])

    r2 = result.coverage(21)
    assert r2.graph.number_edges == 3
    assert r2.graph.edges[0] == (4,7)
    assert r2.graph.edges[1] == (6,4)
    assert r2.graph.edges[2] == (5,6)
    np.testing.assert_allclose(r2.risks, [8,7,6])
    
def test_ApproxPredictor(graph):
    pred = network_hotspot.Predictor(None, graph)
    pred.kernel = mock.Mock()
    pred.kernel.return_value = 1.0
    pred = network_hotspot.ApproxPredictor(pred)

    risks = np.asarray([0,0,0,0], dtype=np.float)
    pred.add_edge(risks, 0, None, 1)
    np.testing.assert_allclose(risks, [1,1,1,1])
    assert pred.kernel.call_args_list == [mock.call(0), mock.call(1.0), mock.call(2.0), mock.call(1.0)]

def test_ApproxPredictorCaching(graph):
    pred = network_hotspot.Predictor(None, graph)
    pred.kernel = mock.Mock()
    pred.kernel.return_value = 1.0
    pred = network_hotspot.ApproxPredictorCaching(pred)

    risks = np.asarray([0,0,0,0], dtype=np.float)
    pred.add_edge(risks, 0, None, 1)
    np.testing.assert_allclose(risks, [1,1,1,1])

def test_ApproxPredictor2(graph2):
    sq2 = np.sqrt(2)
    pred = network_hotspot.Predictor(None, graph2)
    pred.kernel = mock.Mock()
    pred.kernel.return_value = 1.0
    pred = network_hotspot.ApproxPredictor(pred)
    
    risks = np.asarray([0]*9, dtype=np.float)
    pred.add_edge(risks, 0, None, 1)
    np.testing.assert_allclose(risks, [1, sq2/2, 1/4, sq2/4, sq2/2, 1/4, sq2/4, 2/4, 1/8])

def test_ApproxPredictorCaching2(graph2):
    sq2 = np.sqrt(2)
    pred = network_hotspot.Predictor(None, graph2)
    pred.kernel = mock.Mock()
    pred.kernel.return_value = 1.0
    pred = network_hotspot.ApproxPredictorCaching(pred)
    
    risks = np.asarray([0]*9, dtype=np.float)
    pred.add_edge(risks, 0, None, 1)
    np.testing.assert_allclose(risks, [1, sq2/2, 1/4, sq2/4, sq2/2, 1/4, sq2/4, 2/4, 1/8])
