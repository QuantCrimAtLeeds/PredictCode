import pytest

import open_cp.scripted.analysis as analysis

import io
import numpy as np

@pytest.fixture
def counts_csv_file():
    txt = "Predictor,Start time,End time,Number events,1%,2%,3%,4%\n"
    txt += "Pred1,_,_,5,1,2,3,4\n"
    txt += "Pred1,_,_,7,2,1,5,4\n"
    txt += "Pred2,_,_,10,6,3,8,4\n"
    return io.StringIO(txt)

def test_hit_counts_to_beta(counts_csv_file):
    betas = analysis.hit_counts_to_beta(counts_csv_file)
    
    assert set(betas) == {"Pred1", "Pred2"}
    assert set(betas["Pred1"]) == {1,2,3,4}
    assert betas["Pred1"][1].args == pytest.approx((3, 9))
    assert betas["Pred1"][2].args == pytest.approx((3, 9))
    assert betas["Pred1"][3].args == pytest.approx((8, 4))
    assert betas["Pred1"][4].args == pytest.approx((8, 4))

    assert set(betas["Pred2"]) == {1,2,3,4}
    assert betas["Pred2"][1].args == pytest.approx((6, 4))
    assert betas["Pred2"][2].args == pytest.approx((3, 7))
    assert betas["Pred2"][3].args == pytest.approx((8, 2))
    assert betas["Pred2"][4].args == pytest.approx((4, 6))
    
def test_parse_prediction_key():
    x = analysis.parse_prediction_key("RetroHotspotProvider(Weight=Quartic(bandwidth=50))")
    assert x.name == "RetroHotspotProvider"
    assert x.details == {"Weight":"Quartic(bandwidth=50)"}
    
    x = analysis.parse_prediction_key("RetroHotspotProvider ( Weight = Quartic(bandwidth=50)  )")
    assert x.name == "RetroHotspotProvider"
    assert x.details == {"Weight":"Quartic(bandwidth=50)"}

    x = analysis.parse_prediction_key("ProHotspotCtsProvider(Weight=Classic(sb=400, tb=8), DistanceUnit=150)")
    assert x.name == "ProHotspotCtsProvider"
    assert x.details == {"Weight":"Classic(sb=400, tb=8)", "DistanceUnit":150}

    x = analysis.parse_prediction_key("ProHotspotCtsProvider(Weight=Classic(sb=400, tb=8), DistanceUnit=150.566)")
    assert x.name == "ProHotspotCtsProvider"
    assert x.details == {"Weight":"Classic(sb=400, tb=8)", "DistanceUnit":150.566}
    
    x = analysis.parse_prediction_key("NaiveProvider (CountingGridKernel)")
    assert x.name == "NaiveProvider"
    assert x.details == {"CountingGridKernel":None}
    
def test_parse_key_details():
    details = {"TimeKernel":"ExponentialTimeKernel(Scale=1)",
               "SpaceKernel":"GaussianFixedBandwidthProvider(bandwidth=100)"}
    assert analysis.parse_key_details(details) == {
        "TimeKernel":{"ExponentialTimeKernel":{"Scale":1}},
        "SpaceKernel":{"GaussianFixedBandwidthProvider":{"bandwidth":100}}
        }

    details = {'DistanceUnit': 150, 'Weight': 'Classic(sb=2, tb=6)'}
    assert analysis.parse_key_details(details) == {
        "DistanceUnit":150,
        "Weight":{"Classic":{"sb":2, "tb":6}}
        }
    
def test_compute_betas_means_against_max(counts_csv_file):
    betas = analysis.hit_counts_to_beta(counts_csv_file)
    x, d = analysis.compute_betas_means_against_max(betas)
    np.testing.assert_allclose(x, [1,2,3,4])
    assert set(d) == {"Pred1", "Pred2"}
    np.testing.assert_allclose(d["Pred1"], [5/12, 5/6, 5/6, 1])
    np.testing.assert_allclose(d["Pred2"], [1, 1, 1, 3/5])
    
@pytest.fixture
def counts_csv_file_with_zeros():
    txt = "Predictor,Start time,End time,Number events,1%,2%,3%,4%\n"
    txt += "Pred1,_,_,5,0,0,3,4\n"
    txt += "Pred1,_,_,7,0,1,5,4\n"
    txt += "Pred2,_,_,10,6,3,8,4\n"
    return io.StringIO(txt)

def test_compute_betas_means_against_max_with_zeros(counts_csv_file_with_zeros):
    betas = analysis.hit_counts_to_beta(counts_csv_file_with_zeros)
    x, d = analysis.compute_betas_means_against_max(betas)
    np.testing.assert_allclose(x, [1,2,3,4])
    assert set(d) == {"Pred1", "Pred2"}
    np.testing.assert_allclose(d["Pred1"], [0, 5/18, 5/6, 1])
    np.testing.assert_allclose(d["Pred2"], [1, 1, 1, 3/5])

def test_single_hit_counts_to_beta():
    hit_counts = {
        "a" : {1:(0,3), 2:(1,3)},
        "b" : {1:(1,2), 2:(2,2)}
    }
    out = analysis.single_hit_counts_to_beta(hit_counts)
    assert set(out) == {1, 2}
    assert out[1].args == (1, 4)
    assert out[2].args == (3, 2)
