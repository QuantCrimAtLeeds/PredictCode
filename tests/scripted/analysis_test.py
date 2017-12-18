import pytest

import open_cp.scripted.analysis as analysis

import io

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
    
def test_parse_key_details():
    details = {"TimeKernel":"ExponentialTimeKernel(Scale=1)",
               "SpaceKernel":"GaussianFixedBandwidthProvider(bandwidth=100)"}
    assert analysis.parse_key_details(details) == {
        "TimeKernel":{"ExponentialTimeKernel":{"Scale":1}},
        "SpaceKernel":{"GaussianFixedBandwidthProvider":{"bandwidth":100}}
        }
    