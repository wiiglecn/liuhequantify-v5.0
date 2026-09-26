from core.signal_metrics import rank_ic,metric_summary
from core.calibration import fit_temperature,apply_temperature
from core.signal_analysis import signal_correlation,redundancy_groups
from core.ensemble_weighting import normalize_weights,combine_signal_probabilities

def test_metrics_and_rank_ic():
    rows=[({"a":.8,"b":.2},"a"),({"a":.7,"b":.3},"a")]
    assert metric_summary(rows,top_k=(1,))["hit_at_1"]==1.0
    assert rank_ic({"a":.9,"b":.1},"a",["a","b"])>0

def test_temperature_is_train_only_and_deterministic():
    rows=[({"a":.9,"b":.1},"a"),({"a":.8,"b":.2},"a")]
    t=fit_temperature(rows,["a","b"]);p=apply_temperature({"a":.9,"b":.1},["a","b"],t)
    assert t>0 and abs(sum(p.values())-1.0)<1e-9

def test_redundancy_and_weights():
    corr=signal_correlation({"a":[1,2,3],"b":[2,4,6],"c":[3,1,2]})
    assert abs(corr["a"]["b"])>.99
    assert redundancy_groups({"a":[1,2,3],"b":[2,4,6]})
    assert normalize_weights({"a":2,"b":1})["a"]>normalize_weights({"a":2,"b":1})["b"]
    p=combine_signal_probabilities({"a":{"x":.8,"y":.2},"b":{"x":.2,"y":.8}},["x","y"],{"a":1,"b":0})
    assert p["x"]>.7
