from core.v54_research import fit_locked_policy,predict_locked,evaluate_locked_policy

def test_locked_policy_freezes_fit_before_outer():
    inner=[({"a":{"x":.9,"y":.1},"b":{"x":.4,"y":.6}},"x"),
           ({"a":{"x":.8,"y":.2},"b":{"x":.3,"y":.7}},"x")]
    policy=fit_locked_policy(inner,["x","y"],["a","b"])
    outer=[({"a":{"x":.1,"y":.9},"b":{"x":.2,"y":.8}},"y")]
    before=policy.to_dict()
    result=evaluate_locked_policy(policy,outer)
    assert policy.to_dict()==before
    assert result["outer_n"]==1
    assert abs(sum(predict_locked(policy,outer[0][0]).values())-1.0)<1e-9
