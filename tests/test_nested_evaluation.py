from core.nested_evaluation import NestedConfig,evaluate_nested
from core.multiple_testing import bonferroni,benjamini_hochberg

def test_nested_selection_does_not_touch_final_holdout():
    records=list(range(30));seen=[]
    def a(train,c): seen.append(max(train)); return list(c)
    def b(train,c): return list(reversed(c))
    r=evaluate_nested(records,list(range(4)),lambda x:x%4,{"a":a,"b":b},NestedConfig(10,5,1,5,3,5,(1,)))
    assert r.holdout_isolated and all(v<r.holdout_start for v in seen)

def test_multiple_testing():
    x=bonferroni({"a":.01,"b":.20}); assert x["a"]==.02 and x["b"]==.40
    q=benjamini_hochberg({"a":.01,"b":.20}); assert q["a"]<=q["b"]
