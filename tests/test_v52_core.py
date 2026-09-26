from core.evaluation_engine import evaluate_ranked_walk_forward
from core.signal_registry import SignalRegistry,SignalSpec
def test_oos_future_isolation():
    records=list(range(20));seen=[]
    def ranker(train,cands):seen.append((len(train),max(train)));return list(reversed(cands))
    r=evaluate_ranked_walk_forward(records,list(range(4)),lambda x:x%4,ranker,initial_train=10,test_size=5)
    assert r.n==5 and all(n==i and m==i-1 for (n,m),i in zip(seen,range(10,15)))
def test_registry_upsert():
    reg=SignalRegistry();reg.register(SignalSpec("x","v1","t","d",lambda:{}));assert reg.names()==["x"]
    reg.upsert(SignalSpec("x","v2","t","d2",lambda:{},enabled=False));assert reg.names()==[] and reg.get("x").version=="v2"
