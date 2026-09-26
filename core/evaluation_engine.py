"""V5.2 leakage-safe out-of-sample evaluation engine."""
from dataclasses import dataclass,asdict
from math import log,sqrt
from statistics import NormalDist
from typing import Any,Mapping,Sequence,Tuple
EPS=1e-12
@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train:int=60; test_size:int=60; step:int=1; top_k:Tuple[int,...]=(1,3,6)
@dataclass
class FoldResult:
    index:int; train_size:int; actual:Any; ranking:list; probabilities:dict; hit_at_k:dict; logloss:float; brier:float
@dataclass
class EvaluationReport:
    n:int; baseline_top_k:dict; hit_rate_top_k:dict; lift_top_k:dict; logloss:float; brier:float; ece:float; hit_std_error:dict; p_value_vs_baseline:dict; folds:list
    def to_dict(self): return asdict(self)
def normalize_probabilities(raw,candidates):
    vals={c:max(0.0,float(raw.get(c,0.0))) for c in candidates}; total=sum(vals.values())
    if total<=EPS:
        u=1.0/len(candidates) if candidates else 0.0; return {c:u for c in candidates}
    return {c:v/total for c,v in vals.items()}
def ranking_from_probabilities(p): return [k for k,_ in sorted(p.items(),key=lambda kv:(-kv[1],str(kv[0])))]
def _logloss(p,a): return -log(max(EPS,float(p.get(a,0.0))))
def _brier(p,a): return sum((float(v)-(1.0 if k==a else 0.0))**2 for k,v in p.items())
def _se(p,n): return sqrt(max(0.0,p*(1-p)/n)) if n else 0.0
def _pvalue(hits,n,b):
    if n<=0:return 1.0
    se=sqrt(max(EPS,b*(1-b)/n));z=((hits/n)-b)/se
    return 2.0*(1.0-NormalDist().cdf(abs(z)))
def expected_calibration_error(folds,bins=10):
    if not folds:return 0.0
    buckets=[[] for _ in range(bins)]
    for f in folds:
        conf=max(f.probabilities.values()) if f.probabilities else 0.0;ok=int(bool(f.ranking) and f.ranking[0]==f.actual)
        buckets[min(bins-1,int(conf*bins))].append((conf,ok))
    return sum(len(b)/len(folds)*abs(sum(x for x,_ in b)/len(b)-sum(y for _,y in b)/len(b)) for b in buckets if b)
def evaluate_walk_forward(records,candidates,actual_fn,predictor,config=None):
    cfg=config or WalkForwardConfig();candidates=list(candidates)
    if not candidates:raise ValueError("candidate space must not be empty")
    start=cfg.initial_train;end=min(len(records),start+cfg.test_size);folds=[]
    for i in range(start,end,cfg.step):
        train=records[:i];actual=actual_fn(records[i]);raw=predictor(train,candidates)
        if not isinstance(raw,Mapping):raise TypeError("predictor must return candidate->score mapping")
        p=normalize_probabilities(raw,candidates);rank=ranking_from_probabilities(p);hits={k:int(actual in set(rank[:k])) for k in cfg.top_k}
        folds.append(FoldResult(i,len(train),actual,rank,{str(k):float(v) for k,v in p.items()},hits,_logloss(p,actual),_brier(p,actual)))
    n=len(folds);base={k:min(1.0,k/len(candidates)) for k in cfg.top_k};rates={k:(sum(f.hit_at_k[k] for f in folds)/n if n else 0.0) for k in cfg.top_k}
    return EvaluationReport(n,base,rates,{k:rates[k]-base[k] for k in cfg.top_k},sum(f.logloss for f in folds)/n if n else 0.0,sum(f.brier for f in folds)/n if n else 0.0,expected_calibration_error(folds),{k:_se(rates[k],n) for k in cfg.top_k},{k:_pvalue(sum(f.hit_at_k[k] for f in folds),n,base[k]) for k in cfg.top_k},folds)
def evaluate_ranked_walk_forward(records,candidates,actual_fn,rank_predictor,*,initial_train,test_size,step=1,top_k=(1,3,6)):
    candidates=list(candidates);n=len(candidates)
    def predictor(train,cands):
        ranked=list(rank_predictor(train,cands));pos={v:i for i,v in enumerate(ranked)}
        return {v:float(n-pos.get(v,n)) for v in cands}
    return evaluate_walk_forward(records,candidates,actual_fn,predictor,WalkForwardConfig(initial_train,test_size,step,tuple(top_k)))
