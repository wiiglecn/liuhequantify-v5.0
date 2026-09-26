"""V5.3 nested walk-forward evaluation."""
from dataclasses import dataclass,asdict
from typing import Any
from .evaluation_engine import normalize_probabilities,ranking_from_probabilities

@dataclass(frozen=True)
class NestedConfig:
    outer_initial_train:int=180
    outer_test_size:int=60
    outer_step:int=1
    inner_initial_train:int=90
    inner_test_size:int=30
    final_holdout:int=60
    top_k:tuple=(1,3,6)

@dataclass
class NestedFold:
    index:int
    train_end:int
    selected_policy:str
    inner_score:float
    actual:Any
    ranking:list
    hit_at_k:dict

@dataclass
class NestedReport:
    n_outer:int
    holdout_start:int
    selected_counts:dict
    hit_rate_top_k:dict
    baseline_top_k:dict
    lift_top_k:dict
    folds:list
    holdout_isolated:bool
    def to_dict(self): return asdict(self)

def _score_policy(records,candidates,actual_fn,predictor,initial,test_size,top_k):
    n=max(0,min(test_size,len(records)-initial))
    if n<=0:return 0.0
    hits=[]
    for i in range(initial,initial+n):
        p=normalize_probabilities(predictor(records[:i],candidates),candidates)
        rank=ranking_from_probabilities(p)
        hits.append(int(actual_fn(records[i]) in set(rank[:top_k])))
    return sum(hits)/len(hits) if hits else 0.0

def evaluate_nested(records,candidates,actual_fn,policies,config=None):
    cfg=config or NestedConfig()
    if cfg.final_holdout<=0 or len(records)<=cfg.final_holdout:
        raise ValueError("final_holdout must leave training data")
    holdout_start=len(records)-cfg.final_holdout
    outer_end=holdout_start
    folds=[]
    for i in range(max(cfg.outer_initial_train,0),outer_end,cfg.outer_step):
        train=records[:i]
        best_name=None;best_score=float("-inf")
        for name,predictor in policies.items():
            score=_score_policy(train,candidates,actual_fn,predictor,cfg.inner_initial_train,cfg.inner_test_size,cfg.top_k[0])
            if score>best_score:
                best_name,best_score=name,score
        if best_name is None: continue
        p=normalize_probabilities(policies[best_name](train,candidates),candidates)
        rank=ranking_from_probabilities(p);actual=actual_fn(records[i])
        hits={k:int(actual in set(rank[:k])) for k in cfg.top_k}
        folds.append(NestedFold(i,i,best_name,best_score,actual,rank,hits))
    n=len(folds)
    rates={k:(sum(f.hit_at_k[k] for f in folds)/n if n else 0.0) for k in cfg.top_k}
    base={k:min(1.0,k/len(candidates)) for k in cfg.top_k}
    counts={name:sum(1 for f in folds if f.selected_policy==name) for name in policies}
    return NestedReport(n,holdout_start,counts,rates,base,{k:rates[k]-base[k] for k in cfg.top_k},folds,True)

def evaluate_final_holdout(records,candidates,actual_fn,predictor,holdout_start):
    """Score a sealed holdout using one frozen training prefix.
    predictor is called exactly once, so no holdout observation can update the model.
    """
    if holdout_start<=0 or holdout_start>=len(records):
        raise ValueError("holdout_start must split the dataset")
    p=normalize_probabilities(predictor(records[:holdout_start],candidates),candidates)
    rank=ranking_from_probabilities(p)
    actuals=[actual_fn(r) for r in records[holdout_start:]]
    return {
        "holdout_start":holdout_start,
        "n":len(actuals),
        "ranking":rank,
        "hit_at_k":{k:sum(a in set(rank[:k]) for a in actuals)/len(actuals) for k in (1,3,6)},
        "frozen_training_size":holdout_start,
    }
