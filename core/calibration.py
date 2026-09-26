"""V5.4 dependency-free probability calibration."""
from math import exp,log
from typing import Mapping,Sequence
from .evaluation_engine import normalize_probabilities
EPS=1e-12

def _softmax(logits,temperature):
    vals=[v/max(EPS,temperature) for v in logits];m=max(vals);ex=[exp(v-m) for v in vals];s=sum(ex)
    return [v/s for v in ex]

def fit_temperature(rows:Sequence[tuple[Mapping,object]],candidates:Sequence,grid:Sequence[float]|None=None)->float:
    if not rows:return 1.0
    grid=grid or tuple(0.5+i*0.05 for i in range(31));best_t,best_loss=1.0,float("inf")
    for t in grid:
        loss=0.0
        for raw,actual in rows:
            p=normalize_probabilities(raw,candidates);logits=[log(max(EPS,p[c])) for c in candidates]
            q=dict(zip(candidates,_softmax(logits,t)));loss-=log(max(EPS,q.get(actual,0.0)))
        loss/=len(rows)
        if loss<best_loss:best_t,best_loss=float(t),loss
    return best_t

def apply_temperature(raw:Mapping,candidates:Sequence,temperature:float)->dict:
    p=normalize_probabilities(raw,candidates);logits=[log(max(EPS,p[c])) for c in candidates]
    return dict(zip(candidates,_softmax(logits,max(EPS,float(temperature)))))

def calibration_summary(rows,candidates)->dict:
    from .signal_metrics import metric_summary,reliability_curve
    normalized=[(normalize_probabilities(p,candidates),a) for p,a in rows]
    return {"metrics":metric_summary(normalized),"reliability":reliability_curve(normalized)}
